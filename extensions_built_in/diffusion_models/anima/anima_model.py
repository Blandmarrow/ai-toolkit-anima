import os
from typing import TYPE_CHECKING, List, Optional

import torch
import yaml
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, Qwen2Tokenizer, T5TokenizerFast
import huggingface_hub

from toolkit.samplers.custom_flowmatch_sampler import calculate_shift

from toolkit.basic import flush
from toolkit.memory_management import MemoryManager
from toolkit.prompt_utils import PromptEmbeds
from toolkit.accelerator import unwrap_model
from toolkit.util.quantize import get_qtype, quantize, quantize_model
from optimum.quanto import freeze

from ..qwen_image.qwen_image import QwenImageModel
from .model import Anima, build_anima

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

ANIMA_HF_REPO = "circlestone-labs/Anima"
ANIMA_TRANSFORMER_FILE = "split_files/diffusion_models/anima-preview3-base.safetensors"
ANIMA_TE_FILE = "split_files/text_encoders/qwen_3_06b_base.safetensors"
ANIMA_VAE_FILE = "split_files/vae/qwen_image_vae.safetensors"
QWEN3_CONFIG_REPO = "Qwen/Qwen3-0.6B"
QWEN_IMAGE_REPO = "Qwen/Qwen-Image"
T5_TOKENIZER_REPO = "google/t5-v1_1-base"   # 32128-vocab T5 tokenizer


def _remap_vae_keys(state_dict: dict) -> dict:
    """Remap original Anima VAE checkpoint keys to AutoencoderKLQwenImage names.

    The checkpoint uses a flat naming scheme (encoder.downsamples.X, encoder.middle.X,
    decoder.upsamples.X) while diffusers expects hierarchical names (encoder.down_blocks.X,
    encoder.mid_block, decoder.up_blocks.X.resnets/upsamplers).
    """
    # Maps flat upsample index → (block_idx, is_upsampler, inner_idx)
    UPSAMPLE_MAP = {
        0: (0, False, 0), 1: (0, False, 1), 2: (0, False, 2),
        3: (0, True,  0),
        4: (1, False, 0), 5: (1, False, 1), 6: (1, False, 2),
        7: (1, True,  0),
        8: (2, False, 0), 9: (2, False, 1), 10: (2, False, 2),
        11: (2, True,  0),
        12: (3, False, 0), 13: (3, False, 1), 14: (3, False, 2),
    }

    def _resnet_sub(sub: str) -> str:
        # Sequential indices: 0=norm1, 2=conv1, 3=norm2, 6=conv2 (odd indices are activations/dropout)
        if sub.startswith("residual.0."):
            return "norm1." + sub[len("residual.0."):]
        if sub.startswith("residual.2."):
            return "conv1." + sub[len("residual.2."):]
        if sub.startswith("residual.3."):
            return "norm2." + sub[len("residual.3."):]
        if sub.startswith("residual.6."):
            return "conv2." + sub[len("residual.6."):]
        if sub.startswith("shortcut."):
            return "conv_shortcut." + sub[len("shortcut."):]
        return sub  # resample.*, time_conv.* pass through unchanged

    def _middle_key(prefix: str, rest: str) -> str:
        mid_idx_str, sub = rest.split(".", 1)
        mid_idx = int(mid_idx_str)
        if mid_idx == 1:
            return f"{prefix}.mid_block.attentions.0.{sub}"
        resnet_idx = 0 if mid_idx == 0 else 1
        return f"{prefix}.mid_block.resnets.{resnet_idx}." + _resnet_sub(sub)

    remapped = {}
    for k, v in state_dict.items():
        new_k = k

        if k.startswith("conv1."):
            new_k = "quant_conv." + k[len("conv1."):]
        elif k.startswith("conv2."):
            new_k = "post_quant_conv." + k[len("conv2."):]
        elif k.startswith("encoder.conv1."):
            new_k = "encoder.conv_in." + k[len("encoder.conv1."):]
        elif k.startswith("encoder.head.0."):
            new_k = "encoder.norm_out." + k[len("encoder.head.0."):]
        elif k.startswith("encoder.head.2."):
            new_k = "encoder.conv_out." + k[len("encoder.head.2."):]
        elif k.startswith("encoder.downsamples."):
            rest = k[len("encoder.downsamples."):]
            idx, sub = rest.split(".", 1)
            new_k = f"encoder.down_blocks.{idx}." + _resnet_sub(sub)
        elif k.startswith("encoder.middle."):
            new_k = _middle_key("encoder", k[len("encoder.middle."):])
        elif k.startswith("decoder.conv1."):
            new_k = "decoder.conv_in." + k[len("decoder.conv1."):]
        elif k.startswith("decoder.head.0."):
            new_k = "decoder.norm_out." + k[len("decoder.head.0."):]
        elif k.startswith("decoder.head.2."):
            new_k = "decoder.conv_out." + k[len("decoder.head.2."):]
        elif k.startswith("decoder.middle."):
            new_k = _middle_key("decoder", k[len("decoder.middle."):])
        elif k.startswith("decoder.upsamples."):
            rest = k[len("decoder.upsamples."):]
            idx_str, sub = rest.split(".", 1)
            block_idx, is_upsampler, inner_idx = UPSAMPLE_MAP[int(idx_str)]
            if is_upsampler:
                new_k = f"decoder.up_blocks.{block_idx}.upsamplers.{inner_idx}.{sub}"
            else:
                new_k = f"decoder.up_blocks.{block_idx}.resnets.{inner_idx}." + _resnet_sub(sub)

        remapped[new_k] = v

    return remapped


def _load_anima_transformer(path: str, dtype) -> Anima:
    """Load Anima weights from a safetensors file into a freshly built Anima instance."""
    model = build_anima(dtype=dtype, device="cpu")
    state_dict = load_file(path, device="cpu")

    # Keys are stored with a "net." prefix; strip it before loading
    stripped = {}
    for k, v in state_dict.items():
        new_k = k[4:] if k.startswith("net.") else k
        stripped[new_k] = v

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:
        raise RuntimeError(
            f"Anima transformer is missing {len(missing)} keys after loading {path!r}. "
            f"First few: {missing[:5]}"
        )
    if unexpected:
        # Non-fatal — could be future-proofing keys
        import logging
        logging.warning(f"Anima: {len(unexpected)} unexpected keys (ignored)")

    return model.to(dtype)


class AnimaModel(QwenImageModel):
    arch = "anima"
    _qwen_image_keep_visual = False

    def __init__(self, device, model_config, dtype="bf16", custom_pipeline=None, noise_scheduler=None, **kwargs):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.target_lora_modules = ["Anima"]
        self._t5_tokenizer: Optional[T5TokenizerFast] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Anima model")

        transformer_path = self.model_config.name_or_path
        te_path = self.model_config.extras_name_or_path
        vae_path = self.model_config.vae_path

        # --- Transformer ---
        self.print_and_status_update("Loading transformer")
        if not transformer_path.endswith(".safetensors") and not os.path.exists(transformer_path):
            transformer_path = huggingface_hub.hf_hub_download(
                repo_id=ANIMA_HF_REPO, filename=ANIMA_TRANSFORMER_FILE
            )

        if transformer_path.endswith(".safetensors"):
            transformer = _load_anima_transformer(transformer_path, dtype)
        else:
            raise ValueError(
                "Anima transformer must be a .safetensors file or a HF repo ID. "
                f"Got: {transformer_path!r}"
            )

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        if self.model_config.low_vram:
            transformer.to("cpu")
        flush()

        # --- Text Encoder (Qwen3-0.6B CausalLM, text-only) ---
        self.print_and_status_update("Loading text encoder")
        tokenizer_source = te_path

        if te_path is None:
            te_safetensors = huggingface_hub.hf_hub_download(
                repo_id=ANIMA_HF_REPO, filename=ANIMA_TE_FILE
            )
            text_encoder = AutoModelForCausalLM.from_pretrained(QWEN3_CONFIG_REPO, torch_dtype=dtype)
            text_encoder.load_state_dict(load_file(te_safetensors, device="cpu"), strict=True)
            tokenizer_source = QWEN3_CONFIG_REPO
        elif te_path.endswith(".safetensors"):
            text_encoder = AutoModelForCausalLM.from_pretrained(QWEN3_CONFIG_REPO, torch_dtype=dtype)
            text_encoder.load_state_dict(load_file(te_path, device="cpu"), strict=True)
            tokenizer_source = QWEN3_CONFIG_REPO
        else:
            text_encoder = AutoModelForCausalLM.from_pretrained(te_path, torch_dtype=dtype)

        qwen3_tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_source)
        self.processor = None

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        # --- T5 tokenizer (for LLM adapter) ---
        self.print_and_status_update("Loading T5 tokenizer")
        self._t5_tokenizer = T5TokenizerFast.from_pretrained(T5_TOKENIZER_REPO)

        # --- VAE ---
        self.print_and_status_update("Loading VAE")
        from diffusers import AutoencoderKLQwenImage

        if vae_path is None:
            resolved_vae = huggingface_hub.hf_hub_download(
                repo_id=ANIMA_HF_REPO, filename=ANIMA_VAE_FILE
            )
        else:
            resolved_vae = vae_path

        if resolved_vae.endswith(".safetensors"):
            vae = AutoencoderKLQwenImage.from_pretrained(
                QWEN_IMAGE_REPO, subfolder="vae", torch_dtype=dtype
            )
            raw_sd = load_file(resolved_vae, device="cpu")
            vae.load_state_dict(_remap_vae_keys(raw_sd), strict=True)
        else:
            vae = AutoencoderKLQwenImage.from_pretrained(
                resolved_vae, subfolder="vae" if not os.path.isdir(os.path.join(resolved_vae, "vae")) else None,
                torch_dtype=dtype,
            )

        # --- Scheduler ---
        self.noise_scheduler = QwenImageModel.get_train_scheduler()
        self.print_and_status_update("Assembling Anima pipeline")

        text_encoder_list = [text_encoder]
        tokenizer_list = [qwen3_tokenizer]

        if not self.low_vram:
            transformer.to(self.device_torch)

        flush()
        text_encoder_list[0].to(self.device_torch)
        text_encoder_list[0].requires_grad_(False)
        text_encoder_list[0].eval()
        flush()

        self.vae = vae
        self.text_encoder = text_encoder_list
        self.tokenizer = tokenizer_list
        self.model = transformer
        # pipeline attribute expected by base class helpers
        self.pipeline = _AnimaPipelineShim(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=qwen3_tokenizer,
            vae=vae,
            scheduler=self.noise_scheduler,
        )
        self.print_and_status_update("Anima model loaded")

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        """
        Encode text for Anima:
          1. Qwen3-0.6B → hidden states (B, S_qwen, 1024)
          2. T5 tokenizer → token IDs (B, S_t5)
          3. LLM adapter(qwen3_hs, t5_ids) → context (B, S_t5, 1024)
        Returns PromptEmbeds with text_embeds = context.
        """
        te = self.text_encoder[0]
        if te.device != self.device_torch:
            te.to(self.device_torch)

        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        # Qwen3 cannot handle a 0-length token sequence; replace empty prompts with a space
        prompts = [p if p.strip() else " " for p in prompts]

        with torch.no_grad():
            # Qwen3 encoding
            qwen3_inputs = self.tokenizer[0](
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device_torch)

            qwen3_out = te(
                **qwen3_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            # Use last hidden state of the last layer
            qwen3_hs = qwen3_out.hidden_states[-1].to(self.torch_dtype)   # (B, S, 1024)

            # T5 tokenization
            t5_inputs = self._t5_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device_torch)
            t5_ids = t5_inputs.input_ids   # (B, S_t5)

            # LLM adapter — part of the transformer, call without gradient
            anima_net = unwrap_model(self.model)
            # Ensure the llm_adapter is on the right device
            if next(anima_net.llm_adapter.parameters()).device != self.device_torch:
                anima_net.llm_adapter.to(self.device_torch)

            context = anima_net.llm_adapter(
                source_hidden_states=qwen3_hs,
                target_input_ids=t5_ids,
            ).to(self.torch_dtype)  # (B, S_t5, 1024)

            # The model was trained with 512-length context (ComfyUI preprocess_text_embeds
            # always pads to 512 with zeros). Match that distribution here.
            if context.shape[1] < 512:
                context = torch.nn.functional.pad(context, (0, 0, 0, 512 - context.shape[1]))

        pe = PromptEmbeds(context)
        pe.attention_mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.int64)
        return pe

    # ------------------------------------------------------------------
    # Noise prediction
    # ------------------------------------------------------------------

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)

        # latent_model_input: (B, 16, H, W) → (B, 16, 1, H, W)
        x = latent_model_input.to(self.device_torch, self.torch_dtype).unsqueeze(2)
        context = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)

        # Normalize timestep from [0, 1000] scheduler range to [0, 1] continuous flow-matching range
        t_01 = (timestep / 1000.0).to(self.device_torch, self.torch_dtype)
        noise_pred = self.model(x, t_01, context)   # (B, 16, 1, H, W)
        return noise_pred.squeeze(2)                 # (B, 16, H, W)

    # ------------------------------------------------------------------
    # Generation (inference)
    # ------------------------------------------------------------------

    def get_generation_pipeline(self):
        return self.pipeline

    def generate_single_image(self, pipeline, gen_config, conditional_embeds, unconditional_embeds, generator, extra):
        height = gen_config.height
        width = gen_config.width
        num_steps = gen_config.num_inference_steps
        guidance_scale = gen_config.guidance_scale
        do_cfg = guidance_scale > 1.0

        scheduler = QwenImageModel.get_train_scheduler()

        # AutoencoderKLQwenImage uses 16× spatial downsampling
        vae_sf = 16
        num_channels = self.vae.config.z_dim
        latent_h = height // vae_sf
        latent_w = width // vae_sf

        # Dynamic shift requires mu computed from the latent sequence length
        image_seq_len = latent_h * latent_w
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 8192),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 0.9),
        )
        scheduler.set_timesteps(num_steps, device=self.device_torch, mu=mu)

        if generator is not None and generator.device != torch.device(self.device_torch):
            seed = generator.initial_seed()
            generator = torch.Generator(device=self.device_torch).manual_seed(seed)

        if gen_config.latents is not None:
            latents = gen_config.latents.to(self.device_torch, dtype=self.torch_dtype)
        else:
            latents = torch.randn(
                (1, num_channels, latent_h, latent_w),
                generator=generator,
                device=self.device_torch,
                dtype=self.torch_dtype,
            )

        latents = latents * scheduler.init_noise_sigma

        self.model.to(self.device_torch)

        for t in scheduler.timesteps:
            if do_cfg:
                latent_input = torch.cat([latents] * 2)
                ue = unconditional_embeds.text_embeds.to(self.device_torch, self.torch_dtype)
                ce = conditional_embeds.text_embeds.to(self.device_torch, self.torch_dtype)
                # Pad to the same sequence length if necessary
                if ue.shape[1] != ce.shape[1]:
                    max_len = max(ue.shape[1], ce.shape[1])
                    ue = torch.nn.functional.pad(ue, (0, 0, 0, max_len - ue.shape[1]))
                    ce = torch.nn.functional.pad(ce, (0, 0, 0, max_len - ce.shape[1]))
                context = torch.cat([ue, ce], dim=0)
            else:
                latent_input = latents
                context = conditional_embeds.text_embeds.to(self.device_torch, self.torch_dtype)

            latent_input = scheduler.scale_model_input(latent_input, t)
            # Normalize to [0, 1] continuous-time range expected by the model
            t_01 = t.float().item() / 1000.0
            ts = torch.tensor([t_01] * latent_input.shape[0], device=self.device_torch, dtype=self.torch_dtype)

            with torch.no_grad():
                x5d = latent_input.unsqueeze(2)
                noise_pred = self.model(x5d, ts, context).squeeze(2)

            if t == scheduler.timesteps[0]:
                print(f"[Anima DEBUG] context shape={context.shape}, mean={context.float().mean():.4f}, std={context.float().std():.4f}")
                print(f"[Anima DEBUG] noise_pred: mean={noise_pred.float().mean():.4f}, std={noise_pred.float().std():.4f}, abs_max={noise_pred.float().abs().max():.4f}")
                print(f"[Anima DEBUG] latents before step: mean={latents.float().mean():.4f}, std={latents.float().std():.4f}")

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return self._decode_latents_to_pil(latents)

    def _decode_latents_to_pil(self, latents: torch.Tensor) -> Image.Image:
        """Reverse VAE normalisation and decode latents to a PIL image."""
        self.vae.to(self.device_torch)
        self.vae.eval()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1)
            .to(latents.device, latents.dtype)
        )
        vae_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1)
            .to(latents.device, latents.dtype)
        )
        # Reverse normalisation: raw = normalised * vae_std + mean
        raw = latents * vae_std + latents_mean   # (B, C, H, W)

        raw_5d = raw.unsqueeze(2)   # add temporal dim → (B, C, 1, H, W)
        with torch.no_grad():
            decoded = self.vae.decode(raw_5d).sample.squeeze(2)   # (B, 3, H, W)

        # Convert to uint8 PIL image (float32 in [-1, 1])
        img_np = decoded[0].float().permute(1, 2, 0).clamp(-1, 1).add(1).mul(127.5).byte().cpu().numpy()
        return Image.fromarray(img_np)

    # ------------------------------------------------------------------
    # Model-saving
    # ------------------------------------------------------------------

    def save_model(self, output_path, meta, save_dtype):
        from safetensors.torch import save_file

        anima_net = unwrap_model(self.model)
        # Re-add "net." prefix to match the original file format
        state_dict = {f"net.{k}": v.to(save_dtype) for k, v in anima_net.state_dict().items()}
        os.makedirs(output_path, exist_ok=True)
        save_file(state_dict, os.path.join(output_path, "anima_transformer.safetensors"))

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    # ------------------------------------------------------------------
    # Misc overrides
    # ------------------------------------------------------------------

    def get_base_model_version(self):
        return "anima"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["blocks"]

    def get_bucket_divisibility(self):
        # VAE 16× downsampling + patch_spatial=2 → divisible by 32
        return 32

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()


class _AnimaPipelineShim:
    """
    Minimal pipeline-like object to satisfy base-class expectations
    (e.g. set_progress_bar_config, .transformer attribute checks).
    """

    def __init__(self, transformer, text_encoder, tokenizer, vae, scheduler):
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.scheduler = scheduler

    def set_progress_bar_config(self, **kwargs):
        pass
