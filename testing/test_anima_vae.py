"""
Anima VAE round-trip diagnostic.

Loads qwen_image_vae.safetensors with _remap_vae_keys, encodes an image,
decodes it back, and saves both side-by-side so you can see whether the
VAE itself is working correctly (independent of the DiT).

Usage:
    python testing/test_anima_vae.py --vae_path /path/to/qwen_image_vae.safetensors
    python testing/test_anima_vae.py --vae_path /path/to/qwen_image_vae.safetensors --image /path/to/test.png
    python testing/test_anima_vae.py --vae_path /path/to/qwen_image_vae.safetensors --image /path/to/test.png --size 512
"""

import argparse
import os
import sys

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _remap_vae_keys(state_dict: dict) -> dict:
    """Copied from anima_model.py — remap checkpoint keys to AutoencoderKLQwenImage names."""
    UPSAMPLE_MAP = {
        0: (0, False, 0), 1: (0, False, 1), 2: (0, False, 2),
        3: (0, True,  0),
        4: (1, False, 0), 5: (1, False, 1), 6: (1, False, 2),
        7: (1, True,  0),
        8: (2, False, 0), 9: (2, False, 1), 10: (2, False, 2),
        11: (2, True,  0),
        12: (3, False, 0), 13: (3, False, 1), 14: (3, False, 2),
    }

    def _resnet_sub(sub):
        if sub.startswith("residual.0."): return "norm1."  + sub[len("residual.0."):]
        if sub.startswith("residual.2."): return "conv1."  + sub[len("residual.2."):]
        if sub.startswith("residual.3."): return "norm2."  + sub[len("residual.3."):]
        if sub.startswith("residual.6."): return "conv2."  + sub[len("residual.6."):]
        if sub.startswith("shortcut."):   return "conv_shortcut." + sub[len("shortcut."):]
        return sub

    def _middle_key(prefix, rest):
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


def load_anima_vae(vae_path: str, dtype=torch.float32, device="cpu"):
    from diffusers import AutoencoderKLQwenImage

    print(f"\n=== Loading VAE from {vae_path} ===")
    vae = AutoencoderKLQwenImage.from_pretrained(
        "Qwen/Qwen-Image", subfolder="vae", torch_dtype=dtype
    )
    expected_keys = set(vae.state_dict().keys())

    raw_sd = load_file(vae_path, device="cpu")
    remapped = _remap_vae_keys(raw_sd)
    remapped_keys = set(remapped.keys())

    missing  = expected_keys - remapped_keys
    unexpected = remapped_keys - expected_keys

    print(f"  Expected keys : {len(expected_keys)}")
    print(f"  Remapped keys : {len(remapped_keys)}")
    if missing:
        print(f"  MISSING  ({len(missing)}): {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        print(f"  Missing  : 0  OK")
    if unexpected:
        print(f"  UNEXPECTED ({len(unexpected)}): {sorted(unexpected)[:10]}{'...' if len(unexpected) > 10 else ''}")
    else:
        print(f"  Unexpected: 0  OK")

    vae.load_state_dict(remapped, strict=True)
    vae = vae.to(device, dtype=dtype)
    vae.eval()
    print("  VAE loaded successfully.")

    cfg = vae.config
    print(f"\n  z_dim        : {cfg.z_dim}")
    print(f"  latents_mean : {cfg.latents_mean}")
    print(f"  latents_std  : {cfg.latents_std}")

    return vae


def make_test_image(size: int) -> Image.Image:
    """Create a synthetic test image with distinct colour regions."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    half = size // 2
    draw.rectangle([0,    0,    half-1, half-1], fill=(220, 60,  60))   # red
    draw.rectangle([half, 0,    size-1, half-1], fill=(60,  180, 60))   # green
    draw.rectangle([0,    half, half-1, size-1], fill=(60,  80,  220))  # blue
    draw.rectangle([half, half, size-1, size-1], fill=(220, 200, 60))   # yellow
    # draw a white cross through the centre
    draw.line([(half, 0), (half, size-1)], fill=(255, 255, 255), width=3)
    draw.line([(0, half), (size-1, half)], fill=(255, 255, 255), width=3)
    return img


def pil_to_tensor(img: Image.Image, device, dtype) -> torch.Tensor:
    """PIL → (1, 3, H, W) float tensor in [-1, 1]."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device, dtype=dtype)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) float tensor in [-1, 1] → PIL."""
    arr = t[0].float().permute(1, 2, 0).clamp(-1, 1).add(1).mul(127.5).byte().cpu().numpy()
    return Image.fromarray(arr)


def encode(vae, img_t: torch.Tensor) -> torch.Tensor:
    """Encode with the same normalisation as QwenImageModel.encode_images."""
    z_dim = vae.config.z_dim
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(img_t.device, img_t.dtype)
    )
    latents_std_inv = (
        (1.0 / torch.tensor(vae.config.latents_std))
        .view(1, z_dim, 1, 1, 1)
        .to(img_t.device, img_t.dtype)
    )

    x5d = img_t.unsqueeze(2)                                # (1, 3, 1, H, W)
    with torch.no_grad():
        latents = vae.encode(x5d).latent_dist.sample()      # (1, z_dim, 1, H/16, W/16)

    latents = (latents - latents_mean) * latents_std_inv    # normalise
    latents = latents.squeeze(2)                             # (1, z_dim, H/16, W/16)
    return latents


def decode(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode with the same denormalisation as AnimaModel._decode_latents_to_pil."""
    z_dim = vae.config.z_dim
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, z_dim, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, z_dim, 1, 1)
        .to(latents.device, latents.dtype)
    )

    raw = latents * latents_std + latents_mean               # denormalise
    raw5d = raw.unsqueeze(2)                                 # (1, z_dim, 1, H/16, W/16)
    with torch.no_grad():
        decoded = vae.decode(raw5d).sample.squeeze(2)        # (1, 3, H, W)
    return decoded


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a.float(), b.float()).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(4.0 / mse)   # pixel range is [-1,1] so max^2 = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", required=True, help="Path to qwen_image_vae.safetensors")
    parser.add_argument("--image", default=None, help="Path to a test image (optional)")
    parser.add_argument("--size", type=int, default=512, help="Resize / synthetic image size (must be divisible by 16)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="testing/anima_vae_roundtrip.png", help="Where to save the side-by-side result")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype  = dtype_map[args.dtype]
    device = torch.device(args.device)

    size = (args.size // 16) * 16   # must be divisible by 16 for the VAE
    if size != args.size:
        print(f"  Rounding size to {size} (must be divisible by 16)")

    vae = load_anima_vae(args.vae_path, dtype=dtype, device=device)

    if args.image:
        src = Image.open(args.image).convert("RGB").resize((size, size), Image.LANCZOS)
        print(f"\n=== Using image: {args.image} (resized to {size}×{size}) ===")
    else:
        src = make_test_image(size)
        print(f"\n=== Using synthetic test image ({size}×{size}) ===")

    img_t = pil_to_tensor(src, device, dtype)

    print("\n=== Encoding ===")
    latents = encode(vae, img_t)
    print(f"  Latent shape : {tuple(latents.shape)}")
    print(f"  Latent mean  : {latents.float().mean():.4f}")
    print(f"  Latent std   : {latents.float().std():.4f}")
    print(f"  Latent min   : {latents.float().min():.4f}")
    print(f"  Latent max   : {latents.float().max():.4f}")
    expected_std = latents.float().std().item()
    if expected_std < 0.5 or expected_std > 3.0:
        print("  WARNING: Latent std is outside [0.5, 3.0] -- normalisation may be wrong")

    print("\n=== Decoding ===")
    decoded_t = decode(vae, latents)
    print(f"  Decoded shape: {tuple(decoded_t.shape)}")
    print(f"  Decoded mean : {decoded_t.float().mean():.4f}")
    print(f"  Decoded std  : {decoded_t.float().std():.4f}")
    print(f"  Decoded min  : {decoded_t.float().min():.4f}")
    print(f"  Decoded max  : {decoded_t.float().max():.4f}")

    score = psnr(img_t.cpu(), decoded_t.cpu())
    print(f"\n  PSNR (encode->decode round-trip): {score:.2f} dB")
    if score > 25:
        print("  PASS: VAE round-trip looks healthy (PSNR > 25 dB)")
    elif score > 15:
        print("  WARN: Moderate quality loss -- check normalisation constants")
    else:
        print("  FAIL: Very poor reconstruction -- VAE weights or normalisation are likely wrong")

    recon = tensor_to_pil(decoded_t)
    combined = Image.new("RGB", (size * 2, size))
    combined.paste(src,   (0,    0))
    combined.paste(recon, (size, 0))

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    combined.save(out_path)
    print(f"\n  Saved side-by-side comparison -> {out_path}")
    print("  Left = original, Right = VAE round-trip reconstruction\n")


if __name__ == "__main__":
    main()
