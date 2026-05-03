"""
Anima diffusion model architecture.
Ported from ComfyUI (comfy/ldm/cosmos/predict2.py and comfy/ldm/anima/model.py)
without ComfyUI dependencies.
"""

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# 3-D RoPE positional embedding (Cosmos Predict2 style)
# ---------------------------------------------------------------------------

class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        device=None,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.enable_fps_modulation = enable_fps_modulation

        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t

        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2, device=device)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2, device=device)[: (dim_t // 2)].float() / dim_t,
            persistent=False,
        )

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def forward(self, x_B_T_H_W_C, fps=None, device=None, dtype=None):
        return self.generate_embeddings(x_B_T_H_W_C.shape, fps=fps, device=device, dtype=dtype)

    def generate_embeddings(
        self,
        B_T_H_W_C,
        fps=None,
        h_ntk_factor=None,
        w_ntk_factor=None,
        t_ntk_factor=None,
        device=None,
        dtype=None,
    ):
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.to(device=device))
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.to(device=device))
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.to(device=device))

        B, T, H, W, _ = B_T_H_W_C
        seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)

        half_emb_h = torch.outer(seq[:H].to(device=device), h_spatial_freqs)
        half_emb_w = torch.outer(seq[:W].to(device=device), w_spatial_freqs)

        if fps is None or not self.enable_fps_modulation:
            half_emb_t = torch.outer(seq[:T].to(device=device), temporal_freqs)
        else:
            fps_val = fps if isinstance(fps, (int, float)) else fps.float().mean().item()
            half_emb_t = torch.outer(
                seq[:T].to(device=device) / fps_val * self.base_fps, temporal_freqs
            )

        half_emb_h = torch.stack(
            [torch.cos(half_emb_h), -torch.sin(half_emb_h), torch.sin(half_emb_h), torch.cos(half_emb_h)], dim=-1
        )
        half_emb_w = torch.stack(
            [torch.cos(half_emb_w), -torch.sin(half_emb_w), torch.sin(half_emb_w), torch.cos(half_emb_w)], dim=-1
        )
        half_emb_t = torch.stack(
            [torch.cos(half_emb_t), -torch.sin(half_emb_t), torch.sin(half_emb_t), torch.cos(half_emb_t)], dim=-1
        )

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d x -> t h w d x", h=H, w=W),
                repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W),
                repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H),
            ],
            dim=-2,
        )

        return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


# ---------------------------------------------------------------------------
# MiniTrainDIT components (from comfy/ldm/cosmos/predict2.py)
# ---------------------------------------------------------------------------

def _apply_rotary_pos_emb_dit(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Cosmos-style 2x2 rotation-matrix RoPE used in MiniTrainDIT."""
    t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
    t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
    t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
    return t_out


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None, **kwargs):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.ndim > 3:
            x = x.reshape(orig_shape[0], -1, orig_shape[-1])
        out = self.layer2(self.activation(self.layer1(x)))
        return out.reshape(*orig_shape[:-1], out.shape[-1])


class DiTAttention(nn.Module):
    """Multi-head attention used inside MiniTrainDIT blocks."""

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(head_dim)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
    ):
        q = self.q_proj(x)
        ctx = x if context is None else context
        k = self.k_proj(ctx)
        v = self.v_proj(ctx)
        q, k, v = [rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim) for t in (q, k, v)]

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        if self.is_selfattn and rope_emb is not None:
            q = _apply_rotary_pos_emb_dit(q, rope_emb)
            k = _apply_rotary_pos_emb_dit(k, rope_emb)

        return q, k, v

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: (B, S, H, D) → transpose to (B, H, S, D) for SDPA
        B, S, H, D = q.shape
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t)  # (B, H, S, D)
        out = out.transpose(1, 2).reshape(B, S, H * D)        # (B, S, inner_dim)
        return self.output_dropout(self.output_proj(out))

    def forward(self, x, context=None, rope_emb=None, **kwargs):
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(q, k, v)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return rearrange(emb, "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_adaln_lora: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.use_adaln_lora = use_adaln_lora
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora, device=device, dtype=dtype)
        self.activation = nn.SiLU()
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False, device=device, dtype=dtype)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, sample: torch.Tensor):
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        if self.use_adaln_lora:
            return sample, emb      # (emb_B_T_D, adaln_lora_B_T_3D)
        return emb, None


class PatchEmbed(nn.Module):
    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size,
                out_channels,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        # Rearrange into patch tokens: (B, T', H', W', C_flat)
        x = self.proj[0](x)
        B, T2, H2, W2, Cf = x.shape
        # Flatten spatial/temporal dims so quanto's qbytes_mm (max 3D) can handle it
        x = self.proj[1](x.reshape(B, T2 * H2 * W2, Cf))
        return x.reshape(B, T2, H2, W2, -1)


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False, device=device, dtype=dtype),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False, device=device, dtype=dtype),
            )

    def forward(self, x_B_T_H_W_D, emb_B_T_D, adaln_lora_B_T_3D=None):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift, scale = (
                self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift, scale = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift = rearrange(shift, "b t d -> b t 1 1 d")
        scale = rearrange(scale, "b t d -> b t 1 1 d")
        x = self.layer_norm(x_B_T_H_W_D) * (1 + scale) + shift
        B, T, H, W, D = x.shape
        return self.linear(x.reshape(B, T * H * W, D)).reshape(B, T, H, W, -1)


class Block(nn.Module):
    """Transformer block with AdaLN-LoRA modulation (MiniTrainDIT style)."""

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = DiTAttention(x_dim, None, num_heads, x_dim // num_heads, device=device, dtype=dtype)

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = DiTAttention(x_dim, context_dim, num_heads, x_dim // num_heads, device=device, dtype=dtype)

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio), device=device, dtype=dtype)

        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype)
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype)
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype)
            )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_sa, scale_sa, gate_sa = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            shift_ca, scale_ca, gate_ca = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            shift_ml, scale_ml, gate_ml = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_ml, scale_ml, gate_ml = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        def to_1_1(t):
            return rearrange(t, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        def adaLN(x, norm, scale, shift):
            return norm(x) * (1 + to_1_1(scale)) + to_1_1(shift)

        # Self-attention
        normed = adaLN(x_B_T_H_W_D, self.layer_norm_self_attn, scale_sa, shift_sa)
        sa_out = rearrange(
            self.self_attn(rearrange(normed, "b t h w d -> b (t h w) d"), rope_emb=rope_emb_L_1_1_D),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + to_1_1(gate_sa) * sa_out

        # Cross-attention
        normed = adaLN(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_ca, shift_ca)
        ca_out = rearrange(
            self.cross_attn(rearrange(normed, "b t h w d -> b (t h w) d"), crossattn_emb, rope_emb=rope_emb_L_1_1_D),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + to_1_1(gate_ca) * ca_out

        # MLP
        normed = adaLN(x_B_T_H_W_D, self.layer_norm_mlp, scale_ml, shift_ml)
        x_B_T_H_W_D = x_B_T_H_W_D + to_1_1(gate_ml) * self.mlp(normed)

        return x_B_T_H_W_D


class MiniTrainDIT(nn.Module):
    """
    Cosmos Predict2 MiniTrainDIT — a DiT-style video/image transformer with 3-D RoPE
    and optional AdaLN-LoRA modulation.
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        rope_enable_fps_modulation: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.rope_enable_fps_modulation = rope_enable_fps_modulation

        self._build_pos_embed(device=device)

        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim

        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(
                model_channels,
                model_channels,
                use_adaln_lora=use_adaln_lora,
                device=device,
                dtype=dtype,
            ),
        )

        emb_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=emb_in_channels,
            out_channels=model_channels,
            device=device,
            dtype=dtype,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            device=device,
            dtype=dtype,
        )

        self.t_embedding_norm = RMSNorm(model_channels)

    def _build_pos_embed(self, device=None):
        assert self.pos_emb_cls == "rope3d", f"Only rope3d is supported, got {self.pos_emb_cls}"
        self.pos_embedder = VideoRopePosition3DEmb(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
            device=device,
        )

    def prepare_embedded_sequence(self, x_B_C_T_H_W, fps=None, padding_mask=None):
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4],
                    dtype=x_B_C_T_H_W.dtype,
                    device=x_B_C_T_H_W.device,
                )
            else:
                # Resize padding mask to match spatial dims
                import torchvision.transforms.functional as tvf
                from torchvision.transforms import InterpolationMode
                padding_mask = tvf.resize(
                    padding_mask,
                    list(x_B_C_T_H_W.shape[-2:]),
                    interpolation=InterpolationMode.NEAREST,
                )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).expand(-1, -1, x_B_C_T_H_W.shape[2], -1, -1)],
                dim=1,
            )

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        rope_emb = self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device)
        return x_B_T_H_W_D, rope_emb

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )

    def forward(self, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        x_B_T_H_W_D, rope_emb_L_D = self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)

        t_emb_sincos = self.t_embedder[0](timesteps).to(x_B_T_H_W_D.dtype)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](t_emb_sincos)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        rope_emb = rope_emb_L_D.unsqueeze(1).unsqueeze(0)   # (1, L, 1, D//2, 2, 2)

        block_kwargs = dict(
            rope_emb_L_1_1_D=rope_emb,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )

        for block in self.blocks:
            x_B_T_H_W_D = block(x_B_T_H_W_D, t_embedding_B_T_D, context, **block_kwargs)

        x_out = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        return self.unpatchify(x_out)


# ---------------------------------------------------------------------------
# LLM Adapter components (from comfy/ldm/anima/model.py)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_llm(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


class LLMRotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLMAdapterAttention(nn.Module):
    """Attention module used inside LLMAdapter (with Llama-style RoPE)."""

    def __init__(self, query_dim, context_dim, n_heads, head_dim, device=None, dtype=None, **kwargs):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(head_dim)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        B, S_q = x.shape[:2]
        S_k = context.shape[1]

        q_shape = (B, S_q, self.n_heads, self.head_dim)
        kv_shape = (B, S_k, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)    # (B, H, S_q, D)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = _apply_rotary_pos_emb_llm(query_states, cos, sin)
        if position_embeddings_context is not None:
            cos_c, sin_c = position_embeddings_context
            key_states = _apply_rotary_pos_emb_llm(key_states, cos_c, sin_c)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).reshape(B, S_q, -1).contiguous()
        return self.o_proj(attn_output)


class LLMTransformerBlock(nn.Module):
    def __init__(
        self,
        source_dim,
        model_dim,
        num_heads=16,
        mlp_ratio=4.0,
        use_self_attn=False,
        layer_norm=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.use_self_attn = use_self_attn
        norm_cls = nn.LayerNorm if layer_norm else RMSNorm

        if use_self_attn:
            self.norm_self_attn = norm_cls(model_dim)
            self.self_attn = LLMAdapterAttention(
                model_dim, model_dim, num_heads, model_dim // num_heads, device=device, dtype=dtype
            )

        self.norm_cross_attn = norm_cls(model_dim)
        self.cross_attn = LLMAdapterAttention(
            model_dim, source_dim, num_heads, model_dim // num_heads, device=device, dtype=dtype
        )

        self.norm_mlp = norm_cls(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio), device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim, device=device, dtype=dtype),
        )

    def forward(
        self,
        x,
        context,
        target_attention_mask=None,
        source_attention_mask=None,
        position_embeddings=None,
        position_embeddings_context=None,
    ):
        if self.use_self_attn:
            normed = self.norm_self_attn(x)
            x = x + self.self_attn(
                normed,
                mask=target_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )

        normed = self.norm_cross_attn(x)
        x = x + self.cross_attn(
            normed,
            mask=source_attention_mask,
            context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )

        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    """
    Adapter that cross-attends T5-tokenised target embeddings to Qwen3 source
    hidden states, producing the final cross-attention context for Anima.
    """

    def __init__(
        self,
        source_dim=1024,
        target_dim=1024,
        model_dim=1024,
        num_layers=6,
        num_heads=16,
        use_self_attn=True,
        layer_norm=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.embed = nn.Embedding(32128, target_dim)
        self.in_proj = nn.Linear(target_dim, model_dim, device=device, dtype=dtype) if model_dim != target_dim else nn.Identity()
        self.rotary_emb = LLMRotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                LLMTransformerBlock(
                    source_dim, model_dim, num_heads=num_heads,
                    use_self_attn=use_self_attn, layer_norm=layer_norm,
                    device=device, dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(model_dim, target_dim, device=device, dtype=dtype)
        self.norm = RMSNorm(target_dim)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states

        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)

        for block in self.blocks:
            x = block(
                x, context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )

        return self.norm(self.out_proj(x))


# ---------------------------------------------------------------------------
# Anima = MiniTrainDIT + LLMAdapter
# ---------------------------------------------------------------------------

# Default hyperparameters matching the animaOfficial_preview3Base.safetensors weights
ANIMA_CONFIG = dict(
    max_img_h=1024,
    max_img_w=1024,
    max_frames=1,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    mlp_ratio=4.0,
    crossattn_emb_channels=1024,
    pos_emb_cls="rope3d",
    pos_emb_learnable=False,
    pos_emb_interpolation="crop",
    min_fps=1,
    max_fps=30,
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=True,
)

ANIMA_LLM_ADAPTER_CONFIG = dict(
    source_dim=1024,
    target_dim=1024,
    model_dim=1024,
    num_layers=6,
    num_heads=16,
    use_self_attn=True,
    layer_norm=False,
)


class Anima(MiniTrainDIT):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, device=None, dtype=None, **kwargs):
        cfg = {**ANIMA_CONFIG, **kwargs}
        super().__init__(device=device, dtype=dtype, **cfg)
        llm_cfg = {**ANIMA_LLM_ADAPTER_CONFIG}
        self.llm_adapter = LLMAdapter(device=device, dtype=dtype, **llm_cfg)

    def encode_text(self, source_hidden_states, target_input_ids, **kwargs):
        """Run the LLM adapter to produce cross-attention context."""
        return self.llm_adapter(source_hidden_states, target_input_ids, **kwargs)


def build_anima(dtype=None, device=None) -> Anima:
    return Anima(device=device, dtype=dtype)
