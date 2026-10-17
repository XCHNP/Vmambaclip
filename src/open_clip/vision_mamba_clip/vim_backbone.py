"""Bidirectional Vision Mamba (Vim) backbone factory for VisionMambaCLIP.

This module provides ``create_vim``, the factory callable expected by
``vision_mamba_clip_assembly._build_student``.  It returns a bidirectional Vim
backbone and its initialization metadata.

Design notes (from the design summary):
  - The student MUST be a bidirectional Vision Mamba / Vim.
  - VMamba Cross-Scan and MambaVision are explicitly excluded.
  - The adapter extracts the last three encoder layers via hooks.
  - A CLS token is prepended and excluded during pooling.

The bidirectional SSM block scans the sequence in both forward and backward
directions and merges the two branches, following the Vim paper's design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bidirectional SSM core
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Simplified selective state-space model (S6) for one scan direction.

    Implements the discretized recurrence:
        x_{k} = A_bar * x_{k-1} + B_bar * u_k
        y_k   = C_k * x_k + D * u_k

    where A_bar, B_bar are input-dependent (selective) via learned projections.
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = 0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        # Continuous A is initialized as a negative log-spaced diagonal
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_model, -1).contiguous())

        self.D = nn.Parameter(torch.ones(d_model))

        # Input-dependent projections for dt, B, C
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        self.x_to_dbc = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)

        # Initialize dt bias so initial step sizes are in reasonable range
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u: Tensor) -> Tensor:
        """Scan over sequence dimension.

        Args:
            u: [B, L, D] input sequence.

        Returns:
            y: [B, L, D] output sequence.
        """
        B, L, D = u.shape
        N = self.d_state

        dbc = self.x_to_dbc(u)  # [B, L, dt_rank + 2*N]
        dt_raw, B_input, C_input = dbc.split(
            [self.dt_rank, N, N], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))  # [B, L, D]

        # Discretize: A_bar = exp(dt * A), where A is negative
        A = -torch.exp(self.A_log)  # [D, N]
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # [B, L, D, N]
        A_bar = torch.exp(dt_A)

        dt_B = dt.unsqueeze(-1) * B_input.unsqueeze(2)  # [B, L, D, N]

        # Sequential scan (recurrence)
        x = u.new_zeros(B, D, N)  # hidden state
        outputs = []
        for t in range(L):
            x = A_bar[:, t] * x + dt_B[:, t] * u[:, t].unsqueeze(-1)
            y_t = (x * C_input[:, t].unsqueeze(1)).sum(dim=-1)  # [B, D]
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)  # [B, L, D]
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        return y


class BidirectionalSSM(nn.Module):
    """Bidirectional selective SSM: forward scan + backward scan, merged."""

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = 0):
        super().__init__()
        self.forward_ssm = SelectiveSSM(d_model, d_state, dt_rank)
        self.backward_ssm = SelectiveSSM(d_model, d_state, dt_rank)
        # Learnable merge gate (as in Vim paper)
        self.merge_gate = nn.Linear(2 * d_model, d_model, bias=True)

    def forward(self, u: Tensor) -> Tensor:
        """
        Args:
            u: [B, L, D]
        Returns:
            y: [B, L, D]
        """
        y_fwd = self.forward_ssm(u)
        y_bwd = self.backward_ssm(u.flip(dims=[1])).flip(dims=[1])
        y = self.merge_gate(torch.cat([y_fwd, y_bwd], dim=-1))
        return y


# ---------------------------------------------------------------------------
# Vim encoder block and full backbone
# ---------------------------------------------------------------------------

class VimBlock(nn.Module):
    """Single Vim encoder block: LayerNorm -> linear expand -> Bidirectional SSM -> linear project."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand_factor: int = 2,
        dt_rank: int = 0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        d_inner = d_model * expand_factor
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.bidi_ssm = BidirectionalSSM(d_inner, d_state, dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: [B, L, D]
        Returns:
            [B, L, D]
        """
        residual = tokens
        x = self.norm(tokens)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.bidi_ssm(x)
        x = x * F.silu(z)  # gating
        x = self.out_proj(x)
        return residual + self.drop_path(x)


class DropPath(nn.Module):
    """Stochastic depth (drop path) for residual blocks."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(x.new_full(shape, keep))
        return x * mask / keep


class PatchEmbed(nn.Module):
    """Image to patch embedding via non-overlapping convolution."""

    def __init__(self, img_size: int = 256, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = self.norm(x)
        return x


class BidirectionalVim(nn.Module):
    """Bidirectional Vision Mamba (Vim) backbone.

    Architecture:
        Input image → Patch embedding → CLS token prepend → N x VimBlock → output tokens.

    This is the genuine bidirectional Vim following the paper's design:
    each block scans tokens in both forward and backward directions via dual SSMs.

    Attributes:
        layers: nn.ModuleList of VimBlock for hook-based feature extraction.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 24,
        d_state: int = 16,
        expand_factor: int = 2,
        dt_rank: int = 0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Stochastic depth schedule (linearly increasing)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            VimBlock(
                d_model=embed_dim,
                d_state=d_state,
                expand_factor=expand_factor,
                dt_rank=dt_rank,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, images: Tensor) -> Tensor:
        """Extract features from all layers. Hooks placed on self.layers will fire.

        Args:
            images: [B, 3, H, W]

        Returns:
            tokens: [B, 1+N, D] — CLS + patch tokens after final norm.
        """
        B = images.shape[0]
        x = self.patch_embed(images)  # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, D]
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_features(images)

    def set_grad_checkpointing(self, enable: bool = True):
        """Enable/disable gradient checkpointing for memory efficiency."""
        self._grad_checkpointing = enable
        if enable:
            # Override forward to use checkpointing
            original_forward_features = self.forward_features

            def checkpointed_forward(images: Tensor) -> Tensor:
                B = images.shape[0]
                x = self.patch_embed(images)
                cls = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = x + self.pos_embed
                for layer in self.layers:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
                x = self.norm(x)
                return x

            self.forward_features = checkpointed_forward


# ---------------------------------------------------------------------------
# Pre-defined model configurations
# ---------------------------------------------------------------------------

VIM_CONFIGS: Dict[str, dict] = {
    "vim_tiny_patch16": {
        "embed_dim": 192,
        "depth": 24,
        "d_state": 16,
        "expand_factor": 2,
        "patch_size": 16,
    },
    "vim_small_patch16": {
        "embed_dim": 384,
        "depth": 24,
        "d_state": 16,
        "expand_factor": 2,
        "patch_size": 16,
    },
    "vim_base_patch16": {
        "embed_dim": 768,
        "depth": 24,
        "d_state": 16,
        "expand_factor": 2,
        "patch_size": 16,
    },
}


def create_vim(
    *,
    checkpoint_path: Optional[str] = None,
    model_name: str = "vim_small_patch16",
    img_size: int = 256,
    drop_path_rate: float = 0.1,
    **kwargs,
) -> Tuple[nn.Module, dict]:
    """Factory callable for VisionMambaCLIP assembly.

    Returns:
        (backbone, initialization_metadata) as required by the assembly contract.
    """
    if model_name.endswith("_224"):
        model_name = model_name[:-4]

    if model_name not in VIM_CONFIGS:
        raise ValueError(
            f"Unknown Vim model: {model_name!r}. "
            f"Available: {sorted(VIM_CONFIGS.keys())}"
        )
    cfg = VIM_CONFIGS[model_name]
    backbone = BidirectionalVim(
        img_size=img_size,
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        d_state=cfg["d_state"],
        expand_factor=cfg["expand_factor"],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )

    pretrained_loaded = False
    init_source = "random"

    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Vim checkpoint not found: {path}")
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Handle possible position embedding size mismatch
        if "pos_embed" in state_dict:
            pretrained_pos = state_dict["pos_embed"]
            if pretrained_pos.shape != backbone.pos_embed.shape:
                # Interpolate position embedding
                state_dict["pos_embed"] = _interpolate_pos_embed(
                    pretrained_pos, backbone.pos_embed.shape[1]
                )

        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Vim] Missing keys when loading checkpoint: {missing[:10]}...")
        if unexpected:
            print(f"[Vim] Unexpected keys when loading checkpoint: {unexpected[:10]}...")
        pretrained_loaded = True
        init_source = str(path)

    metadata = {
        "identity": f"bidirectional_vim_{model_name}_img{img_size}",
        "pretrained_loaded": pretrained_loaded,
        "initialization_source": init_source,
        "model_name": model_name,
        "embed_dim": cfg["embed_dim"],
        "depth": cfg["depth"],
        "patch_size": cfg["patch_size"],
        "num_patches": (img_size // cfg["patch_size"]) ** 2,
        "img_size": img_size,
        "bidirectional": True,
    }
    return backbone, metadata


def _interpolate_pos_embed(pos_embed: Tensor, target_length: int) -> Tensor:
    """Interpolate position embeddings to handle different input resolutions."""
    N = pos_embed.shape[1]
    if N == target_length:
        return pos_embed
    cls_pos = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]
    dim = patch_pos.shape[-1]
    src_size = int(math.sqrt(patch_pos.shape[1]))
    tgt_size = int(math.sqrt(target_length - 1))
    patch_pos = patch_pos.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos.float(), size=(tgt_size, tgt_size), mode="bicubic", align_corners=False
    ).to(pos_embed.dtype)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat([cls_pos, patch_pos], dim=1)
