"""Hybrid bidirectional-Vim and CNN student specified by VisionMambaCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .adapters import VimAdapter


class SASF(nn.Module):
    """Restore raster-ordered patch tokens and add multiscale spatial context.

    The caller must remove special tokens and supply the actual patch grid;
    inferring a square from the token count can silently scramble geometry.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, in_dim, 3, padding=d, dilation=d,
                          groups=in_dim, bias=False),
                nn.Conv2d(in_dim, in_dim, 1, bias=False),
                nn.GELU(),
            ) for d in (1, 2, 3)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(4 * in_dim, in_dim, 1, bias=False),
            nn.GroupNorm(1, in_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, tokens: Tensor, height: Optional[int] = None, width: Optional[int] = None):
        if tokens.ndim == 4:
            if tokens.shape[1] != self.in_dim:
                raise ValueError(f"SASF expects [B,{self.in_dim},H,W]")
            height, width = tokens.shape[-2:]
            tokens = tokens.flatten(2).transpose(1, 2)
        if tokens.ndim != 3 or tokens.shape[-1] != self.in_dim:
            raise ValueError(f"SASF expects [B,N,{self.in_dim}], got {tuple(tokens.shape)}")
        if height is None or width is None or height <= 0 or width <= 0 or tokens.shape[1] != height * width:
            raise ValueError("SASF patch count must match height * width; exclude CLS and token dropout")
        x = tokens.transpose(1, 2).reshape(tokens.shape[0], self.in_dim, height, width)
        enhanced = x + self.fusion(torch.cat([x] + [branch(x) for branch in self.branches], dim=1))
        self.last_spatial_delta_rms = (enhanced - x).detach().float().square().mean().sqrt()
        return enhanced.flatten(2).transpose(1, 2), self.proj(enhanced.mean(dim=(2, 3)))


@dataclass
class StudentOutput:
    fused_raw: Tensor
    fused_norm: Tensor
    states_raw: Dict[str, Tensor]
    states_norm: Dict[str, Tensor]
    cnn_descriptor: Tensor
    cnn_distill: Tensor
    patch_norm: Tensor
    attention: Tensor
    logits: Optional[Tensor]
    diagnostics: Optional[Dict[str, Tensor]] = None
    cnn_aux_logits: Optional[Tensor] = None


class CNNResidualAux(nn.Module):
    """Independent CNN descriptor; residual and training head are separate factors."""

    def __init__(self, dim, num_classes, *, residual=False, auxiliary=False):
        super().__init__()
        self.cnn_norm = nn.LayerNorm(dim)
        self.fusion_norm = nn.LayerNorm(dim) if residual else None
        self.cnn_gate_logit = nn.Parameter(torch.tensor(-1.38629436112)) if residual else None
        self.cnn_aux_head = nn.Linear(dim, num_classes) if auxiliary else None

    def forward(self, old_fused, cnn_tokens, *, compute_aux=False, disable_cnn=False):
        cnn_global = self.cnn_norm(cnn_tokens.mean(dim=1))
        if disable_cnn:
            cnn_global = torch.zeros_like(cnn_global)
        gate = self.cnn_gate_logit.sigmoid() if self.cnn_gate_logit is not None else old_fused.new_zeros(())
        fused = old_fused
        if self.fusion_norm is not None:
            fused = self.fusion_norm(old_fused + gate * cnn_global)
        aux = self.cnn_aux_head(cnn_global) if compute_aux and self.cnn_aux_head is not None else None
        return fused, aux, gate, cnn_global


class ResNet18ThroughLayer3(nn.Module):
    """Word-supplement default: ResNet18 stem through ``layer3``."""

    output_channels = 256

    def __init__(self, weights=None, return_multiscale=False) -> None:
        super().__init__()
        from torchvision.models import resnet18

        source = resnet18(weights=weights)
        self.stem = nn.Sequential(source.conv1, source.bn1, source.relu, source.maxpool)
        self.layer1 = source.layer1
        self.layer2 = source.layer2
        self.layer3 = source.layer3
        self.return_multiscale = bool(return_multiscale)

    def forward(self, images: Tensor) -> Tensor:
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.return_multiscale:
            return {"mid": x, "deep": self.layer3(x)}
        return self.layer3(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, use_coords: bool = False):
        super().__init__()
        self.use_coords = bool(use_coords)
        input_channels = int(in_channels) + (2 if self.use_coords else 0)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    @staticmethod
    def _coords(feature_map: Tensor) -> Tensor:
        batch, _, height, width = feature_map.shape
        y = torch.linspace(-1.0, 1.0, height, device=feature_map.device, dtype=feature_map.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature_map.device, dtype=feature_map.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, feature_map: Tensor, epsilon: float) -> Tensor:
        source = feature_map
        if self.use_coords:
            source = torch.cat((source, self._coords(source)), dim=1)
        # E: positive weights make the paper's mean sparsity term bounded below.
        return F.softplus(self.net(source)) + float(epsilon)


class StudentClassifier(nn.Module):
    def __init__(self, common_dim: int, num_classes: int):
        super().__init__()
        self.input_linear = nn.Linear(4 * common_dim, common_dim)
        self.batch_norm = nn.BatchNorm1d(common_dim)
        self.activation = nn.ReLU(inplace=True)
        self.final_linear = nn.Linear(common_dim, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        return self.final_linear(self.activation(self.batch_norm(self.input_linear(features))))


class HybridVisionMambaStudent(nn.Module):
    """Vim/CNN student with optional SASF and spatial cross-branch attention."""

    aliases = ("e", "m", "l")

    def __init__(
        self,
        vim: VimAdapter,
        state_dims: Mapping[str, int],
        common_dim: int,
        num_classes: int,
        *,
        attention_heads: int = 8,
        attention_hidden_channels: int = 64,
        use_coords: bool = False,
        epsilon: float = 1e-6,
        cnn_backbone: Optional[nn.Module] = None,
        cnn_channels: Optional[int] = None,
        use_cnn: bool = True,
        compute_logits: bool = True,
        use_sasf: bool = False,
        fusion: str = "concat_projection",
        sasf_stage: str = "l",
        cnn_multiscale: bool = False,
        query_mode: str = "single",
        patch_dropout: float = 0.,
        cnn_auxiliary: bool = False,
        cnn_residual: bool = False,
    ) -> None:
        super().__init__()
        if tuple(state_dims.keys()) != self.aliases:
            raise ValueError("state_dims must have ordered keys ('e', 'm', 'l')")
        if common_dim % attention_heads:
            raise ValueError(
                f"common_dim={common_dim} must be divisible by attention_heads={attention_heads}"
            )
        self.vim = vim
        self.common_dim = int(common_dim)
        self.num_classes = int(num_classes)
        self.epsilon = float(epsilon)
        self.use_cnn = bool(use_cnn)
        self.compute_logits = bool(compute_logits)
        self.use_sasf = bool(use_sasf)
        if (cnn_auxiliary or cnn_residual) and not self.use_cnn:
            raise ValueError("CNN auxiliary/residual paths require the CNN branch")
        self.compute_aux_logits = bool(cnn_auxiliary)
        if sasf_stage not in self.aliases or query_mode not in {"single", "multi"}:
            raise ValueError("Invalid SASF stage or query mode")
        if not 0 <= patch_dropout < 1:
            raise ValueError("patch_dropout must be in [0,1)")
        if (cnn_multiscale or query_mode == "multi" or patch_dropout) and fusion != "mamba_cnn_cross_attention":
            raise ValueError("Multiscale/multi-query/KV dropout require cross-branch fusion")
        self.sasf_stage = sasf_stage
        self.cnn_multiscale = bool(cnn_multiscale)
        self.query_mode = query_mode
        self.patch_dropout = float(patch_dropout)
        self.last_diagnostics = {}
        if fusion not in {"concat_projection", "mamba_cnn_cross_attention"}:
            raise ValueError(f"Unsupported fusion: {fusion}")
        if fusion == "mamba_cnn_cross_attention" and not self.use_cnn:
            raise ValueError("Cross-branch attention requires the CNN branch")
        if self.use_sasf and getattr(vim, "feature_layout", None) != "BCHW" and not callable(getattr(vim, "patch_grid", None)):
            raise ValueError("SASF requires a Vim adapter exposing patch_grid(images)")
        self.fusion = fusion

        self.state_projections = nn.ModuleDict(
            {name: nn.Linear(2 * int(state_dims[name]), common_dim) for name in self.aliases}
        )
        self.global_query = nn.Parameter(torch.empty(1, 1, common_dim))
        nn.init.normal_(self.global_query, std=common_dim ** -0.5)
        self.cross_attention = nn.MultiheadAttention(
            common_dim, attention_heads, batch_first=True
        )

        if self.use_cnn:
            self.cnn_backbone = cnn_backbone or ResNet18ThroughLayer3(weights=None, return_multiscale=cnn_multiscale)
            inferred_channels = getattr(self.cnn_backbone, "output_channels", None)
            actual_channels = cnn_channels or inferred_channels
            if actual_channels is None:
                raise ValueError("cnn_channels is required when the CNN does not expose output_channels")
            self.cnn_channels = int(actual_channels)
            self.cnn_projection = nn.Conv2d(self.cnn_channels, common_dim, kernel_size=1)
            self.spatial_attention = SpatialAttention(
                self.cnn_channels,
                hidden_channels=attention_hidden_channels,
                use_coords=use_coords,
            )
        else:
            self.cnn_backbone = None
            self.cnn_projection = None
            self.spatial_attention = None
            self.cnn_channels = 0

        # P: concat followed by projection, not residual addition.
        self.fusion_projection = nn.Linear(2 * common_dim, common_dim)
        self.classifier = StudentClassifier(common_dim, num_classes)
        self.text_to_classifier = nn.Linear(common_dim, common_dim, bias=False)
        with torch.no_grad():
            self.text_to_classifier.weight.copy_(torch.eye(common_dim))

        # Construct additions last so legacy module initialization stays identical.
        self.sasf = SASF(int(state_dims[sasf_stage]), common_dim) if self.use_sasf else None
        if self.use_sasf:
            self.sasf_alpha = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("sasf_alpha", None)
        if self.fusion == "mamba_cnn_cross_attention":
            self.local_cross_attention = nn.MultiheadAttention(common_dim, attention_heads, batch_first=True)
            self.local_fusion_norm = nn.LayerNorm(common_dim)
        else:
            self.local_cross_attention = None
            self.local_fusion_norm = None
        self.cnn_lateral = nn.Conv2d(128, self.cnn_channels, 1) if cnn_multiscale else None
        self.cnn_residual_aux = CNNResidualAux(common_dim, num_classes, residual=cnn_residual,
                                             auxiliary=cnn_auxiliary) if cnn_auxiliary or cnn_residual else None

    def _project_states(self, hidden: Mapping[str, Tensor]):
        raw: Dict[str, Tensor] = {}
        norm: Dict[str, Tensor] = {}
        for name in self.aliases:
            value = hidden[name]
            if value.ndim == 4:
                value = value.flatten(2).transpose(1, 2)
            pooled = torch.cat((value.mean(dim=1), value.amax(dim=1)), dim=-1)
            raw[name] = self.state_projections[name](pooled)
            norm[name] = F.normalize(raw[name].float(), dim=-1).to(raw[name].dtype)
        return raw, norm

    def _local_features(self, images: Tensor):
        batch = images.shape[0]
        if not self.use_cnn:
            zeros = images.new_zeros((batch, self.common_dim))
            patch_norm = images.new_zeros((batch, 1, self.common_dim))
            attention = images.new_ones((batch, 1, 1, 1))
            return zeros, zeros, patch_norm, attention, patch_norm, patch_norm

        cnn_raw = self.cnn_backbone(images)
        if self.cnn_multiscale:
            if not isinstance(cnn_raw, Mapping) or set(cnn_raw) != {"mid", "deep"}:
                raise RuntimeError("Multiscale CNN must expose layer2/layer3 maps as mid/deep")
            mid = self.cnn_lateral(cnn_raw["mid"])
            deep = F.interpolate(cnn_raw["deep"], size=mid.shape[-2:], mode="bilinear", align_corners=False)
            cnn_raw = (mid + deep) * .5
        if cnn_raw.ndim != 4 or cnn_raw.shape[1] != self.cnn_channels:
            raise RuntimeError(
                f"CNN must return [B,{self.cnn_channels},H,W], got {tuple(cnn_raw.shape)}"
            )
        projected = self.cnn_projection(cnn_raw)
        attention = self.spatial_attention(cnn_raw, self.epsilon)
        descriptor = (attention * projected).mean(dim=(2, 3))
        denominator = attention.sum(dim=(2, 3)).clamp_min(self.epsilon)
        distill = (attention * projected).sum(dim=(2, 3)) / denominator
        patch_raw = projected.flatten(2).transpose(1, 2)
        patch_norm = F.normalize(patch_raw.float(), dim=-1).to(patch_raw.dtype)
        # Preserve locations for K/V. Normalize attention mass to keep token scale
        # independent of the number of CNN positions and the sparsity penalty.
        local_tokens = patch_raw
        if self.local_cross_attention is not None:
            relative_attention = attention / attention.mean(dim=(2, 3), keepdim=True).clamp_min(self.epsilon)
            local_tokens = (projected * relative_attention).flatten(2).transpose(1, 2)
        return descriptor, distill, patch_norm, attention, local_tokens, patch_raw

    def forward(self, images: Tensor, intervention: str = "full", *, return_aux: bool = False) -> StudentOutput:
        if intervention not in {"full", "no_cnn", "no_mamba", "no_sasf", "shuffle_cnn", "shuffle_cnn_values"}:
            raise ValueError(f"Unknown intervention: {intervention}")
        hidden = self.vim(images)
        states_raw, states_norm = self._project_states(hidden)
        if intervention == "no_mamba":
            states_raw = {name: torch.zeros_like(value) for name, value in states_raw.items()}
            states_norm = {name: torch.zeros_like(value) for name, value in states_norm.items()}
        context = torch.stack([states_raw[name] for name in self.aliases], dim=1)
        query = self.global_query.expand(images.shape[0], -1, -1)
        mamba_global = self.cross_attention(query, context, context, need_weights=False)[0][:, 0]
        original_mamba_global = mamba_global
        sasf_delta = torch.zeros_like(mamba_global)
        if self.use_sasf and intervention not in {"no_sasf", "no_mamba"}:
            source = hidden[self.sasf_stage]
            if source.ndim == 4:
                _, spatial_global = self.sasf(source)
            else:
                height, width = self.vim.patch_grid(images)
                _, spatial_global = self.sasf(source, height, width)
            sasf_delta = self.sasf_alpha.tanh() * spatial_global
            mamba_global = mamba_global + sasf_delta
        if intervention == "no_mamba":
            mamba_global = torch.zeros_like(mamba_global)

        cnn_descriptor, cnn_distill, patch_norm, attention, local_tokens, independent_cnn_tokens = self._local_features(images)
        if intervention == "no_cnn":
            cnn_descriptor = torch.zeros_like(cnn_descriptor)
            cnn_distill = torch.zeros_like(cnn_distill)
            local_tokens = torch.zeros_like(local_tokens)
        if self.training and self.patch_dropout:
            keep = max(1, int(local_tokens.shape[1] * (1 - self.patch_dropout)))
            indices = torch.rand(local_tokens.shape[:2], device=local_tokens.device).argsort(dim=1)[:, :keep]
            local_tokens = local_tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, self.common_dim))
        attended = torch.zeros_like(mamba_global)
        mamba_before_fusion = mamba_global
        if self.local_cross_attention is not None:
            queries = mamba_global.unsqueeze(1)
            if self.query_mode == "multi":
                queries = torch.stack([states_raw[name] + (sasf_delta if name == self.sasf_stage else 0)
                                       for name in self.aliases], dim=1)
            keys = values = local_tokens
            if intervention in {"shuffle_cnn", "shuffle_cnn_values"}:
                order = torch.randperm(local_tokens.shape[1], device=local_tokens.device)
                values = values[:, order]
                if intervention == "shuffle_cnn":
                    keys = keys[:, order]
            attended = self.local_cross_attention(
                queries, keys, values, need_weights=False,
            )[0].mean(dim=1)
            if intervention == "no_cnn":
                attended = torch.zeros_like(attended)
            mamba_global = self.local_fusion_norm(mamba_global + attended)
        fusion_input = torch.cat((mamba_global, cnn_descriptor), dim=-1)
        fused_raw = self.fusion_projection(fusion_input)
        aux_logits = None
        cnn_gate = images.new_zeros(())
        independent_cnn_global = images.new_zeros((images.shape[0], self.common_dim))
        if self.cnn_residual_aux is not None:
            fused_raw, aux_logits, cnn_gate, independent_cnn_global = self.cnn_residual_aux(
                fused_raw, independent_cnn_tokens,
                compute_aux=self.compute_aux_logits and (self.training or return_aux),
                disable_cnn=intervention == "no_cnn",
            )
        fused_norm = F.normalize(fused_raw.float(), dim=-1).to(fused_raw.dtype)

        logits = None
        if self.compute_logits:
            classifier_input = torch.cat(
                [states_raw[name] for name in self.aliases] + [fused_raw], dim=-1
            )
            logits = self.classifier(classifier_input)
        self.last_diagnostics = {
            "mamba_rms": mamba_before_fusion.detach().float().square().mean().sqrt(),
            "cnn_rms": cnn_descriptor.detach().float().square().mean().sqrt(),
            "sasf_delta_rms": sasf_delta.detach().float().square().mean().sqrt(),
            "sasf_spatial_delta_rms": self.sasf.last_spatial_delta_rms if self.use_sasf and intervention not in {"no_sasf", "no_mamba"} else images.new_zeros(()),
            "cross_attention_rms": attended.detach().float().square().mean().sqrt(),
            "cnn_kv_tokens": images.new_tensor(local_tokens.shape[1]),
            "sasf_gate": self.sasf_alpha.detach().tanh() if self.sasf_alpha is not None else images.new_zeros(()),
            "sasf_residual_ratio": (sasf_delta.detach().float().norm(dim=-1) /
                                    original_mamba_global.detach().float().norm(dim=-1).clamp_min(self.epsilon)).mean(),
            "cnn_gate": cnn_gate.detach(),
            "cnn_independent_rms": independent_cnn_global.detach().float().square().mean().sqrt(),
            "cross_attn_norm": attended.detach().float().norm(dim=-1).mean(),
            "fused_feat_norm": fused_raw.detach().float().norm(dim=-1).mean(),
        }
        return StudentOutput(
            fused_raw=fused_raw,
            fused_norm=fused_norm,
            states_raw=states_raw,
            states_norm=states_norm,
            cnn_descriptor=cnn_descriptor,
            cnn_distill=cnn_distill,
            patch_norm=patch_norm,
            attention=attention,
            logits=logits,
            diagnostics=self.last_diagnostics,
            cnn_aux_logits=aux_logits,
        )


class CNNOnlyStudent(nn.Module):
    """A true CNN-only control with a 768-D distillation interface; no SSM modules."""

    def __init__(self, cnn_backbone, common_dim=768, num_classes=7, multiscale=False):
        super().__init__()
        self.cnn_backbone = cnn_backbone
        self.cnn_multiscale = bool(multiscale)
        self.cnn_lateral = nn.Conv2d(128, 256, 1) if multiscale else None
        self.cnn_projection = nn.Conv2d(256, common_dim, 1)
        self.fusion_projection = nn.Linear(common_dim, common_dim)
        self.classifier = nn.Linear(common_dim, num_classes)
        self.compute_logits = True
        self.last_diagnostics = {}

    def forward(self, images, intervention="full"):
        if intervention not in {"full", "no_cnn", "no_mamba", "no_sasf", "shuffle_cnn", "shuffle_cnn_values"}:
            raise ValueError("Unknown intervention")
        features = self.cnn_backbone(images)
        if self.cnn_multiscale:
            mid = self.cnn_lateral(features["mid"])
            features = (mid + F.interpolate(features["deep"], mid.shape[-2:], mode="bilinear", align_corners=False)) * .5
        features = self.cnn_projection(features)
        if intervention == "no_cnn":
            features = torch.zeros_like(features)
        pooled = features.mean((2, 3))
        fused = self.fusion_projection(pooled)
        logits = self.classifier(fused) if self.compute_logits else None
        norm = F.normalize(fused.float(), dim=-1)
        tokens = F.normalize(features.flatten(2).transpose(1, 2).float(), dim=-1)
        self.last_diagnostics = {"cnn_rms": pooled.detach().float().square().mean().sqrt()}
        return StudentOutput(fused, norm, {}, {}, pooled, pooled, tokens,
                             features.new_ones((len(images), 1, *features.shape[-2:])), logits, self.last_diagnostics)

