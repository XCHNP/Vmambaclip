"""Hybrid bidirectional-Vim and CNN student specified by VisionMambaCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .adapters import VimAdapter


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


class ResNet18ThroughLayer3(nn.Module):
    """Word-supplement default: ResNet18 stem through ``layer3``."""

    output_channels = 256

    def __init__(self, weights=None) -> None:
        super().__init__()
        from torchvision.models import resnet18

        source = resnet18(weights=weights)
        self.stem = nn.Sequential(source.conv1, source.bn1, source.relu, source.maxpool)
        self.layer1 = source.layer1
        self.layer2 = source.layer2
        self.layer3 = source.layer3

    def forward(self, images: Tensor) -> Tensor:
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
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
    """Main student path: projected states, query aggregation, CNN and concat fusion."""

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

        self.state_projections = nn.ModuleDict(
            {name: nn.Linear(2 * int(state_dims[name]), common_dim) for name in self.aliases}
        )
        self.global_query = nn.Parameter(torch.empty(1, 1, common_dim))
        nn.init.normal_(self.global_query, std=common_dim ** -0.5)
        self.cross_attention = nn.MultiheadAttention(
            common_dim, attention_heads, batch_first=True
        )

        if self.use_cnn:
            self.cnn_backbone = cnn_backbone or ResNet18ThroughLayer3(weights=None)
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

    def _project_states(self, hidden: Mapping[str, Tensor]):
        raw: Dict[str, Tensor] = {}
        norm: Dict[str, Tensor] = {}
        for name in self.aliases:
            value = hidden[name]
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
            return zeros, zeros, patch_norm, attention

        cnn_raw = self.cnn_backbone(images)
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
        return descriptor, distill, patch_norm, attention

    def forward(self, images: Tensor) -> StudentOutput:
        hidden = self.vim(images)
        states_raw, states_norm = self._project_states(hidden)
        context = torch.stack([states_raw[name] for name in self.aliases], dim=1)
        query = self.global_query.expand(images.shape[0], -1, -1)
        mamba_global = self.cross_attention(query, context, context, need_weights=False)[0][:, 0]

        cnn_descriptor, cnn_distill, patch_norm, attention = self._local_features(images)
        fusion_input = torch.cat((mamba_global, cnn_descriptor), dim=-1)
        fused_raw = self.fusion_projection(fusion_input)
        fused_norm = F.normalize(fused_raw.float(), dim=-1).to(fused_raw.dtype)

        logits = None
        if self.compute_logits:
            classifier_input = torch.cat(
                [states_raw[name] for name in self.aliases] + [fused_raw], dim=-1
            )
            logits = self.classifier(classifier_input)
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
        )

