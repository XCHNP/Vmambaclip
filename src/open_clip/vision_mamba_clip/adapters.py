"""Adapters between OpenCLIP/Vim implementations and VisionMambaCLIP.

The adapters deliberately validate interfaces instead of guessing model identity.
In particular, ``VimAdapter`` requires the caller to attest that the supplied
backbone is a bidirectional Vision Mamba (Vim), not VMamba Cross-Scan or
MambaVision.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _infer_clip_output_dim(model: nn.Module) -> int:
    projection = getattr(model, "text_projection", None)
    if isinstance(projection, nn.Linear):
        return int(projection.out_features)
    if isinstance(projection, (Tensor, nn.Parameter)) and projection.ndim == 2:
        return int(projection.shape[-1])

    text_tower = getattr(model, "text", None)
    projection = getattr(text_tower, "text_projection", None)
    if isinstance(projection, nn.Linear):
        return int(projection.out_features)
    if isinstance(projection, (Tensor, nn.Parameter)) and projection.ndim == 2:
        return int(projection.shape[-1])

    visual = getattr(model, "visual", None)
    projection = getattr(visual, "proj", None)
    if isinstance(projection, nn.Linear):
        return int(projection.out_features)
    if isinstance(projection, (Tensor, nn.Parameter)) and projection.ndim == 2:
        return int(projection.shape[-1])
    raise ValueError("Cannot infer CLIP common-space dimension from the supplied model")


class TeacherAdapter(nn.Module):
    """Expose projected CLIP features and a linear FER classification head."""

    def __init__(self, clip_model: nn.Module, num_classes: int, output_dim: Optional[int] = None):
        super().__init__()
        if not hasattr(clip_model, "encode_image") or not hasattr(clip_model, "encode_text"):
            raise TypeError("clip_model must implement encode_image and encode_text")
        self.clip_model = clip_model
        self.output_dim = int(output_dim or _infer_clip_output_dim(clip_model))
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.output_dim, self.num_classes)

    @property
    def logit_scale(self) -> nn.Parameter:
        value = getattr(self.clip_model, "logit_scale", None)
        if value is None:
            raise AttributeError("The wrapped CLIP model has no logit_scale parameter")
        return value

    def encode_image(self, images: Tensor) -> Tensor:
        result = self.clip_model.encode_image(images, normalize=False)
        if result.ndim != 2 or result.shape[-1] != self.output_dim:
            raise RuntimeError(
                f"CLIP image feature must be [B,{self.output_dim}], got {tuple(result.shape)}"
            )
        return result

    def encode_text(self, tokens: Tensor) -> Tensor:
        result = self.clip_model.encode_text(tokens, normalize=False)
        if result.ndim != 2 or result.shape[-1] != self.output_dim:
            raise RuntimeError(
                f"CLIP text feature must be [N,{self.output_dim}], got {tuple(result.shape)}"
            )
        return result

    def forward_classifier(self, images: Tensor) -> Dict[str, Tensor]:
        image_raw = self.encode_image(images)
        return {
            "image_raw": image_raw,
            "image_norm": F.normalize(image_raw.float(), dim=-1),
            "logits": self.classifier(image_raw),
        }

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        return self.forward_classifier(images)

    def text_parameters(self) -> Iterable[nn.Parameter]:
        """Yield text-tower parameters for both OpenCLIP model layouts."""
        text_tower = getattr(self.clip_model, "text", None)
        if isinstance(text_tower, nn.Module):
            yield from text_tower.parameters()
            return
        names = (
            "transformer",
            "token_embedding",
            "ln_final",
        )
        seen = set()
        for name in names:
            module = getattr(self.clip_model, name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        yield parameter
        for name in ("positional_embedding", "text_projection"):
            value = getattr(self.clip_model, name, None)
            if isinstance(value, nn.Parameter) and id(value) not in seen:
                seen.add(id(value))
                yield value
            elif isinstance(value, nn.Module):
                for parameter in value.parameters():
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        yield parameter

    def freeze_text_encoder(self) -> None:
        for parameter in self.text_parameters():
            parameter.requires_grad_(False)
        text_tower = getattr(self.clip_model, "text", None)
        if isinstance(text_tower, nn.Module):
            text_tower.eval()
        else:
            for name in ("transformer", "token_embedding", "ln_final"):
                module = getattr(self.clip_model, name, None)
                if isinstance(module, nn.Module):
                    module.eval()


def _resolve_layer_modules(backbone: nn.Module) -> Sequence[nn.Module]:
    for name in ("layers", "blocks"):
        modules = getattr(backbone, name, None)
        if isinstance(modules, (nn.ModuleList, nn.Sequential)):
            return list(modules)
    encoder = getattr(backbone, "encoder", None)
    if isinstance(encoder, nn.Module):
        for name in ("layers", "blocks"):
            modules = getattr(encoder, name, None)
            if isinstance(modules, (nn.ModuleList, nn.Sequential)):
                return list(modules)
    raise ValueError(
        "Cannot locate Vim encoder layers. Pass layer_modules explicitly; "
        "supported automatic attributes are layers, blocks, encoder.layers, and encoder.blocks."
    )


def patch_grid_from_conv(backbone: nn.Module, images: Tensor):
    """Grid of a raster-flattened Conv2d patch embed, including overlap/padding."""
    patch_embed = getattr(backbone, "patch_embed", None)
    projection = getattr(patch_embed, "proj", None)
    if not isinstance(projection, nn.Conv2d) or isinstance(projection.padding, str):
        raise ValueError("SASF requires an explicit Conv2d patch embedding or a custom patch_grid adapter")
    if getattr(backbone, "flip_img_sequences_ratio", 0) > 0:
        raise ValueError("SASF requires raster patch order without sequence flipping")
    grid = tuple(
        (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
        for size, pad, dilation, kernel, stride in zip(
            images.shape[-2:], projection.padding, projection.dilation,
            projection.kernel_size, projection.stride,
        )
    )
    return grid


class VimAdapter(nn.Module):
    """Extract patch tokens from the final three encoder layers of a real Vim.

    Hooks make this adapter independent of a particular Vim repository while the
    explicit bidirectional assertion prevents silently substituting a different
    Mamba-family architecture.
    """

    aliases = ("e", "m", "l")

    def __init__(
        self,
        backbone: nn.Module,
        *,
        backbone_is_bidirectional: bool,
        layer_modules: Optional[Sequence[nn.Module]] = None,
        layer_indices: Optional[Sequence[int]] = None,
        num_special_tokens: int = 1,
        backbone_identity: str,
    ) -> None:
        super().__init__()
        if not backbone_is_bidirectional:
            raise ValueError(
                "The main configuration requires a bidirectional Vision Mamba/Vim backbone"
            )
        if not backbone_identity.strip():
            raise ValueError("backbone_identity must record the actual Vim model/checkpoint")
        self.backbone = backbone
        self.backbone_identity = backbone_identity
        self.num_special_tokens = int(num_special_tokens)
        if self.num_special_tokens < 0:
            raise ValueError("num_special_tokens must be non-negative")

        modules = list(layer_modules or _resolve_layer_modules(backbone))
        if len(modules) < 3:
            raise ValueError("Vim must expose at least three encoder layers")
        indices = tuple(layer_indices or range(len(modules) - 3, len(modules)))
        if len(indices) != 3 or len(set(indices)) != 3:
            raise ValueError("Exactly three distinct Vim layer indices are required")
        normalized = tuple(i if i >= 0 else len(modules) + i for i in indices)
        if any(i < 0 or i >= len(modules) for i in normalized):
            raise IndexError(f"Vim layer indices {indices} are invalid for {len(modules)} layers")
        if tuple(sorted(normalized)) != normalized:
            raise ValueError("Vim layer indices must be ordered from earlier to later")
        self.layer_indices = normalized
        self.layer_identities = OrderedDict(
            (alias, f"encoder_layer_{index}") for alias, index in zip(self.aliases, normalized)
        )
        self._captured: Dict[int, Tensor] = {}
        self._hook_handles = []
        for index in normalized:
            self._hook_handles.append(modules[index].register_forward_hook(self._capture(index)))

    def _capture(self, index: int):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, Tensor):
                raise TypeError(f"Vim layer {index} produced unsupported {type(output)!r}")
            self._captured[index] = output

        return hook

    @staticmethod
    def _batch_first(tokens: Tensor, batch_size: int) -> Tensor:
        if tokens.ndim != 3:
            raise RuntimeError(f"Vim hidden state must be rank 3, got {tuple(tokens.shape)}")
        if tokens.shape[0] == batch_size:
            return tokens
        if tokens.shape[1] == batch_size:
            return tokens.transpose(0, 1)
        raise RuntimeError(
            f"Cannot identify batch dimension in hidden state {tuple(tokens.shape)} for B={batch_size}"
        )

    def forward(self, images: Tensor) -> Mapping[str, Tensor]:
        self._captured.clear()
        if hasattr(self.backbone, "forward_features"):
            self.backbone.forward_features(images)
        else:
            self.backbone(images)
        missing = [index for index in self.layer_indices if index not in self._captured]
        if missing:
            raise RuntimeError(f"Vim forward did not execute selected encoder layers: {missing}")

        result = OrderedDict()
        for alias, index in zip(self.aliases, self.layer_indices):
            hidden = self._batch_first(self._captured[index], images.shape[0])
            if hidden.shape[1] <= self.num_special_tokens:
                raise RuntimeError(
                    f"Layer {index} has {hidden.shape[1]} tokens, cannot remove "
                    f"{self.num_special_tokens} special token(s)"
                )
            result[alias] = hidden[:, self.num_special_tokens :, :]
        return result

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def patch_grid(self, images: Tensor):
        return patch_grid_from_conv(self.backbone, images)

