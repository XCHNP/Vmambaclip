"""Official hustvl/Vim Small with explicit middle-CLS/residual-stream adapters."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .adapters import patch_grid_from_conv

MODEL_NAME = "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"


class OfficialBackbone(nn.Module):
    def __init__(self, core, provenance):
        super().__init__()
        self.core = core
        self.provenance = provenance
        self.gradient_checkpointing = False
        self.embed_dim = int(core.embed_dim)
        self.cls_index = core.patch_embed.num_patches // 2
        if not core.use_middle_cls_token or core.num_tokens != 1 or core.flip_img_sequences_ratio > 0:
            raise ValueError("This protocol requires one fixed middle CLS without sequence flipping")
        for layer in core.layers:
            original = layer.forward

            def wrap(original_forward):
                def forward(hidden_states, residual=None, inference_params=None):
                    if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                        return checkpoint(original_forward, hidden_states, residual, inference_params,
                                          use_reentrant=False, preserve_rng_state=True)
                    return original_forward(hidden_states, residual, inference_params)
                return forward
            layer.forward = wrap(original)

    @property
    def layers(self):
        return self.core.layers

    def set_grad_checkpointing(self, enabled=True):
        self.gradient_checkpointing = bool(enabled)

    def forward_features(self, images):
        return self.core.forward_features(images)

    def forward(self, images):
        return self.forward_features(images)


class OfficialPatchAdapter(nn.Module):
    aliases = ("e", "m", "l")

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.layer_indices = tuple(range(len(backbone.layers) - 3, len(backbone.layers)))
        self.layer_identities = OrderedDict((alias, f"encoder_layer_{i}") for alias, i in zip(self.aliases, self.layer_indices))
        self._captured = {}
        self._recording = False
        self._handles = [backbone.layers[i].register_forward_hook(self._capture(i)) for i in self.layer_indices]

    def _capture(self, index):
        def hook(module, args, output):
            if not self._recording:
                return
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("Official Vim block must return (mixer_output, residual)")
            hidden, residual = output
            stream = hidden if residual is None else hidden + residual
            self._captured[index] = stream
        return hook

    def forward(self, images):
        self._captured.clear()
        self._recording = True
        try:
            self.backbone.forward_features(images)
        finally:
            self._recording = False
        result = OrderedDict()
        for name, index in zip(self.aliases, self.layer_indices):
            tokens = self._captured[index]
            cls = self.backbone.cls_index
            result[name] = torch.cat((tokens[:, :cls], tokens[:, cls + 1:]), dim=1)
            if result[name].shape[1] != self.backbone.core.patch_embed.num_patches:
                raise RuntimeError("Patch extraction retained an incorrect CLS or patch count")
        self._captured.clear()
        return result

    def patch_grid(self, images):
        return patch_grid_from_conv(self.backbone.core, images)


class MambaCE(nn.Module):
    def __init__(self, backbone, num_classes=7):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, images):
        return self.classifier(self.backbone.forward_features(images))


def create_official_vim(*, checkpoint_path=None, img_size=256, drop_path_rate=0.1, source_root=None):
    project = Path(__file__).resolve().parents[3]
    root = Path(source_root or project / "third_party/Vim")
    provenance = json.loads((root / "official_source.json").read_text())
    if provenance["repository"] != "https://github.com/hustvl/Vim":
        raise ValueError("Unexpected official Vim source repository")
    sys.path.insert(0, str(root / "vim"))
    module = importlib.import_module("models_mamba")
    if Path(module.__file__).resolve() != (root / "vim/models_mamba.py").resolve():
        raise RuntimeError("models_mamba resolved to a different repository")
    mixer_module = importlib.import_module('mamba_ssm.modules.mamba_simple')
    mixer_hash = hashlib.sha256(Path(mixer_module.__file__).read_bytes()).hexdigest()
    official_mixer_hash = hashlib.sha256((root / 'mamba-1p1p1/mamba_ssm/modules/mamba_simple.py').read_bytes()).hexdigest()
    if mixer_hash != official_mixer_hash:
        raise RuntimeError('Installed Mamba mixer differs from the pinned official Vim fork')
    factory = getattr(module, MODEL_NAME)
    core = factory(pretrained=False, img_size=img_size, num_classes=0, drop_path_rate=drop_path_rate)
    if any(getattr(layer.mixer, "bimamba_type", None) != "v2" for layer in core.layers):
        raise RuntimeError("Official backbone is not the expected bidirectional v2 mixer")
    loaded = False
    if checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = payload.get("model", payload)
        state = {key: value for key, value in state.items() if not key.startswith("head.")}
        # A pretrained 224 checkpoint needs middle-CLS-aware interpolation to 256.
        pos = state.get("pos_embed")
        if pos is not None and pos.shape != core.pos_embed.shape:
            source_count = pos.shape[1] - 1
            side = int(source_count ** 0.5)
            source_cls = source_count // 2
            cls = pos[:, source_cls:source_cls + 1]
            patches = torch.cat((pos[:, :source_cls], pos[:, source_cls + 1:]), 1)
            patches = patches.reshape(1, side, side, pos.shape[-1]).permute(0, 3, 1, 2)
            target_side = int(core.patch_embed.num_patches ** 0.5)
            patches = torch.nn.functional.interpolate(patches.float(), size=(target_side, target_side), mode="bicubic", align_corners=False)
            patches = patches.permute(0, 2, 3, 1).reshape(1, target_side ** 2, -1).to(pos.dtype)
            middle = core.patch_embed.num_patches // 2
            state["pos_embed"] = torch.cat((patches[:, :middle], cls, patches[:, middle:]), 1)
        core.load_state_dict(state, strict=True)
        loaded = True
    initialization = hashlib.sha256()
    for key, value in sorted(core.state_dict().items()):
        initialization.update(key.encode())
        initialization.update(value.detach().cpu().contiguous().numpy().tobytes())
    metadata = {
        **provenance, "identity": f"official_hustvl_vim_small_bimambav2_{provenance['commit']}_img{img_size}",
        "model_name": MODEL_NAME, "pretrained_loaded": loaded,
        "initialization_source": str(checkpoint_path) if checkpoint_path else "official architecture random initialization, seed=42",
        "embed_dim": 384, "depth": 24, "image_size": img_size, "patch_size": 16,
        "cls_index": core.patch_embed.num_patches // 2, "patch_count": core.patch_embed.num_patches,
        "hook_semantics": "post-block mixer output plus residual; fixed middle CLS excluded",
        "source_file_sha256": hashlib.sha256((root / "vim/models_mamba.py").read_bytes()).hexdigest(),
        "installed_mixer_sha256": mixer_hash,
        "backbone_initialization_sha256": initialization.hexdigest(),
        "dt_initialization": 'official forward custom dt bias protected; official backward zero bias unchanged',
    }
    return OfficialBackbone(core, metadata), metadata
