"""Pinned official VMamba-T s1l8 (SS2D/Cross-Scan), with spatial stage outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

VMAMBA_MODEL = "vmamba_tiny_s1l8"
VMAMBA_COMMIT = "2ed52ead062a51a64521ed3871d52914bf532876"


def official_vmamba_module():
    root = Path(__file__).resolve().parents[3] / "third_party" / "VMamba"
    provenance = json.loads((root / "official_source.json").read_text(encoding="utf-8"))
    source = root / "vmamba.py"
    digest = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if (provenance["repository"] != "https://github.com/MzeroMiko/VMamba"
            or provenance["commit"] != VMAMBA_COMMIT
            or digest != provenance["source_sha256_lf"]):
        raise RuntimeError("Official VMamba source provenance/hash mismatch")
    name = "_vmambaclip_official_vmamba"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise
    elif Path(sys.modules[name].__file__).resolve() != source.resolve():
        raise RuntimeError("Official VMamba resolved to another source file")
    return sys.modules[name], provenance


def load_vmamba_pretrained(core, path, *, expected_sha256):
    """Load every official classification tensor before discarding its head.

    No partial key loading or Vim checkpoint conversion is permitted.
    """
    path = Path(path)
    digest_state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_state.update(chunk)
    digest = digest_state.hexdigest()
    if not expected_sha256 or digest != expected_sha256:
        raise ValueError("VMamba pretrained checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("VMamba checkpoint must be a state dictionary or contain model/state_dict")
    containers = [name for name in ("model", "state_dict") if name in payload]
    if len(containers) > 1:
        raise ValueError("Ambiguous VMamba checkpoint containers")
    state = dict(payload[containers[0]] if containers else payload)
    if state and all(name.startswith("module.") for name in state):
        state = {name[7:]: value for name, value in state.items()}
    expected = core.state_dict()
    conversions = []
    # These are precisely the aliases in upstream SS2D._load_from_state_dict;
    # the published s1l8 checkpoint predates the official grouped Linear form.
    aliases = {"x_proj_weight": "x_proj.weight", "dt_projs_weight": "dt_projs.weight",
               "dt_projs_bias": "dt_projs.bias", "x_proj_bias": "x_proj.bias"}
    for name in list(state):
        prefix, _, leaf = name.rpartition(".")
        if leaf in aliases and name not in expected:
            target = prefix + "." + aliases[leaf]
            if target in state or target not in expected or not torch.is_tensor(state[name]) or state[name].numel() != expected[target].numel():
                raise ValueError("Invalid official VMamba projection conversion: " + name)
            conversions.append({"source": name, "target": target,
                                "before": list(state[name].shape), "after": list(expected[target].shape)})
            state[target] = state.pop(name).reshape(expected[target].shape)
    if set(state) != set(expected):
        raise ValueError("VMamba checkpoint keys differ from official Tiny s1l8; Vim weights are incompatible")
    for name, value in state.items():
        if not torch.is_tensor(value) or value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError("Invalid VMamba checkpoint tensor: " + name)
    core.load_state_dict(state, strict=True)
    return {"success": True, "sha256": digest, "coverage": 1., "key_count": len(state),
            "official_projection_conversions": conversions}


class VMambaSpatialAdapter(nn.Module):
    """Three actual resolutions from stages 1/2/3, before each downsample.

    Inputs/outputs use BCHW, without CLS, sequence hooks, or inferred grids.
    The upstream SS2D blocks already merge their four scans back to image order.
    """

    aliases = ("e", "m", "l")
    feature_layout = "BCHW"
    layer_indices = (1, 2, 3)

    def __init__(self, core, *, keep_final_norm=False):
        super().__init__()
        if not core.channel_first or list(core.dims) != [96, 192, 384, 768] or len(core.layers) != 4:
            raise ValueError("Expected official channel-first hierarchical VMamba Tiny")
        self.backbone = core
        self.state_dims = OrderedDict(zip(self.aliases, core.dims[1:]))
        self.layer_identities = OrderedDict(zip(self.aliases, ("stage1_stride8", "stage2_stride16", "stage3_stride32")))
        self.final_norm = core.classifier.norm if keep_final_norm else None
        del core.classifier

    def set_grad_checkpointing(self, enabled=True):
        for layer in self.backbone.layers:
            for block in layer.blocks:
                block.use_checkpoint = bool(enabled)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("VMamba expects RGB images [B,3,H,W]")
        if any(size < 32 or size % 32 for size in images.shape[-2:]):
            raise ValueError("VMamba Hybrid requires image dimensions divisible by 32")
        x = self.backbone.patch_embed(images)
        result = OrderedDict()
        for index, layer in enumerate(self.backbone.layers):
            x = layer.blocks(x)
            if index in self.layer_indices:
                alias = self.aliases[index - 1]
                if x.ndim != 4 or x.shape[1] != self.state_dims[alias]:
                    raise RuntimeError("VMamba stage did not return its expected BCHW feature map")
                result[alias] = x
            x = layer.downsample(x)
        return result

    def forward_features(self, images):
        if self.final_norm is None:
            raise RuntimeError("Native VMamba readout requires the official final normalization")
        return self.final_norm(self(images)["l"]).mean(dim=(2, 3))


class VMambaNativeCE(nn.Module):
    def __init__(self, backbone, num_classes=7):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, images):
        return self.classifier(self.backbone.forward_features(images))


def create_official_vmamba(*, checkpoint_path=None, checkpoint_sha256=None,
                          model_name=VMAMBA_MODEL, drop_path_rate=.1,
                          gradient_checkpointing=False, keep_final_norm=False):
    if model_name != VMAMBA_MODEL:
        raise ValueError("Only the reviewed official VMamba-T s1l8 is supported")
    module, provenance = official_vmamba_module()
    # The official named factory ignores kwargs. Build its exact VSSM recipe
    # explicitly so the requested drop path/checkpointing settings actually apply.
    core = module.VSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=drop_path_rate,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=1., ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0., ssm_init="v0",
        forward_type="v05_noz", mlp_ratio=4., mlp_act_layer="gelu",
        mlp_drop_rate=0., gmlp=False, patch_norm=True, norm_layer="ln2d",
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=gradient_checkpointing, posembed=False, imgsize=224,
    )
    load_report = None
    if checkpoint_path is not None:
        load_report = load_vmamba_pretrained(core, checkpoint_path, expected_sha256=checkpoint_sha256)
    adapter = VMambaSpatialAdapter(core, keep_final_norm=keep_final_norm)
    metadata = {
        **provenance, "identity": "official_vmamba_tiny_s1l8_ss2d_" + VMAMBA_COMMIT,
        "pretrained_loaded": load_report is not None,
        "initialization_source": str(checkpoint_path) if load_report else "random initialization; no pretrained weights loaded",
        "load_report": load_report, "stage_indices": [1, 2, 3], "state_dims": dict(adapter.state_dims),
        "feature_layout": "BCHW", "readout": "pre-downsample stage maps; no CLS",
        "selective_scan_backend": "cuda_mamba" if module.WITH_SELECTIVESCAN_MAMBA else "official_pytorch_reference",
        "cross_scan_backend": "triton_when_cuda" if module.WITH_TRITON else "official_pytorch",
    }
    return adapter, metadata
