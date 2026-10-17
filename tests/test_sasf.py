"""CPU interface tests; synthetic backbones do not validate official CUDA kernels."""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from open_clip.vision_mamba_clip.adapters import patch_grid_from_conv
from open_clip.vision_mamba_clip.model import HybridVisionMambaStudent, SASF
from open_clip.vision_mamba_clip.official_vim import OfficialBackbone, OfficialPatchAdapter
from open_clip.vision_mamba_clip.stages import configure_student_stage
from vim_v2.config import prepare, validate
from vim_v2.losses import coefficients, objectives
from vim_v2.optim import configure_stage, groups


class RasterStub(nn.Module):
    """Explicit raster token interface, without claiming to implement Mamba."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 12, 4, stride=4)

    def patch_grid(self, images):
        return patch_grid_from_conv(self, images)

    def forward(self, images):
        tokens = self.patch_embed.proj(images).flatten(2).transpose(1, 2)
        return {"e": tokens, "m": tokens * 1.5, "l": tokens * 2}


class CNNStub(nn.Module):
    output_channels = 8

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=4, padding=1)

    def forward(self, images):
        return self.conv(images)


def student(**kwargs):
    return HybridVisionMambaStudent(
        RasterStub(), {"e": 12, "m": 12, "l": 12}, 8, 7,
        attention_heads=2, cnn_backbone=CNNStub(), **kwargs,
    )


def test_sasf_rectangular_grid_preserves_raster_order():
    module = SASF(4, 8)
    # Turn off the learned residual to inspect the exact reshape round trip.
    for parameter in module.fusion.parameters():
        nn.init.zeros_(parameter)
    tokens = torch.arange(48.).reshape(1, 12, 4)
    spatial, pooled = module(tokens, 3, 4)
    torch.testing.assert_close(spatial, tokens, rtol=0, atol=0)
    torch.testing.assert_close(pooled, module.proj(tokens.mean(1)))
    for bad in (torch.zeros(1, 13, 4), torch.zeros(1, 12, 5), torch.zeros(12, 4)):
        with pytest.raises(ValueError):
            module(bad, 3, 4)


def test_grid_uses_actual_convolution_and_rejects_flipping():
    backbone = RasterStub()
    backbone.patch_embed.proj = nn.Conv2d(3, 12, 3, stride=2, padding=2, dilation=2)
    images = torch.randn(2, 3, 19, 27)
    assert backbone.patch_grid(images) == tuple(backbone.patch_embed.proj(images).shape[-2:])
    backbone.flip_img_sequences_ratio = 1.
    with pytest.raises(ValueError, match="raster"):
        backbone.patch_grid(images)


def test_disabled_and_zero_gate_preserve_legacy_weights_and_outputs():
    torch.manual_seed(42)
    baseline = student().eval()
    torch.manual_seed(42)
    enhanced = student(use_sasf=True).eval()
    baseline_state = baseline.state_dict()
    for name, value in baseline_state.items():
        torch.testing.assert_close(enhanced.state_dict()[name], value, rtol=0, atol=0)
    assert not any(name.startswith(("sasf", "local_")) for name in baseline_state)
    baseline.load_state_dict(baseline_state, strict=True)
    images = torch.randn(2, 3, 16, 24)
    with torch.no_grad():
        a, b = baseline(images), enhanced(images)
    torch.testing.assert_close(a.fused_raw, b.fused_raw, rtol=0, atol=0)
    torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["s1", "s2"])
def test_new_fusion_has_spatial_kv_and_learns_sasf(stage):
    torch.manual_seed(7)
    model = student(use_sasf=True, fusion="mamba_cnn_cross_attention")
    configure_stage(model, stage)
    optimizer_groups = groups(model, {"backbone_lr":.001, "new_lr":.01, "weight_decay":.01})
    names = {name for group in optimizer_groups for name in group["names"]}
    assert {name for name, _ in model.named_parameters() if name.startswith(("sasf", "local_"))} <= names
    optimizer = torch.optim.SGD([p for group in optimizer_groups for p in group["params"]], lr=.05)
    images = torch.randn(3, 3, 16, 24)
    labels = torch.tensor([0, 2, 5])
    attention_shapes = []
    handle = model.local_cross_attention.register_forward_pre_hook(
        lambda module, inputs: attention_shapes.append([tuple(x.shape) for x in inputs[:3]])
    )
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        assert output.fused_raw.shape == (3, 8)
        if stage == "s1":
            loss, _, _ = objectives(None, output.fused_raw, None, coefficients("full_v2", "s1"),
                                     {"feature": torch.randn(3, 8)})
        else:
            loss = F.cross_entropy(output.logits, labels)
        loss.backward()
        assert model.sasf_alpha.grad.abs() > 0
        assert model.local_cross_attention.in_proj_weight.grad.norm() > 0
        assert model.cnn_backbone.conv.weight.grad.norm() > 0
        assert model.vim.patch_embed.proj.weight.grad.norm() > 0
        if step == 0:
            assert model.sasf.proj.weight.grad.count_nonzero() == 0
        else:
            for name, parameter in model.sasf.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                assert parameter.grad.norm() > 0, name
        optimizer.step()
    handle.remove()
    assert attention_shapes == [[(3, 1, 8), (3, 24, 8), (3, 24, 8)]] * 2


def test_legacy_s1_freezes_unreachable_new_fusion():
    model = student(use_sasf=True, fusion="mamba_cnn_cross_attention")
    configure_student_stage(model, "student_s1")
    assert all(not p.requires_grad for p in model.sasf.parameters())
    assert not model.sasf_alpha.requires_grad
    assert all(not p.requires_grad for p in model.local_cross_attention.parameters())
    configure_student_stage(model, "student_s2")
    assert all(p.requires_grad for p in model.sasf.parameters())


def test_new_checkpoint_roundtrip_and_reject_legacy_as_resume(tmp_path):
    model = student(use_sasf=True, fusion="mamba_cnn_cross_attention").eval()
    with torch.no_grad():
        model.sasf_alpha.fill_(.2)
    path = tmp_path / "student.pt"
    torch.save(model.state_dict(), path)
    restored = student(use_sasf=True, fusion="mamba_cnn_cross_attention").eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    images = torch.randn(2, 3, 16, 24)
    with torch.no_grad():
        torch.testing.assert_close(model(images).logits, restored(images).logits, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(student().state_dict(), strict=True)


class TupleBlock(nn.Module):
    def forward(self, hidden, residual=None, inference_params=None):
        return hidden * 2, hidden * 3


class MiddleCLSCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 4
        self.num_tokens = 1
        self.use_middle_cls_token = True
        self.flip_img_sequences_ratio = 0
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 4, 4, stride=4)
        self.patch_embed.num_patches = 6
        self.layers = nn.ModuleList([TupleBlock() for _ in range(3)])

    def forward_features(self, images):
        tokens = self.patch_embed.proj(images).flatten(2).transpose(1, 2)
        tokens = torch.cat([tokens[:, :3], tokens.new_full((len(images), 1, 4), 10000), tokens[:, 3:]], 1)
        for block in self.layers:
            tokens, residual = block(tokens)
        return tokens


@pytest.mark.parametrize("checkpointing", [False, True])
def test_official_patch_adapter_removes_middle_cls_and_adds_residual(checkpointing):
    core = MiddleCLSCore()
    backbone = OfficialBackbone(core, {})
    backbone.set_grad_checkpointing(checkpointing)
    adapter = OfficialPatchAdapter(backbone)
    images = torch.randn(2, 3, 8, 12, requires_grad=True)
    patches = core.patch_embed.proj(images).flatten(2).transpose(1, 2)
    result = adapter(images)
    assert adapter.patch_grid(images) == (2, 3)
    for name, scale in zip(("e", "m", "l"), (5, 10, 20)):
        torch.testing.assert_close(result[name], patches * scale)
    result["l"].sum().backward()
    assert images.grad.norm() > 0
    assert not adapter._captured


def test_v2_plan_records_distinct_model_initialization(tmp_path):
    paths = {"output_root": str(tmp_path / "runs"), "vim": {}, "cnn": {}, "clip_base": {},
             "datasets": {name: {"counts": {"train": 320}, "manifest": {}, "teachers": {}}
                          for name in ("rafdb", "sfew", "affectnet")}}
    path = tmp_path / "paths.json"
    path.write_text(json.dumps(paths), encoding="utf-8")
    plan = prepare(path, tmp_path / "configs", use_sasf=True, cross_branch_fusion=True)
    for row in plan["runs"]:
        config = json.loads(Path(row["config"]).read_text(encoding="utf-8"))
        validate(config)
        if config["student"]["architecture"] == "hybrid_v2":
            assert config["student"]["use_sasf"] is True
            assert config["student"]["fusion"] == "mamba_cnn_cross_attention"
            assert config["init_id"] == "hybrid_v2_sasf_crossattn_pretrained_seed42"
            wrong = copy.deepcopy(config)
            wrong["init_id"] = "hybrid_v2_pretrained_seed42"
            with pytest.raises(ValueError, match="Initialization identity"):
                validate(wrong)
        else:
            assert "use_sasf" not in config["student"]


def test_cross_branch_fusion_rejects_no_cnn():
    with pytest.raises(ValueError, match="requires the CNN"):
        student(use_cnn=False, fusion="mamba_cnn_cross_attention")
