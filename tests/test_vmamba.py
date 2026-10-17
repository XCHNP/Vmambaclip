"""Real official SS2D CPU tests. CUDA/BF16 acceptance is a separate cloud check."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from open_clip.vision_mamba_clip.inference import load_student_for_inference
from open_clip.vision_mamba_clip.official_vmamba import (
    VMAMBA_MODEL, create_official_vmamba, load_vmamba_pretrained,
    official_vmamba_module,
)
from vim_v2.common import fingerprint
from vim_v2.config import prepare, validate
from vim_v2.losses import coefficients, objectives
from vim_v2.models import make_student
from vim_v2.optim import configure_stage, groups, probes


@pytest.fixture
def vmamba_config(tmp_path):
    paths = {"output_root": str(tmp_path / "runs"), "vmamba": {"model_name": VMAMBA_MODEL},
             "cnn": {}, "clip_base": {},
             "datasets": {name: {"counts": {"train": 320}, "manifest": {}, "teachers": {}}
                          for name in ("rafdb", "sfew", "affectnet")}}
    path = tmp_path / "paths.json"
    path.write_text(json.dumps(paths), encoding="utf-8")
    plan = prepare(path, tmp_path / "configs", backbone="vmamba", use_sasf=True, cross_branch_fusion=True)
    configs = [json.loads(Path(row["config"]).read_text(encoding="utf-8")) for row in plan["runs"]]
    return next(config for config in configs if config["run_id"] == "rafdb/full_v2"), configs


def test_vmamba_plan_has_no_vim_cls_or_layer_assumptions(vmamba_config):
    hybrid, configs = vmamba_config
    for config in configs:
        validate(config)
        student = config["student"]
        assert student["official_model_name"] == VMAMBA_MODEL
        assert student["stage_indices"] == [1, 2, 3]
        assert "cls_position_policy" not in student and "layer_indices" not in student
        assert config["init_id"].startswith("vmamba_")
    assert hybrid["init_id"] == "vmamba_hybrid_sasf_crossattn_pretrained_seed42"
    bad = copy.deepcopy(hybrid)
    bad["student"]["official_model_name"] = "vim_small_patch16"
    with pytest.raises(ValueError, match="VMamba"):
        validate(bad)


def test_real_vmamba_matches_official_recipe_and_multiscale_readout():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    adapter, metadata = create_official_vmamba(keep_final_norm=True, drop_path_rate=.1)
    module, _ = official_vmamba_module()
    torch.manual_seed(42)
    upstream = module.vmamba_tiny_s1l8(pretrained=False)
    for name, value in adapter.backbone.state_dict().items():
        torch.testing.assert_close(value, upstream.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(adapter.final_norm.weight, upstream.classifier.norm.weight)
    # Compare the exact upstream final feature readout, without its ImageNet head.
    upstream.classifier.head = torch.nn.Identity()
    upstream.eval(); adapter.eval()
    images = torch.randn(1, 3, 32, 64)
    with torch.no_grad():
        torch.testing.assert_close(adapter.forward_features(images), upstream(images), rtol=1e-5, atol=1e-6)
        result = adapter(torch.randn(1, 3, 256, 256))
    assert {name: tuple(value.shape) for name, value in result.items()} == {
        "e": (1, 192, 32, 32), "m": (1, 384, 16, 16), "l": (1, 768, 8, 8)}
    assert metadata["pretrained_loaded"] is False
    assert sum(isinstance(block.op, module.SS2D) for layer in adapter.backbone.layers for block in layer.blocks) == 14
    with pytest.raises(ValueError, match="divisible by 32"):
        adapter(torch.randn(1, 3, 48, 48))


def test_real_vmamba_sasf_cnn_gradients_and_checkpointed_recompute(vmamba_config):
    config, _ = vmamba_config
    torch.set_num_threads(2)
    model, report = make_student(config, initialize=False)
    assert report["official"]["pretrained_loaded"] is False
    assert model.sasf.in_dim == 768
    with torch.no_grad():
        model.sasf_alpha.fill_(.2)
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 4])
    selected = probes(model)
    assert len([name for name in selected if name.startswith("vmamba.stage_") and ".block_" in name]) == 14
    optimizer_groups = groups(model, config["optimizer"])
    assert next(group for group in optimizer_groups if "vim.backbone.layers.0.blocks.0.op.A_logs" in group["names"])["weight_decay"] == 0.
    for stage in ("s1", "s2"):
        model.zero_grad(set_to_none=True)
        configure_stage(model, stage)
        output = model(images)
        target = {"feature": torch.randn(2, 768), "logits": torch.randn(2, 7)}
        loss, _, _ = objectives(output.logits, output.fused_raw, labels, coefficients("full_v2", stage), target)
        loss.backward()
        for name in ("sasf.branches.0.0.weight", "sasf.fusion.0.weight", "sasf.proj.weight",
                     "local_cross_attention.in_proj_weight", "cnn_backbone.stem.0.weight",
                     "vim.backbone.layers.0.blocks.0.op.in_proj.weight",
                     "vim.backbone.layers.3.blocks.1.op.in_proj.weight"):
            parameter = dict(model.named_parameters())[name]
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.norm() > 0, name
        assert (model.classifier.final_linear.weight.grad is None) == (stage == "s1")


def test_vmamba_inference_reconstructs_without_teacher_or_pretrained_files(vmamba_config, tmp_path):
    config, _ = vmamba_config
    torch.set_num_threads(2)
    model, _ = make_student(config, initialize=False)
    model.eval()
    with torch.no_grad():
        model.sasf_alpha.fill_(.2)
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = model(images).logits
    path = tmp_path / "student.pt"
    torch.save({"model": model.state_dict(), "identity": {"config": config, "config_sha256": fingerprint(config)}}, path)
    with patch("vim_v2.models.check_asset", side_effect=AssertionError("Inference must not load training assets")):
        restored, class_order, transform, _ = load_student_for_inference(str(path))
    assert len(class_order) == 7
    assert tuple(transform.transforms[-1].mean) == tuple(config["preprocess"]["student_mean"])
    with torch.no_grad():
        torch.testing.assert_close(restored(images).logits, expected, rtol=0, atol=0)


def test_vmamba_loader_rejects_wrong_hash_and_vim_weights(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"model": {"pos_embed": torch.zeros(1, 257, 384)}}, path)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    core = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="SHA256"):
        load_vmamba_pretrained(core, path, expected_sha256="wrong")
    with pytest.raises(ValueError, match="Vim weights are incompatible"):
        load_vmamba_pretrained(core, path, expected_sha256=digest)


def test_real_vmamba_checkpointing_preserves_forward_and_backward():
    torch.set_num_threads(2)
    adapter, _ = create_official_vmamba()
    adapter.eval()
    images = torch.randn(1, 3, 32, 32, requires_grad=True)
    outputs, gradients = [], []
    for enabled in (False, True):
        adapter.set_grad_checkpointing(enabled)
        adapter.zero_grad(set_to_none=True)
        images.grad = None
        result = adapter(images)
        loss = sum(value.square().mean() for value in result.values())
        loss.backward()
        outputs.append(result["l"].detach().clone())
        gradients.append({"input": images.grad.clone(),
                          "first_ss2d": adapter.backbone.layers[0].blocks[0].op.in_proj.weight.grad.clone(),
                          "last_ss2d": adapter.backbone.layers[3].blocks[1].op.in_proj.weight.grad.clone()})
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-5, atol=1e-6)
