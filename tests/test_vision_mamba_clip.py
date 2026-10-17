import copy
import json

import pytest
import torch
from torch import nn

from open_clip.vision_mamba_clip.adapters import TeacherAdapter, VimAdapter
from open_clip.vision_mamba_clip.checkpoint import (
    load_stage_checkpoint,
    save_stage_checkpoint,
)
from open_clip.vision_mamba_clip.config import resolve_warmup_steps
from open_clip.vision_mamba_clip.losses import (
    Stage1Loss,
    Stage1LossConfig,
    Stage2Loss,
    Stage2LossConfig,
    _distribution_kl_from_log_probs,
)
from open_clip.vision_mamba_clip.metrics import ClassificationMetrics
from open_clip.vision_mamba_clip.model import HybridVisionMambaStudent
from open_clip.vision_mamba_clip.stages import (
    configure_student_stage,
    configure_teacher_stage,
    freeze_teacher_for_student,
)
from open_clip.vision_mamba_clip.text_anchors import load_prompt_schema
from open_clip.vision_mamba_clip.training import (
    student_s1_step,
    student_s2_step,
    teacher_t1_step,
    teacher_t2_step,
)


class TinyVimBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width)

    def forward(self, tokens):
        return tokens + torch.tanh(self.projection(self.norm(tokens)))


class TinyBidirectionalVim(nn.Module):
    """Interface stub only; it is never reported as a real Vim smoke test."""

    def __init__(self, width=12):
        super().__init__()
        self.patch = nn.Conv2d(3, width, kernel_size=4, stride=4)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.layers = nn.ModuleList([TinyVimBlock(width) for _ in range(5)])

    def forward_features(self, images):
        tokens = self.patch(images).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls.expand(images.shape[0], -1, -1), tokens), dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


class TinyCNN(nn.Module):
    output_channels = 8

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),
        )

    def forward(self, images):
        return self.net(images)


class TinyVisual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images):
        return self.net(images).flatten(1)


class TinyCLIP(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.visual = TinyVisual(dim)
        self.token_embedding = nn.Embedding(32, dim)
        self.positional_embedding = nn.Parameter(torch.zeros(4, dim))
        self.text_projection = nn.Parameter(torch.eye(dim))
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def encode_image(self, images, normalize=False):
        result = self.visual(images)
        return torch.nn.functional.normalize(result, dim=-1) if normalize else result

    def encode_text(self, tokens, normalize=False):
        result = self.token_embedding(tokens).mean(dim=1) @ self.text_projection
        return torch.nn.functional.normalize(result, dim=-1) if normalize else result


def build_models():
    vim = VimAdapter(
        TinyBidirectionalVim(),
        backbone_is_bidirectional=True,
        backbone_identity="synthetic-interface-stub",
        num_special_tokens=1,
    )
    student = HybridVisionMambaStudent(
        vim,
        {"e": 12, "m": 12, "l": 12},
        common_dim=8,
        num_classes=7,
        attention_heads=2,
        cnn_backbone=TinyCNN(),
        cnn_channels=8,
    )
    teacher = TeacherAdapter(TinyCLIP(8), num_classes=7)
    return student, teacher


def test_student_contract_concat_fusion_and_label_free_inference():
    student, teacher = build_models()
    images = torch.randn(2, 3, 32, 32)
    student.eval()
    teacher.eval()
    with torch.no_grad():
        output = student(images)
        teacher_output = teacher(images)
    assert student.vim.layer_indices == (2, 3, 4)
    assert list(student.vim.layer_identities) == ["e", "m", "l"]
    assert output.fused_raw.shape == (2, 8)
    assert all(value.shape == (2, 8) for value in output.states_raw.values())
    assert output.cnn_descriptor.shape == (2, 8)
    assert output.patch_norm.shape == (2, 64, 8)
    assert output.attention.shape == (2, 1, 8, 8)
    assert output.logits.shape == (2, 7)
    assert teacher_output["image_raw"].shape == (2, 8)
    assert student.fusion_projection.in_features == 16
    assert student.classifier.final_linear.weight.shape == (7, 8)
    assert student.text_to_classifier.weight.shape == (8, 8)


def test_vim_adapter_rejects_unattested_non_bidirectional_backbone():
    with pytest.raises(ValueError, match="bidirectional"):
        VimAdapter(
            TinyBidirectionalVim(),
            backbone_is_bidirectional=False,
            backbone_identity="unknown",
        )


def test_kl_directions_match_manual_reference_and_differ():
    p = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64)
    q = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    p_to_q = _distribution_kl_from_log_probs(p.log(), q.log())
    q_to_p = _distribution_kl_from_log_probs(q.log(), p.log())
    manual = (p * (p.log() - q.log())).sum()
    assert torch.allclose(p_to_q, manual)
    assert not torch.allclose(p_to_q, q_to_p)


def test_complete_s1_and_s2_gradient_paths_are_finite():
    torch.manual_seed(2)
    student, teacher = build_models()
    anchors = torch.nn.functional.normalize(torch.randn(7, 8), dim=-1)
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([1, 5])
    freeze_teacher_for_student(teacher)
    teacher_bn_before = teacher.clip_model.visual.net[1].running_mean.clone()

    s1 = Stage1Loss(Stage1LossConfig())
    params = configure_student_stage(student, "student_s1")
    report = student_s1_step(student, teacher, s1, images, anchors)
    report["total"].backward()
    assert all(torch.isfinite(value) for value in report.values())
    assert student.state_projections["e"].weight.grad is not None
    assert student.cnn_projection.weight.grad is not None
    assert student.fusion_projection.weight.grad is None
    assert student.global_query.grad is None
    assert student.classifier.final_linear.weight.grad is None
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert torch.equal(teacher_bn_before, teacher.clip_model.visual.net[1].running_mean)
    assert {id(p) for p in params} == {
        id(p) for p in student.parameters() if p.requires_grad
    }

    student.zero_grad(set_to_none=True)
    s2 = Stage2Loss(Stage2LossConfig())
    configure_student_stage(student, "student_s2")
    report = student_s2_step(student, teacher, s2, images, labels, anchors)
    report["total"].backward()
    assert all(torch.isfinite(value) for value in report.values())
    assert student.global_query.grad is not None
    assert student.fusion_projection.weight.grad is not None
    assert student.classifier.final_linear.weight.grad is not None
    assert student.text_to_classifier.weight.grad is not None


def test_teacher_stage_objectives_and_freezing():
    _, teacher = build_models()
    images = torch.randn(3, 3, 32, 32)
    labels = torch.tensor([0, 0, 2])
    anchors = torch.nn.functional.normalize(torch.randn(7, 8), dim=-1)
    text_before = teacher.clip_model.token_embedding.weight.detach().clone()

    parameters = configure_teacher_stage(teacher, "teacher_t1")
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    t1 = teacher_t1_step(teacher, images, labels, anchors)
    t1["total"].backward()
    optimizer.step()
    assert torch.isfinite(t1["total"])
    assert t1["same_class_off_diagonal_pairs"].item() == 2
    assert teacher.clip_model.visual.net[0].weight.grad is not None
    assert teacher.logit_scale.grad is not None
    assert teacher.classifier.weight.grad is None
    assert torch.equal(text_before, teacher.clip_model.token_embedding.weight)

    teacher.zero_grad(set_to_none=True)
    parameters = configure_teacher_stage(teacher, "teacher_t2")
    t2 = teacher_t2_step(teacher, images, labels)
    t2["total"].backward()
    assert torch.isfinite(t2["total"])
    assert teacher.classifier.weight.grad is not None
    assert teacher.logit_scale.grad is None
    assert all(parameter.requires_grad is False for parameter in teacher.text_parameters())
    assert teacher.clip_model.positional_embedding.requires_grad is False


def test_warmup_conflict_is_never_silent():
    with pytest.raises(ValueError, match="paper schedule is not effective"):
        resolve_warmup_steps(100, paper_warmup_steps=10_000, strict=True)
    assert resolve_warmup_steps(
        100, paper_warmup_steps=10_000, strict=False, engineering_warmup_ratio=0.05
    ) == 5


def test_classification_metrics_include_recall_f1_and_confusion():
    accumulator = ClassificationMetrics(3)
    logits = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
    ])
    accumulator.update(logits, torch.tensor([0, 1, 2, 2]))
    result = accumulator.compute()
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["per_class_recall"] == pytest.approx([1.0, 1.0, 0.5])
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert 0 < result["macro_f1"] < 1


def test_spatial_target_detach_and_numerical_boundaries():
    torch.manual_seed(4)
    student, teacher = build_models()
    student.eval()
    teacher.eval()
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    anchors = torch.nn.functional.normalize(torch.randn(7, 8), dim=-1)
    output = student(images)
    output.patch_norm = torch.zeros_like(output.patch_norm, requires_grad=True)
    output.attention = torch.full_like(output.attention, 1e-30, requires_grad=True)
    teacher_output = teacher(images)
    criterion = Stage2Loss(Stage2LossConfig(
        lambda_task=0.0,
        lambda_distill=0.0,
        lambda_anchor=0.0,
        lambda_au=1.0,
        gamma_sparse=0.0,
        tau_low=1e-3,
        detach_spatial_target=True,
    ))
    report = criterion(
        output,
        student,
        teacher_output["image_raw"],
        torch.tensor([[1e4, -1e4, 0, 2, 3, 4, 5]]).repeat(2, 1),
        labels,
        anchors,
    )
    report["total"].backward()
    assert torch.isfinite(report["total"])
    assert output.patch_norm.grad is None
    assert output.attention.grad is not None
    assert torch.isfinite(output.attention.grad).all()


def _metadata(stage):
    return {
        "stage": stage,
        "model_identity": "synthetic-interface-stub",
        "initialization_source": "random-test",
        "common_dim": 8,
        "class_order": [str(i) for i in range(7)],
        "data_manifest_hash": "data-hash",
        "prompt_schema_hash": "prompt-hash",
        "preprocess": {"size": 32},
        "seed": 1,
        "actual_steps": 1,
    }


def test_checkpoint_round_trip_and_identity_rejection(tmp_path):
    student, _ = build_models()
    student.eval()
    images = torch.randn(2, 3, 32, 32)
    expected = student(images).logits.detach()
    path = tmp_path / "student_s1.pt"
    save_stage_checkpoint(str(path), student, _metadata("student_s1"))

    restored, _ = build_models()
    load_stage_checkpoint(
        str(path),
        restored,
        expected_stage="student_s1",
        common_dim=8,
        class_order=[str(i) for i in range(7)],
        data_manifest_hash="data-hash",
        prompt_schema_hash="prompt-hash",
    )
    restored.eval()
    assert torch.allclose(expected, restored(images).logits, atol=1e-6)
    with pytest.raises(ValueError, match="Expected student_s2"):
        load_stage_checkpoint(
            str(path),
            restored,
            expected_stage="student_s2",
            common_dim=8,
            class_order=[str(i) for i in range(7)],
        )
    with pytest.raises(FileExistsError):
        save_stage_checkpoint(str(path), student, _metadata("student_s1"))


def test_prompt_order_mismatch_is_rejected(tmp_path):
    value = {
        "class_order": ["1", "0"],
        "prompts": {"1": "sourced prompt one", "0": "sourced prompt zero"},
        "source": "unit-test source",
    }
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_prompt_schema(str(path), ["0", "1"])

