"""Explicit module/gradient policies for all four training stages."""

from __future__ import annotations

from typing import Iterable, List, Optional

from torch import nn

from .adapters import TeacherAdapter
from .model import HybridVisionMambaStudent


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _enable_module(module: Optional[nn.Module], enabled: bool = True) -> None:
    if module is None:
        return
    _set_requires_grad(module, enabled)
    module.train(enabled)


def configure_teacher_stage(teacher: TeacherAdapter, stage: str) -> List[nn.Parameter]:
    if stage not in {"teacher_t1", "teacher_t2"}:
        raise ValueError(f"Unsupported teacher stage: {stage}")
    teacher.eval()
    _set_requires_grad(teacher, False)
    teacher.freeze_text_encoder()
    visual = getattr(teacher.clip_model, "visual", None)
    if not isinstance(visual, nn.Module):
        raise ValueError("Teacher CLIP must expose its visual tower as .visual")
    _enable_module(visual, True)
    if stage == "teacher_t1":
        teacher.logit_scale.requires_grad_(True)
        teacher.classifier.eval()
    else:
        teacher.logit_scale.requires_grad_(False)
        _enable_module(teacher.classifier, True)
    return [parameter for parameter in teacher.parameters() if parameter.requires_grad]


def freeze_teacher_for_student(teacher: TeacherAdapter) -> None:
    _set_requires_grad(teacher, False)
    teacher.eval()


def configure_student_stage(
    student: HybridVisionMambaStudent,
    stage: str,
    *,
    train_text_projection: bool = True,
) -> List[nn.Parameter]:
    if stage not in {"student_s1", "student_s2"}:
        raise ValueError(f"Unsupported student stage: {stage}")
    if stage == "student_s2":
        _set_requires_grad(student, True)
        student.train()
        if not train_text_projection:
            _set_requires_grad(student.text_to_classifier, False)
            student.text_to_classifier.eval()
        return [parameter for parameter in student.parameters() if parameter.requires_grad]

    # S1 only trains modules reached by the five specified losses.
    student.eval()
    _set_requires_grad(student, False)
    _enable_module(student.vim, True)
    _enable_module(student.state_projections, True)
    _enable_module(student.cnn_backbone, True)
    _enable_module(student.cnn_projection, True)
    _enable_module(student.spatial_attention, True)
    # Explicitly keep the unreachable aggregation/fusion/classifier path frozen.
    student.cross_attention.eval()
    student.fusion_projection.eval()
    student.classifier.eval()
    student.text_to_classifier.eval()
    student.global_query.requires_grad_(False)
    return [parameter for parameter in student.parameters() if parameter.requires_grad]


def assert_no_trainable_parameters(module: nn.Module, name: str) -> None:
    trainable = [key for key, value in module.named_parameters() if value.requires_grad]
    if trainable:
        raise RuntimeError(f"{name} unexpectedly has trainable parameters: {trainable[:10]}")

