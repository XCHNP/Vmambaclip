"""Small, reusable four-stage training steps.

This module intentionally does not hide checkpoint identity checks or data
construction. The CLI composes these steps with the shared FER protocol.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .adapters import TeacherAdapter
from .losses import (
    Stage1Loss,
    Stage2Loss,
    duplicate_label_statistics,
    teacher_classification_loss,
    teacher_contrastive_loss,
    teacher_prototype_loss,
)
from .model import HybridVisionMambaStudent


def _autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None or device.type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def teacher_t1_step(
    teacher: TeacherAdapter,
    images: Tensor,
    labels: Tensor,
    text_anchors: Tensor,
) -> Dict[str, Tensor]:
    image_raw = teacher.encode_image(images)
    loss = teacher_prototype_loss(image_raw, text_anchors, labels, teacher.logit_scale,
                                  getattr(teacher, 'class_weights', None))
    report = {"total": loss, "effective_logit_scale": teacher.logit_scale.detach().float().clamp(max=4.605170185988092).exp()}
    report.update({key: torch.tensor(value, device=loss.device) for key, value in duplicate_label_statistics(labels).items()})
    return report


def teacher_t2_step(
    teacher: TeacherAdapter,
    images: Tensor,
    labels: Tensor,
) -> Dict[str, Tensor]:
    output = teacher.forward_classifier(images)
    return {"total": teacher_classification_loss(output["logits"], labels)}


def student_s1_step(
    student: HybridVisionMambaStudent,
    teacher: TeacherAdapter,
    criterion: Stage1Loss,
    images: Tensor,
    text_anchors: Tensor,
) -> Dict[str, Tensor]:
    with torch.no_grad():
        teacher_image = teacher.encode_image(images)
    return criterion(student(images), teacher_image, text_anchors)


def student_s2_step(
    student: HybridVisionMambaStudent,
    teacher: TeacherAdapter,
    criterion: Stage2Loss,
    images: Tensor,
    labels: Tensor,
    text_anchors: Tensor,
) -> Dict[str, Tensor]:
    with torch.no_grad():
        teacher_output = teacher.forward_classifier(images)
    output = student(images)
    return criterion(
        output,
        student,
        teacher_output["image_raw"],
        teacher_output["logits"],
        labels,
        text_anchors,
    )


def optimizer_step(
    report: Dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    scaler=None,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        report["total"].backward()
        optimizer.step()
    else:
        scaler.scale(report["total"]).backward()
        scaler.step(optimizer)
        scaler.update()


def gradient_summary(module: nn.Module) -> Dict[str, Dict[str, float]]:
    result = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        result[name] = {
            "norm": float(grad.norm().item()),
            "max_abs": float(grad.abs().max().item()),
            "finite": bool(torch.isfinite(grad).all().item()),
        }
    return result

