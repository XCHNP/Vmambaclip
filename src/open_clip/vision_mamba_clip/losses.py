"""Numerically explicit losses for the four VisionMambaCLIP stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import HybridVisionMambaStudent, StudentOutput


def _finite(name: str, value: Tensor) -> Tensor:
    if not torch.isfinite(value).all():
        raise FloatingPointError(
            f"{name} is non-finite: shape={tuple(value.shape)}, dtype={value.dtype}"
        )
    return value


def _cosine_distance(first: Tensor, second: Tensor) -> Tensor:
    return (1.0 - F.cosine_similarity(first.float(), second.float(), dim=-1)).mean()


def teacher_contrastive_loss(
    image_raw: Tensor, text_batch_raw: Tensor, logit_scale: Tensor
) -> Tensor:
    """T1 one-to-one symmetric CLIP loss (same-class negatives are retained)."""
    if image_raw.shape != text_batch_raw.shape:
        raise ValueError(
            f"T1 paired features must have identical shape, got "
            f"{tuple(image_raw.shape)} and {tuple(text_batch_raw.shape)}"
        )
    image = F.normalize(image_raw.float(), dim=-1)
    text = F.normalize(text_batch_raw.float(), dim=-1)
    scale = logit_scale.float().exp()
    _finite("T1 logit scale", scale)
    similarities = scale * image @ text.T
    target = torch.arange(image.shape[0], device=image.device)
    loss = 0.5 * (
        F.cross_entropy(similarities, target) + F.cross_entropy(similarities.T, target)
    )
    return _finite("T1 loss", loss)


def teacher_prototype_loss(
    image_raw: Tensor, text_anchors: Tensor, labels: Tensor, logit_scale: Tensor,
    class_weights: Tensor = None,
) -> Tensor:
    """T1 class-prototype CE; same-class images are never treated as negatives."""
    if image_raw.ndim != 2 or text_anchors.ndim != 2 or image_raw.shape[1] != text_anchors.shape[1]:
        raise ValueError("T1 image/anchor features must be [B,D] and [C,D]")
    if labels.ndim != 1 or labels.shape[0] != image_raw.shape[0]:
        raise ValueError("T1 requires one canonical label per image")
    image = F.normalize(image_raw.float(), dim=-1)
    anchors = F.normalize(text_anchors.detach().float(), dim=-1)
    scale = logit_scale.float().clamp(max=4.605170185988092).exp()
    logits = scale * image @ anchors.T
    weights = class_weights.to(logits) if class_weights is not None else None
    return _finite("T1 prototype CE", F.cross_entropy(logits, labels, weight=weights))


def duplicate_label_statistics(labels: Tensor) -> Dict[str, int]:
    _, counts = labels.detach().view(-1).unique(return_counts=True)
    return {
        "batch_size": int(labels.numel()),
        "unique_classes": int(counts.numel()),
        "same_class_off_diagonal_pairs": int(((counts * (counts - 1)).sum()).item()),
    }


def teacher_classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return _finite("T2 loss", F.cross_entropy(logits.float(), labels))


@dataclass(frozen=True)
class Stage1LossConfig:
    lambda_it: float = 1.0
    lambda_align: float = 1.0
    lambda_cnn: float = 1.0
    lambda_tv: float = 1e-4
    lambda_sparse: float = 1e-3
    layer_weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    tau_it: float = 2.0

    def validate(self) -> None:
        if self.tau_it <= 0:
            raise ValueError("tau_it must be positive")
        if len(self.layer_weights) != 3 or any(w < 0 for w in self.layer_weights):
            raise ValueError("layer_weights must contain three non-negative values")
        if abs(sum(self.layer_weights) - 1.0) > 1e-6:
            raise ValueError("layer_weights must sum to one")


@dataclass(frozen=True)
class Stage2LossConfig:
    lambda_task: float = 1.0
    lambda_distill: float = 0.2
    beta_logit: float = 1.0
    lambda_anchor: float = 0.05
    lambda_au: float = 0.1
    gamma_sparse: float = 1e-3
    tau_logit: float = 2.0
    tau_low: float = 0.1
    detach_spatial_target: bool = True
    epsilon: float = 1e-6

    def validate(self) -> None:
        if self.tau_logit <= 0 or self.tau_low <= 0:
            raise ValueError("distillation temperatures must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


def _distribution_kl_from_log_probs(log_p: Tensor, log_q: Tensor) -> Tensor:
    """Mean batch KL(P || Q), with no extra division by class/spatial count."""
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()


class Stage1Loss(nn.Module):
    aliases = ("e", "m", "l")

    def __init__(self, config: Stage1LossConfig):
        super().__init__()
        config.validate()
        self.config = config

    def forward(
        self,
        student: StudentOutput,
        teacher_image_raw: Tensor,
        text_anchors: Tensor,
    ) -> Dict[str, Tensor]:
        teacher_norm = F.normalize(teacher_image_raw.detach().float(), dim=-1)
        anchors = F.normalize(text_anchors.detach().float(), dim=-1)
        teacher_log_p = F.log_softmax(
            teacher_norm @ anchors.T / self.config.tau_it, dim=-1
        )

        loss_it = teacher_norm.new_zeros(())
        loss_align = teacher_norm.new_zeros(())
        for name, weight in zip(self.aliases, self.config.layer_weights):
            student_norm = F.normalize(student.states_norm[name].float(), dim=-1)
            student_log_p = F.log_softmax(
                student_norm @ anchors.T / self.config.tau_it, dim=-1
            )
            # P: teacher || student.
            loss_it = loss_it + weight * _distribution_kl_from_log_probs(
                teacher_log_p, student_log_p
            )
            loss_align = loss_align + weight * _cosine_distance(
                student.states_raw[name], teacher_image_raw.detach()
            )
        loss_it = loss_it * (self.config.tau_it ** 2)
        loss_cnn = _cosine_distance(student.cnn_distill, teacher_image_raw.detach())

        attention = student.attention.float()
        dh = attention[:, :, 1:, :] - attention[:, :, :-1, :]
        dw = attention[:, :, :, 1:] - attention[:, :, :, :-1]
        loss_tv = (
            dh.square().flatten(1).sum(dim=1) + dw.square().flatten(1).sum(dim=1)
        ).mean()
        loss_sparse = attention.mean()

        weighted_it = self.config.lambda_it * loss_it
        weighted_align = self.config.lambda_align * loss_align
        weighted_cnn = self.config.lambda_cnn * loss_cnn
        weighted_tv = self.config.lambda_tv * loss_tv
        weighted_sparse = self.config.lambda_sparse * loss_sparse
        total = weighted_it + weighted_align + weighted_cnn + weighted_tv + weighted_sparse
        report = {
            "it": loss_it,
            "align": loss_align,
            "cnn_pre": loss_cnn,
            "tv": loss_tv,
            "sparse": loss_sparse,
            "weighted_it": weighted_it,
            "weighted_align": weighted_align,
            "weighted_cnn_pre": weighted_cnn,
            "weighted_tv": weighted_tv,
            "weighted_sparse": weighted_sparse,
            "total": total,
        }
        for name, value in report.items():
            _finite(f"S1 {name}", value)
        return report


class Stage2Loss(nn.Module):
    def __init__(self, config: Stage2LossConfig):
        super().__init__()
        config.validate()
        self.config = config

    def forward(
        self,
        student_output: StudentOutput,
        student_model: HybridVisionMambaStudent,
        teacher_image_raw: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
        text_anchors: Tensor,
    ) -> Dict[str, Tensor]:
        if student_output.logits is None:
            raise ValueError("S2 requires student logits")
        logits = student_output.logits.float()
        teacher_logits = teacher_logits.detach().float()
        anchors = F.normalize(text_anchors.detach().float(), dim=-1)
        if anchors.shape != student_model.classifier.final_linear.weight.shape:
            raise ValueError(
                "Text anchors and final classifier weight must both be [C,D], got "
                f"{tuple(anchors.shape)} and "
                f"{tuple(student_model.classifier.final_linear.weight.shape)}"
            )
        if labels.min() < 0 or labels.max() >= anchors.shape[0]:
            raise IndexError("S2 labels are outside the text-anchor class order")

        loss_ce = F.cross_entropy(logits, labels)
        loss_feat = _cosine_distance(student_output.fused_raw, teacher_image_raw.detach())

        log_p_student = F.log_softmax(logits / self.config.tau_logit, dim=-1)
        log_p_teacher = F.log_softmax(teacher_logits / self.config.tau_logit, dim=-1)
        # P: student || teacher.
        loss_logit = (self.config.tau_logit ** 2) * _distribution_kl_from_log_probs(
            log_p_student, log_p_teacher
        )
        loss_distill = loss_feat + self.config.beta_logit * loss_logit

        classifier_weight = student_model.classifier.final_linear.weight.float()
        projected_text = student_model.text_to_classifier(anchors.to(
            dtype=student_model.text_to_classifier.weight.dtype
        )).float()
        loss_anchor = _cosine_distance(classifier_weight, projected_text)

        target_text = anchors[labels]
        similarity = (student_output.patch_norm.float() * target_text[:, None, :]).sum(dim=-1)
        minimum = similarity.amin(dim=-1, keepdim=True)
        maximum = similarity.amax(dim=-1, keepdim=True)
        normalized = (similarity - minimum) / (maximum - minimum).clamp_min(
            self.config.epsilon
        )
        if self.config.detach_spatial_target:
            normalized = normalized.detach()
        log_q_target = F.log_softmax(normalized / self.config.tau_low, dim=-1)
        log_p_attention = F.log_softmax(student_output.attention.float().flatten(1), dim=-1)
        if log_p_attention.shape != log_q_target.shape:
            raise RuntimeError(
                f"Attention and AU target spatial shapes differ: "
                f"{tuple(log_p_attention.shape)} vs {tuple(log_q_target.shape)}"
            )
        # P: student attention || target.
        loss_au_kl = _distribution_kl_from_log_probs(log_p_attention, log_q_target)
        loss_sparse = student_output.attention.float().mean()
        loss_au = loss_au_kl + self.config.gamma_sparse * loss_sparse

        weighted_task = self.config.lambda_task * loss_ce
        weighted_distill = self.config.lambda_distill * loss_distill
        weighted_anchor = self.config.lambda_anchor * loss_anchor
        weighted_au = self.config.lambda_au * loss_au
        total = weighted_task + weighted_distill + weighted_anchor + weighted_au
        report = {
            "ce": loss_ce,
            "feature": loss_feat,
            "logit": loss_logit,
            "distill": loss_distill,
            "weight_align": loss_anchor,
            "au_kl": loss_au_kl,
            "sparse": loss_sparse,
            "au_spatial": loss_au,
            "weighted_task": weighted_task,
            "weighted_distill": weighted_distill,
            "weighted_anchor": weighted_anchor,
            "weighted_au": weighted_au,
            "total": total,
        }
        for name, value in report.items():
            _finite(f"S2 {name}", value)
        return report

