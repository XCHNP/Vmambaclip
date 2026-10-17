"""Classification metrics that do not require scikit-learn."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


class ClassificationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)

    def update(self, logits: Tensor, labels: Tensor) -> None:
        predictions = logits.detach().argmax(dim=-1).view(-1).cpu().to(torch.int64)
        labels = labels.detach().view(-1).cpu().to(torch.int64)
        if labels.numel() != predictions.numel():
            raise ValueError("Prediction and label counts differ")
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
            raise IndexError("Metric labels are outside the configured class range")
        indices = labels * self.num_classes + predictions
        self.confusion += torch.bincount(
            indices, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict:
        matrix = self.confusion.float()
        true_positive = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
        precision = torch.where(
            predicted > 0, true_positive / predicted, torch.zeros_like(predicted)
        )
        denominator = precision + recall
        f1 = torch.where(
            denominator > 0,
            2 * precision * recall / denominator,
            torch.zeros_like(denominator),
        )
        total = matrix.sum()
        accuracy = true_positive.sum() / total if total > 0 else total.new_zeros(())
        return {
            "accuracy": float(accuracy.item()),
            "macro_f1": float(f1.mean().item()),
            "uar": float(recall.mean().item()),
            "macro_recall": float(recall.mean().item()),
            "balanced_accuracy": float(recall.mean().item()),
            "per_class_precision": [float(value) for value in precision.tolist()],
            "per_class_f1": [float(value) for value in f1.tolist()],
            "per_class_support": [int(value) for value in support.tolist()],
            "class_coverage": int((support > 0).sum().item()),
            "macro_definition": "unweighted mean over all configured classes; absent classes contribute zero",
            "row_normalized_confusion_matrix": (matrix / support.clamp_min(1).unsqueeze(1)).tolist(),
            "per_class_recall": [float(value) for value in recall.tolist()],
            "confusion_matrix": self.confusion.tolist(),
            "sample_count": int(total.item()),
        }

