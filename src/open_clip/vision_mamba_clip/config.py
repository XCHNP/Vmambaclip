"""Configuration loading and protocol checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping


def load_config(path: str) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    config["config_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return config


def validate_config(config: Mapping) -> None:
    for key in (
        "protocol", "teacher", "student", "data", "stage1", "stage2", "optimizer",
        "stage_epochs", "stage_controls", "checkpoints",
    ):
        if key not in config:
            raise ValueError(f"Configuration is missing {key!r}")
    protocol = config["protocol"]
    if int(protocol["image_size"]) != 256:
        raise ValueError("The main VisionMambaCLIP protocol requires 256x256 images")
    if int(protocol["num_classes"]) != 7:
        raise ValueError("The RAF-DB main protocol requires C=7")
    if config["teacher"]["architecture"] != "ViT-L-14":
        raise ValueError("The main teacher must be CLIP ViT-L/14")
    if config["student"]["family"] != "bidirectional_vim":
        raise ValueError("The main student must use bidirectional Vim")
    if config["student"]["fusion"] not in {"concat_projection", "mamba_cnn_cross_attention"}:
        raise ValueError("Unsupported student fusion")
    if type(config["student"].get("use_sasf", False)) is not bool:
        raise ValueError("student.use_sasf must be boolean")
    if config["student"]["fusion"] == "mamba_cnn_cross_attention" and not config["student"].get("use_cnn", True):
        raise ValueError("Cross-branch attention requires CNN features")
    if config["student"].get("use_cnn", True) and config["student"]["cnn"] != "resnet18_layer3":
        raise ValueError("The default local branch must be ResNet18 through layer3")
    if not config["student"].get("use_cnn", True) and float(config["stage2"]["lambda_au"]) != 0:
        raise ValueError("The no-CNN ablation must disable the AU spatial loss")
    if int(config["optimizer"]["gradient_accumulation_steps"]) != 1:
        raise ValueError("This runner currently requires gradient_accumulation_steps=1")
    if config.get("profile") == "full_paper_target":
        for key in ("beta_logit", "lambda_anchor", "lambda_au"):
            if float(config["stage2"][key]) <= 0:
                raise ValueError(f"Full configuration may not disable S2 {key}")
        for key in ("lambda_it", "lambda_align", "lambda_cnn", "lambda_tv", "lambda_sparse"):
            if float(config["stage1"][key]) <= 0:
                raise ValueError(f"Full configuration may not disable S1 {key}")
    for stage in ("teacher_t1", "teacher_t2", "student_s1", "student_s2"):
        if int(config["stage_epochs"][stage]) <= 0:
            raise ValueError(f"stage_epochs.{stage} must be positive")
        if stage not in config["stage_controls"]:
            raise ValueError(f"stage_controls.{stage} is required")


def resolve_warmup_steps(
    total_optimizer_steps: int,
    *,
    paper_warmup_steps: int,
    strict: bool,
    engineering_warmup_ratio: float = 0.05,
) -> int:
    if total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    if 0 <= paper_warmup_steps < total_optimizer_steps:
        return int(paper_warmup_steps)
    if strict:
        raise ValueError(
            f"warmup_steps={paper_warmup_steps} must be below total optimizer "
            f"steps={total_optimizer_steps}; paper schedule is not effective for this run"
        )
    if not 0 <= engineering_warmup_ratio < 1:
        raise ValueError("engineering_warmup_ratio must be in [0,1)")
    return min(total_optimizer_steps - 1, round(total_optimizer_steps * engineering_warmup_ratio))

