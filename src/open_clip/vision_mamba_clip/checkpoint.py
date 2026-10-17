"""Identity-preserving checkpoints for T1/T2/S1/S2."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import nn


VALID_STAGES = {"teacher_t1", "teacher_t2", "student_s1", "student_s2"}
REQUIRED_METADATA = {
    "stage",
    "model_identity",
    "initialization_source",
    "common_dim",
    "class_order",
    "data_manifest_hash",
    "prompt_schema_hash",
    "preprocess",
    "seed",
    "actual_steps",
}


def validate_metadata(metadata: Mapping) -> None:
    missing = REQUIRED_METADATA.difference(metadata)
    if missing:
        raise ValueError(f"Checkpoint metadata is missing: {sorted(missing)}")
    if metadata["stage"] not in VALID_STAGES:
        raise ValueError(f"Invalid checkpoint stage: {metadata['stage']!r}")
    if int(metadata["common_dim"]) <= 0:
        raise ValueError("common_dim must be positive")
    if not metadata["class_order"]:
        raise ValueError("class_order may not be empty")


def save_stage_checkpoint(
    path: str,
    model: nn.Module,
    metadata: Mapping,
    *,
    optimizer=None,
    scheduler=None,
    extra_state: Optional[Mapping] = None,
) -> None:
    """Atomically save a new checkpoint and never overwrite an existing stage."""
    validate_metadata(metadata)
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "metadata": dict(metadata),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "extra_state": dict(extra_state or {}),
    }
    handle, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(handle)
    try:
        torch.save(payload, temporary)
        try:
            os.replace(temporary, target)
        except PermissionError:
            # Some managed Windows filesystems disallow rename/replace even within
            # one writable directory. Exclusive creation preserves the essential
            # no-overwrite invariant in that environment.
            with target.open("xb") as output:
                torch.save(payload, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_payload(path: str, map_location="cpu") -> Dict:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model" not in payload or "metadata" not in payload:
        raise ValueError(f"Not a VisionMambaCLIP stage checkpoint: {path}")
    validate_metadata(payload["metadata"])
    return payload


def read_checkpoint_metadata(path: str, map_location="cpu") -> Mapping:
    """Read validated metadata without constructing or mutating a model."""
    return dict(_load_payload(path, map_location=map_location)["metadata"])


def validate_checkpoint_identity(
    metadata: Mapping,
    *,
    expected_stage: str,
    common_dim: int,
    class_order: Sequence[str],
    data_manifest_hash: Optional[str] = None,
    prompt_schema_hash: Optional[str] = None,
) -> None:
    if metadata["stage"] != expected_stage:
        raise ValueError(
            f"Expected {expected_stage} checkpoint, received {metadata['stage']}"
        )
    if int(metadata["common_dim"]) != int(common_dim):
        raise ValueError(
            f"Checkpoint common_dim={metadata['common_dim']} does not match {common_dim}"
        )
    if list(metadata["class_order"]) != [str(value) for value in class_order]:
        raise ValueError("Checkpoint class order does not match the active data protocol")
    if data_manifest_hash and metadata["data_manifest_hash"] != data_manifest_hash:
        raise ValueError("Checkpoint data manifest hash does not match the active split")
    if prompt_schema_hash and metadata["prompt_schema_hash"] != prompt_schema_hash:
        raise ValueError("Checkpoint prompt schema hash does not match the active anchors")


def load_stage_checkpoint(
    path: str,
    model: nn.Module,
    *,
    expected_stage: str,
    common_dim: int,
    class_order: Sequence[str],
    data_manifest_hash: Optional[str] = None,
    prompt_schema_hash: Optional[str] = None,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
) -> Mapping:
    payload = _load_payload(path, map_location=map_location)
    validate_checkpoint_identity(
        payload["metadata"],
        expected_stage=expected_stage,
        common_dim=common_dim,
        class_order=class_order,
        data_manifest_hash=data_manifest_hash,
        prompt_schema_hash=prompt_schema_hash,
    )
    # Strict loading is intentional: critical missing keys may never be hidden.
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError("Checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("Checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    return payload["metadata"]


def required_teacher_stage(student_stage: str) -> str:
    mapping = {"student_s1": "teacher_t1", "student_s2": "teacher_t2"}
    try:
        return mapping[student_stage]
    except KeyError as error:
        raise ValueError(f"No teacher checkpoint relation for {student_stage!r}") from error

