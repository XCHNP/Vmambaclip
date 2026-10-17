"""Strict, traceable AU prompt loading and frozen CLIP text anchors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .adapters import TeacherAdapter


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptSchema:
    class_order: List[str]
    prompts: Dict[str, str]
    source: str
    schema_hash: str


def load_prompt_schema(path: str, expected_class_order: Sequence[str]) -> PromptSchema:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    required = {"class_order", "prompts", "source"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"Prompt schema is missing fields: {sorted(missing)}")
    class_order = [str(item) for item in value["class_order"]]
    expected = [str(item) for item in expected_class_order]
    if class_order != expected:
        raise ValueError(
            f"Prompt class_order {class_order} does not match dataset class_order {expected}"
        )
    prompts = {str(key): str(text) for key, text in value["prompts"].items()}
    if set(prompts) != set(class_order):
        raise ValueError("Prompt keys must cover the class order exactly")
    empty = [name for name in class_order if not prompts[name].strip()]
    if empty:
        raise ValueError(f"Prompts may not be empty: {empty}")
    if not str(value["source"]).strip():
        raise ValueError("Prompt schema must name a traceable source/version")
    return PromptSchema(class_order, prompts, str(value["source"]), canonical_hash(value))


class TextAnchorBank(nn.Module):
    def __init__(self, anchors: Tensor, metadata: Mapping):
        super().__init__()
        if anchors.ndim != 2 or not torch.isfinite(anchors).all():
            raise ValueError("Text anchors must be a finite [C,D] tensor")
        self.register_buffer("anchors", F.normalize(anchors.float(), dim=-1))
        self.metadata = dict(metadata)

    @property
    def class_order(self) -> List[str]:
        return list(self.metadata["class_order"])

    def validate_class_order(self, class_order: Sequence[str]) -> None:
        actual = [str(value) for value in class_order]
        if actual != self.class_order:
            raise ValueError(
                f"Text-anchor order {self.class_order} does not match labels {actual}"
            )

    def forward(self) -> Tensor:
        return self.anchors


def encode_prompt_schema(
    schema: PromptSchema,
    tokenizer: Callable[[Sequence[str]], Tensor],
    teacher: TeacherAdapter,
    *,
    text_encoder_identity: str,
) -> TextAnchorBank:
    if not text_encoder_identity.strip():
        raise ValueError("text_encoder_identity must name the actual CLIP checkpoint")
    prompts = [schema.prompts[name] for name in schema.class_order]
    tokens = tokenizer(prompts)
    if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[0] != len(prompts):
        raise RuntimeError("Tokenizer must return integer token IDs with shape [C,context_length]")
    if tokens.shape[1] == 0:
        raise ValueError("Tokenizer returned a zero-length token sequence")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"Tokenizer returned {tokens.dtype}; integer token IDs are required")

    teacher.freeze_text_encoder()
    teacher.clip_model.eval()
    device = next(teacher.parameters()).device
    with torch.no_grad():
        raw = teacher.encode_text(tokens.to(device))
        anchors = F.normalize(raw.float(), dim=-1).cpu()
    metadata = {
        "class_order": schema.class_order,
        "prompts": schema.prompts,
        "prompt_source": schema.source,
        "prompt_schema_hash": schema.schema_hash,
        "text_encoder_identity": text_encoder_identity,
        "token_ids": tokens.detach().cpu().tolist(),
        "token_shape": list(tokens.shape),
        "anchor_hash": hashlib.sha256(anchors.numpy().tobytes()).hexdigest(),
        "semantic_anchor_count_interpretation": "E: K equals C; one AU description per class",
    }
    return TextAnchorBank(anchors, metadata)


def class_name_ablation_schema(class_order: Sequence[str]) -> PromptSchema:
    """Explicit ablation only; never used as a fallback for missing AU descriptions."""
    order = [str(value) for value in class_order]
    payload = {
        "class_order": order,
        "prompts": {name: name for name in order},
        "source": "E: class-name-only ablation generated from dataset identifiers",
    }
    return PromptSchema(order, payload["prompts"], payload["source"], canonical_hash(payload))

