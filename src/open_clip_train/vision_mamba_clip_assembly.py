"""Default assembly for the VisionMambaCLIP stage runner.

It uses the repository's OpenCLIP and FER data implementations while requiring a
real external Vim factory. Missing assets are errors, never random fallbacks.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.model import get_model_preprocess_cfg
from open_clip.transform import image_transform
from open_clip.vision_mamba_clip import (
    FERDataset,
    HybridVisionMambaStudent,
    TeacherAdapter,
    VimAdapter,
    build_protocol,
    class_name_ablation_schema,
    encode_prompt_schema,
    load_prompt_schema,
    load_stage_checkpoint,
    read_checkpoint_metadata,
)


def _required_path(value, purpose: str) -> str:
    if not value:
        raise FileNotFoundError(f"Missing {purpose} path in configuration")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{purpose} does not exist: {path}")
    return str(path)


def _factory(spec: str):
    if not spec or ":" not in spec:
        raise ValueError(
            "student.factory must identify a real Vim callable as module.submodule:callable"
        )
    module_name, name = spec.split(":", 1)
    value = getattr(importlib.import_module(module_name), name)
    if not callable(value):
        raise TypeError(f"Vim factory is not callable: {spec}")
    return value


def _build_student(config, device, image_size, common_dim):
    student_cfg = config["student"]
    result = _factory(student_cfg["factory"])(
        checkpoint_path=student_cfg.get("checkpoint"),
        **student_cfg.get("factory_kwargs", {}),
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("Vim factory must return (backbone, initialization_metadata)")
    backbone, initialization = result
    if not isinstance(backbone, nn.Module):
        raise TypeError("Vim factory did not return an nn.Module")
    required = {"identity", "pretrained_loaded", "initialization_source"}
    if required.difference(initialization):
        raise ValueError(f"Vim initialization metadata requires {sorted(required)}")
    if student_cfg.get("checkpoint") and not initialization["pretrained_loaded"]:
        raise RuntimeError("Vim checkpoint was configured but the factory reports it was not loaded")
    from open_clip.vision_mamba_clip.official_vim import OfficialBackbone, OfficialPatchAdapter
    if isinstance(backbone, OfficialBackbone):
        vim = OfficialPatchAdapter(backbone).to(device)
        if student_cfg.get("layer_indices") is not None and tuple(student_cfg["layer_indices"]) != vim.layer_indices:
            raise ValueError("Official Vim assembly requires the final three layers")
    else:
        vim = VimAdapter(
            backbone,
            backbone_is_bidirectional=True,
            backbone_identity=initialization["identity"],
            layer_indices=student_cfg.get("layer_indices"),
            num_special_tokens=student_cfg["num_special_tokens"],
        ).to(device)
    with torch.no_grad():
        hidden = vim(torch.zeros(2, 3, image_size, image_size, device=device))
    state_dims = {name: int(value.shape[-1]) for name, value in hidden.items()}
    student = HybridVisionMambaStudent(
        vim,
        state_dims,
        common_dim,
        config["protocol"]["num_classes"],
        attention_heads=student_cfg["attention_heads"],
        use_coords=student_cfg["use_coords"],
        epsilon=config["epsilon"],
        use_cnn=student_cfg["use_cnn"],
        use_sasf=student_cfg.get("use_sasf", False),
        fusion=student_cfg.get("fusion", "concat_projection"),
    ).to(device)
    return student, initialization


def build(*, config, stage: str, device):
    protocol = build_protocol(config["data"])
    class_order = protocol.class_order
    if len(class_order) != config["protocol"]["num_classes"]:
        raise ValueError("Discovered FER classes do not match protocol.num_classes")

    original_clip = _required_path(config["teacher"]["checkpoint"], "original CLIP checkpoint")
    clip_model, preprocess_train, _preprocess_val = create_model_and_transforms(
        config["teacher"]["architecture"],
        pretrained=original_clip,
        force_image_size=config["protocol"]["image_size"],
        device=device,
        require_pretrained=True,
    )
    teacher = TeacherAdapter(clip_model, len(class_order)).to(device)
    preprocess = get_model_preprocess_cfg(clip_model)
    expected_prompt_hash = None

    # Load the teacher stage before encoding anchors; its text tower must be the
    # same frozen identity used throughout the run.
    loaded = {}
    teacher_source = original_clip
    teacher_stage = None
    if stage == "teacher_t2":
        teacher_stage = "teacher_t1"
    elif stage == "student_s1":
        teacher_stage = "teacher_t1"
    elif stage == "student_s2":
        teacher_stage = "teacher_t2"
    if teacher_stage:
        teacher_source = _required_path(
            config["checkpoints"][teacher_stage], f"{teacher_stage} checkpoint"
        )

    prompt_path = config["data"].get("au_prompt_schema")
    if config["data"].get("text_mode") == "explicit_class_name_ablation":
        schema = class_name_ablation_schema(class_order)
    else:
        schema = load_prompt_schema(_required_path(prompt_path, "AU prompt schema"), class_order)
    expected_prompt_hash = schema.schema_hash

    if teacher_stage:
        load_stage_checkpoint(
            teacher_source,
            teacher,
            expected_stage=teacher_stage,
            common_dim=teacher.output_dim,
            class_order=class_order,
            data_manifest_hash=protocol.manifest_hash,
            prompt_schema_hash=expected_prompt_hash,
            map_location=device,
        )
        loaded["teacher" if stage.startswith("student_") else "teacher_init"] = teacher_stage

    tokenizer = get_tokenizer(config["teacher"]["architecture"])
    anchors = encode_prompt_schema(
        schema,
        tokenizer,
        teacher,
        text_encoder_identity=f"{config['teacher']['architecture']}:{teacher_source}",
    ).to(device)
    train_loader = DataLoader(
        FERDataset(protocol, "train", preprocess_train),
        batch_size=config["optimizer"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("workers", 0),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        FERDataset(protocol, "val", _preprocess_val),
        batch_size=config["optimizer"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("workers", 0),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    bundle = {
        "teacher": teacher,
        "text_anchors": anchors,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "loaded_checkpoint_stages": loaded,
    }
    initialization_source = teacher_source
    model_identity = f"CLIP {config['teacher']['architecture']}"
    if stage.startswith("student_"):
        student, student_initialization = _build_student(
            config, device, config["protocol"]["image_size"], teacher.output_dim
        )
        initialization_source = student_initialization["initialization_source"]
        model_identity = student_initialization["identity"]
        if stage == "student_s2":
            student_source = _required_path(
                config["checkpoints"]["student_s1"], "student_s1 checkpoint"
            )
            load_stage_checkpoint(
                student_source,
                student,
                expected_stage="student_s1",
                common_dim=teacher.output_dim,
                class_order=class_order,
                data_manifest_hash=protocol.manifest_hash,
                prompt_schema_hash=expected_prompt_hash,
                map_location=device,
            )
            loaded["student_init"] = "student_s1"
            initialization_source = student_source
        bundle["student"] = student

    bundle["metadata"] = {
        "model_identity": model_identity,
        "initialization_source": initialization_source,
        "common_dim": teacher.output_dim,
        "class_order": class_order,
        "data_manifest_hash": protocol.manifest_hash,
        "prompt_schema_hash": expected_prompt_hash,
        "text_encoder_identity": anchors.metadata["text_encoder_identity"],
        "text_anchor_metadata": anchors.metadata,
        "preprocess": preprocess,
        "seed": config["data"]["validation_seed"],
        "actual_steps": 0,
        "data_summary": protocol.summary(),
        "vim_selected_layers": (
            bundle["student"].vim.layer_identities if "student" in bundle else None
        ),
        "cnn_truncation": "resnet18.layer3" if config["student"]["use_cnn"] else None,
        "cnn_initialization": config["student"]["cnn_initialization"],
        "vim_initialization": (
            student_initialization if stage.startswith("student_") else None
        ),
        "attention": config["student"]["attention_parameterization"],
        "fusion": config["student"]["fusion"],
        "use_sasf": config["student"].get("use_sasf", False),
        "classifier_input": config["student"]["classifier_input"],
    }
    return bundle


def build_inference(*, config, split: str, device):
    """Build student-only inference; no teacher, labels-to-model, or anchors."""
    if split not in {"val", "test"}:
        raise ValueError("Inference split must be explicitly 'val' or 'test'")
    protocol = build_protocol(config["data"])
    checkpoint_path = _required_path(
        config["checkpoints"]["student_s2"], "student_s2 checkpoint"
    )
    checkpoint_metadata = read_checkpoint_metadata(checkpoint_path, map_location="cpu")
    preprocess = checkpoint_metadata["preprocess"]
    transform = image_transform(
        image_size=preprocess.get("size", config["protocol"]["image_size"]),
        is_train=False,
        mean=preprocess.get("mean"),
        std=preprocess.get("std"),
        resize_mode=preprocess.get("resize_mode"),
        interpolation=preprocess.get("interpolation"),
    )
    student, initialization = _build_student(
        config,
        device,
        config["protocol"]["image_size"],
        int(checkpoint_metadata["common_dim"]),
    )
    load_stage_checkpoint(
        checkpoint_path,
        student,
        expected_stage="student_s2",
        common_dim=int(checkpoint_metadata["common_dim"]),
        class_order=protocol.class_order,
        data_manifest_hash=protocol.manifest_hash,
        map_location=device,
    )
    loader = DataLoader(
        FERDataset(protocol, split, transform),
        batch_size=config["optimizer"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("workers", 0),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    return {
        "student": student,
        "eval_loader": loader,
        "checkpoint_metadata": checkpoint_metadata,
        "student_initialization": initialization,
        "data_summary": protocol.summary(),
    }

