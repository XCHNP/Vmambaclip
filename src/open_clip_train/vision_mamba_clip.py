"""VisionMambaCLIP four-stage runner.

Model construction is delegated to an experiment assembly callable because this
repository snapshot does not contain a real Vim implementation/checkpoint. This
keeps the training loop executable without inventing a substitute backbone.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
from pathlib import Path

import torch

from open_clip.vision_mamba_clip import (
    build_protocol,
    Stage1Loss,
    Stage1LossConfig,
    Stage2Loss,
    Stage2LossConfig,
    configure_student_stage,
    configure_teacher_stage,
    freeze_teacher_for_student,
    load_config,
    required_teacher_stage,
    resolve_warmup_steps,
    save_stage_checkpoint,
)
from open_clip.vision_mamba_clip.training import (
    gradient_summary,
    optimizer_step,
    student_s1_step,
    student_s2_step,
    teacher_t1_step,
    teacher_t2_step,
)
from open_clip.vision_mamba_clip.metrics import ClassificationMetrics


def _load_callable(spec: str):
    if ":" not in spec:
        raise ValueError("Factory must use module.submodule:callable syntax")
    module_name, callable_name = spec.split(":", 1)
    value = getattr(importlib.import_module(module_name), callable_name)
    if not callable(value):
        raise TypeError(f"Assembly target is not callable: {spec}")
    return value


def _validate_assembly(bundle, stage: str):
    required = {"teacher", "text_anchors", "train_loader", "val_loader", "metadata"}
    if stage.startswith("student_"):
        required.add("student")
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Experiment assembly is missing: {sorted(missing)}")
    identities = bundle.get("loaded_checkpoint_stages", {})
    if stage == "teacher_t2" and identities.get("teacher_init") != "teacher_t1":
        raise ValueError("T2 must initialize from a validated teacher_t1 checkpoint")
    if stage.startswith("student_"):
        expected = required_teacher_stage(stage)
        if identities.get("teacher") != expected:
            raise ValueError(f"{stage} must use a validated {expected} teacher checkpoint")
    if stage == "student_s2" and identities.get("student_init") != "student_s1":
        raise ValueError("S2 must initialize from a validated student_s1 checkpoint")


def _stage_loss(config, stage: str):
    if stage == "student_s1":
        value = config["stage1"]
        return Stage1Loss(Stage1LossConfig(
            lambda_it=value["lambda_it"],
            lambda_align=value["lambda_align"],
            lambda_cnn=value["lambda_cnn"],
            lambda_tv=value["lambda_tv"],
            lambda_sparse=value["lambda_sparse"],
            layer_weights=tuple(value["layer_weights"]),
            tau_it=value["tau_it"],
        ))
    if stage == "student_s2":
        value = config["stage2"]
        return Stage2Loss(Stage2LossConfig(
            lambda_task=value["lambda_task"],
            lambda_distill=value["lambda_distill"],
            beta_logit=value["beta_logit"],
            lambda_anchor=value["lambda_anchor"],
            lambda_au=value["lambda_au"],
            gamma_sparse=value["gamma_sparse"],
            tau_logit=value["tau_logit"],
            tau_low=value["tau_low"],
            detach_spatial_target=value["detach_spatial_target"],
            epsilon=config["epsilon"],
        ))
    return None


def _make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def factor(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _precision_context(config, device):
    requested = config["optimizer"]["amp_dtype"]
    if requested == "float32":
        return None, "float32", None
    if requested == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            return None, "float32", "requested BF16 is unsupported on this CUDA device"
        if device.type not in {"cuda", "cpu"}:
            return None, "float32", f"requested BF16 is unsupported on {device.type}"
        return torch.bfloat16, "bfloat16", None
    if requested == "float16" and device.type == "cuda":
        return torch.float16, "float16", None
    return None, "float32", f"unsupported requested amp_dtype={requested!r}; used float32"


def _step(stage, teacher, student, criterion, images, labels, anchors):
    if stage == "teacher_t1":
        return teacher_t1_step(teacher, images, labels, anchors)
    if stage == "teacher_t2":
        return teacher_t2_step(teacher, images, labels)
    if stage == "student_s1":
        return student_s1_step(student, teacher, criterion, images, anchors)
    return student_s2_step(student, teacher, criterion, images, labels, anchors)


def _validation_metric(stage, teacher, student, criterion, loader, anchors, device, amp_dtype):
    teacher.eval()
    if student is not None:
        student.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None else torch.autocast(device_type=device.type, enabled=False)
            )
            with context:
                report = _step(stage, teacher, student, criterion, images, labels, anchors)
            loss_sum += float(report["total"].detach().cpu()) * images.shape[0]
            sample_count += images.shape[0]
            if stage == "teacher_t2":
                correct += int((teacher.forward_classifier(images)["logits"].argmax(-1) == labels).sum())
            elif stage == "student_s2":
                correct += int((student(images).logits.argmax(-1) == labels).sum())
    if sample_count == 0:
        raise RuntimeError("Validation loader is empty")
    if stage in {"teacher_t2", "student_s2"}:
        return correct / sample_count, "accuracy", True
    return loss_sum / sample_count, "loss", False


def run_stage(args) -> None:
    config = load_config(args.config)
    assembly = _load_callable(args.assembly)
    device = torch.device(args.device)
    bundle = assembly(config=config, stage=args.stage, device=device)
    _validate_assembly(bundle, args.stage)

    teacher = bundle["teacher"].to(device)
    anchors = bundle["text_anchors"]
    if hasattr(anchors, "anchors"):
        anchors = anchors.anchors
    anchors = anchors.detach().to(device)
    student = bundle.get("student")
    if student is not None:
        student = student.to(device)

    if args.stage.startswith("teacher_"):
        parameters = configure_teacher_stage(teacher, args.stage)
        training_model = teacher
        epochs = int(config["stage_epochs"][args.stage])
    else:
        freeze_teacher_for_student(teacher)
        parameters = configure_student_stage(
            student,
            args.stage,
            train_text_projection=config["student"]["train_text_projection"],
        )
        training_model = student
        epochs = int(config["stage_epochs"][args.stage])
    if not parameters:
        raise RuntimeError(f"No trainable parameters were selected for {args.stage}")

    optimizer_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=optimizer_cfg["lr"],
        betas=tuple(optimizer_cfg["betas"]),
        eps=optimizer_cfg["eps"],
        weight_decay=optimizer_cfg["weight_decay"],
    )
    loader = bundle["train_loader"]
    requested_steps = epochs * len(loader)
    if args.stage == "student_s1":
        loss_config = config["stage1"]
    elif args.stage == "student_s2":
        loss_config = config["stage2"]
    elif args.stage == "teacher_t1":
        loss_config = {"objective": "seven-class prototype cross-entropy", "maximum_logit_scale": 100.0}
    else:
        loss_config = {"objective": "P: supervised cross-entropy only"}
    max_steps = args.max_steps or config["stage_controls"][args.stage].get("max_steps")
    total_steps = min(requested_steps, int(max_steps)) if max_steps else requested_steps
    warmup = resolve_warmup_steps(
        total_steps,
        paper_warmup_steps=optimizer_cfg["paper_warmup_steps"],
        strict=optimizer_cfg["strict_warmup"],
        engineering_warmup_ratio=optimizer_cfg.get("engineering_warmup_ratio", 0.05),
    )
    scheduler = _make_scheduler(optimizer, total_steps, warmup)
    criterion = _stage_loss(config, args.stage)
    amp_dtype, actual_precision, precision_fallback = _precision_context(config, device)
    checkpointing_status = "disabled"
    if optimizer_cfg.get("gradient_checkpointing"):
        target = teacher.clip_model if args.stage.startswith("teacher_") else student.vim.backbone
        setter = getattr(target, "set_grad_checkpointing", None)
        if callable(setter):
            setter(True)
            checkpointing_status = "enabled"
        else:
            checkpointing_status = "requested_but_unsupported"

    actual_steps = 0
    last_report = None
    best_metric = None
    best_epoch = None
    best_state = None
    for epoch in range(epochs):
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None else torch.autocast(device_type=device.type, enabled=False)
            )
            with context:
                report = _step(args.stage, teacher, student, criterion, images, labels, anchors)
            optimizer_step(report, optimizer)
            scheduler.step()
            actual_steps += 1
            last_report = {name: float(value.detach().cpu()) for name, value in report.items()}
            print(json.dumps({"stage": args.stage, "step": actual_steps, "losses": last_report}))
            if actual_steps >= total_steps:
                break
        if actual_steps >= total_steps:
            pass
        metric, metric_name, maximize = _validation_metric(
            args.stage,
            teacher,
            student,
            criterion,
            bundle["val_loader"],
            anchors,
            device,
            amp_dtype,
        )
        print(json.dumps({"stage": args.stage, "epoch": epoch + 1, f"val_{metric_name}": metric}))
        if best_metric is None or (metric > best_metric if maximize else metric < best_metric):
            best_metric = metric
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in training_model.state_dict().items()
            }
        if actual_steps >= total_steps:
            break
        if args.stage.startswith("teacher_"):
            configure_teacher_stage(teacher, args.stage)
        else:
            freeze_teacher_for_student(teacher)
            configure_student_stage(
                student,
                args.stage,
                train_text_projection=config["student"]["train_text_projection"],
            )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    training_model.load_state_dict(best_state, strict=True)

    metadata = dict(bundle["metadata"])
    metadata.update({
        "stage": args.stage,
        "actual_steps": actual_steps,
        "warmup_steps": warmup,
        "requested_optimizer_steps": requested_steps,
        "loss_config": loss_config,
        "last_losses": last_report,
        "gradient_summary": gradient_summary(training_model),
        "loaded_checkpoint_stages": bundle.get("loaded_checkpoint_stages", {}),
        "selected_by_validation": {"epoch": best_epoch, "metric": metric_name, "value": best_metric},
        "actual_amp_dtype": actual_precision,
        "amp_fallback_reason": precision_fallback,
        "gradient_checkpointing_status": checkpointing_status,
    })
    save_stage_checkpoint(
        args.output,
        training_model,
        metadata,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def inspect_data(args) -> None:
    config = load_config(args.config)
    data = dict(config["data"])
    if args.root:
        data["root"] = args.root
    protocol = build_protocol(data)
    print(json.dumps(protocol.summary(), ensure_ascii=False, indent=2))
    if args.manifest:
        protocol.write_manifest(args.manifest)


def evaluate_student(args) -> None:
    config = load_config(args.config)
    device = torch.device(args.device)
    assembly = _load_callable(args.assembly)
    bundle = assembly(config=config, split=args.split, device=device)
    if "student" not in bundle or "eval_loader" not in bundle:
        raise ValueError("Inference assembly must return student and eval_loader")
    student = bundle["student"].to(device).eval()
    metrics = ClassificationMetrics(config["protocol"]["num_classes"])
    with torch.no_grad():
        for batch in bundle["eval_loader"]:
            images, labels = batch[0].to(device), batch[1].to(device)
            # Labels are used only by metrics and never enter student.forward.
            output = student(images)
            metrics.update(output.logits, labels)
    result = {
        "split": args.split,
        "prediction": "student classifier argmax",
        "metrics": metrics.compute(),
        "checkpoint_metadata": bundle["checkpoint_metadata"],
        "data_summary": bundle.get("data_summary"),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        target = Path(args.output)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect-data")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.add_argument("--root")
    inspect_parser.add_argument("--manifest")
    inspect_parser.set_defaults(func=inspect_data)

    run_parser = subparsers.add_parser("run-stage")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument(
        "--stage",
        required=True,
        choices=("teacher_t1", "teacher_t2", "student_s1", "student_s2"),
    )
    run_parser.add_argument("--assembly", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run_parser.add_argument("--max-steps", type=int)
    run_parser.set_defaults(func=run_stage)

    eval_parser = subparsers.add_parser("evaluate-student")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--assembly", required=True)
    eval_parser.add_argument("--split", required=True, choices=("val", "test"))
    eval_parser.add_argument("--output")
    eval_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    eval_parser.set_defaults(func=evaluate_student)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

