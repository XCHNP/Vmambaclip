"""Standalone student inference script for VisionMambaCLIP FER.

Usage:
    python -m open_clip.vision_mamba_clip.inference \
        --checkpoint checkpoints/student_s2.pt \
        --image path/to/face.jpg \
        --device cuda

Design notes:
    - Inference uses ONLY the student model and its classifier head.
    - No teacher, text anchors, or AU prompts are needed at inference time.
    - Prediction is simply argmax over the student's classification logits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn
from PIL import Image

from .checkpoint import read_checkpoint_metadata, _load_payload
from .model import HybridVisionMambaStudent


def _build_transform(preprocess: dict, image_size: int):
    """Build a torchvision transform from checkpoint metadata."""
    from torchvision import transforms

    size = preprocess.get("size", image_size)
    mean = preprocess.get("mean", (0.48145466, 0.4578275, 0.40821073))
    std = preprocess.get("std", (0.26862954, 0.26130258, 0.27577711))

    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_student_for_inference(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple:
    """Load a trained student model ready for inference.

    Returns:
        (model, class_order, transform, metadata)
    """
    payload = _load_payload(checkpoint_path, map_location=device)
    metadata = payload["metadata"]
    class_order = metadata["class_order"]
    common_dim = int(metadata["common_dim"])
    num_classes = len(class_order)

    # Reconstruct student via the Vim factory
    from .vim_backbone import create_vim
    from .adapters import VimAdapter

    vim_init = metadata.get("vim_initialization", {})
    model_name = vim_init.get("model_name", "vim_small_patch16")
    img_size = vim_init.get("img_size", 256)

    backbone, _ = create_vim(
        model_name=model_name,
        img_size=img_size,
    )
    vim = VimAdapter(
        backbone,
        backbone_is_bidirectional=True,
        backbone_identity=vim_init.get("identity", f"bidirectional_vim_{model_name}"),
        num_special_tokens=1,
    )

    # Probe state dims
    with torch.no_grad():
        hidden = vim(torch.zeros(1, 3, img_size, img_size))
    state_dims = {name: int(v.shape[-1]) for name, v in hidden.items()}

    student = HybridVisionMambaStudent(
        vim,
        state_dims,
        common_dim,
        num_classes,
        compute_logits=True,
    )

    student.load_state_dict(payload["model"], strict=True)
    student = student.to(device).eval()

    preprocess = metadata.get("preprocess", {})
    transform = _build_transform(preprocess, img_size)

    return student, class_order, transform, metadata


@torch.no_grad()
def predict_single(
    model: nn.Module,
    image: Image.Image,
    transform,
    class_order: List[str],
    device: str = "cpu",
) -> Dict:
    """Run inference on a single image.

    Returns:
        Dict with 'predicted_class', 'predicted_index', 'probabilities', 'logits'.
    """
    tensor = transform(image).unsqueeze(0).to(device)
    output = model(tensor)
    logits = output.logits[0].float()
    probs = torch.softmax(logits, dim=0)
    pred_idx = int(logits.argmax().item())

    return {
        "predicted_class": class_order[pred_idx],
        "predicted_index": pred_idx,
        "probabilities": {
            name: float(probs[i].item())
            for i, name in enumerate(class_order)
        },
        "logits": [float(v) for v in logits.tolist()],
    }


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    images: Tensor,
    class_order: List[str],
) -> List[Dict]:
    """Run inference on a batch of preprocessed image tensors.

    Args:
        model: Trained student model in eval mode.
        images: [B, 3, H, W] preprocessed tensor.
        class_order: List of class names.

    Returns:
        List of prediction dicts per sample.
    """
    output = model(images)
    logits = output.logits.float()
    probs = torch.softmax(logits, dim=1)
    pred_indices = logits.argmax(dim=1)

    results = []
    for i in range(images.shape[0]):
        results.append({
            "predicted_class": class_order[int(pred_indices[i].item())],
            "predicted_index": int(pred_indices[i].item()),
            "probabilities": {
                name: float(probs[i, j].item())
                for j, name in enumerate(class_order)
            },
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="VisionMambaCLIP student inference")
    parser.add_argument("--checkpoint", required=True, help="Path to student_s2 checkpoint")
    parser.add_argument("--image", required=True, help="Path to face image")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Optional path to save prediction JSON")
    args = parser.parse_args()

    model, class_order, transform, metadata = load_student_for_inference(
        args.checkpoint, device=args.device
    )

    image = Image.open(args.image).convert("RGB")
    result = predict_single(model, image, transform, class_order, device=args.device)
    result["checkpoint"] = args.checkpoint
    result["image"] = args.image

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
