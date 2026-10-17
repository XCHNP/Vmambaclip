"""Export the exact FERProtocol splits for generic OpenCLIP CSV training.

Usage: PYTHONPATH=src python -m open_clip_train.predata_fer --config CONFIG --output-dir DIR
Use --inspect-only to inspect without writing. Output directories must not exist.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

from PIL import Image

# Load the exact shared data implementation without importing the entire
# OpenCLIP model package into a memory-limited CPU preprocessing process.
_data_path = Path(__file__).resolve().parents[1] / "open_clip/vision_mamba_clip/data.py"
_spec = importlib.util.spec_from_file_location("_fer_preprocessing_data", _data_path)
_data_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _data_module
_spec.loader.exec_module(_data_module)
build_protocol = _data_module.build_protocol


def _verify_images(protocol, mode, samples_per_class):
    if mode not in {"full", "structure-and-sample"}:
        raise ValueError("Unknown image verification mode")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    total = sum(len(samples) for samples in protocol.splits.values())
    errors, groups = [], {}
    checked, decoded = 0, 0
    last_progress = time.monotonic()
    for split, samples in protocol.splits.items():
        for sample in samples:
            try:
                with Image.open(Path(protocol.root) / sample.path) as image:
                    if mode == "full":
                        image.convert("RGB").load()
                        decoded += 1
                    else:
                        image.verify()
                        groups.setdefault((split, sample.label), []).append(sample.path)
            except Exception as exc:
                errors.append({"split": split, "path": sample.path, "error": str(exc)})
            checked += 1
            if time.monotonic() - last_progress >= 30 or checked == total:
                print(f"Image {mode} check: {checked}/{total}, errors={len(errors)}", flush=True)
                last_progress = time.monotonic()
    if mode == "structure-and-sample":
        rng = random.Random(protocol.validation_seed)
        for (split, label), paths in sorted(groups.items()):
            selected = rng.sample(sorted(paths), min(samples_per_class, len(paths)))
            for relative in selected:
                try:
                    with Image.open(Path(protocol.root) / relative) as image:
                        image.convert("RGB").load()
                        decoded += 1
                except Exception as exc:
                    errors.append({"split": split, "path": relative, "error": str(exc)})
            print(f"Decode sample: {split}/{protocol.class_order[label]}, {len(selected)} images, errors={len(errors)}", flush=True)
    if errors:
        raise ValueError(f"{len(errors)} unreadable images; first errors: {errors[:10]}")
    return {"mode": mode, "files_checked": checked, "images_decoded": decoded,
            "sample_seed": protocol.validation_seed, "samples_per_class_per_split": samples_per_class if mode != "full" else None,
            "note": "JPEG structure verification checks headers; sampled mode does not decode every JPEG pixel stream" if mode != "full" else "All images fully decoded"}


def preprocess(config_path, output_dir=None, *, inspect_only=False, verify_images=False,
               verification_mode="full", samples_per_class=100):
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    data = config["data"]
    started = time.monotonic()
    print(f"Building {data.get('dataset')} data protocol...", flush=True)
    protocol = build_protocol(data)
    total = sum(len(samples) for samples in protocol.splits.values())
    print(f"Protocol contains {total} images; built in {time.monotonic() - started:.1f}s", flush=True)
    if verify_images:
        verification = _verify_images(protocol, verification_mode, samples_per_class)
    else:
        verification = {"mode": "not-run"}
    summary = protocol.summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if inspect_only:
        return summary
    if output_dir is None:
        raise ValueError("--output-dir is required unless --inspect-only is used")
    prompt_path = data.get("au_prompt_schema")
    if not prompt_path:
        raise ValueError("CSV export requires data.au_prompt_schema")
    with Path(prompt_path).open(encoding="utf-8") as handle:
        schema = json.load(handle)
    if schema.get("class_order") != protocol.class_order or not schema.get("source"):
        raise ValueError("AU prompt schema requires matching canonical class_order and a recorded source")
    prompts = schema["prompts"]
    if set(prompts) != set(protocol.class_order) or any(not isinstance(p, str) or not p.strip() for p in prompts.values()):
        raise ValueError("AU prompt schema requires one nonempty text prompt for each canonical class")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    protocol.write_manifest(str(target / "manifest.json"))
    for split, samples in protocol.splits.items():
        with (target / f"{split}.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "caption", "labels"])
            for sample in samples:
                writer.writerow([str(Path(protocol.root) / sample.path), prompts[sample.class_name], sample.label])
    with (target / "input_contract.json").open("x", encoding="utf-8") as handle:
        json.dump({
            "dataset": data.get("dataset"), "config": str(Path(config_path).resolve()),
            "manifest_hash": protocol.manifest_hash,
            "prompt_schema": str(Path(prompt_path).resolve()), "prompt_source": schema["source"],
            "csv_arguments": {"dataset_type": "csv", "csv_separator": ",", "csv_img_key": "image_path", "csv_caption_key": "caption"},
            "note": "Generic OpenCLIP consumes image/text pairs. VisionMambaCLIP reads the same protocol and its canonical labels directly.",
            "image_decode_verified": verify_images and verification_mode == "full",
            "image_verification": verification,
        }, handle, ensure_ascii=False, indent=2)
    print(f"Exported shared FER splits to {target.resolve()}")
    return summary


def main(default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config, required=default_config is None)
    parser.add_argument("--output-dir")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--verify-images", action="store_true", help="Check images before writing; full decode by default")
    parser.add_argument("--verification-mode", choices=("full", "structure-and-sample"), default="full")
    parser.add_argument("--samples-per-class", type=int, default=100)
    args = parser.parse_args()
    preprocess(args.config, args.output_dir, inspect_only=args.inspect_only, verify_images=args.verify_images,
               verification_mode=args.verification_mode, samples_per_class=args.samples_per_class)


if __name__ == "__main__":
    main()
