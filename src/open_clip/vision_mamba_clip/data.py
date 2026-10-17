"""Shared, traceable seven-class FER protocol for loaders and CSV preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CANONICAL_EMOTION_ORDER = [
    "Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised",
]
# Only RAF's raw annotation convention is established in the repository.
# Renumbered SFEW/AffectNet folders require data.label_mapping in their configs.
RAW_LABEL_TO_INDEX = {
    "rafdb": {"1": 6, "2": 2, "3": 1, "4": 3, "5": 5, "6": 0, "7": 4},
}


def _dataset_key(dataset):
    return re.sub(r"[^a-z0-9]", "", str(dataset).lower())


def _resolve_index_mapping(dataset, label_mapping=None):
    source = label_mapping if label_mapping is not None else RAW_LABEL_TO_INDEX.get(_dataset_key(dataset))
    if not isinstance(source, Mapping) or not source:
        raise ValueError(f"Dataset {dataset!r} requires an explicitly confirmed data.label_mapping")
    mapping = {}
    for raw, value in source.items():
        if isinstance(value, str) and value in CANONICAL_EMOTION_ORDER:
            index = CANONICAL_EMOTION_ORDER.index(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            index = value
        else:
            raise ValueError(f"Invalid canonical label {value!r} for raw label {raw!r}")
        mapping[str(raw)] = index
    if sorted(mapping.values()) != list(range(7)):
        raise ValueError("label_mapping must cover canonical indices 0..6 exactly once")
    return mapping


def _image_files(directory):
    with os.scandir(directory) as entries:
        return sorted(Path(entry.path) for entry in entries
                      if Path(entry.name).suffix.lower() in IMAGE_SUFFIXES and entry.is_file())


def _split_training(pool, validation_ratio, validation_seed):
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    by_class = {}
    for relative, label in pool:
        by_class.setdefault(label, []).append(relative)
    if set(by_class) != set(range(7)):
        raise ValueError("Training data must contain all seven canonical classes")
    rng = random.Random(validation_seed)
    splits = {"train": [], "val": []}
    for label in sorted(by_class):
        paths = sorted(by_class[label])
        if len(paths) < 2:
            raise ValueError(f"Class {CANONICAL_EMOTION_ORDER[label]} needs at least two training images")
        rng.shuffle(paths)
        # Preserve the existing RAF 90/10 rounding and seeded shuffle.
        n_train = max(1, min(len(paths) - 1, int(len(paths) * (1 - validation_ratio))))
        for split, selected in (("train", paths[:n_train]), ("val", paths[n_train:])):
            splits[split].extend(FERSample(path, label, CANONICAL_EMOTION_ORDER[label], split) for path in selected)
    return splits


@dataclass(frozen=True)
class FERSample:
    path: str
    label: int
    class_name: str
    split: str


@dataclass
class FERProtocol:
    root: str
    class_order: List[str]
    splits: Dict[str, List[FERSample]]
    manifest_hash: str
    validation_seed: int
    validation_ratio: float
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def _build(cls, base, class_order, splits, validation_seed, validation_ratio, metadata=None):
        seen = {}
        resolved_parents = {}
        for split, samples in splits.items():
            if not samples:
                raise ValueError(f"Split {split!r} contains no images")
            for sample in samples:
                path = Path(sample.path)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Sample path escapes dataset root: {sample.path}")
                absolute = base / path
                parent = absolute.parent
                if parent not in resolved_parents:
                    resolved_parents[parent] = parent.resolve()
                resolved = absolute.resolve() if absolute.is_symlink() else resolved_parents[parent] / absolute.name
                if base not in resolved.parents:
                    raise ValueError(f"Sample resolves outside dataset root: {sample.path}")
                resolved_key = resolved.as_posix()
                if resolved_key in seen:
                    raise ValueError(f"Duplicate image path in {seen[resolved_key]} and {split}: {sample.path}")
                seen[resolved_key] = split
        del seen, resolved_parents
        payload = {
            "class_order": list(class_order), "validation_seed": validation_seed,
            "validation_ratio": validation_ratio, "metadata": metadata or {},
            "splits": {name: [s.__dict__ for s in samples] for name, samples in splits.items()},
        }
        hasher = hashlib.sha256()
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"))
        buffer, size = [], 0
        for chunk in encoder.iterencode(payload):
            buffer.append(chunk)
            size += len(chunk)
            if size >= 65536:
                hasher.update("".join(buffer).encode("utf-8"))
                buffer, size = [], 0
        if buffer:
            hasher.update("".join(buffer).encode("utf-8"))
        return cls(str(base), list(class_order), dict(splits), hasher.hexdigest(), validation_seed, validation_ratio, metadata or {})

    @classmethod
    def from_image_folders(
        cls, root, *, dataset="rafdb", train_directory="Train", test_directory="Test",
        validation_directory=None, validation_usage="selection", validation_ratio=0.1,
        validation_seed=42, label_mapping=None, excluded_paths=(), exclusion_metadata=None,
    ):
        """Preserve named splits; optionally hold out selection data from training.

        validation_usage='selection': supplied validation stays val, optional Test stays test.
        validation_usage='evaluation': supplied validation becomes test; derive val from Train.
        The latter records its original source name and never claims an official Test split.
        """
        mapping = _resolve_index_mapping(dataset, label_mapping)
        base = Path(root).resolve()
        if not base.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {base}")
        if validation_usage not in {"selection", "evaluation"}:
            raise ValueError("validation_usage must be 'selection' or 'evaluation'")
        if validation_usage == "evaluation" and (not validation_directory or test_directory is not None):
            raise ValueError("evaluation usage requires validation_directory and test_directory=null")
        directories = [name for name in (train_directory, validation_directory, test_directory) if name is not None]
        if len(set(directories)) != len(directories):
            raise ValueError("Train, validation and test directories must be distinct")
        if validation_directory is None and test_directory is None:
            raise ValueError("Supply a validation_directory or test_directory")
        excluded = set(excluded_paths)
        if len(excluded) != len(excluded_paths):
            raise ValueError("Duplicate exclusion paths")
        for relative in excluded:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
                raise ValueError(f"Invalid relative exclusion path: {relative}")
        excluded_seen = set()
        removed = Counter()

        def scan(name):
            directory = (base / name).resolve()
            if base not in directory.parents or not directory.is_dir():
                raise FileNotFoundError(f"Split directory missing or outside dataset root: {directory}")
            folders = sorted(p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith("."))
            if set(folders) != set(mapping):
                raise ValueError(f"{name}: class folders {folders} do not match mapping {sorted(mapping)}")
            pool = []
            for raw in sorted(folders, key=lambda raw: mapping[raw]):
                files = _image_files(directory / raw)
                retained = 0
                for path in files:
                    relative = path.relative_to(base).as_posix()
                    if relative in excluded:
                        excluded_seen.add(relative)
                        removed[name] += 1
                        continue
                    pool.append((relative, mapping[raw]))
                    retained += 1
                if not retained:
                    raise ValueError(f"No retained images in class directory: {directory / raw}")
            return pool

        def samples(pool, split):
            return [FERSample(p, label, CANONICAL_EMOTION_ORDER[label], split) for p, label in pool]

        training = scan(train_directory)
        sources = {"train": train_directory}
        if validation_directory and validation_usage == "selection":
            splits = {"train": samples(training, "train"), "val": samples(scan(validation_directory), "val")}
            ratio = 0.0
            sources["val"] = validation_directory
        else:
            splits = _split_training(training, validation_ratio, validation_seed)
            ratio = validation_ratio
            sources["val"] = f"stratified holdout from {train_directory}"
        if validation_directory and validation_usage == "evaluation":
            splits["test"] = samples(scan(validation_directory), "test")
            sources["test"] = validation_directory
        elif test_directory is not None:
            splits["test"] = samples(scan(test_directory), "test")
            sources["test"] = test_directory
        metadata = {"dataset": _dataset_key(dataset), "raw_label_mapping": mapping,
                    "split_sources": sources, "validation_usage": validation_usage,
                    "format": "folders", "ignored_hidden_directories": True}
        if excluded - excluded_seen:
            raise ValueError(f"Exclusion paths not found in configured splits: {sorted(excluded - excluded_seen)[:5]}")
        if excluded:
            metadata["exclusions"] = dict(exclusion_metadata or {},
                                          excluded_count=len(excluded_seen),
                                          excluded_counts_by_source=dict(sorted(removed.items())))
        return cls._build(base, CANONICAL_EMOTION_ORDER, splits, validation_seed, ratio, metadata)

    @classmethod
    def from_label_file(
        cls, root, label_path, *, dataset="rafdb", train_prefix="train_", test_prefix="test_",
        strip_suffix="_aligned", validation_seed=42, validation_ratio=0.1, label_mapping=None,
    ):
        mapping = _resolve_index_mapping(dataset, label_mapping)
        base = Path(root).resolve()
        label_file = Path(label_path).resolve()
        if not base.is_dir() or not label_file.is_file():
            raise FileNotFoundError(f"Image directory or annotation file missing: {base}, {label_file}")
        if not train_prefix or not test_prefix or train_prefix.startswith(test_prefix) or test_prefix.startswith(train_prefix):
            raise ValueError("Train/test prefixes must be nonempty and unambiguous")
        raw_labels = {}
        with label_file.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 2:
                    raise ValueError(f"Malformed annotation at {label_file}:{number}")
                name, raw = parts
                if name in raw_labels:
                    raise ValueError(f"Duplicate annotation: {name}")
                if raw not in mapping:
                    raise ValueError(f"Unmapped raw label {raw!r} at line {number}")
                raw_labels[name] = raw
        train, test, matched, failures = [], [], set(), []
        ignored = 0
        for path in _image_files(base):
            name = path.name
            if not (name.startswith(train_prefix) or name.startswith(test_prefix)):
                ignored += 1
                continue
            lookup = name
            if lookup not in raw_labels and strip_suffix and path.stem.endswith(strip_suffix):
                lookup = path.stem[:-len(strip_suffix)] + path.suffix
            if lookup not in raw_labels:
                failures.append(name)
                continue
            if lookup in matched:
                raise ValueError(f"Multiple images match annotation {lookup}")
            matched.add(lookup)
            target = test if name.startswith(test_prefix) else train
            target.append((path.relative_to(base).as_posix(), mapping[raw_labels[lookup]]))
        missing = sorted(set(raw_labels) - matched)
        if failures or missing:
            raise ValueError(f"Annotation/image mismatch: {len(failures)} unlabeled images, {len(missing)} missing images; examples={failures[:3] + missing[:3]}")
        if not test:
            raise ValueError(f"No {test_prefix}* samples found under {base}")
        splits = _split_training(train, validation_ratio, validation_seed)
        splits["test"] = [FERSample(p, label, CANONICAL_EMOTION_ORDER[label], "test") for p, label in test]
        metadata = {"dataset": _dataset_key(dataset), "raw_label_mapping": mapping, "format": "label_file",
                    "annotation_sha256": hashlib.sha256(label_file.read_bytes()).hexdigest(),
                    "split_sources": {"train": train_prefix, "val": f"stratified holdout from {train_prefix}", "test": test_prefix},
                    "ignored_non_split_images": ignored}
        return cls._build(base, CANONICAL_EMOTION_ORDER, splits, validation_seed, validation_ratio, metadata)

    def summary(self):
        return {
            "root": self.root, "class_order": self.class_order,
            "label_mapping": {name: i for i, name in enumerate(self.class_order)},
            "split_sizes": {name: len(samples) for name, samples in self.splits.items()},
            "class_counts": {name: dict(sorted(Counter(s.class_name for s in samples).items())) for name, samples in self.splits.items()},
            "manifest_hash": self.manifest_hash, "validation_seed": self.validation_seed,
            "validation_ratio": self.validation_ratio,
            "validation_source": self.metadata.get("split_sources", {}).get("val", "stratified holdout"),
            "metadata": self.metadata,
        }

    def write_manifest(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary()
        payload["samples"] = {name: [s.__dict__ for s in samples] for name, samples in self.splits.items()}
        with target.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_protocol(data_config: Mapping):
    dataset = data_config.get("dataset", "rafdb")
    label_file = data_config.get("label_file")
    fmt = data_config.get("format", "label_file" if label_file else "folders")
    common = dict(dataset=dataset, validation_seed=int(data_config.get("validation_seed", 42)),
                  validation_ratio=float(data_config.get("validation_ratio", 0.1)),
                  label_mapping=data_config.get("label_mapping"))
    expected = data_config.get("class_order")
    if expected is not None and expected != "discover_from_actual_directories" and list(expected) != CANONICAL_EMOTION_ORDER:
        raise ValueError("data.class_order differs from canonical seven-class order")
    if fmt == "label_file":
        if data_config.get("exclusions_file"):
            raise ValueError("exclusions_file currently requires folders format")
        if not label_file:
            raise ValueError("data.format='label_file' requires data.label_file")
        return FERProtocol.from_label_file(data_config["root"], label_file,
            train_prefix=data_config.get("train_prefix", "train_"),
            test_prefix=data_config.get("test_prefix", "test_"),
            strip_suffix=data_config.get("strip_suffix", "_aligned"), **common)
    if fmt == "folders":
        exclusion_args = {}
        if data_config.get("exclusions_file"):
            exclusion_file = Path(data_config["exclusions_file"])
            contents = exclusion_file.read_bytes()
            exclusions = json.loads(contents)
            if _dataset_key(exclusions.get("dataset")) != _dataset_key(dataset):
                raise ValueError("Exclusion file dataset does not match data.dataset")
            records = exclusions.get("excluded_samples")
            if not isinstance(records, list) or not records or any(
                not isinstance(record, dict) or not isinstance(record.get("path"), str)
                or not isinstance(record.get("reason"), str) or not record["reason"]
                for record in records
            ):
                raise ValueError("Exclusion file requires nonempty path/reason records")
            exclusion_args = {
                "excluded_paths": [record["path"] for record in records],
                "exclusion_metadata": {"source": exclusions.get("source"),
                    "file_sha256": hashlib.sha256(contents).hexdigest(), "entries": records},
            }
        return FERProtocol.from_image_folders(data_config["root"],
            train_directory=data_config.get("train_directory", "Train"),
            test_directory=data_config.get("test_directory", "Test"),
            validation_directory=data_config.get("validation_directory"),
            validation_usage=data_config.get("validation_usage", "selection"), **exclusion_args, **common)
    raise ValueError(f"Unknown data.format={fmt!r}; expected 'label_file' or 'folders'")


class FERDataset:
    """Map-style dataset accepted by PyTorch DataLoader without eager torch imports."""
    def __init__(self, protocol, split, transform: Optional[Callable] = None):
        if split not in protocol.splits:
            raise KeyError(f"Unknown split {split!r}; available={sorted(protocol.splits)}")
        self.root = Path(protocol.root)
        self.samples = protocol.splits[split]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(self.root / sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.label, sample.path
