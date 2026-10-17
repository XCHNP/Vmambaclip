"""CPU-only regression tests; no real dataset labels are assumed."""

import importlib.util
import hashlib
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

# Avoid importing the entire model stack for data-only regression checks.
MODULE_PATH = Path(__file__).resolve().parents[1] / "src/open_clip/vision_mamba_clip/data.py"
spec = importlib.util.spec_from_file_location("fer_data_under_test", MODULE_PATH)
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)


class FERProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mapping = {str(i): name for i, name in enumerate(data.CANONICAL_EMOTION_ORDER)}

    def folders(self, names=("train", "val")):
        for split in names:
            for raw in self.mapping:
                directory = self.root / split / raw
                directory.mkdir(parents=True)
                for i in range(10 if split == "train" else 2):
                    Image.new("RGB", (5, 5), (int(raw), i, 1)).save(directory / f"{i}.png")
        (self.root / "val/0/.ipynb_checkpoints").mkdir()

    def config(self, **kwargs):
        config = dict(dataset="sfew", root=str(self.root), format="folders", label_mapping=self.mapping,
                      train_directory="train", validation_directory="val", test_directory=None)
        config.update(kwargs)
        return config

    def test_unknown_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "confirmed"):
            data.build_protocol(self.config(label_mapping=None))

    def test_affectnet_confirmed_config_mapping_and_split_policy(self):
        project = MODULE_PATH.parents[3]
        config = json.loads((project / "configs/affectnet_config.json").read_text())["data"]
        self.assertEqual(data._resolve_index_mapping("affectnet", config["label_mapping"]),
                         {"0": 4, "1": 3, "2": 5, "3": 6, "4": 2, "5": 1, "6": 0})
        self.assertEqual(config["validation_usage"], "evaluation")
        self.assertIsNone(config["test_directory"])
        self.assertEqual(config["validation_ratio"], 0.1)

    def test_sfew_confirmed_config_mapping_and_split_policy(self):
        config = json.loads((MODULE_PATH.parents[3] / "configs/sfew_config.json").read_text())["data"]
        self.assertEqual(data._resolve_index_mapping("sfew", config["label_mapping"]), {str(i): i for i in range(7)})
        self.assertEqual(config["validation_usage"], "evaluation")
        self.assertIsNone(config["test_directory"])
        self.assertEqual(config["validation_ratio"], 0.1)

    def test_exclusions_are_applied_before_training_holdout(self):
        self.folders()
        exclusions = self.root / "excluded.json"
        records = [{"path": "train/0/0.png", "reason": "duplicate"},
                   {"path": "val/3/0.png", "reason": "conflicting annotation"}]
        exclusions.write_text(json.dumps({"dataset": "sfew", "source": "test fixture", "excluded_samples": records}))
        config = self.config(validation_usage="evaluation", exclusions_file=str(exclusions))
        protocol = data.build_protocol(config)
        paths = {s.path for samples in protocol.splits.values() for s in samples}
        self.assertFalse(paths & {record["path"] for record in records})
        self.assertEqual(len(paths), 82)
        self.assertEqual(protocol.metadata["exclusions"]["excluded_counts_by_source"], {"train": 1, "val": 1})
        self.assertTrue((self.root / "train/0/0.png").exists())
        self.assertEqual(protocol.manifest_hash, data.build_protocol(config).manifest_hash)

    def test_exclusions_reject_missing_paths_and_wrong_dataset(self):
        self.folders()
        exclusions = self.root / "excluded.json"
        record = {"path": "train/0/missing.png", "reason": "duplicate"}
        exclusions.write_text(json.dumps({"dataset": "sfew", "excluded_samples": [record]}))
        with self.assertRaisesRegex(ValueError, "not found"):
            data.build_protocol(self.config(exclusions_file=str(exclusions)))
        exclusions.write_text(json.dumps({"dataset": "rafdb", "excluded_samples": [record]}))
        with self.assertRaisesRegex(ValueError, "dataset does not match"):
            data.build_protocol(self.config(exclusions_file=str(exclusions)))

    def test_structural_and_sample_verification_records_coverage(self):
        sys.path.insert(0, str(MODULE_PATH.parents[2]))
        from open_clip_train.predata_fer import _verify_images

        self.folders()
        protocol = data.build_protocol(self.config(validation_usage="evaluation"))
        with contextlib.redirect_stdout(io.StringIO()):
            report = _verify_images(protocol, "structure-and-sample", 1)
        self.assertEqual(report["files_checked"], 84)
        self.assertEqual(report["images_decoded"], 21)
        self.assertEqual(report["sample_seed"], 42)

    def test_corrupt_file_aborts_verification(self):
        sys.path.insert(0, str(MODULE_PATH.parents[2]))
        from open_clip_train.predata_fer import _verify_images

        self.folders()
        (self.root / "train/0/broken.jpg").write_bytes(b"broken image")
        protocol = data.build_protocol(self.config())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "unreadable images"):
            _verify_images(protocol, "structure-and-sample", 1)

    def test_existing_validation_is_preserved_without_fake_test(self):
        self.folders()
        protocol = data.build_protocol(self.config())
        self.assertEqual({s: len(v) for s, v in protocol.splits.items()}, {"train": 70, "val": 14})
        self.assertEqual(protocol.validation_ratio, 0.0)
        self.assertTrue(all(s.path.startswith("val/") for s in protocol.splits["val"]))
        self.assertEqual(protocol.summary()["metadata"]["split_sources"]["val"], "val")

    def test_validation_as_final_evaluation_has_independent_selection(self):
        self.folders()
        protocol = data.build_protocol(self.config(validation_usage="evaluation", validation_ratio=0.2))
        self.assertEqual({s: len(v) for s, v in protocol.splits.items()}, {"train": 56, "val": 14, "test": 14})
        self.assertTrue(all(s.path.startswith("train/") for s in protocol.splits["val"]))
        self.assertTrue(all(s.path.startswith("val/") for s in protocol.splits["test"]))
        self.assertEqual(protocol.manifest_hash, data.build_protocol(self.config(validation_usage="evaluation", validation_ratio=0.2)).manifest_hash)
        paths = [s.path for samples in protocol.splits.values() for s in samples]
        self.assertEqual(len(paths), len(set(paths)))

    def test_tiff_and_rgb_conversion(self):
        self.folders()
        path = self.root / "train/0/extra.tif"
        Image.new("L", (5, 5)).save(path)
        protocol = data.build_protocol(self.config())
        dataset = data.FERDataset(protocol, "train")
        index = next(i for i, s in enumerate(dataset.samples) if s.path.endswith("extra.tif"))
        image, label, _ = dataset[index]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(label, 0)
        import torch
        from torch.utils.data import DataLoader

        tensor_dataset = data.FERDataset(protocol, "train", lambda image: torch.zeros(3, 5, 5))
        images, labels, paths = next(iter(DataLoader(tensor_dataset, batch_size=2)))
        self.assertEqual(tuple(images.shape), (2, 3, 5, 5))
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertEqual(len(paths), 2)

    def test_duplicate_mapping_and_unknown_folder_are_rejected(self):
        self.folders()
        bad = dict(self.mapping, **{"0": "Happy"})
        with self.assertRaisesRegex(ValueError, "exactly once"):
            data.build_protocol(self.config(label_mapping=bad))
        (self.root / "train/7").mkdir()
        with self.assertRaisesRegex(ValueError, "do not match"):
            data.build_protocol(self.config())

    def labels(self):
        rows = []
        for raw in range(1, 8):
            for i in range(10):
                name = f"train_{raw}_{i}.jpg"
                Image.new("RGB", (5, 5)).save(self.root / name.replace(".jpg", "_aligned.jpg"))
                rows.append(f"{name} {raw}")
            name = f"test_{raw}.jpg"
            Image.new("RGB", (5, 5)).save(self.root / name.replace(".jpg", "_aligned.jpg"))
            rows.append(f"{name} {raw}")
        labels = self.root / "labels.txt"
        labels.write_text("\n".join(rows), encoding="utf-8")
        return labels

    def test_raf_alignment_mapping_and_ratio(self):
        labels = self.labels()
        protocol = data.build_protocol(dict(dataset="rafdb", root=str(self.root), label_file=str(labels), validation_ratio=0.2))
        self.assertEqual({s: len(v) for s, v in protocol.splits.items()}, {"train": 56, "val": 14, "test": 7})
        self.assertEqual(next(s.label for s in protocol.splits["test"] if s.path == "test_1_aligned.jpg"), 6)
        hash_payload = {
            "class_order": protocol.class_order, "validation_seed": protocol.validation_seed,
            "validation_ratio": protocol.validation_ratio, "metadata": protocol.metadata,
            "splits": {split: [s.__dict__ for s in samples] for split, samples in protocol.splits.items()},
        }
        expected_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(protocol.manifest_hash, expected_hash)
        target = self.root / "manifest.json"
        protocol.write_manifest(str(target))
        self.assertEqual(json.loads(target.read_text())["manifest_hash"], protocol.manifest_hash)
        with self.assertRaises(FileExistsError):
            protocol.write_manifest(str(target))

    def test_raf_unlabeled_images_are_not_silently_skipped(self):
        labels = self.labels()
        Image.new("RGB", (5, 5)).save(self.root / "train_extra_aligned.jpg")
        with self.assertRaisesRegex(ValueError, "unlabeled images"):
            data.FERProtocol.from_label_file(str(self.root), str(labels))

    def test_symlink_overlap_is_rejected(self):
        self.folders()
        (self.root / "val/0/0.png").unlink()
        (self.root / "val/0/0.png").symlink_to(self.root / "train/0/0.png")
        with self.assertRaisesRegex(ValueError, "Duplicate image path"):
            data.build_protocol(self.config())

    def test_csv_and_runner_use_identical_samples_and_labels(self):
        import csv

        sys.path.insert(0, str(MODULE_PATH.parents[2]))
        from open_clip_train.predata_fer import preprocess
        from open_clip_train.data import CsvDataset

        self.folders()
        prompt_file = self.root / "prompts.json"
        prompt_file.write_text(json.dumps({
            "class_order": data.CANONICAL_EMOTION_ORDER,
            "prompts": {name: f"test prompt for {name}" for name in data.CANONICAL_EMOTION_ORDER},
            "source": "synthetic integration test, not real AU annotations",
        }), encoding="utf-8")
        config_file = self.root / "config.json"
        config = self.config(validation_usage="evaluation", au_prompt_schema=str(prompt_file))
        config_file.write_text(json.dumps({"data": config}), encoding="utf-8")
        output = self.root / "export"
        with contextlib.redirect_stdout(io.StringIO()):
            preprocess(str(config_file), str(output), verify_images=True)
        protocol = data.build_protocol(config)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["manifest_hash"], protocol.manifest_hash)
        for split, samples in protocol.splits.items():
            with (output / f"{split}.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([(row["image_path"], int(row["labels"])) for row in rows],
                             [(str(self.root / sample.path), sample.label) for sample in samples])
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = CsvDataset(str(output / "train.csv"), lambda image: image,
                                 "image_path", "caption", sep=",", tokenizer=lambda texts: [[1]])
        self.assertEqual(dataset[0][0].mode, "RGB")
        tsv = self.root / "input.tsv"
        with tsv.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["image_path", "caption"])
            writer.writerow([str(self.root / "train/0/0.png"), "a test face"])
        with contextlib.redirect_stdout(io.StringIO()):
            tab_dataset = CsvDataset(str(tsv), lambda image: image,
                                     "image_path", "caption", sep="\t", tokenizer=lambda texts: [[1]])
        self.assertEqual(len(tab_dataset), 1)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            preprocess(str(config_file), str(output))


if __name__ == "__main__":
    unittest.main()
