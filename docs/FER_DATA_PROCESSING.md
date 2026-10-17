# FER data processing

The CSV preprocessing entry and VisionMambaCLIP runner now use `build_protocol`
from `src/open_clip/vision_mamba_clip/data.py`. Both use identical canonical
labels, samples, splits and manifest hashes. The canonical output order is:

`Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised` (indices 0–6).

## Actual directories inspected on 2026-10-13

- RAF aligned images (current layout after user flattened extraction): `/root/autodl-tmp/open_clip-main/rafdb`
- SFEW (current flattened layout): `/root/autodl-tmp/open_clip-main/sfew`, containing `train/0..6` and `val/0..6`.
- AffectNet: `/root/autodl-tmp/open_clip-main/affectnet`, containing `train/0..6` and `val/0..6`, including TIFF files.

Do not move or rename the original images. The old RAF CSV uses stale Windows
paths and mixes test-prefixed filenames into a train directory; it is not the
annotation source for this procedure. `affectnet.csv` references a different
image tree (`datasets/AffectNet/train_set/images/...`) and does not establish
the semantic meaning of the downloaded numeric class directories.

## Current confirmed scope (2026-10-13)

RAF-DB, SFEW and AffectNet are in the active processing scope. The user resumed
RAF-DB after preparing `rafdb_labels.txt` (15,339 rows), and resumed SFEW after
confirming its raw 0–6 order as Angry/Disgusted/Fearful/Happy/Neutral/Sad/Surprised.
The user requested data processing only;
real-image decoding and structural audits are not part of these exports.

The user confirmed AffectNet folders 0–6 mean Neutral, Happy, Sad, Surprised,
Fearful, Disgusted, Angry respectively, giving canonical indices 4,3,5,6,2,1,0.
`configs/affectnet_config.json` records this mapping explicitly and reserves
downloaded `val` for final evaluation. A per-class 10% holdout from train selects
checkpoints, with seed 42. Stage checkpoint paths use a separate affectnet directory.
SFEW uses the same evaluation/selection separation, with its identity raw-to-canonical
mapping recorded in `configs/sfew_config.json` and separate stage checkpoint paths.

SFEW/AffectNet require `data.label_mapping`: exactly seven raw labels mapped
to canonical integer indices or canonical emotion names. They have no default
numeric mapping. Seven-class mode rejects extra class folders rather than
silently treating contempt or invalid categories as one of the seven emotions.

For either downloaded folder dataset, configure `format="folders"`, the actual
`root`, `train_directory="train"`, `validation_directory="val"` and
`test_directory=null`. Choose one explicit policy:

- `validation_usage="selection"`: preserve downloaded train and val; no test split is invented.
- `validation_usage="evaluation"`: preserve downloaded val as the final evaluation
  (`test` runner slot), and hold out `validation_ratio` of train for model selection.
  The manifest records `test`'s source as `val`; reports must describe it as
  downloaded validation, not official test.

RAF uses `format="label_file"`, original raw mapping 1→6, 2→2, 3→1, 4→3,
5→5, 6→0, 7→4, and deterministic per-class training holdout. Missing images,
missing labels, duplicate annotations and repeated annotation matches raise
errors. The configured holdout ratio and filename prefixes are honored.

## Export after confirmation

Run from `/root/autodl-tmp/open_clip-main` in the `clip` environment:

```bash
PYTHONPATH=src /root/miniconda3/envs/clip/bin/python -m open_clip_train.predata_fer \
  --config configs/affectnet_config.json --inspect-only

PYTHONPATH=src /root/miniconda3/envs/clip/bin/python -m open_clip_train.predata_fer \
  --config configs/affectnet_config.json --output-dir artifacts/preprocessed/affectnet

PYTHONPATH=src /root/miniconda3/envs/clip/bin/python -m open_clip_train.predata_fer \
  --config configs/rafdb_config.json --output-dir artifacts/preprocessed/rafdb

PYTHONPATH=src /root/miniconda3/envs/clip/bin/python -m open_clip_train.predata_fer \
  --config configs/sfew_config.json --output-dir artifacts/preprocessed/sfew
```

The commands above do not open or decode real images. They enumerate image
paths, match labels, create deterministic splits, and write outputs.
Use a new output directory; existing exports are never overwritten. Every
available split gets its own CSV with `image_path,caption,labels`, plus a
manifest and input contract. Captions come from the configured AU prompt schema,
whose canonical order and source must match. Such captions are derived prompts,
not original dataset AU annotations. Generic OpenCLIP still consumes image/text
pairs; VisionMambaCLIP consumes canonical labels through the shared protocol.

Generic OpenCLIP arguments for these CSVs:
`--dataset-type csv --csv-separator ',' --csv-img-key image_path --csv-caption-key caption`.

`--verify-images` without a mode decodes every image before creating any output.
`--verification-mode structure-and-sample` checks all file structures/headers
and fully decodes up to 100 images per class per split with a recorded seed.
JPEG header checks alone do not validate every compressed pixel stream. This
mode is optional and was stopped at the user's request. Current exports record
`image_verification.mode="not-run"` in `input_contract.json`. Data assembly
also checks resolved path uniqueness across splits, including symlinks. This is
not a full image-content duplicate audit.

## Verification and compatibility

`/root/miniconda3/envs/clip/bin/python tests/test_fer_data_protocol.py` runs fifteen
temporary-data tests, including loader/export parity, canonical labels, both
validation policies, configurable RAF holdout, TIFF/RGB handling, separator
handling, annotation mismatches, path overlaps, and overwrite refusal.

Manifest metadata now records raw mapping and split provenance, so manifest
hashes differ from older protocol versions. Existing stage checkpoints remain
subject to the runner's manifest identity checks; do not bypass a mismatch.
The subsequent directory-cache and streaming-hash optimizations preserve the
same serialized protocol hash while lowering preprocessing memory consumption.
The preprocessing entry loads the shared data source without loading the whole
OpenCLIP model package. This supports the CPU instance's 2 GB memory limit.

Original changed files are preserved in `artifacts/fer_data_backup_20261013/`.
All exported CSV rows are checked against their corresponding manifest and raw
labels, with split counts and hashes recorded in each export's `audit.json`.
Model training is a separate operation and has not been started by this data task.

## Current export results after approved SFEW cleanup

| Dataset | Train | Model-selection val | Final evaluation |
| --- | ---: | ---: | ---: |
| RAF-DB | 11040 | 1231 | 3068 |
| SFEW (cleaned export) | 763 | 88 | 402 |
| AffectNet | 255507 | 28393 | 3500 |

All fifteen regression tests and all CSV/manifest/raw-label consistency audits
passed. The CSV consistency audit does not imply that the source dataset is free
from duplicate image content. RAF-DB and AffectNet have no cross-source matching
basenames. SFEW has 148 such basenames; targeted size/SHA256 comparisons found
148 distinct training files byte-identical to final-evaluation files (149 pairs).

The image `21_001703120_00000012.png` is identical at `train/6/`, `val/6/` and
`val/3/`; it had both Surprised and Happy labels in final evaluation.
The user approved excluding the 148 duplicate training files and both conflicting
evaluation copies. The explicit list is `configs/sfew_exclusions.json`; the
shared loader applies it before drawing the training holdout, ensuring CSV export
and direct training use the same retained samples. Counts, reasons and exclusion
file hash are recorded in the cleaned manifest. Missing exclusion paths are errors.

No original image was changed or deleted. The unfiltered export and original
duplicate evidence are backed up at
`artifacts/preprocessed/sfew_before_cleanup_20261013/`. The cleaned export at
`artifacts/preprocessed/sfew/` has no cross-source matching basenames and none of
the previously identified duplicate pairs. Its final evaluation is a curated
402-image subset of downloaded val, and must be reported as such.
`cleanup_audit.json` verifies that exactly the approved 150 samples were excluded
and all their original files still exist. RAF-DB and AffectNet exports are unaffected.
