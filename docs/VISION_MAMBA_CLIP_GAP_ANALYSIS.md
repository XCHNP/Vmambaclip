# VisionMambaCLIP implementation gap analysis

This record was made before implementing the task specification. It distinguishes
repository facts from the requested design; it is not an experimental result.

## Repository facts (R)

- `src/open_clip/model.py::CLIP` already exposes `encode_image` and
  `encode_text`. Both return the projected CLIP common-space feature when
  `normalize=False`; the repository does not expose a FER classifier adapter.
- `src/open_clip_train/main.py` and `train.py` implement generic OpenCLIP
  contrastive training. They do not implement T1/T2/S1/S2 checkpoint identities,
  FER labels, Vision Mamba hidden-state extraction, or the requested losses.
- `src/open_clip_train/data.py` supports caption CSV/WebDataset inputs and an
  ImageNet-specific `ImageFolder` path. It does not preserve a FER label protocol,
  make a traceable validation split, or emit a manifest hash.
- No Vision Mamba/Vim, hybrid encoder, ResNet18 local branch, AU prompt table, or
  VisionMambaCLIP checkpoint was found in `src/` or `tests/`.
- The checked Python environments do not contain `mamba_ssm`, `vim`, or a
  `vision_mamba` package. Therefore a real Vim forward cannot be claimed in this
  repository snapshot. The implementation must accept a real external Vim while
  rejecting an absent or incompatible backbone clearly.
- The local `src/pretrained_models/open_clip_config.json` and checkpoint tensors
  identify a ViT-B/32 model: image size 224, visual width 768, patch size 32, and
  projected dimension 512. The inspected checkpoint SHA-256 is
  `1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad`.
  This is not the required ViT-L/14 teacher. It remains a
  separately identified local asset and must not be relabelled as ViT-L/14.
- `RAF_DB/Train` and `RAF_DB/Test` contain numeric class directories `1` through
  `7`. The repository provides no authoritative numeric-to-emotion-name mapping
  and no AU prompt source. Code therefore discovers and hashes the numeric order;
  formal AU training requires an explicit, matching prompt schema.
- There is no Git metadata in this source snapshot, so changes cannot be reported
  as a Git diff or commit.

## Target differences

| Responsibility | Existing implementation | Required implementation |
|---|---|---|
| Teacher | bare CLIP common-space encoders | adapter plus linear FER head and frozen text boundary |
| Student global branch | absent | last-three-layer bidirectional Vim adapter, mean/max projection, learned-query cross-attention |
| Student local branch | absent | ResNet18 through `layer3`, non-negative spatial attention, distinct GAP/distillation pooling |
| Fusion/classifier | absent | concat+projection and 4D-to-D BN/ReLU MLP with `[C,D]` final weight |
| Text anchors | absent | strict schema/order validation, frozen normalized buffer, hashes and token metadata |
| Losses | generic CLIP loss only | exact T1, T2, five-term S1, and five-component S2 objectives/directions |
| Stage control | absent | explicit train/eval and `requires_grad` policies for T1/T2/S1/S2 |
| Checkpoints | generic epoch checkpoint | immutable stage identity and protocol validation |
| FER data | absent | discovered label order, deterministic validation split, counts and manifest hash |

Engineering choices (E) introduced by the implementation are documented in
`docs/VISION_MAMBA_CLIP_IMPLEMENTATION.md` and in configuration `source` fields.
