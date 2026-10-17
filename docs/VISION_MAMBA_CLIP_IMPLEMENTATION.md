# VisionMambaCLIP implementation guide

This implementation follows the supplied task specification. `P`, `W`, `E`, and
`R` retain the meanings defined there. It is an implementation and structural
validation record, not a reproduced accuracy claim.

## Tensor flow

```text
images [B,3,256,256]
  |
  +-- real bidirectional Vim -- last 3 layers -- remove special tokens
  |       |                         hidden e/m/l [B,N_i,C_i]
  |       +-- mean || max -- Linear(2C_i,D) -- state_i [B,D]
  |                                      |
  |          learned query [1,1,D] -- cross-attention over 3 states
  |                                      |
  |                                mamba_global [B,D]
  |
  +-- ResNet18 through layer3 -- cnn_raw [B,Ccnn,Hc,Wc]
          |-- Conv1x1(Ccnn,D) -- patch_norm [B,Hc*Wc,D]
          |-- softplus attention -- A [B,1,Hc,Wc]
          |-- mean(A * projected) -- cnn_descriptor [B,D]
          +-- sum(A * projected)/sum(A) -- cnn_distill [B,D]

mamba_global || cnn_descriptor -- Linear(2D,D) -- fused_raw [B,D]
e || m || l || fused -- Linear(4D,D)-BN-ReLU-Linear(D,C) -- logits [B,C]
```

The student forward call receives images only. Labels, teachers, text targets,
and spatial KL construction live exclusively in loss/training code.

## File responsibilities

- `adapters.py`: projected CLIP teacher and strict last-three-layer Vim hook
  adapter. A caller must explicitly attest the backbone is bidirectional Vim.
- `model.py`: ResNet18-layer3 branch, positive spatial attention, distinct local
  pooling interfaces, learned-query aggregation, concat fusion, and MLP head.
- `losses.py`: exact T1/T2 objectives and complete S1/S2 losses, with KL
  directions written explicitly and finite-value failures surfaced.
- `stages.py`: train/eval and `requires_grad` boundaries for all four stages.
- `text_anchors.py`: sourced prompt schema, exact class-order checks, frozen CLIP
  text encoding, normalized buffer, token IDs, identities, and hashes.
- `data.py`: actual-folder label discovery, fixed stratified validation split,
  class counts, mapping, and manifest hash.
- `checkpoint.py`: immutable stage checkpoints, strict state loading, and
  stage/dimension/class/data/prompt identity validation.
- `training.py`: composable small-batch steps and gradient summaries.
- `open_clip_train/vision_mamba_clip.py`: shared four-stage training runner.
- `open_clip_train/vision_mamba_clip_assembly.py`: OpenCLIP/RAF assembly plus a
  required external real-Vim factory; missing assets never fall back silently.
- `configs/vision_mamba_clip/`: complete target, smoke, and named single-change
  ablations.
- `artifacts/raf_db_protocol.json`: generated inventory of the present RAF-DB
  folders and deterministic validation split.

## Real commands and checkpoint relationships

Install the project normally (`pip install -e .`) so its declared dependencies
are available. Before formal training, set these fields in a run-specific copy of
`full_paper_target.json`:

1. `teacher.checkpoint`: real pretrained CLIP ViT-L/14 checkpoint.
2. `student.factory`: `package.module:callable` for a real bidirectional Vim.
   The callable receives `checkpoint_path` plus `factory_kwargs` and must return
   `(backbone, metadata)`. Metadata must contain `identity`,
   `pretrained_loaded`, and `initialization_source`.
3. `student.checkpoint`: the real Vim initialization checkpoint, if used.
4. `data.au_prompt_schema`: sourced JSON prompt table in exact discovered class
   order.
5. Each completed stage path in `checkpoints` before starting its dependent
   stage.

Data audit:

```powershell
python -m open_clip_train.vision_mamba_clip inspect-data `
  --config configs/vision_mamba_clip/full_paper_target.json `
  --root RAF_DB --manifest artifacts/raf_db_protocol.json
```

Four stages (use a unique output path; overwriting is refused):

```powershell
python -m open_clip_train.vision_mamba_clip run-stage --config RUN.json --stage teacher_t1 --assembly open_clip_train.vision_mamba_clip_assembly:build --output checkpoints/teacher_t1.pt
python -m open_clip_train.vision_mamba_clip run-stage --config RUN.json --stage teacher_t2 --assembly open_clip_train.vision_mamba_clip_assembly:build --output checkpoints/teacher_t2.pt
python -m open_clip_train.vision_mamba_clip run-stage --config RUN.json --stage student_s1 --assembly open_clip_train.vision_mamba_clip_assembly:build --output checkpoints/student_s1.pt
python -m open_clip_train.vision_mamba_clip run-stage --config RUN.json --stage student_s2 --assembly open_clip_train.vision_mamba_clip_assembly:build --output checkpoints/student_s2.pt
```

Student-only evaluation (loads neither teacher nor text anchors):

```powershell
python -m open_clip_train.vision_mamba_clip evaluate-student --config RUN.json --assembly open_clip_train.vision_mamba_clip_assembly:build_inference --split test --output results/student_s2_test.json
```

The loader enforces:

```text
original_clip -> teacher_t1 -> teacher_t2
student_init  -> student_s1  -> student_s2
teacher_t1 supervises student_s1
teacher_t2 supervises student_s2
```

## Engineering decisions (E)

- `K=C`: one sourced AU description per class.
- Special tokens are removed before Vim pooling; selected physical layer indices
  are stored, while e/m/l remain interface aliases.
- Per-layer mean/max concatenation is projected to D, then three D-dimensional
  states are the key/value sequence for learned-query cross-attention.
- Spatial attention is `softplus(logits)+epsilon`; coordinates default off.
- The fusion descriptor uses attention-weighted GAP; the distillation descriptor
  uses attention-sum normalization. These remain separate tensors.
- Classifier input is `state_e || state_m || state_l || fused`; hidden size is D.
- `phi` starts as identity, is frozen in S1, and is trainable in the main S2 run.
- S2 spatial `Norm` is per-sample min-max; constant maps become zero maps and
  therefore uniform target distributions. The target is detached by default.
- Text encoders are frozen for all stages. T1 trains visual+logit scale; T2 trains
  visual+linear head. S1 freezes unreachable query/attention/fusion/classifier/
  phi modules; S2 trains the complete student.
- A deterministic 10% per-class validation subset is derived from `Train` with
  seed 42 because the repository has no explicit validation directory.
- Initial loss coefficients and per-stage 20 epochs in the full config are startup
  choices, not paper-validated optima. Strict warmup rejects 10,000 steps when a
  stage has too few optimizer steps.

## Validation completed on this snapshot

- Unit suite: shape/output contract, concat fusion, last-three-layer identity,
  rejection of unattested non-bidirectional backbones, asymmetric KL references,
  complete S1/S2 gradients and freezes, constant-map/low-attention/low-temperature/
  large-logit finite behavior, detached spatial target, strict prompt order,
  checkpoint round-trip/mismatch/no-overwrite, and label-free inference.
- Synthetic-interface small batches: complete S1 and S2 losses each backpropagate.
  These tests are explicitly stubs and make no real-Vim claim.
- Actual local CLIP asset: the repository's ViT-B/32 checkpoint ran B=2 at
  256x256 after positional interpolation on CUDA. Image/text features were
  `[2,512]`, logits `[2,7]`, all finite. This validates the adapter only and is not
  the requested ViT-L/14 teacher. Checkpoint SHA-256:
  `1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad`.
- Actual RAF folders: class order `1..7`; train/validation/test counts
  `12429/1381/3056`; manifest SHA-256
  `56f520e1f2868b0435fcc5ddf98162eae1643dfbfd06a2ce7aa30c2942bd632a`.

## Unverified or missing (R)

- Required pretrained CLIP ViT-L/14 checkpoint.
- A real bidirectional Vim implementation and checkpoint; consequently the real
  Vim forward, cost, throughput, and memory acceptance remain unverified.
- A sourced seven-class AU prompt table and authoritative mapping from numeric RAF
  folders to emotion names. No table was fabricated.
- Formal T1/T2/S1/S2 training, validation selection, convergence, accuracy, and
  performance profiling.
- DDP local-to-global logits/rank offsets/gather gradients.
- Actual BF16/checkpointing fallback metadata in a formal hardware run.

