# T1-5 — Soft Pseudo-Label Precompute (CPU-only Kaggle kernel)

Pre-computes **SOFT** pseudo-labels for the **10,592 unlabeled `train_soundscapes`** using the
best inference-only PUBLIC ensemble, so the noisy-student training set (T2-A) is ready the instant
GPU resets. Built by Build-Agent B1 for the BirdCLEF+ 2026 sprint.

See: `sprint_2026-05-29_architecture/00_MASTER_BOARD.md` (T1-5),
`agents/A6_prior_winners.md` (noisy-student protocol), `01_LATEST_PUBLIC_095.md` (0.950 anchor).

---

## What it does

1. Lists the 10,592 unlabeled soundscapes (skips files already in `train_soundscapes_labels.csv` if present).
2. For each file: load → mono → 32 kHz → exactly 60 s → **12 windows of 5 s** (the repo convention,
   matching `inference_notebooks/` and `g124_2025pre_pseudo/infer.py`).
3. Runs two PUBLIC branches on **CPU via onnxruntime**:
   - **Perch v2** (`perch_v2_no_dft.onnx`): per-5s-window raw waveform `(N, 160000)` → 234-class
     species head (sigmoid). The ONNX also exposes a 1536-D embedding.
   - **Distilled-SED** (`sed_fold*.onnx`): log-mel `(N, 1, 256, T)` (n_fft=2048, hop=512, n_mels=256,
     fmin=20, fmax=16000, top_db=80, per-clip standardized — verbatim from `exp019_fast.py`) →
     `0.5*sigmoid(clip_logits) + 0.5*sigmoid(frame_max)`, fold-averaged.
4. **Rank-blends** them with the 0.950 anchor's dominant-branch weights:
   `z = 0.60 * R(p_perch) + 0.40 * R(p_sed)`, where `R` = class-wise percentile rank
   (computed per shard of ~1000 files = ~12k rows, plenty for stable per-class ranks).
5. Applies the **A6 soft-label power-transform** (5th-place verified):
   `power_transform(p) = clamp( p*(p>0.3) + p**2 , 0, 1 )`.
6. Saves per-window + per-file soft labels and a manifest.

If only one branch is mountable, it gracefully falls back to that branch alone (rank renormalizes).
If the Perch ONNX has no 234-class head, the kernel uses SED-only.

---

## Output schema (the noisy-student consumption contract for B2)

### `pseudo_soft_per_window.parquet`  (≈ 127,104 rows = 10,592 × 12)
| column          | type     | meaning |
|-----------------|----------|---------|
| `filename`      | str      | source `.ogg` name (e.g. `BC2026_xxx.ogg`) |
| `start_sec`     | int      | window start in seconds: 0,5,10,…,55 |
| `window_idx`    | int      | 0..11 |
| `row_id`        | str      | `{stem}_{end_sec}` (e.g. `BC2026_xxx_5`) — same id convention as submissions |
| `win_confidence`| float32  | max soft score in this window (per-window confidence) |
| `<234 species>` | float32  | post-power-transform SOFT scores, one column per `primary_label` |

### `pseudo_soft_per_window.npz`  (compact mirror for fast training-time load)
`soft` `(N,234) float32`, `filename` `(N,)`, `start_sec` `(N,) int32`, `window_idx` `(N,) int32`,
`classes` `(234,)` (column order == species order in `sample_submission.csv`).

### `pseudo_soft_per_file.parquet`  (10,592 rows)
| column            | meaning |
|-------------------|---------|
| `filename`        | source `.ogg` name |
| `file_confidence` | mean over the 12 windows of `win_confidence` (how "callful" the file is) |
| `<234 species>`   | per-file soft labels = **max over the 12 windows** (the repo's `file_pred` rule) |

### `manifest.json`
Records `n_classes`, `n_windows_per_file`, `window_sec`, `sr`, the blend weights, the soft-label
transform constants, and the downstream-mixing contract.

> If the Kaggle image lacks a parquet engine the tables are written as `.csv` instead (same columns);
> `manifest.json["files"]` always points at whatever was actually written.

---

## Consumption contract for B2's noisy-student trainer (READ THIS)

- These are **SOFT teacher probabilities AFTER the power-transform** (`p*(p>0.3)+p**2`). Do **not**
  re-apply the power-transform.
- They are **NOT** mixed with hard labels yet. The unlabeled soundscapes have no hard label, so in the
  training loop apply the A6 rule:
  ```python
  # hard = 0 for pure-unlabeled rows (no ground truth)
  target = PSEUDO_ALPHA * soft + (1 - PSEUDO_ALPHA) * hard   # PSEUDO_ALPHA = 0.7
  ```
  i.e. for the unlabeled soundscapes the effective target is simply `0.7 * soft`.
- Column order of the 234 scores == `sample_submission.csv` / `taxonomy.csv` `primary_label` order.
- Use `win_confidence` / `file_confidence` if you want to gate or weight pseudo rows (A6 caps the
  pseudo-ratio ≤ ~0.4 and 2nd-place selects by recall — both optional, applied at training time).
- Window framing matches training: 12 × 5 s windows per 60 s file, `row_id = {stem}_{end_sec}`.

---

## How to run on Kaggle (CPU)

1. Create a **CPU** notebook/script kernel (GPU OFF, internet OFF).
2. Attach inputs:
   - `birdclef-2026` competition data (auto).
   - `tuckerarrants/perch-v2-no-dft-onnx`  ← Perch ONNX **+ the onnxruntime CPU wheel** (verified live).
   - `tuckerarrants/bc2026-distilled-sed-public`  ← `sed_fold*.onnx` (verified live).
   - (fallback Perch: `rishikeshjani/perch-onnx-for-birdclef-2026`.)
3. Push with `kernel-metadata.json` (already set: `enable_gpu=false`, `enable_internet=false`).
4. Outputs land in `/kaggle/working/`. Package them as a dataset with `dataset-metadata.json`
   (`kaggle datasets create -p .` after copying the outputs + this json), or "Save Version" and
   "New Dataset" from the kernel output.

### Local / smoke test
Env overrides: `PL_LIMIT_FILES` (cap files), `PL_OUT_DIR`, `PL_BATCH_FILES`, `PL_NUM_WORKERS`,
`PL_SHARD_FILES`, `PL_ONLY_UNLABELED`.
```bash
PL_LIMIT_FILES=8 PL_OUT_DIR=/tmp/out python precompute_pseudo_labels.py
```

---

## Expected Kaggle CPU runtime

Two ONNX branches, 8-file batched calls, 4 IO-prefetch threads, 4 intra-op threads.
Repo baselines: Perch-only ≈0.33 f/s un-batched (`sub2`), batched ≈ much faster. Combined
Perch+SED batched throughput lands around **~1.0–1.5 files/s**:

| throughput | 10,592 files |
|-----------:|-------------:|
| 1.5 files/s | ~2.0 h |
| 1.0 files/s | ~2.9 h |
| 0.8 files/s | ~3.7 h |

**Comfortably inside Kaggle's 12 h CPU kernel limit.** (This is a utility kernel — it is NOT the
90-min submission kernel; that budget does not apply here.) Tune `PL_BATCH_FILES` up if RAM allows.
The run is shard-checkpoint friendly (rank-blend done per ~1000-file shard) so memory stays flat.
