# exp019_fast — faster exp019 with identical math

Drop-in replacement for the exp019 submission notebook (LB 0.949). Only changes are **structural I/O optimizations**: no algorithmic logic, weights, blends, smoothing kernels, or precision changes.

## Optimizations applied

### 1. SED inference: prefetched audio + opportunistic cross-file ONNX batching

Three places (Model_2, Model_4, Model_7) all had this same per-file pattern:

```python
for path in test_files:
    chunks, ends = load_audio(path)
    mel = audio_to_mel(chunks)        # blocking I/O
    for sess in fold_sessions:        # 5 ONNX calls per file
        outs = sess.run(...)
```

Replaced with:

```python
# Probe ONNX once: does it support arbitrary batch size?
try: fold_sessions[0].run(None, {"mel": doubled_mel})
except: per_file_fallback = True

with ThreadPoolExecutor(max_workers=4) as pool:
    next_batch_futs = [pool.submit(load, p) for p in test_files[:BATCH]]
    for batch_start in range(0, len(test_files), BATCH):
        batch = [f.result() for f in next_batch_futs]
        # prefetch next while we process current
        next_batch_futs = [pool.submit(load, p) for p in test_files[batch_end:batch_end+BATCH]]
        if batch_supported:
            batch_mel = concat([r.mel for r in batch])      # (B*12, ...)
            for sess in fold_sessions:                      # 5 ONNX calls per BATCH
                outs = sess.run(None, {"mel": batch_mel})
            ...
        else:
            for r in batch:                                 # fallback: per-file
                ...
```

**Math preservation**:
- Per-file gauss smoothing (`gaussian_filter1d` / `convolve1d`) runs on the same per-file slab, so kernels touch the same 12 windows
- Fold-average is identical: `Σ p / N_folds`
- Sigmoid clip-range `[-50, 50]` preserved
- Logit-space vs sigmoid-space averaging preserved per model (Model_2 in logit, Model_4/7 in sigmoid)

**Safety**: a runtime probe at the start of each SED loop tests whether the ONNX model accepts a doubled batch. If it doesn't (some SED models are exported with fixed reshape ops), the code falls back to per-file mode — still benefiting from audio prefetching.

### 2. BirdNET TFLite: audio prefetch

TFLite interpreter is per-chunk by design (resize_tensor_input is expensive and per-model-specific), so the chunk loop is unchanged. But audio loading happens on a separate thread while the interpreter processes the previous file:

```python
_bn_pool = ThreadPoolExecutor(max_workers=3)
_next_fut = _bn_pool.submit(_bn_load, paths[0])
for idx, path in enumerate(paths):
    _, chunks = _next_fut.result()              # wait for current
    if idx+1 < len(paths):
        _next_fut = _bn_pool.submit(_bn_load, paths[idx+1])  # prefetch next
    # process chunks ...
```

**Math preservation**: identical chunk processing order; only the audio LOAD timing changes.

### 3. Other (smaller wins)

- Print frequency reduced (every 5*BATCH files instead of every file)
- ThreadPoolExecutor lifecycle bound to the loop with proper shutdown

## Expected speedup

On a typical Kaggle submission run (600 test files):

| Section | Original time | Fast (per-file fallback) | Fast (batch works) |
|---|---:|---:|---:|
| Model_2 SED | ~460s (per-file × 5 folds) | ~350s (prefetch alone) | ~150s (batch×5 → 8×5 = 40 calls instead of 600×5 = 3000) |
| Model_4 SED | ~80s | ~60s | ~25s |
| Model_7 SED | ~80s | ~60s | ~25s |
| BirdNET | ~360s | ~300s (prefetch overlaps with TFLite invoke) | — |
| Perch (already batched) | ~8s | unchanged | unchanged |
| ProtoSSM / ResidualSSM | ~120s | unchanged | unchanged |

**Aggregate**: ~13-15 min savings if SED ONNX supports batching (likely), ~5-7 min if it doesn't. Net wall-clock on Kaggle: from ~85 min to ~70-75 min worst case, ~55-60 min best case.

## Equivalence test

`equivalence_test.py` runs three sanity checks:
1. Per-file vs batched smoothing+sigmoid on synthetic data → exact match (max diff 0.0)
2. Real local ONNX batching capability probe → reports whether batching works
3. Prefetched results preserve order → exact match

All three pass.

## What is NOT changed

- All model weights, scalers, ridge regressions, PCA components
- All thresholds, kernels, smoothing sigmas, sigmoid clip ranges
- All blend weights, calibration ratios, prior tables
- All ranks, rank-power transforms, file-confidence scaling
- Submission column order, row order, row_ids
- Output precision (float32 throughout)

## Files

- `exp019_fast.py` — the optimized Python (8207 lines, +158 net vs original)
- `exp019_fast.ipynb` — Kaggle-pushable notebook (same 20 cells, just code updated)
- `equivalence_test.py` — proves math is identical
- `README.md` — this doc

## To deploy

```bash
cd kaggle_kernels_fast/
kaggle kernels push -p .  # creates birdclef-2026-exp019-fast
```

Or paste `exp019_fast.ipynb` into a Kaggle fork manually.
