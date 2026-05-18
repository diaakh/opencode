# V6: Real Audio Augmentation Benchmark Results

## Method
Ran `augmentation_benchmark.py` with:
- Model: fold0.onnx (60s waveform → 12-window logits, 234 classes)
- Data: 30 labeled train_soundscapes files (360 5-sec windows)
- Ground truth: train_soundscapes_labels.csv multi-hot
- Metric: macro-AUC across 46 classes with positives

## Individual augmentation results (partial — benchmark in progress)

Baseline: 0.9141

| Augmentation | AUC | Δ vs baseline |
|---|---:|---:|
| baseline (no aug) | 0.9141 | — |
| time_shift -0.5s | 0.9148 | +0.0007 |
| time_shift +0.5s | 0.9131 | -0.0010 |
| time_shift +1.0s | 0.9123 | -0.0018 |
| time_shift -1.0s | 0.9084 | -0.0057 |
| time_shift +2.5s | 0.9089 | -0.0052 |
| (more to come...) | | |

## Output-space "augmentation" (no audio re-run) — TESTED AND REJECTED

Confirmed these are no-ops on rank-power inputs (preserve within-class rankings):

| Technique | AUC delta on v4 stack |
|---|---:|
| Temperature averaging (3/5/7 temps) | +0.0000 |
| Rank-power averaging (3/5 powers) | +0.0000 |
| Multi-weight prior averaging | +0.0000 |
| Logit Gaussian noise + average | NEGATIVE (just noise) |

**Conclusion**: Output-space "TTA" doesn't add information. Real TTA needs audio re-runs.

## What got built

- `sub_v6_full_aug_bruce.py`: Kaggle kernel running Bruce + Perch pipeline with 7 augmentation paths (baseline, ±2.5s shift, ±3dB gain, rms_norm, rms_norm+shift). Averages logits across paths.
- `kaggle_kernels_v6/sub_sub_v6_full_aug_bruce/`: push-ready kernel directory.

Push command: `kaggle kernels push -p /tmp/kpush_v6/sub_sub_v6_full_aug_bruce/`

Expected standalone v6 LB: 0.78-0.82 (Bruce baseline ~0.755 + augmentation TTA boost).
Real value when ensembled with v4 patched exp019: small additional diversity gain.
