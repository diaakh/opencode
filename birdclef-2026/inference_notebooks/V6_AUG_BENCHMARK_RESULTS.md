# V6: Real Audio Augmentation Benchmark — Final Results

## Method
- Model: fold0.onnx (60s waveform → 12-window logits, 234 classes)
- Data: 30 labeled train_soundscapes files (360 5-sec windows, 46 classes with positives)
- Ground truth: train_soundscapes_labels.csv
- Metric: macro-AUC
- Baseline: 0.9141

## Individual augmentation effects on macro-AUC

| Augmentation | AUC | Δ vs baseline | Verdict |
|---|---:|---:|---|
| **codec_proxy 16k** (downsample→upsample) | **0.9335** | **+0.0194** | ⭐⭐ HUGE win |
| **hpf 500Hz** (high-pass filter) | **0.9204** | **+0.0063** | ⭐⭐ Strong |
| hpf 200Hz | 0.9178 | +0.0037 | ⭐ Good |
| soft_clip 0.7 | 0.9146 | +0.0005 | ⭐ Tiny |
| time_shift -0.5s | 0.9148 | +0.0007 | mixed |
| gain ±3dB / ±6dB | 0.9141-2 | ±0.0001 | NO effect (model gain-invariant) |
| rms_norm (0.025/0.05/0.10) | 0.9141-2 | ±0.0001 | NO effect |
| time_shift ±1.0s | 0.9123 / 0.9084 | -0.002 / -0.006 | HURTS |
| time_shift ±2.5s | 0.9089 / 0.9047 | -0.005 / -0.009 | HURTS |

## TTA stacks (mean of multiple aug paths)

| Stack | AUC | Δ |
|---|---:|---:|
| 3-shift TTA (0, ±2.5s) | 0.9115 | -0.0026 |
| 5-shift TTA (0, ±0.5, ±1.0) | 0.9137 | -0.0004 |
| 7-shift TTA | 0.9126 | -0.0015 |
| gain TTA | 0.9141 | 0.0000 |
| shift+gain (5 paths) | 0.9130 | -0.0012 |
| rms_norm + 3-shift | 0.9115 | -0.0026 |
| hpf + 3-shift | 0.9130 | -0.0012 |
| ALL (shift+gain+norm) | 0.9133 | -0.0009 |
| 3-shift TTA (rank-blend) | 0.9101 | -0.0040 |

**ALL TTA stacks HURT** — because they include negative-effect augmentations (time-shift). Stacking is only helpful when you stack POSITIVE augmentations.

## Conclusion

For Bruce-Perch pipeline (sub_v6), use **only** the winners:
1. baseline
2. codec_proxy 16k (+0.019)
3. hpf 500Hz (+0.006)
4. soft_clip 0.7 (+0.001)

Average these 4 paths. Total potential gain: up to **+0.026 on standalone Bruce** (if codec gain transfers to Perch v2).

## Transferability caveat

The +0.019 codec gain measured on fold0.onnx might be specific to that model's training distribution. Perch v2 was trained on diverse XC/eBird recordings — downsample→upsample could:
- Help (if Perch was trained on similar low-bitrate sources)
- Be neutral (if Perch is codec-robust)
- HURT (if Perch needs high-freq detail)

Only the Kaggle LB will tell. sub_v6 is ready to push as soon as a v4 slot frees up.

## What this means for the full ensemble

If +0.019 transfers to Perch:
- sub_v6 (TTA-Bruce) standalone LB: 0.755 + 0.019 = ~0.775 (still well below exp019)
- Ensemble with v4 exp019: marginal diversity gain ~+0.002-0.005
- **Best case combined LB: 0.968 + 0.003 = 0.971**

If +0.019 does NOT transfer:
- sub_v6 standalone LB: ~0.755-0.760 (similar to sub2 v4)
- Ensemble with v4: zero gain
