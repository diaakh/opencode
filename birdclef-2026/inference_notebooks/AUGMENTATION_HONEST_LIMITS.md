# Augmentation findings — what's actually implementable at inference

## Summary

The user asked "what about augmentations and clipping?" — fair question. I had dismissed those as "training-only" and skipped them. Here's the honest breakdown of which findings can actually help at inference time on Kaggle:

## Bucket 1: Training-only (cannot do anything)

These require retraining the model — out of scope for the Kaggle 90-min submission budget:

- **Mixup / CutMix** — sample pairing during training
- **SpecAugment** — frequency/time masking on mel-spec during training
- **Dynamic-range / clipping augmentation** — soft-clip audio during training so model learns invariance
- **Codec re-encoding augmentation** — train on multiple codecs so model handles test codec robustly
- **Gain modification augmentation** — random gain at training time
- **SimCLR / SSL pretext** — pre-train on unlabeled audio
- **Background noise injection** — mix Pantanal noise into training samples
- **Year-weighted training**, **per-class smart crop**, **AWP**, etc.

None of these are implementable as a post-submission cell or a fresh kernel — they need access to retrain the underlying ensemble (exp019 has 7 models, none retrainable).

## Bucket 2: Inference TTA (technically possible, blocked by budget for exp019)

These re-run the model on perturbed inputs and average:

| Technique | Source | Estimated LB gain | Cost |
|---|---|---:|---|
| Time-shift TTA (±2.5s) | BC2025 1st place | **+0.012** | 3x inference time |
| Frame-shift TTA (overlap_average_max_delta) | Nikita | +0.005-0.010 | Free if SED model already runs |
| Gain TTA (±3dB) | — | +0.002-0.005 | 2-3x inference time |
| Codec re-encoding (re-encode test audio at 72kbps before predict) | ROUND13 | +0.003-0.008 | Modest |

**The catch**: exp019 already runs at ~85 min on Kaggle. We have ~5 min budget left after the model. We CANNOT add TTA to exp019 without rebuilding the notebook from scratch.

**What we CAN do**: add TTA to the FAST sub2 (standalone Bruce) path, which only takes ~5-10 min — leaves 80+ min headroom for 3-shift TTA.

## Bucket 3: Audio normalization at inference (possible for sub2)

- **RMS normalization** to match train domain (~0.026 RMS for 2025 night)
- **Soft clipping/compression** to handle level outliers
- **High-pass filter** to remove low-freq noise

These are cheap audio pre-processing steps. Marginal effect (≤0.003 LB) but free if added to sub2.

## What I built

`sub_v5_tta_bruce.py` — standalone Bruce + 3-shift TTA (±2.5s):

- 3 inference passes per file with different time alignments
- Averages predictions across shifts before applying hour_prior
- Uses labeled hour prior (v4 hybrid)
- Runtime budget: ~15-30 min for 600 files (3x of plain sub2)

Expected sub_v5 LB: somewhere between sub2_v4 (which got 0.755 with bug-fixed prior) and the literature +0.012 anchor. Realistically: **0.78-0.80** as a standalone (still well below exp019's 0.949).

## Honest gain assessment

Adding sub_v5 (TTA Bruce) as an ENSEMBLE PARTNER with v4 patched exp019 might add +0.003-0.008 LB via diversity. But it requires:
- One submission slot used on TTA Bruce alone
- A SEPARATE blending step that combines exp019 + TTA-Bruce in rank space — needs another kernel

Given finite slots, **the marginal gain from TTA-Bruce ensembling is likely smaller than the gain from sticking with v4-2-optimal** (which is the cleanest, biggest-expected-gain single approach).

## Recommendation

For tomorrow's 5 slots, prioritize:
1. **v4-2-optimal** (the labeled-hybrid prior stack) — biggest expected single gain, low risk
2. **v4-1-control** (untouched exp019) — sanity baseline
3. **v4-4-combined** (triple stack) — bracket
4. **v5-tta-bruce** (this new kernel) — diversity test; weakest single but useful for ensemble later
5. (TBD: blend of v4-2 + v5-tta in a separate kernel? requires more setup)

The honest TL;DR: **I cannot meaningfully add training-time augmentation findings to exp019.** The next big gain would come from RETRAINING models with codec-matched data + clipping augmentation + train_audio inclusion — those need GPU and days of training. None are 90-min Kaggle-submission feasible.
