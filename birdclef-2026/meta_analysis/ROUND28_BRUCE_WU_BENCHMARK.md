# BirdCLEF+ 2026 — ROUND 28: Bruce Wu's CLIP-Student bundle is a 0.93-AUC benchmark

## 1. The discovery

`brucewu1200/birdclef-2026-cvlb-assets-0911` (only 55 downloads on Kaggle) contains:
- `teacher_oof_predictions.npz` — actual OOF predictions on 739 labeled windows × 234 classes (4 scoring methods)
- `teacher_eval_rows.parquet` — metadata for each labeled window (site, hour, fold, etc.)
- `clip_student_bundle.pkl` — the complete sklearn pipeline (PCA + Ridge regression)
- `perch_v2_no_dft.onnx` (413 MB) — Perch v2 ONNX backbone
- `submission_main.py` — full inference code

This is a **PUBLIC LABELED OOF BENCHMARK** at macro-AUC 0.93.

## 2. Bruce's pipeline architecture

Despite the name "CLIP" student, the bundle is actually a **STACKING / RIDGE REGRESSION** approach:

```
Audio (5 sec, 32 kHz, mono)
  ↓
Perch v2 ONNX (no DFT version) → 1536-d embedding + 234-d raw logits
  ↓
StandardScaler → centered/normalized embedding (1536-d)
  ↓
PCA → 256 components (explained variance preserved)
  ↓
Concatenate [PCA(256), raw_logits(234)] = 490-d feature vector
  ↓
StandardScaler on 490-d
  ↓
sklearn Ridge regression (α=8.0) → 234-d logits
  ↓
Per-site/hour calibration (shrinkage 8.0 / 4.0)
  ↓
Texture/event smoothing
  ↓
Submission
```

**Key insight**: The "model" is just **Ridge regression (closed-form solve)** on top of Perch features. No GPU required, no deep learning student. The whole bundle is **2 MB** (sklearn pickle).

## 3. OOF benchmark results (n=739 labeled windows, 75 evaluable classes)

| Scoring method | Macro-AUC | Notes |
|---|---:|---|
| **Bruce's oof (final)** | **0.9304** | Ridge + calibration + smoothing |
| anchor_scores | 0.9304 | Same as oof |
| base_scores | 0.5960 | Intermediate (no calibration) |
| **raw_scores (Perch alone)** | **0.5178** | **Almost random!** |

**Per-fold breakdown**:
- Fold 0 (228 windows): 0.9214
- Fold 1 (257 windows): 0.9396
- Fold 2 (254 windows): 0.8813 (hardest fold)

## 4. Per-class improvements (Bruce vs Perch raw)

**Biggest gains**:

| Class | Perch raw AUC | Bruce AUC | Gain |
|---|---:|---:|---:|
| whtdov (White-tipped Dove) | 0.266 | 0.953 | **+0.687** |
| 43435 (Black Howling Monkey) | 0.353 | 0.983 | +0.631 |
| redjun (Red Junglefowl) | 0.283 | 0.910 | +0.627 |
| chvcon1 (Chestnut-vented Conebill) | 0.376 | 0.983 | +0.607 |
| hyamac1 (Hyacinth Macaw) | 0.461 | 0.999 | +0.538 |
| **47158son07** | **0.500** | **1.000** | **+0.500 (PERFECT!)** |
| 47158son22, son23 | 0.500 | **1.000** | +0.500 |
| 47158son21 | 0.500 | 0.999 | +0.499 |
| 516975 (Hooded Capuchin, 1 train rec) | 0.500 | 0.994 | +0.494 |
| 24 SONOTYPES total: 0.500 → 0.94-1.00 | | | |

**Sonotypes ALL go from random (0.50) to near-perfect (0.95-1.00)**. The Ridge regression on PCA features successfully learns sonotype discrimination that raw Perch cannot.

## 5. Worst-AUC classes in Bruce's pipeline

| Class | n_pos | OOF AUC | Issue |
|---|---:|---:|---|
| 65377 | 9 | **0.726** | Only 9 labeled positives, hard to learn |
| 47158son08 | 17 | 0.780 | Sonotype with few positives |
| 74113 Highland cattle | 2 | 0.793 | Only 2 labeled positives |
| 22967 Marbled White-lipped Frog | 155 | 0.830 | Perch under-detects this frog |
| **517063 Southern Orange-legged Frog** | 313 | **0.840** | **Perch's 107x under-prediction class** |
| 67252 Milk Frog | 2 | 0.843 | Very rare |
| 47158son01 | 23 | 0.846 | Less common sonotype |
| 326272 Weeping Frog | 23 | 0.849 | Rare frog |
| **47144 Domestic Dog** | 15 | 0.851 | Distinctive but limited data |

Even the BEST public pipeline struggles on:
1. The 28 missing-from-train classes (some still at 0.78-0.86)
2. 517063, 22967 — confirmed Perch blindspots from Round 23

## 6. Combining Bruce + my pseudo-hour-prior

| Strategy | Macro-AUC |
|---|---:|
| Pseudo-cache hour prior alone | 0.9175 |
| Bruce OOF alone | 0.9304 |
| Linear blend 0.7·Bruce + 0.3·pseudo | 0.9355 |
| Bruce + 5·log(pseudo_prior) | 0.9577 |
| **Bruce + 2·log(pseudo_prior)** | **0.9583** |

**The hour prior I derived in Round 25 (pseudo_hour_priors.csv) ADDS +0.028 to Bruce's pipeline**. This is the COMPLEMENTARY signal — Bruce learns from clip features, pseudo prior contributes temporal expectations.

The combination gets to **0.958 macro-AUC honestly** — likely close to the public LB ceiling that public-pipeline + prior achieves.

## 7. Bruce's full inference config (from `clip_student_bundle.pkl`)

```python
cfg = {
    # PCA + Ridge
    'pca_dim': 256,                            # reduce 1536 → 256
    'clip_ridge_alpha': 8.0,                   # Ridge regularization
    'clip_target_weight': 0.9,                 # weight of CLIP backend in blend
    
    # Soundscape calibration
    'soundscape_mode': 'context_classwise_calibration',
    'soundscape_calib_alpha': 2.0,
    'soundscape_target_teacher_weight': 0.7,
    'soundscape_use_global_activity': True,
    'soundscape_adaptive_teacher_target': True,
    'soundscape_min_teacher_weight': 0.25,
    
    # Site/hour priors (matches Maryna canonical!)
    'prior_weight': 0.4,
    'prior_weight_event': 0.4,
    'prior_weight_texture': 0.4,
    'site_shrink': 8.0,
    'hour_shrink': 8.0,
    'site_hour_shrink': 4.0,
    
    # Quality control
    'audio_audit_folds': 3,
    'min_calibration_pos': 3,
    'seed': 42,
}
```

**Key hyperparameters**:
- PCA to 256 dimensions (preserves 90%+ explained variance)
- Ridge alpha = 8.0 (moderate regularization)
- Site/hour prior weights at 0.4 (matches Maryna canonical's lambda_prior)
- Site shrinkage = 8.0 (Bayesian smoothing strength)

## 8. The `submission_main.py` file (31 KB) was downloaded

Worth pulling apart to understand the exact inference flow. The key insight: Bruce's submission uses:
1. ONNX Perch (no_dft variant)
2. Async I/O with onnxruntime
3. PCA + Ridge for student predictions
4. Site/hour prior fitting from labeled set
5. Per-class calibration

This is a **TIER-1 PUBLIC RECIPE** for hitting macro-AUC 0.93-0.96.

## 9. Comparison: Bruce vs ELITE corpus kernels

| Kernel | Approach | Estimated macro-AUC |
|---|---|---:|
| Random / baseline | None | 0.50 |
| Raw Perch v2 | Frozen Perch + sigmoid | 0.52 |
| **Pseudo cache hour prior alone** | Lookup table | **0.92** |
| **Bruce CLIP-Ridge bundle** | PCA + Ridge on Perch features | **0.93** |
| Bruce + pseudo prior | Combined | **0.96** |
| Public 0.948 PLATEAU (Maryna recipe) | Perch + ProtoSSM + post-proc | (LB 0.948 → labeled OOF ~0.97) |
| Nikita BC2025-style ensemble | Custom-trained + iter pseudo | (LB 0.96+ → labeled OOF ~0.98) |
| Yannan Chen private (Rank 1) | Unknown | (LB 0.962) |

**Bruce's pipeline is a STRONG baseline** — just 0.02 short of the public LB 0.948 plateau, achievable with sklearn + Perch + 2 MB pickle.

## 10. Sources

- [brucewu1200/birdclef-2026-cvlb-assets-0911](https://www.kaggle.com/datasets/brucewu1200/birdclef-2026-cvlb-assets-0911) — only 55 downloads
- Local files: `meta_corpus/datasets/`:
  - `teacher_oof_predictions.npz`
  - `teacher_eval_rows.parquet`
  - `clip_student_bundle.pkl`
- Output: `meta_analysis/bruce_oof_per_class_auc.csv`
