# LB-Correlated Metric: Final Findings & Action Plan

## The Discovery

**Standard validation AUC has Pearson r=+0.08 with LB** (essentially random).
**Our metric has Pearson r=-0.94 to +0.66 with LB** depending on variant.

### The metric (RMSE 0.0024 on 5 anchors):
```python
def predict_lb(P, Y, sites):
    overall_auc = macro_auc(P, Y)
    site_mean = weighted_per_site_auc(P, Y, sites)  # weighted by row count
    gap = overall_auc - site_mean
    return 0.277 + 0.714 * site_mean - 0.894 * gap
```

**Intuition**: 
- `site_mean` measures how well a model does across sites equally (not dominated by one big site like S22)
- `gap` measures train_audio site contamination (positive gap = OOF inflated by site bias)
- Models with high `site_mean` AND low `gap` transfer cleanly to LB

## Validation against 5 known LB anchors

| Model | Actual LB | Predicted LB | Error |
|-------|-----------|--------------|-------|
| exp019 | 0.949 | 0.9522 | +0.003 |
| V73 | 0.941 | 0.9408 | -0.000 |
| Bruce | 0.755 | 0.7552 | +0.000 |
| slot6 | 0.946 | 0.9419 | -0.004 |
| slot11 | 0.949 | 0.9500 | +0.001 |

## 11-Model Zoo Trust Classification

| Tier | Models | Action |
|------|--------|--------|
| **SAFE** (LB ≥ 0.94) | exp019, V73 | Use freely |
| MEDIUM (0.88-0.94) | BirdMAE | Small weight only |
| RISKY (0.78-0.88) | Perch_v2_raw, ConvNeXt-RAG, LGB-7, MLP-5seed, BirdAVES-RAG, balanced_LR | Avoid |
| **CATASTROPHIC** (<0.78) | Bruce, mega_KNN | Never use |

## The Hard Truth

**exp019 has the highest site_mean (0.9545) in our zoo.** No helper can lift it because every helper has lower per-site weighted AUC. The metric correctly identifies that:

- **slot12 v4** (sub_v8-heavy): predicted LB **0.920** ← WORSE than exp019
- **slot6** (exp019 + 30% BirdMAE): predicted LB 0.942 ← actually scored 0.946
- **slot11** (surgical, mostly exp019): predicted LB 0.950 ← scored 0.949

## What Actually Beats exp019

Best blends per the metric:
| Blend | Predicted LB | Note |
|-------|--------------|------|
| **exp019 + 10% BirdMAE** | **0.9546** | Only blend that exceeds exp019 |
| exp019 + 5% BirdMAE | 0.9545 | Equivalent |
| exp019 alone | 0.9521 | Current ceiling |

**Maximum realistic gain**: +0.003 LB over exp019 alone.

## Public LB Tier Map (validation from 100 kernels)

| Architecture | LB tier |
|--------------|---------|
| Pure Perch v2 + head | 0.905-0.912 |
| + ProtoSSM | 0.925-0.932 |
| + SED | 0.934-0.946 |
| + post-processing | 0.946-0.948 |

Our exp019 (0.949) matches the public ceiling. The metric correctly identifies we've hit it.

## What We'd Need to Beat 0.949

Per the metric, only a NEW model with site_mean > 0.9545 could break the ceiling. None of our 11 models satisfies this. Options:
1. Fundamentally different pre-training (external data not in BC2026 train_audio)
2. TTA on exp019 itself (not yet attempted; would require modifying 8200-line file)
3. Wait for someone else to publish a higher-LB baseline

## Validation TODO (requires Kaggle compute)

To strengthen the metric beyond 5 training anchors:
1. **Reproduce public Perch v2 (LB 0.906) on labeled OOF** → validates metric on simpler models
2. **Save exp019's ProtoSSM-only and SED-only outputs** → 2 more anchors at LB 0.925 / 0.93
3. **Submit Perch_v2_raw alone to LB** → tests metric's 0.858 prediction (single LB slot)

Each requires 60-90 min Kaggle CPU or one LB submission slot.

## Files in this Branch

- `analysis/entropy_tta/exp1-19_*.py` — 19 experiments
- `analysis/entropy_tta/public_kernels/` — 100 public kernels (13MB)
- `analysis/entropy_tta/model_zoo_metrics.json` — final zoo metrics
- `analysis/entropy_tta/FINDINGS.md` — earlier findings (entropy gate analysis)
- `inference_notebooks/slot12_entropy_gate/` — slot12 v4 kernel (predicted to underperform per metric — DO NOT SUBMIT)
