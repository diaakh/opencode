# Metric Reality Check: Public OOF Data Revealed Overfit

## Discovery from public kernel outputs

Downloaded 15 high-LB public kernels' outputs and found 8 contained OOF
predictions on labeled train_soundscapes:

| Kernel | LB | OOF data |
|--------|-----|----------|
| safar1 (LB 0.948) | 0.948 | Perch v2 raw scores on 708 rows |
| mtoshidesu (LB 0.947) | 0.947 | Perch v2 raw scores |
| youssefmo (LB 0.948) | 0.948 | Perch v2 raw scores |
| mattiaangeli (LB 0.943) | 0.943 | Perch v2 raw scores |
| afr1ste (LB 0.946) | 0.946 | Perch v2 raw scores |
| needless090 (LB 0.934) | 0.934 | oof_base + oof_prior |
| koushikrudra (LB 0.928) | 0.928 | oof_base |

## What this revealed

The metric (LB ≈ 0.277 + 0.714·site_mean − 0.894·gap) **fits internal anchors with RMSE 0.002** but **fails on public OOF predictions**:

| Source | overall_auc | site_mean | gap | predicted LB | actual LB | error |
|--------|-------------|-----------|-----|--------------|-----------|-------|
| safar1 Perch scores | 0.799 | 0.769 | +0.030 | 0.798 | 0.948 | **-0.150** |
| needless090 oof_base | 0.771 | 0.561 | +0.210 | 0.490 | 0.934 | **-0.444** |
| koushikrudra oof_base | 0.770 | 0.549 | +0.221 | 0.472 | 0.928 | **-0.457** |

## Why it broke

1. **safar1's "Perch v2 scores"** is just the BASE model output before ProtoSSM/SED. The kernel's LB 0.948 is for the full pipeline. So the score 0.948 attribution is wrong for the OOF predictions.

2. **Different implementations have different OOF behavior**:
   - Our P_perch20: macro-AUC 0.874 on 41 active classes
   - safar1 scores: macro-AUC 0.799 on same 41 classes
   - Rank correlation between our P_perch20 and safar1's scores: only 0.664
   
   Same Perch v2 model, different inference pipelines, different OOF results.

3. **Refit with 13 anchors** gives RMSE 0.050 (max error 0.17). The original 5-anchor RMSE 0.002 was massively overfit.

## Honest revised conclusions

**The metric DOES work for**:
- Ranking models WITHIN our own pipeline (Bruce << exp019 << safe ensemble)
- Identifying catastrophic risks (mega_KNN site_std 0.186 → very low LB)
- Confirming exp019 is at the ceiling of our zoo

**The metric DOES NOT reliably do**:
- Predict absolute LB for our blends to 0.001 precision
- Generalize across different Perch v2 inference pipelines
- Compare across teams/implementations

## What this means for next steps

For our own ensemble decisions:
- Trust the **rank order** the metric gives
- Trust the **trust tier** classification (SAFE / MEDIUM / RISKY / CATASTROPHIC)
- Distrust **absolute LB predictions** to better than ±0.05

For validating new blends pre-submission:
- Compute metric using OUR pipeline's OOF predictions
- A NEW blend with site_mean above exp019's 0.9545 is a positive signal
- A NEW blend below 0.85 is likely catastrophic
- Mid-range (0.85-0.95) needs empirical LB testing

## True ranking (refit on all 13 anchors):

LB ≈ 0.9815 − 0.0582·site_mean − 0.1431·gap (RMSE 0.050)

This fit has nearly-flat slope on site_mean — meaning site_mean alone doesn't strongly distinguish LB tiers across DIFFERENT implementations. The original "site_mean works!" finding was an artifact of testing only our internal anchors which share the same inference pipeline.

## Take-away

The metric is a **rank-order classifier** for our zoo, not a precise LB predictor. Use it to triage candidates, but actual LB still requires a submission to know.
