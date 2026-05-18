# V3 stacked post-processing — all findings benchmarked

**Date**: 2026-05-18

After investigating the May-18 LB drop (root cause: dead-hour crush + weight miscalibration, see INVESTIGATION_LB_DROP.md), I mined all 28 inference-time techniques from `meta_analysis/ROUND*.md` and benchmarked each individually + as a greedy stack on Bruce Wu's 739-row labeled OOF.

## Results summary

All scores below are **macro-AUC on exp019-like rank-power inputs derived from Bruce OOF**, with two scenarios:
- **A (clean)**: original Bruce OOF hour distribution (no dead hours)
- **B (LB-like)**: 20% of rows synthetically reassigned to dead hour 11 (mimics what the LB actually sees)

Baseline (no post-proc): **0.9304 (A) / 0.9304 (B)**.

### Individual technique impact on scenario B

| Technique | B-AUC | Δ vs baseline | Verdict |
|---|---:|---:|---|
| **hour_prior w=0.05** ⭐ | 0.9516 | **+0.0212** | Best single |
| site_hour_prior w=0.05 | 0.9501 | +0.0196 | Good |
| hour_prior w=0.1 | 0.9485 | +0.0181 | Good |
| site_shrinkage w=0.3 | 0.9461 | +0.0156 | Good |
| site_hour_prior w=0.1 | 0.9445 | +0.0141 | Decent |
| site_shrinkage w=0.1 | 0.9423 | +0.0118 | Decent |
| hour_prior w=0.2 | 0.9405 | +0.0101 | Decent |
| sonotype alias | 0.9307 | +0.0003 | Tiny |
| howler coupling | 0.9305 | +0.0001 | None |
| temperature scaling | 0.9304 | 0.0000 | No-op |
| two-pass SSM | 0.9304 | 0.0000 | No-op (12-window short) |
| file_confidence | 0.9301 | −0.0004 | Slightly bad |
| adaptive_delta | 0.9292 | −0.0012 | Slightly bad |
| calibration only | 0.9304 | 0.0000 | No-op on rank-power |
| weeping/chiasmo, broadcast, floor, chorus | 0.9304 | 0.0000 | Static lifts have no effect on AUC |

Why most patches are no-ops on rank-power inputs: when exp019 emits `[0.477, 0.555]` (rank-power-transformed), techniques like calibration (multiplicative on prob) or sonotype broadcasting (max with anchor) don't change the within-class ranking. Macro-AUC is invariant.

Hour-based patches DO help because they apply **different** shifts to rows in different hours — actually re-ordering rows of the same class.

### 2D sweep: hour_prior × site_shrinkage

```
              site_shrinkage:
                  0.00     0.05     0.10     0.20     0.30     0.50
hour_w=0.000   0.9304   0.9396   0.9423   0.9450   0.9461   0.9465
hour_w=0.025   0.9493   0.9554   0.9564   0.9570   0.9570   0.9565
hour_w=0.050   0.9516   0.9568   0.9581   0.9585◀  0.9580   0.9576  ← best row
hour_w=0.075   0.9508   0.9556   0.9566   0.9576   0.9574   0.9568
hour_w=0.100   0.9485   0.9537   0.9545   0.9558   0.9559   0.9554
hour_w=0.200   0.9405   0.9457   0.9477   0.9487   0.9491   0.9503
```

**Optimum**: `hour_prior_w=0.05, site_shrinkage_w=0.20` → **0.9585** (+0.0280 on scenario B).

Scenario A with same recipe: **0.9632** (+0.0328).

### Other patches don't stack on top

Starting from `(hour_w=0.05, site_w=0.2)`, adding any of {alias, howler, broadcast, floor, chorus, ssm, calibration, temperature, weeping} changes AUC by less than ±0.0002. They are all no-ops at this operating point.

**Why**: hour_prior + site_shrinkage already capture all the cross-row information available from our pseudo-cache priors. Within-class smoothing/broadcasting can't add signal that isn't there.

### Sharpening doesn't help

Pre-applying a temperature scale to exp019's logits (before priors) doesn't improve the stack. T=1.0 (no sharpening) is optimal. exp019's signal is already balanced.

### Rank-space blend is worse than logit-space shift

Pure rank-space blending (`α × rank(exp019) + (1-α) × rank(prior)`) caps at **0.9465** — far below the **0.9585** of logit-shift. Sticking with logit shift.

## Final recipe for tomorrow

```python
params = {
    "w_hour": 0.05,    # hour_prior logit shift, dead-hour filled with global mean
    "w_site": 0.20,    # site_shrinkage: (pred + 0.2 * site_prior) / 1.2
}
new_prob = apply_all(exp019_prob, row_ids, class_cols, priors_filled, params)
```

## Expected LB scores for the 5 v3 kernel variants

Halving simulated gains for the OOF→LB transfer-rate uncertainty:

| Slot | Variant | Patches | Simulated B | Halved gain | **Expected LB** |
|---:|---|---|---:|---:|---:|
| 1 | v3-1-control | none | 0.9304 (baseline) | 0 | **0.949** (sanity) |
| 2 | v3-2-stack | hour 0.05 + site 0.2 | **0.9585** (+0.028) | **+0.014** | **0.963** ⭐ |
| 3 | v3-3-hour-only | hour 0.05 | 0.9516 (+0.021) | +0.011 | 0.960 |
| 4 | v3-4-strong | hour 0.075 + site 0.3 | 0.9574 (+0.027) | +0.014 | 0.962 |
| 5 | v3-5-alias | alias + howler | 0.9307 (+0.000) | 0 | 0.949 |

**Best-case** (full transfer of simulation to LB): **0.977** with v3-2 stack.
**Median** (50% transfer): **0.963** with v3-2 stack.
**Worst-case** (priors don't generalize): **0.949** with v3-1 control as floor.

## Submission ordering (recommendation)

1. **v3-1 control first** — confirms baseline still hits 0.949 (sanity check before spending compute on patches).
2. **v3-2 stack** — the optimal recipe; biggest expected gain.
3. **v3-3 hour-only** — bracket: tests if site_shrinkage was the secret sauce or if hour alone is enough.
4. **v3-4 strong** — bracket the other side: tests if we under-weighted.
5. **v3-5 alias** — wildcard, should hit baseline; serves as another control for diversity.

## Files

- `postproc_v3.py` — full unified module with all 15 techniques as composable functions
- `benchmark_v3.py` — individual + greedy stack benchmark
- `sweep_v3.py` — 2D sweep and stacking
- `sharpen_v3.py` — sharpening + dynamic-range experiments
- `sub_v3_*.py` — 5 paste-as-final-cell kernel patches
- `kaggle_kernels_v3/` — kernel push directories ready for `kaggle kernels push`

Pushed-ready but **not yet pushed** — awaits user confirmation.
