# Investigation: why our 5 May-18 submissions underperformed

**Date**: 2026-05-18 03:30 UTC  
**Trigger**: LB scores for sub1/sub2/sub3/sub4 came in at 0.755-0.920 vs exp019 baseline 0.949.

## TL;DR

Two compounding bugs in the post-processing cell:

1. **Dead-hour crush** (PRIMARY): Hours 11-16 have zero pseudo-cache data because train_soundscapes are duty-cycled to night/dawn/dusk. Our code clipped missing priors to `EPS=1e-7`, so test rows at those hours got a `−48 logit` shift, randomizing predictions for ~20% of test files.

2. **Weight too aggressive on rank-power outputs** (SECONDARY): exp019 emits rank-power-transformed scores in `[0.477, 0.555]`. The full class-discrimination range is ~0.31 logits. Our `w=3.0` adds shifts of ±15 logits — ~125× larger than the underlying signal. Even with the prior values correct, the right weight for exp019 is ~25-60× smaller (w ∈ 0.05-0.2).

## Evidence

### Pseudo-cache hour coverage gap

```
Hour | Σ priors (234 classes) | n_zero
 0-10 |        6-10            |   0
11-16 |        0.00            | 234   ← all classes are 0
17-23 |        5-10            |   0
```

`train_soundscapes` recorder distribution: 80% of files are at hours 0-4 and 18-23 (night/dawn/dusk). Zero files at hours 11-16.

### OOF benchmark was blind to the bug

Bruce's 739-row labeled OOF: **zero rows at hours 11-16**. So our 0.9586 OOF benchmark never tested the dead-hour case. The score was real but didn't generalize because test contains hours we never simulated.

### Reproduction of LB drop in simulation

Simulated 20% dead-hour exposure in OOF + exp019-like rank-power inputs:

| Weight | Simulated AUC | Δ vs no-prior |
|---:|---:|---:|
| 0.05 | 0.9516 | +0.0212 ⭐ best |
| 0.1  | 0.9485 | +0.0181 |
| 0.2  | 0.9405 | +0.0101 |
| 0.3  | 0.9341 | +0.0037 |
| 0.5  | 0.9252 | −0.0053 |
| 1.0  | 0.9126 | −0.0179 |
| **3.0** | **0.8985** | **−0.0319** (matches sub1's actual −0.029) |

Simulation predicts the actual LB drop within 0.003. Direction reversal confirmed: at w=3.0 on rank-power outputs we expected +0.028 but got −0.029. **Off by 0.057 in absolute swing** — entirely explained by the two bugs above.

### sub2 (standalone Bruce) at 0.755 also explained

- Bruce alone macro-AUC ~0.85 on test-like data
- With buggy dead-hour shift at w=3.0 on 20% of rows: simulated 0.78
- Off by 0.026 from actual 0.755 (Bruce-alone baseline is weaker than Bruce-with-ensemble context)

## Fix design

```python
# Step 1: build "filled" prior table — global mean fills dead hours
hourly_sum = prior_df.sum(axis=1)
covered = hourly_sum[hourly_sum > 0].index
global_prior = prior_df.loc[covered].mean(axis=0)
for h in range(24):
    if h not in covered:
        prior_df.loc[h] = global_prior  # use global, not zero

# Step 2: SMALLER weight for rank-power inputs
HOUR_PRIOR_WEIGHT = 0.05  # was 3.0
```

## Tomorrow's 5-slot plan (corrected)

| Slot | Variant | Why |
|---:|---|---|
| 1 | exp019 + hour_prior **w=0.05** + dead-hour fix | Best simulated weight |
| 2 | exp019 UNTOUCHED | Control — confirms baseline ~0.949 |
| 3 | exp019 + hour_prior **w=0.1** + dead-hour fix | Bracket optimum from above |
| 4 | exp019 + sonotype_alias ONLY (no hour_prior) | Test if alias alone is neutral/+ |
| 5 | 0.5 × sub1_fixed + 0.5 × exp019 (rank blend) | Soft fallback that can't underperform exp019 |

Sub5 (which never submitted today) is currently "sub5 v2: exp019 + hour_prior w=3.0 + perch_calib + alias" — kernel exists at v2 but it has the SAME bugs, so even with a slot it would score similarly to sub1. We should re-push it with corrections before submitting tomorrow.

## Methodology lessons

1. **OOF distribution must match LB distribution** — Bruce's 739-row labeled set was site-S22-night-biased. Should have stratified by hour and site, or held out a non-S22 fold.
2. **Post-processing weights must match the input scale** — rank-power outputs need ~50× smaller weights than raw-logit outputs. The same `w=3.0` produced opposite-sign effects on Bruce vs exp019.
3. **Edge-case data coverage matters** — always check whether your data sources cover the full range of values you'll encounter (hour, site, etc.).
