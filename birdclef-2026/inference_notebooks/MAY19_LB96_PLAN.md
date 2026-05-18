# May 19 submission plan — push LB to 96+

**Goal:** LB macro-AUC ≥ 0.96
**Baseline:** strongest public exp019 LB ~0.949
**Headroom needed:** +0.011 minimum

## The thesis

Public exp019 already gets us to 0.949. Our local research adds:
1. **combined_hour_prior** (pseudo + iNat hybrid) — dense across all 24 hours
2. **calibrated weight** based on input dynamic range (auto-scaled)
3. **dead-hour fix** baked into the prior (iNat fills hours 11-16)
4. **sonotype aliases** (free patch — broadcast paired call types)

The May 18 submissions used `w=3.0` on exp019's rank-power outputs ([0.477, 0.555])
which **crushed** the LB (sub1 = 0.92 vs exp019 = 0.949). The fix is `w=0.5` on
that scale per labeled-OOF simulation with the **combined** prior (the
previous `w=0.05` finding assumed the buggy dead-hour-zero prior).

## Slot allocation (in submit order — first is safest)

| Slot | Notebook | Patches | Expected LB |
|---:|---|---|---:|
| 1 | exp019 fork + sub_v7 cell | hour_prior(auto-w) + sonotype_aliases | 0.96–0.97 ⭐ |
| 2 | exp019 fork + sub_v7 cell | hour_prior only (no aliases) | 0.96 (control) |
| 3 | exp019 vanilla | none | 0.949 (baseline anchor) |
| 4 | exp019 + sub_v7 + perch_calib | full stack | 0.96–0.97 (risky calib) |
| 5 | sub2_bruce_standalone + sub_v7 cell | Bruce-based + combined prior | 0.85–0.92 (diversity test) |

**Why slot 1 first:** isolates the single highest-confidence move (combined
prior at auto-scaled weight). If it beats baseline, slots 2/4 build on it.

**Why slot 3 control:** we never confirmed our team's actual exp019 baseline
from this account — assumed 0.949 from CONTEXT.md. Sub3 anchors the gap.

**Why slot 5:** if Bruce + our extras at honest weights gets close to exp019,
that proves the path is robust. If it bombs, exp019 IS the right base.

## sub_v7 logic

```python
def auto_weight(prob):
    span = np.percentile(prob, 99) - np.percentile(prob, 1)
    if span < 0.05: return 0.1     # very compressed
    if span < 0.15: return 0.5     # rank-power compressed (exp019)
    if span < 0.5: return 1.5      # mid
    return 2.5                     # wide / raw probability
```

Calibrated on labeled OOF (combined prior, no dead-hour clipping):

| Span class | Sample | Optimal w | OOF AUC | Δ |
|---|---|---:|---:|---:|
| Very compressed (0.02) | extreme rank-power | 0.1 | 0.9505 | +0.092 |
| Rank-power (0.08) | exp019 | 0.5 | 0.9506 | +0.064 |
| Mid (0.87) | rescaled Bruce | 2.5 | 0.9528 | +0.094 |
| Wide (0.96) | raw Bruce | 3.0 | 0.9534 | +0.095 |

## Risks

1. **OOF distribution doesn't match LB.** Labeled OOF is 100% night hours
   (S22-night biased). Test set hour mix is unknown. The iNat fill for hours
   11-16 is plausible but unvalidated on LB.

2. **exp019's compressed scale may distort more than simulation predicts.**
   The May 18 LB had a 0.057 absolute swing vs simulation. Auto-weight at
   `w=0.5` is mid-range of optimum (sim says 0.5-1.0 plateau); should be safe.

3. **`sonotype_aliases` patch on test could over-broadcast.** It assumes the
   sono15/16 and sono22/23 pairs share recording labels. On labeled OOF
   neutral; on LB could be +0.001 to -0.003.

## Pre-submission checklist

- [ ] Re-upload `birdclef-2026-priors-research` dataset with `combined_hour_prior.csv`
  (was added today; old version only has `pseudo_hour_priors.csv`).
- [ ] Confirm exp019 notebook fork still runs successfully on Kaggle.
- [ ] Verify the sub_v7 cell auto-detects the right weight by printing `span`
  before submitting (the first run output will show `AUTO-SCALED weight: w=X`).
- [ ] If span detection lands at `w=0.1` or `w=2.5` (extremes) on exp019,
  manually override to `w=0.5` — that's where the simulation peak is.

## How to submit

```
1. Fork exp019 notebook on Kaggle ("Copy & Edit")
2. Add Data → adkasd/birdclef-2026-priors-research (after re-upload)
3. Scroll to last cell, click "+ Add code cell" below it
4. Paste the entire contents of sub_v7_combined_prior_autoscale.py
5. Save & Run All
6. Verify the auto-weight printout matches table above
7. Submit to Competition
```

## What 96 LB unlocks

A 0.96+ LB score puts us in the top 20-30 of public LB (depending on the day).
The OOF→LB gap we've seen is ~0.02; getting LB 0.96 means OOF 0.98 territory
is plausible on a less-biased validation set.

Further moves AFTER hitting 0.96:
- Run the standalone Bruce+megaKNN+Probe+Perch+prior recipe (our 0.9613 OOF)
  as an independent base, then blend with exp019 at 50/50 rank-mean.
- Push the 93MB knn_index.pkl as a private Kaggle dataset to enable megaKNN
  at inference time.
- Train a frog-specialist GPU model on Babych BC2025 extra-data (17k frog
  observations) to break the Amphibia ceiling.
