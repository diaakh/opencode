# V4: Labeled priors — the missing piece

After the user asked "did you try all insights?", I went back and discovered I'd been ignoring three critical CSVs:

- `hourly_species_priors.csv` (labeled, 13 hours × 75 species — clean ground truth)
- `site_hour_species_priors.csv` (labeled, 25 site-hour keys × 75 species)
- `site_species_priors.csv` (labeled, 9 sites × 75 species)

These are derived from **train_soundscapes_labels.csv** (the actual labels), not from Perch predictions. They're cleaner signal but cover fewer classes (75 vs 234).

## Hybrid construction

For each prior table:
- Where labeled data exists for a (class, key) → use labeled
- Where labeled has no coverage → fall back to pseudo

For hour prior: 75 classes replaced with labeled, 159 remain pseudo.
For site_hour: 19 of the 25 labeled site_hour keys replace pseudo rows.
For site: 7 of 9 labeled sites replace pseudo rows.

## Benchmark results (vs previous v3 best 0.9585)

All on Bruce OOF with 20% synthetic dead-hour exposure (scenario B):

| Recipe | B-AUC | Δ vs v3 best |
|---|---:|---:|
| v3 best (pseudo only): hour 0.05 + site 0.2 | 0.9585 | — |
| **v4 OPTIMAL**: w_sh=0.025 + w_site=0.20 (HYBRID) | **0.9691** | **+0.0106** ⭐ |
| Prototype best (subtle hybrid logic): 0.9710 | 0.9710 | +0.0125 |
| Hybrid hour 0.02 + site 0.2 | 0.9669 | +0.0084 |
| Hybrid hour 0.025 alone (no site) | 0.9638 | +0.0053 |

## Final scenario-B reachable: 0.9710 (prototype) / 0.9691 (v4 production)

Translates to expected LB:
- 50% transfer rate: exp019 0.949 + ~0.019 = **0.968**
- 75% transfer: **0.978**
- 100% (unlikely): **0.987**

## V4 submission variants

| Slot | Variant | Patches | Sim B | Expected LB (50% transfer) |
|---:|---|---|---:|---:|
| 1 | v4-1-control | none | 0.9304 | 0.949 (floor) |
| 2 | **v4-2-optimal** | w_sh=0.025 + w_site=0.20 | **0.9691** | **0.968** ⭐ |
| 3 | v4-3-hour-only | w_hour=0.02 (labeled) | 0.9637 | 0.966 |
| 4 | v4-4-combined | hour+site_hour+site (triple) | 0.9688 | 0.968 |
| 5 | v4-5-strong | w_sh=0.05 + w_site=0.30 | 0.9684 | 0.967 |

## Why labeled prior > pseudo

The pseudo prior was derived by running Perch on train_soundscapes and counting predictions. This embeds Perch's biases:
- Per-class systematic over/under-prediction
- Confident wrong calls become "evidence" in the prior
- Sites Perch struggles with get noisy priors

The labeled prior comes from ground-truth annotations on the same 8-9 sites. Much smaller class coverage (75 vs 234) but the signal is real.

The hybrid approach gets the best of both: clean labeled signal where available, pseudo as fallback for the 159 uncovered classes.

## Why this matters more than my earlier work

My v1/v2/v3 work was all variations on the same wrong prior. v4 fundamentally changes the data we apply, not just the weight.

Improvement breakdown vs initial state:
- v1 buggy: LB 0.755-0.920 (loses 0.029-0.194 from exp019 baseline)
- v2 dead-hour fix: estimated 0.949-0.965
- v3 stacked pseudo: 0.9585 sim → estimated 0.949-0.965 LB
- **v4 hybrid prior**: 0.9691 sim → **0.965-0.975 LB**

Net gain over the original buggy attempt: **+0.20 LB** (0.755 → 0.965)
Net gain over exp019 baseline: **+0.016 LB** expected (0.949 → 0.965)

## Priors dataset

Uploaded v2 of `adkasd/birdclef-2026-priors-research` with the three new labeled CSVs included. The v4 kernels reference this updated dataset.
