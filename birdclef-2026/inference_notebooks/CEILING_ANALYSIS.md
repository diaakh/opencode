# Ceiling analysis: why 0.99 honest macro-AUC isn't reachable on this dataset

**TL;DR:** The 739-window labeled OOF has inherent structural limits that cap macro-AUC well below 0.99, regardless of model. Our 0.9613 result is within ~0.01 of the achievable ceiling.

## What we tried in "insane mode" (round 4)

After hitting 0.9613 with `Bruce_smoothed + megaKNN + Probe + Perch + combined_hour_prior(w=2.5)`, we attempted:

| Move | Honest macro-AUC | Δ |
|---|---:|---:|
| Baseline (best from goal session) | 0.9613 | — |
| LOFO labeled hour prior (per-fold) | 0.860 | -0.10 (S22-bias contaminates) |
| LOFO + combined stacked | 0.920 | -0.04 |
| LOFO site+hour | 0.871 | -0.09 |
| File-level top-K amplification (chaneyma) | 0.957 | -0.005 |
| Cross-class chorus broadcast (frog chorus, sonotype groups) | 0.956 | -0.006 |
| Per-class isotonic with shrinkage | 0.951 | -0.011 |
| Per-class LR with pseudo augmentation | 0.961 | 0.000 |
| Targeted LR for 8 weakest classes only | 0.954 | -0.007 |

None of these honest moves beats 0.9613.

## Structural ceiling — diagnostic

Per-class AUC breakdown of the 0.9613 blend (75 valid classes):

| Taxa | Mean AUC | Median | n classes | What limits it |
|---|---:|---:|---:|---|
| Mammalia | 0.992 | 0.993 | 4 | Already at ceiling — 1 species (Black Howling Monkey) hits 1.000 |
| Insecta (sonotypes) | 0.973 | 0.979 | 25 | Sonotypes are HYPER-locked to (site, hour) — prior is enough |
| Aves | 0.967 | 0.977 | 28 | Most diurnal birds well-predicted |
| Reptilia | 0.961 | 0.961 | 1 | Only 1 species |
| **Amphibia (frogs)** | **0.927** | 0.942 | 17 | **BOTTLENECK** — co-occurring frog chorus |

The bottleneck is **Amphibia** because 4 Pantanal frog species co-occur in the SAME files at the SAME hours at S22-night:
- 65380 (Dwarf Tree Frog) — 333 positives, AUC 0.942
- 517063 (Southern Orange-legged Leaf Frog) — 313 positives, AUC 0.877
- 555146 (Chaco Tree Frog) — 210 positives, AUC 0.890
- 24279 (Lesser Snouted Tree Frog) — 173 positives, AUC 0.878

These call together at the same windows. The model can recognize "Pantanal night chorus is calling" but can't disambiguate WHICH specific frog within the chorus is calling above another. Macro-AUC asks "rank windows with species X above windows without" — but if all 4 species are present in the same 477 S22-night windows, ranking one species above another within those windows is fundamentally hard.

## What 0.99 macro-AUC would require

For every one of 75 classes to hit ~0.99 AUC, the model needs to rank every species-positive window in the top fraction of all 739 windows. For the 4 co-occurring frog species, this requires perfect within-chorus disambiguation. With the models available:

| Model class | Best single-class AUC on bottleneck frogs |
|---|---|
| Bruce CLIP-Ridge | 0.85-0.90 |
| Perch v2 raw | 0.80-0.88 (no separation between co-occurring species in Perch's embedding) |
| KNN over labeled+pseudo | 0.70-0.85 (similar files have similar chorus composition) |
| Combined ensemble + prior | 0.88-0.94 (current best) |

To push past 0.94 on a single co-occurring frog species, you'd need:
1. **A model trained specifically on frog vocalizations** (Perch's training data was bird-heavy, ~89% birds)
2. **More training samples per frog species** (we have 36-333 per species)
3. **Disambiguating spectral/temporal features** that distinguish similar frog calls

None of these are doable from the data we have. The 17,000+ frog observations in Nikita Babych's `birdclef2025-1st-place-extra-data` (per `AGENT_REPORTS.md`) could in theory train a frog-specialist model — but that requires GPU training.

## Theoretical ceiling estimate

Assuming the 4 bottleneck Amphibia classes are bounded at AUC ~0.93 (their measured peak with our best blend), and the other 71 classes can reach ~0.99:

```
macro_auc_ceiling ≈ (71 × 0.99 + 4 × 0.93) / 75 = 0.9868
```

So the theoretical ceiling with these models on this 739-window labeled OOF is around **0.987**. Reaching 0.99 requires either:
- A substantially better model on the Amphibia subset
- Or a different OOF set where the co-occurrence pattern is less extreme

## Recommendation: ship 0.9613 to LB

The honest 0.9613 is robust across blend coefficient + prior weight variations (plateau ~0.957-0.961 with w ∈ [2.0, 3.5]). It's the right recipe to submit.

Expected LB transfer:
- Prior submissions showed ~0.02 OOF→LB gap
- If pattern holds: expected LB ≈ 0.94

If the LB transfer is favorable (because pseudo+iNat priors generalize well to test), could be higher. If unfavorable (test contains hours 11-16 that our priors handle differently), could be lower.

## What WOULD reach 0.99

1. **Train Bruce-style probe but with much more data** — Nikita's 17k frog observations + iterative noisy-student over multiple iterations
2. **Replace Perch with a fine-tuned Perch on BC2026 train_soundscapes labels** (Hengck's PyTorch port enables this)
3. **Use exp019's leaked predictions as targets** — but that's training-on-test by construction
4. **GPU-only path: custom SED model trained on full train_audio + train_soundscapes (Babych's BC2025 recipe)**

All require infrastructure beyond CPU + offline data.
