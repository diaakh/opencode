# May 19 submission plan — honest path to LB 0.96+

**Goal:** LB macro-AUC ≥ 0.96
**Status:** sub_v8 is benchmark-validated; expected runtime 21 min for 600 files

## Honest OOF state

| Recipe | OOF | LB (if measured) |
|---|---:|---:|
| Bruce alone | 0.867 | — |
| Bruce + KNN + Probe + Perch + prior | 0.9595 | — |
| sub2 Bruce standalone + prior w=3.0 | 0.9586 | **0.755** (severe OOF→LB gap) |
| exp019 + hour_prior w=3.0 | "0.997" (leaky) | **0.920** |
| exp019 + hour_prior w=2.0 + alias | "0.99x" (leaky) | 0.920 |
| **exp019 vanilla** | — | **0.949** ← public baseline |
| **V73 5-fold** | 0.667 (Aves-only model, drags on others) | **0.941** |
| **V73 + Raunak blend** | — | 0.931 |
| sub_v8 with stackers + clean prototype | **0.9710 (honest)** | **?** (untested) |

**Critical insight from May 18 submissions:** the hour-prior at w=3.0 HURT
LB by ~0.03. The prior is overfit to labeled-OOF site/hour distribution.
On LB it crushes legitimate signal.

## Slot allocation (5 LB submissions available)

| Slot | Strategy | Expected LB | Risk |
|---:|---|---:|---|
| 1 | **exp019 vanilla** (no prior) | 0.949 anchor | none |
| 2 | **V73 5-fold + small w_prior=0.5** | 0.94-0.95 | low |
| 3 | **exp019 + V73 5-fold rank-blend (50/50)** | 0.95-0.96 | low |
| 4 | **sub_v8 full pipeline standalone** (0.9710 OOF) | 0.85-0.93 | medium |
| 5 | **exp019 + sub_v8 rank-blend (70/30)** | 0.94-0.96 ⭐ best | low |

## Why slot 5 is the high-confidence push:

- **exp019 anchors at 0.949** (proven LB)
- **sub_v8 adds orthogonal Bruce-derived signal** (not Perch-pseudo-labeling)
- A 70/30 rank-blend keeps 70% of the proven recipe + adds 30% diversity
- If sub_v8 transfers at even 0.85 LB, the blend lifts toward 0.95
- If sub_v8 transfers at 0.92, blend can hit 0.96+

## What canceled the prior-heavy plan:

The May 18 results showed: hour_prior w=3.0 → LB 0.920 (vs 0.949 baseline).
The prior was tuned on labeled OOF where S22-night dominates, but the
test set has different site/hour distribution. So:
- DROP heavy prior weights (w ≥ 2.0)
- Use w=0.5 only as a gentle adjustment if needed
- Trust per-class model signals over hour priors

## Sub_v8 details

Recipe: Bruce_smoothed + megaKNN + Probe + Perch + balanced LR + hour LR
+ LGB-7 stacker + 5-seed MLP + prototype-sim + combined_hour_prior(w=2.0)

Critical fixes shipped this session:
1. **sklearn 1.6/1.8 compat** — pickled LRs lost `multi_class` attribute;
   restored via _sklearn_compat_fix() walk. Sub_v7 would have CRASHED on
   LB without this.
2. **Prototype bundle uses clean DB** — rebuilt without labeled-OOF rows
   to ensure LB inference matches the deployed signal.
3. **Benchmark on 30 train_soundscapes files**: 64s = 2.13 s/file → 21
   min for 600 files. 4× headroom.

## Sub_v8 deployment checklist

- [x] sub_v8_full_recipe_standalone.py with all stackers
- [x] Compat fix applied
- [x] Datasets attached: comp + Bruce + Perch + KNN + probe + priors
- [x] prototype_bundle.pkl uploaded (clean DB, blend_alpha=0.10)
- [x] Benchmark confirms 21 min runtime
- [ ] Push to LB Slot 4 tomorrow

## What CAN'T fix the 0.99 OOF wall

Already exhausted (all gave 0 or negative on clean measurements):
- Multi-prototype-per-class (KMeans on pure rows)
- Multi-output MLP (PyTorch + sklearn variants)
- ExtraTrees, RandomForest per class
- Multi-SVD MLP ensemble
- Big MLP with concat features
- LOFO labeled prototypes
- KNN-of-Bruce-predictions
- Multi-prototype mean/median/medoid
- Within-file rank/mean-centered/std/max
- Per-class LDA discriminator
- Soft-RAG / Full-RAG / GMM density
- Context-centered embedding subtraction
- File-grouped per-class temperature
- Sigmoid temperature scaling (doesn't affect AUC)

The 0.99 wall is structural — 4 Pantanal frogs co-occurring at S22-night
have per-class AUC bounded ~0.81-0.94 because they're acoustically
mixed in the same windows. To break it requires:
- GPU-trained model on full train_audio + frog extras (e.g., Babych 17k)
- Hengck PyTorch Perch fine-tune
- Different OOF set with no chorus co-occurrence
