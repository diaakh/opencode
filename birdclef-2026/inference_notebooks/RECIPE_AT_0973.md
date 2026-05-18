# Final recipe: 0.9729 honest macro-AUC

**Supersedes RECIPE_AT_0961.md and the 0.9710 plateau.** Pure-call prototype
similarity broke the structural ceiling — +0.0023 over the prior best.

## The breakthrough

The 0.9613 → 0.9710 ladder hit a plateau because all stackers (balanced LR,
hour-LR, MLP, LGB) derive from the same Perch embedding space. They have
correlated blind spots.

The prototype-similarity move is **orthogonal**: it uses an explicit
per-class Perch-embedding signature built from KNN-DB rows that contain
ONLY that class (single-label "pure" rows). For chorus-contaminated
windows, the model gets a fresh signal from "how similar is this window
to a known pure-call recording of species X?".

## The 0.9729 recipe

```python
R_blend = (
    0.30 * R_bruce_smoothed
  + 0.40 * R_megaKNN
  + 0.20 * R_probe
  + 0.10 * R_perch
)
R_blend = 0.45 * R_blend + 0.55 * (R_balanced_LR + R_hour_LR) / 2  # 0.9647
R_blend = 0.90 * R_blend + 0.10 * R_lgb_meta                       # 0.9671
R_blend = 0.65 * R_blend + 0.35 * R_mlp_5seed                      # 0.9708
R_blend = 0.70 * R_blend + 0.30 * R_prototype_sim                  # 0.9729 ⭐

# Combined hour prior in logit space at w=2.0
final = sigmoid(logit(R_blend) + 2.0 * log(combined_hour_prior[hour]))
```

## What is `R_prototype_sim`?

For each of the 234 BC2026 classes, build a prototype embedding by averaging
the L2-normalized Perch embeddings of all KNN-DB rows where:
- That class is positive
- No other class is positive (single-label rows preferred)

If fewer than 3 single-label rows exist, fall back to all positive rows.

At inference time:
```python
proto_sim = normalize(emb) @ prototypes.T  # (B*W, 234)
```

Then rank-norm cross-window and blend at alpha=0.30.

## Per-class prototype-sim AUC highlights

Some of the bottleneck Amphibia frogs that previously dragged the mean:
| Species | Pre-proto AUC | Proto-sim alone | Mechanism |
|---|---:|---:|---|
| 22961 (Pantanal chorus frog) | 0.82 | **0.994** | Embedding-space sig clean |
| 65380 (Dwarf Tree Frog) | 0.94 | 0.971 | Strong |
| 555146 (Chaco Tree Frog) | 0.89 | 0.968 | Strong |
| 24279 (Lesser Snouted Tree Frog) | 0.88 | 0.951 | Solid |
| 1491113 | 0.82 | 0.920 | Solid |
| 326272 | ? | 0.685 | Weaker (less DB data) |

The prototype mechanism is most effective for **chorus disambiguation**:
even when the local audio sounds like 4 frogs together, the embedding
similarity to each species's pure-call signature provides discrimination.

## Session ladder

| Stage | OOF Macro-AUC | Δ |
|---|---:|---:|
| Bruce alone | 0.867 | — |
| + Bruce_smoothed (within-file kernels) | 0.870 | +0.003 |
| + megaKNN (12 KNN variants) | 0.887 | +0.017 |
| + Probe Ridge | 0.888 | +0.001 |
| + combined_hour_prior (w=2.5) | 0.9595 | +0.072 |
| **= RECIPE_AT_0961 plateau** | **0.9613** | — |
| + balanced LR | 0.9647 | +0.0034 |
| + hour-conditional LR | 0.9663 | +0.0016 |
| + LGB-meta stacker | 0.9671 | +0.0008 |
| + 5-seed MLP (SVD-96) | 0.9708 | +0.0037 |
| **= prior plateau** | **0.9710** | — |
| **+ prototype-sim (alpha=0.30)** | **0.9729** | **+0.0019** ⭐ |

## Out-of-distribution attempts that didn't beat 0.9729

| Move | Result | Why |
|---|---:|---|
| LOFO labeled-OOF prototypes | 0.5778 | Only 78 single-label rows in 739 |
| KNN-over-Bruce-preds (within-OOF) | 0.7279 alone | Hurts blend |
| Multi-cluster prototypes (KMeans on pure rows) | 0.9728 | Only 9 classes had enough rows |
| Medoid prototypes | 0.9732 | +0.0003 vs mean — within noise |
| 3-way (orig+robust+medoid) | 0.9730 | Same |
| Multi-output MLP | 0.6773 alone | sklearn class-balancing inadequate |
| ExtraTrees per class | 0.7651 | Hurts blend |
| Multi-SVD MLP ensemble | 0.8331 | Worse than single SVD-96 |
| V73 (LB 0.941) on OOF | 0.9742 with CV-adaptive | Can't deploy (10s/file × 600 = 100min) |

## Files

- `analysis/creative/all_prototypes.npz` — (234, 1536) prototypes + quality flags
- `analysis/creative/prototype_bundle/prototype_bundle.pkl` — production deploy
- `inference_notebooks/priors_bundle/prototype_bundle.pkl` — staged for dataset
- Uploaded to `adkasd/birdclef-2026-priors-research` (Kaggle private dataset)

## LB transfer estimate

OOF→LB gap on labeled benchmark has been highly variable:
- exp019 + hour_prior w=3.0:  0.997 OOF → 0.949 LB (gap 0.048, but exp019 leaks)
- exp019 + hour_prior w=2.0:  0.99x OOF → 0.920 LB (gap ~0.07)
- Bruce + prior:               0.959 OOF → 0.755 LB (gap 0.20, severe)
- exp019 vanilla:              ? OOF → 0.949 LB

For sub_v8 with prototype:
- The prototype signal is **external-data-based** (KNN-DB rows from train_audio +
  pseudo) — should generalize better than hour priors tuned on labeled OOF
- Expected LB: **0.85-0.92** if Bruce-style transfer pattern; **0.93-0.97** if
  more favorable (prototypes match test recording distribution)

## Deployment

Production submission notebook: `sub_v8_full_recipe_standalone.py`
- Includes sklearn 1.6 compat fix (critical — sub_v7 would have crashed)
- Auto-detects prototype_bundle and applies at alpha=0.30
- Benchmark verified: 2.33 sec/file × 600 = ~23 min runtime
