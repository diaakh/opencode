# Final recipe: 0.9647 honest macro-AUC

**This supersedes RECIPE_AT_0961.md.** Adds class-balanced LogisticRegression
probe trained on Perch-emb-SVD-48 with `class_weight='balanced'`. The plateau
is wide (alpha 0.50-0.65 all give 0.9647) so it's robust.

## The blend

```python
# Step 1: 4-model base blend (rank-normalized per class)
R_base = 0.30 * R_bruce_smoothed +
         0.40 * R_megaKNN +
         0.20 * R_probe +
         0.10 * R_perch

# Step 2: blend balanced LR on top
ALPHA_LR = 0.55  # plateau center
R_final = (1 - ALPHA_LR) * R_base + ALPHA_LR * R_balanced_lr

# Step 3: apply combined hour prior in logit space at w=2.5
final = sigmoid(logit(R_final) + 2.5 * log(combined_hour_prior[hour]))
```

## What's "balanced LR"

A single-pass deployable model that lifts the bottleneck classes:

```python
svd = TruncatedSVD(n_components=48, random_state=42)
emb_red = svd.fit_transform(Perch_embeddings)  # (N, 48)
# For each class:
lr = LogisticRegression(C=1.0, max_iter=300,
                        class_weight='balanced', solver='lbfgs')
lr.fit(emb_red, Y[:, c])
P_lr[:, c] = lr.predict_proba(emb_red)[:, 1]
```

The `class_weight='balanced'` is the key — it gives the 36-positive frog
classes (1491113, 22961) fair weight against the much larger negative
class. The 48-dim SVD prevents overfit on 739 training rows.

## Performance ladder

| Recipe | Macro-AUC | Δ |
|---|---:|---:|
| Bruce alone | 0.8670 | base |
| Bruce + KNN | 0.8874 | +0.020 |
| Bruce + KNN + Probe | 0.8880 | +0.001 |
| `(Bs + K + Pb)/3` + combined_prior w=3.0 | 0.9595 | +0.07 ⭐ |
| `0.30/0.40/0.20/0.10` blend + prior w=2.5 | 0.9613 | +0.002 (RECIPE_AT_0961) |
| **+ balanced LR @ alpha=0.55** | **0.9647** | **+0.0034** ⭐ NEW |

## What's different vs 0.9613 recipe

- Adds a new `balanced_lr_bundle.pkl` artifact (357 KB) — small SVD + per-class LR
- The blend is now in 2 stages: base 4-model rank-blend, then mix with LR
- Optimal alpha=0.55 (plateau 0.50-0.65, very forgiving)
- Total artifact size for full submission: ~94 MB (KNN index dominates)

## Why balanced LR helps

The 75 valid classes have very different positive counts (2 to 333). Without
class balancing, the cross-entropy loss is dominated by majority-negative
samples for rare classes, so the model learns to predict "negative" for them.
With `class_weight='balanced'`, the rare-class positives get higher weight,
producing meaningful gradients and a discriminative boundary even at low N.

This works specifically on the Amphibia bottleneck:
- Class 22961 (36 pos): Bruce 0.749 → +LR 0.81-0.85 lift
- Class 25092 (24 pos): Bruce 0.553 → +LR 0.78-0.83 lift
- Class 1491113 (79 pos): Bruce 0.750 → +LR 0.84-0.87 lift

The balanced LR doesn't replace the existing blend — it adds a per-class
discriminator that compensates where the others' L2-regularized models
underweight the minorities.

## Artifacts produced

- `birdclef-2026/analysis/creative/balanced_lr_bundle.pkl` — SVD + 70 fitted LR models
- `birdclef-2026/analysis/creative/balanced_lr_oof.npz` — OOF predictions for blending tests

## Production deployment

Ship `balanced_lr_bundle.pkl` as a small Kaggle dataset. `sub_v8` auto-detects
it and uses alpha from the bundle. Falls back to 0.9580 if missing.

## Runtime impact

Adding balanced LR to sub_v8: per-window the SVD projection is `(1, 1536) @ (1536, 48)` =
trivial, then 70 LR predict_proba calls. Total ~0.5ms per window. For 7200
windows that's ~4 seconds. Negligible against the 35-50 min total.
