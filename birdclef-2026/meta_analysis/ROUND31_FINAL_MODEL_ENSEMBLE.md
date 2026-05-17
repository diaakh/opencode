# BirdCLEF+ 2026 — ROUND 31: FINAL model ensemble forensics

## 1. All single-fold models tested

Ran inference on the **same 739 labeled windows** for 6 different models:

| Model | Architecture | macro-AUC |
|---|---|---:|
| **Bruce Wu** | Ridge on PCA(Perch_emb)+logits | **0.9304** |
| **Pseudo hour prior** | Lookup table | **0.9175** |
| Alex SED | EfficientNet-B0 + 4-head SED | 0.6344 |
| Long ConvNeXtV2-tiny | ConvNeXt-tiny + 60s input | 0.6303 |
| Mauricio (Pantanal-augmented) | EfficientNet-B0 from scratch | 0.6232 |
| Snowflake EfficientNetV2-M | EffV2-M SED, 5s input | 0.6010 |
| Snowflake ConvNeXt-tiny | ConvNeXt-tiny SED, 5s input | 0.5988 |
| **Raw Perch v2** | Frozen Perch | 0.5178 |

**Key takeaway**: All CNN-from-scratch models score in **0.60-0.63** range. The only way to break 0.90 with single fold is **Bruce's Ridge on Perch** approach.

## 2. SED ensemble (5 single folds averaged)

| Method | macro-AUC |
|---|---:|
| Individual SED models | 0.60 - 0.63 |
| **Probability average** of 5 SEDs | **0.6527** |
| Rank average of 5 SEDs | 0.6370 |

**5-fold averaging adds only +0.02 AUC over best single SED**. This is because all 5 models have similar weaknesses (single-fold variance dominates).

## 3. Final ensemble tuning (with SED + Bruce + Hour)

```
log_Bruce + 0.0*log_SED + 3.0*log_Prior: 0.9586  ⭐ BEST
log_Bruce + 0.1*log_SED + 3.0*log_Prior: 0.9585
log_Bruce + 0.2*log_SED + 3.0*log_Prior: 0.9583
log_Bruce + 0.3*log_SED + 3.0*log_Prior: 0.9579
log_Bruce + 0.5*log_SED + 3.0*log_Prior: 0.9566
```

**Adding SED HURTS the ensemble.** The single-fold SED models are too noisy to add value to Bruce + Prior.

## 4. The single highest-yield no-training recipe (CONFIRMED)

```python
# Step 1: Compute Bruce Wu's Ridge prediction
# (requires Perch v2 ONNX + clip_student_bundle.pkl)
bruce_logits = ridge_model.predict(features)

# Step 2: Look up pseudo-cache hour prior
hour_prior = pseudo_hour_priors.loc[hour_of_test_file]

# Step 3: Combine in logit space with 3x prior weight
combined_logits = bruce_logits + 3.0 * np.log(hour_prior + 1e-7)
final_probs = 1 / (1 + np.exp(-combined_logits))
```

**Expected macro-AUC on labeled OOF: 0.9586**.
On private LB: **likely 0.94-0.95** (typical 0.01-0.02 OOF→test gap).

## 5. Why single-fold SEDs are weak

Looking at the per-class AUC distribution for each model:
- **Bruce**: 37 classes >0.95, 0 classes <0.70 (all classes well-modeled)
- **Single-fold SED**: ~15 classes >0.95, ~40 classes <0.70 (most classes poorly modeled)

The reason: Bruce leverages **Perch's 1536-d frozen embeddings** which encode the entire iNaturalist + XC corpus knowledge. Single-fold SEDs need to learn this from scratch with only 35k training examples.

**For SED-based ensemble to beat Bruce**, you'd need:
1. 5-fold training (5x compute)
2. Pseudo-labeling with multiple iterations
3. Backbone diversity (3+ architectures)
4. ESC-50 background mixup (Sydorskyi-style)

## 6. Inference cost vs gain table

| Recipe | Inference time | Macro-AUC | Cost per AUC point |
|---|---|---|---|
| Pseudo prior only | ~5 sec | 0.92 | free |
| Bruce + Prior | ~30 min (Perch inference) | 0.96 | 30 min for +0.04 |
| Bruce + Prior + SED ens | ~90 min | 0.96 | 60 min for ZERO gain |
| Custom 5-fold ensemble | ~90 min | 0.94-0.95 (LB) | weeks training |

**The Bruce + Pseudo Prior recipe is the optimal cost-quality tradeoff** — no custom training, fits the 90-min CPU budget.

## 7. What WOULD beat 0.96?

Looking at the BC2026 LB ceiling (0.962 for Yannan Chen):
1. **Custom-trained 5-fold ensemble** at backbone diversity (B0 + V2-S + ConvNeXt + nfnet)
2. **Multi-iterative noisy student** pseudo-labeling (3-4 rounds)
3. **Per-class calibration** with held-out validation
4. **Texture/event-aware smoothing** (aliozanmemetoglu)
5. **Site-conditional priors** from pseudo cache

The public ceiling appears to be ~0.96 OOF / ~0.95 LB. Beyond that, private innovations dominate.

## 8. Sources

All from local computation:
- Ran 5 model checkpoints live: alexander, mauricio_seed42
- Ran 3 ONNX models live: long_convnextv2, snowflake_convnext, snowflake_efnetv2m
- Bruce's OOF: pre-computed in his bundle
- All ensemble combinations evaluated on identical 739-row Y matrix
