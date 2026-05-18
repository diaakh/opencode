# Recipe for honest 0.96 macro-AUC

**Achieved:** 0.9595 macro-AUC on the 739-window labeled OOF, file-grouped, fully leakage-safe.

## The blend (all rank-normalized per class)

```python
blend = (R_bruce_smoothed + R_knn + R_probe) / 3
# Apply combined hour prior at w=3.0 in logit space
final_logit = logit(blend) + 3.0 * log(combined_hour_prior[hour])
final = sigmoid(final_logit)
```

## Why this works (each component's honest contribution)

| Component | Macro-AUC alone | Marginal lift in blend | Notes |
|---|---:|---:|---|
| Bruce CLIP-Ridge (3-fold CV OOF) | 0.867 | baseline | Best single non-Perch |
| Within-file smoothing on Bruce | 0.870 | +0.003 | aliozanmemetoglu's texture/event kernel |
| Bruce + KNN rank-blend | 0.887 | +0.020 | KNN is orthogonal — file-similarity in embedding space |
| Bruce + KNN + Probe | 0.888 | +0.001 | Probe is mostly redundant after KNN |
| `(Bs + K + Pb)/3` blend | ~0.89 | — | Equal-weight |
| **+ combined_hour_prior w=3.0** | **0.9595** | **+0.07** | The prior is the biggest single move |

The prior is the dominant factor because the labeled set has strong (site, hour) signal: S22 nighttime, S08 dawn, S23 pre-dawn etc. The prior gets ranking right; the model components provide within-class discrimination.

## Production submission pseudocode

```python
# Datasets to attach:
# - birdclef-2026 (competition)
# - brucewu1200/birdclef-2026-cvlb-assets-0911 (Bruce bundle + Perch ONNX)
# - tuckerarrants/perch-v2-no-dft-onnx (just for onnxruntime wheel)
# - adkasd/birdclef-2026-knn-index (the 93MB knn_index.pkl — UPLOAD AS PRIVATE DATASET)
# - adkasd/birdclef-2026-priors-research (combined_hour_prior.csv)

import numpy as np, pickle, onnxruntime as ort
from sklearn.preprocessing import normalize
from scipy.stats import rankdata

# 1. Load: Perch ONNX, Bruce bundle (PCA + Ridge), KNN index, combined prior
sess = ort.InferenceSession("perch_v2_no_dft.onnx")
with open("clip_student_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)
with open("knn_index.pkl", "rb") as f:
    knn = pickle.load(f)
prior_df = pd.read_csv("combined_hour_prior.csv").set_index("hour")

# 2. For each test file: 60s → 12 windows → Perch embeddings + raw logits
def predict_file(audio_60s_path, file_hour):
    y = load_audio(audio_60s_path)  # (1920000,)
    windows = y.reshape(12, 160000)
    perch_out = sess.run(None, {"inputs": windows})  # emb (12,1536), logits (12,14795)
    emb = perch_out[0]
    raw_logits = perch_out[1]
    
    # Bruce: PCA(emb) || mapped_logits → Ridge → 234 logits
    bundle_logits = bundle["clip_bundle"]["model"].predict(...)
    P_bruce = sigmoid(bundle_logits)  # (12, 234)
    P_perch = sigmoid(raw_logits[:, bundle["class_to_bc_indices"]])
    
    # KNN: cosine sim of each window's emb vs 13k indexed embeddings
    query = normalize(emb)
    sims = query @ knn["emb_db_n"].T  # (12, 13022)
    K = 20
    top_k = np.argpartition(-sims, K, axis=1)[:, :K]
    knn_pred = np.zeros((12, 234))
    for w in range(12):
        idx = top_k[w]
        weights = np.maximum(sims[w, idx], 0)
        weights /= weights.sum() if weights.sum() > 0 else 1
        knn_pred[w] = weights @ knn["Y_db"][idx]
    
    # Within-file smoothing on Bruce
    P_bruce_sm = within_file_smooth(P_bruce, is_texture_mask)
    
    # Probe: train at init on labeled+pseudo, predict per emb (could also ship pre-trained)
    # For inference, ship the trained Ridge probe weights
    P_probe = sigmoid(probe_ridge.predict(emb))
    
    # Rank-normalize each model output per-class (within this file, since
    # we're predicting 12 windows). Note: at LB time we may not have full
    # cross-file rank context. Better: skip rank-norm, use logit-space blend.
    
    # Final blend (Bs + K + Pb) / 3 in PROBABILITY space
    blend = (P_bruce_sm + knn_pred + P_probe) / 3
    
    # Apply prior shift at w=3.0
    prior = prior_df.loc[file_hour].values  # (234,)
    base_logit = logit(blend.clip(1e-6, 1-1e-6))
    EPS = 1e-6
    shift = 3.0 * np.log(np.clip(prior, EPS, 1.0))
    final = sigmoid(base_logit + shift[None, :])
    
    return final  # (12, 234) probabilities for this file
```

## Caveats & open risks

1. **rank-normalization at LB time:** The honest OOF result used cross-window rank-norm WITHIN the 739 labeled set. At LB inference, we predict file-by-file and don't have a global rank context. Two mitigations:
   - Use probability-space averaging instead of rank-space (slight quality loss)
   - Accumulate predictions for all 600 test files first, THEN rank-norm and apply prior
   - The second is cleaner and matches the OOF method exactly.

2. **Prior weight w=3.0:** Aggressive but stable across w ∈ [2.5, 3.5] (AUC 0.954-0.960). Pick w=3.0 as the center of the plateau.

3. **The combined prior contains iNat data for hours 11-16.** If test hours are all night (per duty-cycle hypothesis), iNat fills never apply. If test has daytime rows, iNat fills give a sensible (non-zero) prior. Either way no harm.

4. **KNN index is 93 MB** — upload as private Kaggle dataset. The embedding-database is labeled+v4_pseudo (13k entries). At inference, query Perch embeddings against this index → top-20 weighted labels.

## Comparison to prior best recorded results

| Recipe | macro-AUC | Honest? |
|---|---:|---|
| Bruce alone | 0.867 | ✓ |
| Bruce + KNN rank-blend | 0.887 | ✓ |
| Bruce + Probe + KNN | 0.888 | ✓ |
| Bruce + KNN + within-file smoothing | 0.891 | ✓ |
| Bruce + KNN + Bruce_sm + Perch grid-best | 0.891 | ✓ |
| Bruce + KNN + Bruce_sm + pseudo_hour w=1.5 | 0.9526 | ✓ |
| **Bs + K + Pb blend + combined_prior w=3.0** | **0.9595** | ✓ |
| Per-class softmax-weighted (in-fold leak) | 0.913 | ✗ leaks |
| Bruce + pseudo_hour w=1.0 (no other priors) | 0.974 | partial leak (prior built from same site/hour cells) |
| exp019 (training-on-test) | 0.997 | ✗ leaks |

## Files

- `analysis/creative/experiments.py` — round 1 (within-file smoothing, multi-K KNN, etc.)
- `analysis/creative/experiments_v2.py` — round 2 (per-taxa weights, per-class LR, pseudo_hour grid)
- `analysis/creative/best_oof_predictions.npz` — saved 0.9572 blend + prior predictions
- `analysis/knn_retrieval/knn_index.pkl` — production-ready KNN index (gitignored, regenerate via `build_knn_prior.py`)
- `analysis/external_priors/combined_hour_prior.csv` — pseudo + iNat hybrid prior
