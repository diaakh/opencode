"""Experiment 38: Use 127k unlabeled rows with pseudo-labels to validate/extend metric.

backtracking/birdclef2026-pseudo-cache-v1 has:
- 127,104 rows × 234 classes
- Perch v2 raw scores (anchor for one tier)
- Soft pseudo-labels from an ensemble teacher (LB ~0.932)
- Perch v2 embeddings (1536-dim)

Strategy:
1. Treat soft pseudo-labels as probabilistic "labels" — Y_pseudo
2. Compute Perch v2's macro-AUC against Y_pseudo
3. See if Perch v2 features on 127k rows give stronger metric features than on 739 labeled
4. Identify which classes are well-pseudo-labeled (high consensus)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr, pearsonr

print("Loading pseudo data (127k rows)...")
scores = np.load("/tmp/pseudo_data/pseudo_scores.npy").astype(np.float32)
soft = np.load("/tmp/pseudo_data/pseudo_soft.npy").astype(np.float32)
meta = pd.read_parquet("/tmp/pseudo_data/pseudo_meta.parquet")
print(f"scores: {scores.shape}, range [{scores.min():.2f}, {scores.max():.2f}]")
print(f"soft (pseudo labels): {soft.shape}, range [{soft.min():.4f}, {soft.max():.4f}]")
print(f"meta: {meta.shape}, sites: {meta['site'].nunique()}")

# Convert scores to probs (sigmoid)
P_perch_unl = 1.0 / (1.0 + np.exp(-scores))

# Statistics
print(f"\nSites distribution:")
print(meta['site'].value_counts().head(10))

# Pseudo-label thresholds — class positivity rates
print(f"\nPseudo-label class positivity (at threshold 0.5):")
pos_rate = (soft > 0.5).sum(axis=0)
print(f"  Classes with > 100 pseudo-positives: {(pos_rate > 100).sum()}")
print(f"  Classes with > 1000 pseudo-positives: {(pos_rate > 1000).sum()}")
print(f"  Max pseudo-positives in a single class: {pos_rate.max()}")

# Use pseudo-labels to compute Perch v2 macro-AUC on 127k unlabeled
# Treat soft > 0.5 as "positive" pseudo-label, rest as negative
Y_pseudo_hard = (soft > 0.5).astype(np.float32)
print(f"\nHard pseudo-label totals: {int(Y_pseudo_hard.sum())} positives across {Y_pseudo_hard.shape[0]} rows")

# Compute Perch v2 OOF-style AUC against hard pseudo-labels (per class)
aucs = []
for c in range(234):
    if Y_pseudo_hard[:, c].sum() < 10: continue
    if Y_pseudo_hard[:, c].sum() == 127104: continue
    if scores[:, c].max() == scores[:, c].min(): continue
    try: aucs.append(roc_auc_score(Y_pseudo_hard[:, c], scores[:, c]))
    except: pass
print(f"\nPerch v2 raw vs pseudo-label macro-AUC: {np.mean(aucs):.4f} ({len(aucs)} classes)")

# Compute per-site AUC
sites = meta['site'].values
site_data = []
for site in sorted(set(sites)):
    mask = sites == site
    if mask.sum() < 100: continue
    Y_s = Y_pseudo_hard[mask]
    P_s = scores[mask]
    s_aucs = []
    for c in range(234):
        if Y_s[:, c].sum() < 5 or Y_s[:, c].sum() == mask.sum(): continue
        if P_s[:, c].max() == P_s[:, c].min(): continue
        try: s_aucs.append(roc_auc_score(Y_s[:, c], P_s[:, c]))
        except: pass
    if s_aucs:
        site_data.append((mask.sum(), np.mean(s_aucs), site))

counts = np.array([c for c, _, _ in site_data])
sa = np.array([a for _, a, _ in site_data])
weighted_site_mean = (counts / counts.sum() * sa).sum()
print(f"Perch v2 vs pseudo-label site_mean: {weighted_site_mean:.4f}")
print(f"Site breakdown:")
for c, a, s in site_data[:10]:
    print(f"  Site {s}: {c} rows, AUC {a:.4f}")

# === ANCHOR our metric ===
# We know public Perch v2 + trained head gets LB ~0.91
# These scores are RAW Perch v2 = no trained head
# Predicted Perch v2 raw LB tier: ~0.80-0.90?

# Apply our metric: LB ≈ 0.279 - 0.897*overall_auc + 1.609*site_mean
overall_auc_unl = np.mean(aucs)
predicted_lb = 0.279 - 0.897 * overall_auc_unl + 1.609 * weighted_site_mean
print(f"\n=== Metric prediction for Perch v2 raw (on 127k unlabeled) ===")
print(f"overall_auc (vs pseudo): {overall_auc_unl:.4f}")
print(f"site_mean (vs pseudo): {weighted_site_mean:.4f}")
print(f"predicted LB: {predicted_lb:.4f}")
print(f"Public Perch v2+head LB: 0.906-0.912")
print(f"Perch v2 raw expected LB (subtract trained head boost): 0.80-0.85?")
