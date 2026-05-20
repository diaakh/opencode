"""Experiment 35: Use ACTUAL probability distributions and entropies as features.

Instead of aggregating to overall_auc/site_mean, use:
- Probability quantiles (p1, p10, p50, p90, p99) — shape of distribution
- Entropy quantiles (per-cell, per-row)
- Top-1, top-3 probability mass
- Tail concentration (fraction > 0.9, < 0.1)
- Bimodality measures
- Per-class probability std
- Logit distribution characteristics
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr, pearsonr
from itertools import combinations
import re

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp = ex["P_exp019"]
Y = ex["Y"]
row_fn = ex["row_filename"]
N, C = Y.shape

def site_of(fn):
    parts = str(fn).split('_')
    for p in parts:
        if p.startswith('S') and len(p) <= 3 and p[1:].isdigit():
            return p
    return "S??"
sites = np.array([site_of(fn) for fn in row_fn])

v73 = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]
bmae = np.load("/tmp/birdmae_oof_kernel/birdmae_labeled_oof.npz", allow_pickle=True)["P_birdmae"]
perch = np.load("/tmp/perch20_blend.npz", allow_pickle=True)["P_perch20"]
bruce = np.load('/tmp/bruce_oof/labeled_oof_perch_bruce.npz', allow_pickle=True)['P_bruce']

def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 1) / (N - 1)
    return R

R_exp = rank_norm(P_exp); R_v73 = rank_norm(v73); R_bmae = rank_norm(bmae); R_bruce = rank_norm(bruce); R_perch = rank_norm(perch)

HOUR_RE = re.compile(r"_(\d{8})_(\d{2})\d{4}")
hours = np.array([int(HOUR_RE.search(str(f)).group(2)) if HOUR_RE.search(str(f)) else -1 for f in row_fn])
hc = np.zeros((24, C), dtype=np.float32)
for i, h in enumerate(hours):
    if 0 <= h < 24: hc[h] += Y[i]
hp = hc / (hc.sum(axis=0, keepdims=True) + 1e-6)
P_hour = np.zeros((N, C), dtype=np.float32)
for i, h in enumerate(hours):
    if 0 <= h < 24: P_hour[i] = hp[h]
P_hour = P_hour + 1e-7 * np.random.RandomState(42).rand(N, C)
R_hour = rank_norm(P_hour)

# Anchors with their RAW predictions (not just ranks)
RAW_ANCHORS = [
    ("exp019", P_exp, 0.949),
    ("V73", v73, 0.941),
    ("Bruce", bruce, 0.755),
    ("slot6_recon", 0.7*P_exp + 0.3*bmae, 0.946),
    ("slot11_recon", 0.97*P_exp + 0.03*(bruce + perch)/2, 0.949),
    ("sub1_v3", 0.75*P_exp + 0.25*P_hour, 0.920),
]

def distribution_features(P):
    """Compute distribution-shape features from raw probabilities."""
    p = P.flatten()
    eps = 1e-7
    p_clip = np.clip(p, eps, 1-eps)
    # Per-cell entropy
    H = -(p_clip * np.log(p_clip) + (1-p_clip) * np.log(1-p_clip)) / np.log(2)
    
    # Per-class probability stats
    p_class_mean = P.mean(axis=0)
    p_class_std = P.std(axis=0)
    p_class_max = P.max(axis=0)
    
    # Top-k per row
    top1 = np.sort(P, axis=1)[:, -1]  # top probability per row
    top3 = np.sort(P, axis=1)[:, -3:].mean(axis=1)
    top10 = np.sort(P, axis=1)[:, -10:].mean(axis=1)
    
    return {
        # Probability quantiles
        "p_p01": np.percentile(p, 1),
        "p_p10": np.percentile(p, 10),
        "p_p50": np.percentile(p, 50),
        "p_p90": np.percentile(p, 90),
        "p_p99": np.percentile(p, 99),
        # Concentration
        "p_frac_gt_09": (p > 0.9).mean(),
        "p_frac_lt_01": (p < 0.1).mean(),
        "p_frac_mid": ((p > 0.3) & (p < 0.7)).mean(),
        # Entropy stats
        "H_mean": H.mean(),
        "H_std": H.std(),
        "H_p10": np.percentile(H, 10),
        "H_p50": np.percentile(H, 50),
        "H_p90": np.percentile(H, 90),
        "H_frac_low": (H < 0.3).mean(),
        "H_frac_high": (H > 0.8).mean(),
        # Top-k stats
        "top1_mean": top1.mean(),
        "top1_std": top1.std(),
        "top3_mean": top3.mean(),
        "top10_mean": top10.mean(),
        # Per-class shape
        "class_max_mean": p_class_max.mean(),
        "class_std_mean": p_class_std.mean(),
        "class_mean_max": p_class_mean.max(),
        "class_mean_min": p_class_mean.min(),
        # Bimodality
        "p_var": p.var(),
        "p_skew": ((p - p.mean())**3).mean() / max(p.std()**3, 1e-6),
        "p_kurt": ((p - p.mean())**4).mean() / max(p.std()**4, 1e-6),
        # Bimodality coefficient
        "bimodality": ((p > 0.7) | (p < 0.3)).mean() / max(((p > 0.4) & (p < 0.6)).mean(), 1e-6),
    }

features = []
for name, P, lb in RAW_ANCHORS:
    f = distribution_features(P)
    f["name"] = name
    f["lb"] = lb
    features.append(f)

# Display key features
print("="*100)
print("DISTRIBUTION-SHAPE FEATURES (using actual probabilities and entropies)")
print("="*100)
FEATS_TO_SHOW = ["p_p10", "p_p50", "p_p90", "H_mean", "H_p10", "H_p90", "top1_mean", "top3_mean", "bimodality", "p_skew"]
print(f"{'Anchor':<13s}", end="")
for fn in FEATS_TO_SHOW:
    print(f"| {fn:<11s}", end="")
print(f"| LB")
for f in features:
    print(f"{f['name']:<13s}", end="")
    for fn in FEATS_TO_SHOW:
        print(f"| {f[fn]:>10.4f} ", end="")
    print(f"| {f['lb']:.3f}")

ALL_FEATURES = list(features[0].keys())
ALL_FEATURES = [f for f in ALL_FEATURES if f not in ("name", "lb")]
print(f"\nTotal distribution features: {len(ALL_FEATURES)}")

# Individual correlations
y = np.array([f["lb"] for f in features])
print(f"\n--- Individual correlation with LB ---")
correlations = []
for fn in ALL_FEATURES:
    x = np.array([f[fn] for f in features])
    if np.std(x) < 1e-6: continue
    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    correlations.append((fn, r, rho))
correlations.sort(key=lambda x: -abs(x[2]))
for fn, r, rho in correlations[:15]:
    print(f"  {fn:<25s}: Pearson r={r:+.3f}, Spearman ρ={rho:+.3f}")

# Best 1-feature, 2-feature LOO
X = np.array([[f[fn] for fn in ALL_FEATURES] for f in features])
print(f"\n--- 1-feature LOO (top 8) ---")
results = []
for i, fn in enumerate(ALL_FEATURES):
    if np.std(X[:, i]) < 1e-6: continue
    preds = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(X):
        r = LinearRegression()
        r.fit(X[tr][:, [i]].reshape(-1, 1), y[tr])
        preds[te] = r.predict(X[te][:, [i]].reshape(-1, 1))
    rmse = np.sqrt(((preds - y)**2).mean())
    rho, _ = spearmanr(preds, y)
    results.append((fn, rmse, rho))
results.sort(key=lambda x: x[1])
for fn, rmse, rho in results[:8]:
    print(f"  {fn:<25s}: LOO RMSE {rmse:.4f}, ρ {rho:+.3f}")

print(f"\n--- 2-feature LOO (top 10) ---")
results2 = []
for i, j in combinations(range(len(ALL_FEATURES)), 2):
    if np.std(X[:, i]) < 1e-6 or np.std(X[:, j]) < 1e-6: continue
    preds = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(X):
        try:
            r = LinearRegression()
            r.fit(X[tr][:, [i, j]], y[tr])
            preds[te] = r.predict(X[te][:, [i, j]])
        except: preds[te] = y[tr].mean()
    rmse = np.sqrt(((preds - y)**2).mean())
    try: rho, _ = spearmanr(preds, y)
    except: rho = 0
    results2.append((ALL_FEATURES[i], ALL_FEATURES[j], rmse, rho))
results2.sort(key=lambda x: x[2])
print(f"{'F1':<25s} {'F2':<25s} | LOO RMSE | LOO ρ")
for f1, f2, rmse, rho in results2[:10]:
    print(f"{f1:<25s} {f2:<25s} | {rmse:.4f}   | {rho:+.3f}")
