"""Experiment 32: Add UNLABELED-derived features to the metric.

For each anchor, compute features that DON'T require labels:
- mean prediction entropy
- agreement with ensemble consensus (rank corr per row)
- distribution skewness
- fraction of confident predictions
- per-site prediction distribution stability

Then test if combining labeled + unlabeled features beats labeled-only.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr, pearsonr
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
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

# Load all our models
v73 = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]
bmae = np.load("/tmp/birdmae_oof_kernel/birdmae_labeled_oof.npz", allow_pickle=True)["P_birdmae"]
perch = np.load("/tmp/perch20_blend.npz", allow_pickle=True)["P_perch20"]
bruce = np.load('/tmp/bruce_oof/labeled_oof_perch_bruce.npz', allow_pickle=True)['P_bruce']

def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 1) / (N - 1)
    return R

R_exp = rank_norm(P_exp)
R_v73 = rank_norm(v73)
R_bmae = rank_norm(bmae)
R_bruce = rank_norm(bruce)

# Build hour prior
HOUR_RE = re.compile(r"_(\d{8})_(\d{2})\d{4}")
hours = np.array([int(HOUR_RE.search(str(f)).group(2)) if HOUR_RE.search(str(f)) else -1 for f in row_fn])
hour_counts = np.zeros((24, C), dtype=np.float32)
for i, h in enumerate(hours):
    if 0 <= h < 24: hour_counts[h] += Y[i]
hour_prior = hour_counts / (hour_counts.sum(axis=0, keepdims=True) + 1e-6)
P_hour = np.zeros((N, C), dtype=np.float32)
for i, h in enumerate(hours):
    if 0 <= h < 24: P_hour[i] = hour_prior[h]
P_hour = P_hour + 1e-7 * np.random.RandomState(42).rand(N, C)
R_hour = rank_norm(P_hour)

# All anchors
anchors_list = [
    ("exp019", R_exp, 0.949),
    ("V73", R_v73, 0.941),
    ("Bruce", R_bruce, 0.755),
    ("slot6_recon", 0.7*R_exp + 0.3*R_bmae, 0.946),
    ("slot11_recon", 0.97*R_exp + 0.03*(R_bruce + rank_norm(perch))/2, 0.949),
    ("sub1_v3", 0.75*R_exp + 0.25*R_hour, 0.920),
]

# Build ENSEMBLE CONSENSUS (mean of all anchors' predictions)
# This is our "pseudo-label" proxy
stack = np.stack([P for _, P, _ in anchors_list], axis=0)  # (n_anchors, N, C)
consensus = stack.mean(axis=0)  # (N, C)

def site_mean(P, m=None):
    if m is None: m = np.ones(N, dtype=bool)
    site_data = []
    for site in sorted(set(sites[m])):
        sub = (sites == site) & m
        if sub.sum() < 15: continue
        aucs = []
        for c in range(C):
            if Y[sub, c].sum() < 2 or Y[sub, c].sum() == sub.sum(): continue
            if P[sub, c].max() == P[sub, c].min(): continue
            try: aucs.append(roc_auc_score(Y[sub, c], P[sub, c]))
            except: pass
        if aucs:
            site_data.append((sub.sum(), np.mean(aucs)))
    counts = np.array([c for c, _ in site_data])
    sa = np.array([a for _, a in site_data])
    return (counts / counts.sum() * sa).sum()

def macro_auc(P):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs) if aucs else 0

# Build per-anchor features
def extract_full_features(P_anchor_idx):
    """All features INCLUDING unlabeled-derived."""
    name, P, lb = anchors_list[P_anchor_idx]
    
    # LABELED features
    oa = macro_auc(P)
    sm = site_mean(P)
    gap = oa - sm
    
    # UNLABELED-derived features (no labels needed!)
    # 1. Per-row Spearman corr with ensemble consensus
    # Use LEAVE-ONE-OUT consensus
    other_stack = np.delete(stack, P_anchor_idx, axis=0)
    loo_consensus = other_stack.mean(axis=0)  # (N, C)
    rhos_row = []
    for r in range(N):
        try:
            rho, _ = spearmanr(P[r], loo_consensus[r])
            if not np.isnan(rho): rhos_row.append(rho)
        except: pass
    consensus_agreement_row = np.mean(rhos_row)
    
    # 2. Per-class Spearman corr with consensus
    rhos_cls = []
    for c in range(C):
        if P[:, c].max() == P[:, c].min(): continue
        if loo_consensus[:, c].max() == loo_consensus[:, c].min(): continue
        try:
            rho, _ = spearmanr(P[:, c], loo_consensus[:, c])
            if not np.isnan(rho): rhos_cls.append(rho)
        except: pass
    consensus_agreement_cls = np.mean(rhos_cls)
    
    # 3. Prediction entropy (no labels needed)
    eps = 1e-7
    p_c = np.clip(P, eps, 1-eps)
    H_cell = -(p_c * np.log(p_c) + (1-p_c) * np.log(1-p_c)) / np.log(2)
    H_mean = H_cell.mean()
    H_frac_confident = (H_cell < 0.3).mean()
    
    # 4. Distribution stability per site (unlabeled-style)
    site_pred_std = []
    for site in sorted(set(sites)):
        m = sites == site
        if m.sum() < 5: continue
        # Compute mean prediction per class per site
        site_pred_std.append(P[m].mean(axis=0))
    site_pred_std = np.array(site_pred_std)
    site_var = site_pred_std.var(axis=0).mean()  # average variance of class means across sites
    
    # 5. Top-K consensus diversity (no labels)
    # For each row, top-3 predicted classes for this model
    top3_match_consensus = 0
    for r in range(N):
        top3_self = set(np.argsort(P[r])[::-1][:3])
        top3_cons = set(np.argsort(loo_consensus[r])[::-1][:3])
        top3_match_consensus += len(top3_self & top3_cons) / 3
    top3_match_consensus /= N
    
    # 6. Self-similarity: same-file rows should agree more
    # Per (filename, neighbor windows), compute pred similarity
    # Skip for now
    
    return {
        "name": name, "lb": lb,
        # Labeled
        "overall_auc": oa, "site_mean": sm, "gap": gap,
        # Unlabeled-derived
        "consensus_row_corr": consensus_agreement_row,
        "consensus_cls_corr": consensus_agreement_cls,
        "entropy_mean": H_mean,
        "frac_confident": H_frac_confident,
        "site_pred_var": site_var,
        "top3_match_consensus": top3_match_consensus,
    }

print("Computing features (including unlabeled)...")
features = []
for i in range(len(anchors_list)):
    f = extract_full_features(i)
    features.append(f)

# Display
print(f"\n{'Anchor':<15s} | overall | site_mean | gap     | row_corr | cls_corr | entropy | site_var | top3_m | LB")
print("-"*120)
for f in features:
    print(f"{f['name']:<15s} | {f['overall_auc']:.4f}  | {f['site_mean']:.4f}    | {f['gap']:+.4f} | {f['consensus_row_corr']:+.3f}   | {f['consensus_cls_corr']:+.3f}   | {f['entropy_mean']:.3f}   | {f['site_pred_var']:.4f}  | {f['top3_match_consensus']:.3f} | {f['lb']:.3f}")

# Correlation with LB
print("\n--- Single feature correlation with LB ---")
features_list = ["overall_auc", "site_mean", "gap", "consensus_row_corr", "consensus_cls_corr", "entropy_mean", "frac_confident", "site_pred_var", "top3_match_consensus"]
y = np.array([f["lb"] for f in features])
for fn in features_list:
    x = np.array([f[fn] for f in features])
    if np.std(x) < 1e-6: continue
    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    print(f"  {fn:<25s}: Pearson r={r:+.3f}, Spearman ρ={rho:+.3f}")

# Try 2-feature combos including unlabeled features
print("\n--- 2-feature LOO with unlabeled features ---")
X = np.array([[f[fn] for fn in features_list] for f in features])
results = []
for i, j in combinations(range(len(features_list)), 2):
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
    results.append((features_list[i], features_list[j], rmse, rho))
results.sort(key=lambda x: x[2])
print(f"{'F1':<25s} {'F2':<25s} | LOO RMSE | LOO ρ")
for f1, f2, rmse, rho in results[:8]:
    print(f"{f1:<25s} {f2:<25s} | {rmse:.4f}   | {rho:+.3f}")
