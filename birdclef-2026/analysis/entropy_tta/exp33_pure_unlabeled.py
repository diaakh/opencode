"""Experiment 33: Can a pure UNLABELED-feature metric predict LB?

If yes, we can score NEW blends without labels — on any data.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
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

anchors_list = [
    ("exp019", R_exp, 0.949),
    ("V73", R_v73, 0.941),
    ("Bruce", R_bruce, 0.755),
    ("slot6_recon", 0.7*R_exp + 0.3*R_bmae, 0.946),
    ("slot11_recon", 0.97*R_exp + 0.03*(R_bruce + R_perch)/2, 0.949),
    ("sub1_v3", 0.75*R_exp + 0.25*R_hour, 0.920),
]
stack = np.stack([P for _, P, _ in anchors_list], axis=0)

def extract_features(P, anchor_idx):
    """All features, both labeled and unlabeled-derived."""
    # Labeled features
    aucs_o = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs_o.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    overall_auc = np.mean(aucs_o)
    
    site_data = []
    for site in sorted(set(sites)):
        m = sites == site
        if m.sum() < 15: continue
        a = []
        for c in range(C):
            if Y[m, c].sum() < 2 or Y[m, c].sum() == m.sum(): continue
            if P[m, c].max() == P[m, c].min(): continue
            try: a.append(roc_auc_score(Y[m, c], P[m, c]))
            except: pass
        if a:
            site_data.append((m.sum(), np.mean(a)))
    counts = np.array([c for c, _ in site_data])
    sa = np.array([a for _, a in site_data])
    site_mean = (counts / counts.sum() * sa).sum()
    
    # UNLABELED features (no labels needed)
    other_stack = np.delete(stack, anchor_idx, axis=0)
    loo_consensus = other_stack.mean(axis=0)
    
    rhos_row = []
    for r in range(N):
        try:
            rho, _ = spearmanr(P[r], loo_consensus[r])
            if not np.isnan(rho): rhos_row.append(rho)
        except: pass
    consensus_row = np.mean(rhos_row)
    
    rhos_cls = []
    for c in range(C):
        if P[:, c].max() == P[:, c].min(): continue
        if loo_consensus[:, c].max() == loo_consensus[:, c].min(): continue
        try:
            rho, _ = spearmanr(P[:, c], loo_consensus[:, c])
            if not np.isnan(rho): rhos_cls.append(rho)
        except: pass
    consensus_cls = np.mean(rhos_cls)
    
    eps = 1e-7
    p_c = np.clip(P, eps, 1-eps)
    H = -(p_c * np.log(p_c) + (1-p_c) * np.log(1-p_c)) / np.log(2)
    entropy_mean = H.mean()
    
    site_means_per_class = []
    for site in sorted(set(sites)):
        m = sites == site
        if m.sum() < 5: continue
        site_means_per_class.append(P[m].mean(axis=0))
    site_pred_var = np.array(site_means_per_class).var(axis=0).mean()
    
    top3_self_match = 0
    for r in range(N):
        t_self = set(np.argsort(P[r])[::-1][:3])
        t_cons = set(np.argsort(loo_consensus[r])[::-1][:3])
        top3_self_match += len(t_self & t_cons) / 3
    top3_self_match /= N
    
    return {
        "overall_auc": overall_auc,
        "site_mean": site_mean,
        "gap": overall_auc - site_mean,
        "consensus_row": consensus_row,
        "consensus_cls": consensus_cls,
        "entropy_mean": entropy_mean,
        "site_pred_var": site_pred_var,
        "top3_self_match": top3_self_match,
    }

feats = []
for i, (n, P, lb) in enumerate(anchors_list):
    f = extract_features(P, i)
    f["name"] = n
    f["lb"] = lb
    feats.append(f)

# Pure unlabeled features (no labels)
UNLAB_FEATURES = ["consensus_row", "consensus_cls", "entropy_mean", "site_pred_var", "top3_self_match"]
y = np.array([f["lb"] for f in feats])

print("="*80)
print("PURE UNLABELED METRIC: features that don't require labels")
print("="*80)
print(f"{'Anchor':<15s}", end=" ")
for fn in UNLAB_FEATURES: print(f"| {fn:<14s}", end="")
print(f"| LB")
for f in feats:
    print(f"{f['name']:<15s}", end=" ")
    for fn in UNLAB_FEATURES: print(f"| {f[fn]:>+.4f}      ", end="")
    print(f"| {f['lb']:.3f}")

print("\n--- 2-feature LOO with PURE UNLABELED features ---")
X = np.array([[f[fn] for fn in UNLAB_FEATURES] for f in feats])
results = []
for i, j in combinations(range(len(UNLAB_FEATURES)), 2):
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
    results.append((UNLAB_FEATURES[i], UNLAB_FEATURES[j], rmse, rho))
results.sort(key=lambda x: x[2])
print(f"{'F1':<25s} {'F2':<25s} | LOO RMSE | LOO ρ")
for f1, f2, rmse, rho in results[:5]:
    print(f"{f1:<25s} {f2:<25s} | {rmse:.4f}   | {rho:+.3f}")

# Try ALL features (8 features, 6 anchors → can fit max 5 with degrees of freedom)
print("\n--- 3-feature LOO with ALL features ---")
ALL_FEATURES = ["overall_auc", "site_mean", "gap", "consensus_row", "consensus_cls", "entropy_mean", "site_pred_var", "top3_self_match"]
X_all = np.array([[f[fn] for fn in ALL_FEATURES] for f in feats])
results3 = []
for trip in combinations(range(len(ALL_FEATURES)), 3):
    preds = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(X_all):
        try:
            r = LinearRegression()
            r.fit(X_all[tr][:, list(trip)], y[tr])
            preds[te] = r.predict(X_all[te][:, list(trip)])
        except: preds[te] = y[tr].mean()
    rmse = np.sqrt(((preds - y)**2).mean())
    try: rho, _ = spearmanr(preds, y)
    except: rho = 0
    results3.append((trip, rmse, rho))
results3.sort(key=lambda x: x[1])
print(f"{'Features':<70s} | LOO RMSE | LOO ρ")
for trip, rmse, rho in results3[:8]:
    fns = ", ".join(ALL_FEATURES[t] for t in trip)
    print(f"{fns:<70s} | {rmse:.4f}   | {rho:+.3f}")

# Build a HYBRID metric: best 2 labeled + 1 unlabeled
print("\n--- Hybrid: best labeled + 1 unlabeled feature ---")
i_oa = ALL_FEATURES.index("overall_auc")
i_sm = ALL_FEATURES.index("site_mean")
results_hybrid = []
for u_name in UNLAB_FEATURES:
    u_idx = ALL_FEATURES.index(u_name)
    preds = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(X_all):
        try:
            r = LinearRegression()
            r.fit(X_all[tr][:, [i_oa, i_sm, u_idx]], y[tr])
            preds[te] = r.predict(X_all[te][:, [i_oa, i_sm, u_idx]])
        except: preds[te] = y[tr].mean()
    rmse = np.sqrt(((preds - y)**2).mean())
    rho, _ = spearmanr(preds, y)
    results_hybrid.append((u_name, rmse, rho))
for un, rmse, rho in results_hybrid:
    print(f"  overall_auc + site_mean + {un:<25s} | RMSE {rmse:.4f} | ρ {rho:+.3f}")
