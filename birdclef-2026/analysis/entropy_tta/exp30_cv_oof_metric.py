"""Experiment 30: Fit a SEPARATE metric for CV-OOF protocol public anchors.

8 CV-OOF anchors discovered:
  baidalinadilzhan (0.904), saurabhrajvarma (0.922), 
  mtoshidesu_0928 (0.928), koushikrudra (0.928),
  needless090_oof_base (0.934), needless090_oof_prior (0.934),
  itshyao_oof_base (0.949), itshyao_oof_prior (0.949)

If we can fit a metric here too, we have two protocols: non-CV and CV.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr
from itertools import combinations

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
Y_full = ex["Y"]
row_fn_full = ex["row_filename"]
row_start = ex["row_start_sec"]
N_full, C = Y_full.shape

def site_of(fn):
    parts = str(fn).split('_')
    for p in parts:
        if p.startswith('S') and len(p) <= 3 and p[1:].isdigit():
            return p
    return "S??"
sites_full = np.array([site_of(fn) for fn in row_fn_full])

def our_key(fn, start):
    return f"{str(fn).replace('.ogg', '')}_{int(start)}"
our_idx = {our_key(row_fn_full[i], row_start[i]): i for i in range(N_full)}

safar_meta = pd.read_parquet("/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_meta.parquet")
their_keys = safar_meta["row_id"].astype(str).values
pub_map = [(i, our_idx[k]) for i, k in enumerate(their_keys) if k in our_idx]
print(f"Public→ours: {len(pub_map)} rows")

CV_ANCHORS = [
    ("baidalinadilzhan", "/tmp/pub_outputs/baidalinadilzhan_perch-improved-lb-0-904/perch_cache/full_oof_meta_features.npz", "oof_base", 0.904),
    ("saurabhrajvarma", "/tmp/pub_outputs/saurabhrajvarma_birdclef-2026-audio-classification-0-922/perch_cache/full_oof_meta_features.npz", "oof_base", 0.922),
    ("mtoshidesu_0928", "/tmp/pub_outputs/mtoshidesu_0-928-bird26-reproduce-perch-protossm-resssm/perch_cache/full_oof_meta_features.npz", "oof_base", 0.928),
    ("koushikrudra", "/tmp/pub_outputs/koushikrudra_0-928-winner-position/perch_cache/full_oof_meta_features.npz", "oof_base", 0.928),
    ("needless090_b", "/tmp/pub_outputs/needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s/perch_cache/full_oof_meta_features.npz", "oof_base", 0.934),
    ("needless090_p", "/tmp/pub_outputs/needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s/perch_cache/full_oof_meta_features.npz", "oof_prior", 0.934),
    ("itshyao_b", "/tmp/pub_outputs/itshyao_birdclef-2026-s106-eos5-0949-safealign2/perch_cache/full_oof_meta_features.npz", "oof_base", 0.949),
    ("itshyao_p", "/tmp/pub_outputs/itshyao_birdclef-2026-s106-eos5-0949-safealign2/perch_cache/full_oof_meta_features.npz", "oof_prior", 0.949),
]

def compute_features(P_aligned, mask):
    """Extract metric features from an aligned prediction matrix."""
    # Overall AUC
    Y_m = Y_full[mask]; P_m = P_aligned[mask]
    aucs = []
    for c in range(C):
        if Y_m[:, c].sum() < 2 or Y_m[:, c].sum() == mask.sum(): continue
        if P_m[:, c].max() == P_m[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y_m[:, c], P_m[:, c]))
        except: pass
    overall = np.mean(aucs) if aucs else 0
    n_active = len(aucs)
    # Site mean
    site_data = []
    for site in sorted(set(sites_full[mask])):
        sub = (sites_full == site) & mask
        if sub.sum() < 15: continue
        s_aucs = []
        for c in range(C):
            if Y_full[sub, c].sum() < 2 or Y_full[sub, c].sum() == sub.sum(): continue
            if P_aligned[sub, c].max() == P_aligned[sub, c].min(): continue
            try: s_aucs.append(roc_auc_score(Y_full[sub, c], P_aligned[sub, c]))
            except: pass
        if s_aucs:
            site_data.append((sub.sum(), np.mean(s_aucs)))
    counts = np.array([c for c, _ in site_data])
    site_aucs = np.array([a for _, a in site_data])
    w = counts / counts.sum()
    site_mean = (w * site_aucs).sum()
    site_std = np.sqrt((w * (site_aucs - site_mean)**2).sum())
    site_min = site_aucs.min()
    # Per-class stats
    cls_aucs = np.array(aucs)
    cls_std = cls_aucs.std()
    cls_min = cls_aucs.min()
    return {
        "overall": overall, "site_mean": site_mean, "site_std": site_std,
        "site_min": site_min, "cls_std": cls_std, "cls_min": cls_min,
        "gap": overall - site_mean, "n_active": n_active
    }

cv_data = []
for name, path, key, lb in CV_ANCHORS:
    d = np.load(path)
    if key not in d.keys(): continue
    pred = d[key]
    P_prob = 1.0/(1.0 + np.exp(-pred)) if pred.min() < -1 else pred
    P_aligned = np.full((N_full, C), np.nan, dtype=np.float32)
    for ti, oi in pub_map:
        P_aligned[oi] = P_prob[ti]
    mask = ~np.isnan(P_aligned).any(axis=1)
    feat = compute_features(P_aligned, mask)
    feat["name"] = name
    feat["lb"] = lb
    cv_data.append(feat)

print(f"\n{'Anchor':<25s} | overall | site_mean | gap     | n_active | actual LB")
print("-"*90)
for d in cv_data:
    print(f"{d['name']:<25s} | {d['overall']:.4f}  | {d['site_mean']:.4f}    | {d['gap']:+.4f} | {d['n_active']:<8d} | {d['lb']:.3f}")

# Try fitting a metric on CV data alone
features_to_try = ["overall", "site_mean", "site_std", "site_min", "cls_std", "cls_min", "gap"]
X = np.array([[d[f] for f in features_to_try] for d in cv_data])
y = np.array([d["lb"] for d in cv_data])

print("\n--- Single feature correlations with LB on 8 CV anchors ---")
for i, f in enumerate(features_to_try):
    if np.std(X[:, i]) < 1e-6: continue
    r, _ = pearsonr(X[:, i], y)
    rho, _ = spearmanr(X[:, i], y)
    print(f"  {f:<15s}: Pearson r={r:+.3f}, Spearman ρ={rho:+.3f}")

# 1- and 2-feature LOO models
print("\n--- 2-feature LOO models (top 5) ---")
results = []
for i, j in combinations(range(len(features_to_try)), 2):
    if np.std(X[:, i]) < 1e-6 or np.std(X[:, j]) < 1e-6: continue
    preds = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(X):
        r = LinearRegression()
        try:
            r.fit(X[tr][:, [i, j]], y[tr])
            preds[te] = r.predict(X[te][:, [i, j]])
        except:
            preds[te] = y[tr].mean()
    rmse = np.sqrt(((preds - y)**2).mean())
    try: rho, _ = spearmanr(preds, y)
    except: rho = 0
    results.append((features_to_try[i], features_to_try[j], rmse, rho))
results.sort(key=lambda x: x[2])
print(f"{'F1':<15s} {'F2':<15s} | RMSE   | Spearman")
for f1, f2, rmse, rho in results[:5]:
    print(f"{f1:<15s} {f2:<15s} | {rmse:.4f} | {rho:+.3f}")

# === COMBINED PROTOCOL-AGNOSTIC METRIC ===
# What if we use a feature that's STABLE across CV and non-CV?
# Try: max(overall_auc, site_mean) - this normalizes the protocol difference
print("\n--- Protocol-agnostic candidates ---")
# Combine our internal + CV data
internal_data = [
    {"overall": 0.9618, "site_mean": 0.9545, "site_std": 0.0997, "site_min": 0.622, "cls_std": 0.097, "cls_min": 0.558, "gap": 0.0073, "n_active": 70, "lb": 0.949, "name": "exp019"},
    {"overall": 0.6665, "site_mean": 0.7833, "site_std": 0.1049, "site_min": 0.507, "cls_std": 0.309, "cls_min": 0.019, "gap": -0.1168, "n_active": 70, "lb": 0.941, "name": "V73"},
    {"overall": 0.8585, "site_mean": 0.7747, "site_std": 0.1129, "site_min": 0.510, "cls_std": 0.142, "cls_min": 0.434, "gap": 0.0838, "n_active": 70, "lb": 0.755, "name": "Bruce"},
    {"overall": 0.9733, "site_mean": 0.9546, "site_std": 0.0912, "site_min": 0.654, "cls_std": 0.063, "cls_min": 0.697, "gap": 0.0187, "n_active": 70, "lb": 0.946, "name": "slot6_recon"},
    {"overall": 0.9618, "site_mean": 0.9532, "site_std": 0.1010, "site_min": 0.615, "cls_std": 0.097, "cls_min": 0.564, "gap": 0.0086, "n_active": 70, "lb": 0.949, "name": "slot11_recon"},
    {"overall": 0.9728, "site_mean": 0.9427, "site_std": 0.0979, "site_min": 0.631, "cls_std": 0.061, "cls_min": 0.701, "gap": 0.0301, "n_active": 70, "lb": 0.920, "name": "sub1_v3"},
]

all_data = internal_data + cv_data
# Add a 'cls_min' feature — minimum per-class AUC (deflated by both protocols)
print(f"\n{'Anchor':<25s} | overall | site_mean | cls_min | LB")
print("-"*70)
for d in all_data:
    print(f"{d['name']:<25s} | {d['overall']:.4f}  | {d['site_mean']:.4f}    | {d.get('cls_min', 0):.4f}  | {d['lb']:.3f}")

# Try cls_min as a stable feature
X2 = np.array([[d["overall"], d["site_mean"], d.get("cls_min", 0)] for d in all_data])
y2 = np.array([d["lb"] for d in all_data])

# LOO test
print("\n--- Cross-protocol LOO with overall+site_mean+cls_min ---")
preds = np.zeros_like(y2)
for tr, te in LeaveOneOut().split(X2):
    r = LinearRegression()
    r.fit(X2[tr], y2[tr])
    preds[te] = r.predict(X2[te])
rmse = np.sqrt(((preds - y2)**2).mean())
rho, _ = spearmanr(preds, y2)
print(f"  3-feat LOO RMSE: {rmse:.4f}, ρ: {rho:+.3f}")

# Try just cls_min
X3 = np.array([[d.get("cls_min", 0)] for d in all_data])
r = LinearRegression()
preds = np.zeros_like(y2)
for tr, te in LeaveOneOut().split(X3):
    rr = LinearRegression()
    rr.fit(X3[tr], y2[tr])
    preds[te] = rr.predict(X3[te])
rmse = np.sqrt(((preds - y2)**2).mean())
rho, _ = spearmanr(preds, y2)
print(f"  cls_min only LOO RMSE: {rmse:.4f}, ρ: {rho:+.3f}")
