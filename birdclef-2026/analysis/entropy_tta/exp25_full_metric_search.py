"""Experiment 25: Comprehensive LB-correlation metric search.

Use EVERYTHING:
- 5 internal OOF anchors (exp019, V73, Bruce, slot6_recon, slot11_recon)
- 8 public OOF anchors (safar1, mtoshidesu, ..., needless090, koushikrudra)
- 2 reconstructions (sub1 75/25, sub2 Bruce+Perch)

Features (15+):
- overall_auc
- weighted site_mean, site_std, site_min, site_max
- per-class AUC mean, std, min, top-decile, bottom-decile
- ECE calibration error
- entropy stats (mean, fraction high-conf)
- gap (overall - site_mean)
- prediction std (per class avg)
- mean confidence
- spearman corr with Perch v2

Regression methods:
- Linear, Ridge, Lasso
- Random Forest (small)
- LOO CV for true error estimate
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from scipy.stats import rankdata, spearmanr
from numpy.linalg import lstsq
import re

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
Y_full = ex["Y"]
row_fn = ex["row_filename"]
row_start = ex["row_start_sec"]
N_full, C = Y_full.shape

def site_of(fn):
    parts = str(fn).split('_')
    for p in parts:
        if p.startswith('S') and len(p) <= 3 and p[1:].isdigit():
            return p
    return "S??"
sites_full = np.array([site_of(fn) for fn in row_fn])

# Build our row keys to match public
def our_key(fn, start):
    return f"{str(fn).replace('.ogg', '')}_{int(start)}"
our_idx = {our_key(row_fn[i], row_start[i]): i for i in range(N_full)}

# Load all models
v73 = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]
bmae = np.load("/tmp/birdmae_oof_kernel/birdmae_labeled_oof.npz", allow_pickle=True)["P_birdmae"]
P_perch20 = np.load("/tmp/perch20_blend.npz", allow_pickle=True)["P_perch20"]
bruce = np.load('/tmp/bruce_oof/labeled_oof_perch_bruce.npz', allow_pickle=True)['P_bruce']
P_exp = ex["P_exp019"]

def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 1) / (N_full - 1)
    return R

# === ANCHORS ===
# Internal anchors (use full 739 rows)
anchors = []

# Add internal
anchors.append({"name": "exp019", "P": P_exp, "is_logit": False, "lb": 0.949, "N": N_full})
anchors.append({"name": "V73", "P": v73, "is_logit": False, "lb": 0.941, "N": N_full})
anchors.append({"name": "Bruce", "P": bruce, "is_logit": False, "lb": 0.755, "N": N_full})

# slot6 reconstruction
slot6 = 0.7 * rank_norm(P_exp) + 0.3 * rank_norm(bmae)
anchors.append({"name": "slot6_recon", "P": slot6, "is_logit": False, "lb": 0.946, "N": N_full})

# slot11 reconstruction
sub_proxy = rank_norm(bruce) * 0.5 + rank_norm(P_perch20) * 0.5  # rough
slot11 = 0.97 * rank_norm(P_exp) + 0.03 * sub_proxy
anchors.append({"name": "slot11_recon", "P": slot11, "is_logit": False, "lb": 0.949, "N": N_full})

# sub1 reconstruction (75% exp + 25% hour_prior)
HOUR_RE = re.compile(r"_(\d{8})_(\d{2})\d{4}")
hours = np.array([int(HOUR_RE.search(str(f)).group(2)) if HOUR_RE.search(str(f)) else -1 for f in row_fn])
hour_counts = np.zeros((24, C), dtype=np.float32)
for i, h in enumerate(hours):
    if 0 <= h < 24:
        hour_counts[h] += Y_full[i]
hour_prior = hour_counts / (hour_counts.sum(axis=0, keepdims=True) + 1e-6)
P_hour = np.zeros((N_full, C), dtype=np.float32)
for i, h in enumerate(hours):
    if 0 <= h < 24:
        P_hour[i] = hour_prior[h]
P_hour = P_hour + 1e-7 * np.random.RandomState(42).rand(N_full, C)
sub1 = 0.75 * rank_norm(P_exp) + 0.25 * rank_norm(P_hour)
anchors.append({"name": "sub1_v3", "P": sub1, "is_logit": False, "lb": 0.920, "N": N_full})

# === Public anchors (aligned to our 649 rows) ===
public_kernels = [
    ("safar1_perch_logits", "/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_arrays.npz",
     "/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_meta.parquet",
     "scores", 0.948),
    ("mtoshidesu_perch", "/tmp/pub_outputs/mtoshidesu_birdclef-2026-0-947-lb-public-pipeline-reproduced/cache/perch_arrays.npz",
     "/tmp/pub_outputs/mtoshidesu_birdclef-2026-0-947-lb-public-pipeline-reproduced/cache/perch_meta.parquet",
     "scores", 0.947),
    ("needless090_oof_base", "/tmp/pub_outputs/needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s/perch_cache/full_oof_meta_features.npz",
     None, "oof_base", 0.934),
    ("needless090_oof_prior", "/tmp/pub_outputs/needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s/perch_cache/full_oof_meta_features.npz",
     None, "oof_prior", 0.934),
    ("koushikrudra_oof_base", "/tmp/pub_outputs/koushikrudra_0-928-winner-position/perch_cache/full_oof_meta_features.npz",
     None, "oof_base", 0.928),
]

# Build alignment from public 708 to our 739
# Look at safar's meta to get keys
safar_meta = pd.read_parquet("/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_meta.parquet")
their_keys = safar_meta["row_id"].astype(str).values
public_to_our_idx = []  # list of (their_idx, our_idx)
for i, k in enumerate(their_keys):
    if k in our_idx:
        public_to_our_idx.append((i, our_idx[k]))
print(f"Public→ours alignment: {len(public_to_our_idx)} rows")

# Build P matrices aligned to OUR 739 rows (filled with NaN for unmatched)
for kname, path, meta_path, key, lb in public_kernels:
    d = np.load(path)
    if key not in d.keys():
        print(f"  SKIP {kname}: no key {key}")
        continue
    pub_pred = d[key]
    # Convert logits to probabilities
    if pub_pred.dtype == np.float32 and pub_pred.min() < -1:
        # likely logits
        P_prob = 1.0 / (1.0 + np.exp(-pub_pred))
    else:
        P_prob = pub_pred
    # Align to our 739 rows
    P_aligned = np.full((N_full, C), np.nan, dtype=np.float32)
    for their_i, our_i in public_to_our_idx:
        P_aligned[our_i] = P_prob[their_i]
    anchors.append({"name": kname, "P": P_aligned, "is_logit": False, "lb": lb, "N": N_full})

print(f"\nTotal anchors: {len(anchors)}")

# === FEATURE EXTRACTION ===
def safe_auc(p, y):
    if y.sum() < 2 or y.sum() == len(y): return np.nan
    if p.max() == p.min(): return np.nan
    try: return roc_auc_score(y, p)
    except: return np.nan

def extract_features(P, Y, sites, name):
    """Extract many features from prediction matrix."""
    # Only use rows where P is not NaN (for public anchors with partial coverage)
    valid_mask = ~np.isnan(P).any(axis=1)
    P_v = P[valid_mask]
    Y_v = Y[valid_mask]
    sites_v = sites[valid_mask]
    n = len(P_v)
    
    # Per-class AUCs (full data)
    class_aucs = []
    for c in range(P_v.shape[1]):
        a = safe_auc(P_v[:, c], Y_v[:, c])
        if not np.isnan(a):
            class_aucs.append(a)
    class_aucs = np.array(class_aucs)
    if len(class_aucs) == 0:
        return None
    
    # Per-site AUCs
    site_data = []
    for site in sorted(set(sites_v)):
        m = sites_v == site
        if m.sum() < 15: continue
        s_aucs = []
        for c in range(P_v.shape[1]):
            a = safe_auc(P_v[m, c], Y_v[m, c])
            if not np.isnan(a):
                s_aucs.append(a)
        if s_aucs:
            site_data.append((m.sum(), np.mean(s_aucs)))
    if not site_data:
        return None
    counts = np.array([c for c, _ in site_data])
    sa = np.array([a for _, a in site_data])
    w = counts / counts.sum()
    s_mean = (w * sa).sum()
    s_var = (w * (sa - s_mean)**2).sum()
    s_std = np.sqrt(s_var)
    
    # Calibration ECE
    p_flat = P_v.flatten()
    y_flat = Y_v.flatten()
    valid = ~np.isnan(p_flat)
    p_flat = p_flat[valid]
    y_flat = y_flat[valid]
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(p_flat, bins[1:-1])
    ece = 0
    for b in range(10):
        m = bin_idx == b
        if m.sum() < 30: continue
        ece += (m.sum() / len(p_flat)) * abs(p_flat[m].mean() - y_flat[m].mean())
    
    # Entropy
    eps = 1e-7
    p_c = np.clip(P_v, eps, 1-eps)
    h_per_cell = -(p_c * np.log(p_c) + (1-p_c) * np.log(1-p_c)) / np.log(2)
    h_mean = h_per_cell.mean()
    
    # Distribution stats
    p_std_per_class = P_v.std(axis=0)
    p_skew = ((P_v - P_v.mean()) ** 3).mean() / max(P_v.std() ** 3, 1e-6)
    
    return {
        "overall_auc": class_aucs.mean(),
        "n_active": len(class_aucs),
        "site_mean": s_mean,
        "site_std": s_std,
        "site_min": sa.min(),
        "site_max": sa.max(),
        "site_range": sa.max() - sa.min(),
        "n_sites": len(site_data),
        "gap": class_aucs.mean() - s_mean,
        "cls_auc_std": class_aucs.std(),
        "cls_auc_min": class_aucs.min(),
        "cls_auc_q25": np.percentile(class_aucs, 25),
        "cls_auc_q75": np.percentile(class_aucs, 75),
        "cls_auc_iqr": np.percentile(class_aucs, 75) - np.percentile(class_aucs, 25),
        "ece": ece,
        "entropy_mean": h_mean,
        "frac_high_conf": (h_per_cell < 0.3).mean(),
        "frac_uncertain": (h_per_cell > 0.7).mean(),
        "pred_std_mean": p_std_per_class.mean(),
        "pred_skew": p_skew,
        "n_valid_rows": int(valid_mask.sum()),
    }

# Extract features
print(f"\nExtracting features for {len(anchors)} anchors...")
features = []
names = []
lbs = []
for a in anchors:
    f = extract_features(a["P"], Y_full, sites_full, a["name"])
    if f is not None:
        features.append(f)
        names.append(a["name"])
        lbs.append(a["lb"])

feat_df = pd.DataFrame(features, index=names)
feat_df["lb"] = lbs
print(f"\nFeatures extracted ({len(features)} anchors × {len(features[0]) if features else 0} features):")
print(feat_df.to_string())

# Save
feat_df.to_csv(f"{ETT}/anchor_features.csv")
print(f"\nSaved: {ETT}/anchor_features.csv")
