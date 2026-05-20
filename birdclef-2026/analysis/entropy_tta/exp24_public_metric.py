"""Experiment 24: Apply our LB metric to public kernel OOF predictions.

Sources (all 708 rows × 234 classes):
- safar1 scores: Perch v2 raw logits, kernel LB 0.948 (full pipeline)
- mtoshidesu, youssefmo: same Perch v2 logits, LB 0.947, 0.948
- mattiaangeli: same Perch v2 logits, LB 0.943
- afr1ste: same, LB 0.946  
- needless090 oof_base: their base model, LB 0.934 (full)
- needless090 oof_prior: with prior applied, LB 0.934
- koushikrudra oof_base: their base, LB 0.928

For Perch v2 raw logits: actually corresponds to LB 0.91-0.92 tier (matches
public pure Perch v2 baselines)

For oof_base from needless090/koushikrudra: their FULL prediction (not just
Perch v2), should match their kernel LB.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import re

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
Y_full = ex["Y"]
row_fn_full = ex["row_filename"]
classes_full = ex["classes"]
N_full, C = Y_full.shape

# Load safar1 meta to get row_id alignment
safar_meta = pd.read_parquet("/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_meta.parquet")
safar_row_ids = safar_meta["row_id"].astype(str).values  # 708 rows
safar_sites = safar_meta["site"].astype(str).values  # 708 sites

# Build map: row_id → our Y index
# Our row_fn is filename like "BC2026_Train_0001_S08_20250606_030007.ogg"
# Their row_id is "BC2026_Train_0001_S08_20250606_030007_5"  (filename + _ + start_sec)
# Our row_start_sec gives the start
row_start = ex["row_start_sec"]

def build_y_key(fn, start):
    fn_no_ext = str(fn).replace(".ogg", "")
    return f"{fn_no_ext}_{int(start)}"

y_key_to_idx = {build_y_key(row_fn_full[i], row_start[i]): i for i in range(N_full)}
print(f"Our Y has {len(y_key_to_idx)} row_ids")

# Map their rows to ours
mapping = []
for i, rid in enumerate(safar_row_ids):
    if rid in y_key_to_idx:
        mapping.append((i, y_key_to_idx[rid]))
matched = len(mapping)
print(f"safar matched: {matched}/708 to our 739 labels")

# Now align all the predictions
public_files = {
    "safar1_perch_scores":   ("/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_arrays.npz", "scores", 0.948),
    "mtoshidesu_perch":      ("/tmp/pub_outputs/mtoshidesu_birdclef-2026-0-947-lb-public-pipeline-reproduced/cache/perch_arrays.npz", "scores", 0.947),
    "youssefmo_perch":       ("/tmp/pub_outputs/youssefmo942009_lb-0-948/cache/perch_arrays.npz", "scores", 0.948),
    "mattiaangeli_perch":    ("/tmp/pub_outputs/mattiaangeli_birdclef-2026-0-943-better-blend/cache/perch_arrays.npz", "scores", 0.943),
    "afr1ste_perch":         ("/tmp/pub_outputs/afr1ste_birdclef-2026-0-946-updated-perch-sed/cache/perch_arrays.npz", "scores", 0.946),
    "needless090_oof_base":  ("/tmp/pub_outputs/needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s/perch_cache/full_oof_meta_features.npz", "oof_base", 0.934),
    "needless090_oof_prior": ("/tmp/pub_outputs/needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s/perch_cache/full_oof_meta_features.npz", "oof_prior", 0.934),
    "koushikrudra_oof_base": ("/tmp/pub_outputs/koushikrudra_0-928-winner-position/perch_cache/full_oof_meta_features.npz", "oof_base", 0.928),
}

def site_of(fn):
    parts = str(fn).split('_')
    for p in parts:
        if p.startswith('S') and len(p) <= 3 and p[1:].isdigit():
            return p
    return "S??"
our_sites = np.array([site_of(fn) for fn in row_fn_full])

# Build aligned predictions matrix for each source
def align(pub_pred, mapping, N_target):
    """pub_pred: (708, 234), map to (739, 234) of our Y."""
    P = np.full((N_target, pub_pred.shape[1]), np.nan, dtype=np.float32)
    for src_i, dst_i in mapping:
        P[dst_i] = pub_pred[src_i]
    return P

def macro_auc(P, mask=None):
    """Compute macro AUC, optionally on a subset."""
    if mask is None:
        mask = ~np.isnan(P).any(axis=1)
    Y_m = Y_full[mask]
    P_m = P[mask]
    aucs = []
    for c in range(C):
        if Y_m[:, c].sum() < 2 or Y_m[:, c].sum() == mask.sum(): continue
        if P_m[:, c].max() == P_m[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y_m[:, c], P_m[:, c]))
        except: pass
    return np.mean(aucs) if aucs else 0

def site_mean_metric(P, mask=None):
    if mask is None:
        mask = ~np.isnan(P).any(axis=1)
    site_data = []
    for site in sorted(set(our_sites[mask])):
        sub_mask = (our_sites == site) & mask
        if sub_mask.sum() < 15: continue
        Y_s = Y_full[sub_mask]
        P_s = P[sub_mask]
        aucs = []
        for c in range(C):
            if Y_s[:, c].sum() < 2 or Y_s[:, c].sum() == sub_mask.sum(): continue
            if P_s[:, c].max() == P_s[:, c].min(): continue
            try: aucs.append(roc_auc_score(Y_s[:, c], P_s[:, c]))
            except: pass
        if aucs:
            site_data.append((sub_mask.sum(), np.mean(aucs)))
    counts = np.array([c for c, _ in site_data])
    aucs = np.array([a for _, a in site_data])
    w = counts / counts.sum()
    mean = (w * aucs).sum()
    return mean

print("\n" + "="*100)
print("PUBLIC OOF PREDICTIONS → METRIC COMPUTATION")
print("="*100)
print(f"{'Source':<25s} | matched | overall | site_mean | gap     | predicted | actual LB | error")
print("-"*100)
new_anchors = []
for name, (path, key, lb) in public_files.items():
    try:
        d = np.load(path)
        pred = d[key]
        if pred.shape != (708, 234):
            print(f"  {name}: shape mismatch {pred.shape}, skip")
            continue
        # Convert logits to probabilities (sigmoid)
        P_prob = 1 / (1 + np.exp(-pred))
        P_aligned = align(P_prob, mapping, N_full)
        mask = ~np.isnan(P_aligned).any(axis=1)
        oa = macro_auc(P_aligned, mask)
        sm = site_mean_metric(P_aligned, mask)
        gap = oa - sm
        pred_lb = 0.277 + 0.714 * sm - 0.894 * gap
        new_anchors.append({"name": name, "oa": oa, "sm": sm, "gap": gap, "pred": pred_lb, "lb": lb})
        print(f"{name:<25s} | {int(mask.sum()):<7d} | {oa:.4f}  | {sm:.4f}    | {gap:+.4f} | {pred_lb:.4f}    | {lb:.3f}     | {pred_lb-lb:+.4f}")
    except Exception as e:
        print(f"  {name}: ERR {e}")

# Now: refit metric with these as additional anchors
print("\n" + "="*100)
print("REFIT METRIC WITH ALL ANCHORS (internal + public)")
print("="*100)
# Internal anchors
internal = [
    ("exp019", 0.9618, 0.9545, 0.949),
    ("V73", 0.6665, 0.7833, 0.941),
    ("Bruce", 0.8585, 0.7747, 0.755),
    ("slot6_recon", 0.9733, 0.9546, 0.946),
    ("slot11_recon", 0.9618, 0.9532, 0.949),
]
all_anchors_data = []
for name, oa, sm, lb in internal:
    all_anchors_data.append({"name": name, "oa": oa, "sm": sm, "gap": oa-sm, "lb": lb})
for a in new_anchors:
    all_anchors_data.append(a)

print(f"\n{'Anchor':<26s} | overall | site_mean | gap     | actual LB")
print("-"*80)
for a in all_anchors_data:
    print(f"{a['name']:<26s} | {a['oa']:.4f}  | {a['sm']:.4f}    | {a['gap']:+.4f} | {a['lb']:.3f}")

# Fit
from numpy.linalg import lstsq
X = np.array([[1, a["sm"], a["gap"]] for a in all_anchors_data])
y = np.array([a["lb"] for a in all_anchors_data])
coefs, _, _, _ = lstsq(X, y, rcond=None)
pred = X @ coefs
errors = pred - y
print(f"\nRefined fit ({len(y)} anchors):")
print(f"  LB ≈ {coefs[0]:.4f} + {coefs[1]:.4f}*site_mean + {coefs[2]:.4f}*gap")
print(f"  RMSE: {np.sqrt((errors**2).mean()):.4f}, max error: {np.abs(errors).max():.4f}")
print()
print(f"{'Anchor':<26s} | pred    | actual  | error")
for i, a in enumerate(all_anchors_data):
    print(f"{a['name']:<26s} | {pred[i]:.4f}  | {y[i]:.4f}  | {pred[i]-y[i]:+.4f}")
