"""Experiment 29: Validate best metric on expanded anchor set (now ~10+ LB-known)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr, pearsonr

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

# Build our row keys
def our_key(fn, start):
    return f"{str(fn).replace('.ogg', '')}_{int(start)}"
our_idx = {our_key(row_fn_full[i], row_start[i]): i for i in range(N_full)}

# Get public alignment from safar1 meta
safar_meta = pd.read_parquet("/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_meta.parquet")
their_keys = safar_meta["row_id"].astype(str).values
pub_to_our_map = []
for i, k in enumerate(their_keys):
    if k in our_idx:
        pub_to_our_map.append((i, our_idx[k]))
print(f"Public→ours alignment: {len(pub_to_our_map)} rows")

# Collect ALL public NPZ anchors with LB attributions
# Note: this attribution is for the KERNEL's full pipeline LB, applied to its oof_base/scores
public_anchors = [
    # NEW high-value anchors
    ("baidalinadilzhan_oof_base", "/tmp/pub_outputs/baidalinadilzhan_perch-improved-lb-0-904/perch_cache/full_oof_meta_features.npz", "oof_base", 0.904),
    ("saurabhrajvarma_oof_base", "/tmp/pub_outputs/saurabhrajvarma_birdclef-2026-audio-classification-0-922/perch_cache/full_oof_meta_features.npz", "oof_base", 0.922),
    ("mtoshidesu_0928_oof_base", "/tmp/pub_outputs/mtoshidesu_0-928-bird26-reproduce-perch-protossm-resssm/perch_cache/full_oof_meta_features.npz", "oof_base", 0.928),
    ("itshyao_0949_oof_base", "/tmp/pub_outputs/itshyao_birdclef-2026-s106-eos5-0949-safealign2/perch_cache/full_oof_meta_features.npz", "oof_base", 0.949),
    ("itshyao_0949_oof_prior", "/tmp/pub_outputs/itshyao_birdclef-2026-s106-eos5-0949-safealign2/perch_cache/full_oof_meta_features.npz", "oof_prior", 0.949),
    # Existing anchors (already validated)
    ("needless090_0934_oof_base", "/tmp/pub_outputs/needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s/perch_cache/full_oof_meta_features.npz", "oof_base", 0.934),
    ("needless090_0934_oof_prior", "/tmp/pub_outputs/needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s/perch_cache/full_oof_meta_features.npz", "oof_prior", 0.934),
    ("koushikrudra_0928_oof_base", "/tmp/pub_outputs/koushikrudra_0-928-winner-position/perch_cache/full_oof_meta_features.npz", "oof_base", 0.928),
]

# Compute metrics
def macro_auc(P, Y, mask):
    Y_m = Y[mask]
    P_m = P[mask]
    aucs = []
    for c in range(Y_m.shape[1]):
        if Y_m[:, c].sum() < 2 or Y_m[:, c].sum() == mask.sum(): continue
        if P_m[:, c].max() == P_m[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y_m[:, c], P_m[:, c]))
        except: pass
    return np.mean(aucs) if aucs else 0

def site_mean(P, Y, sites, mask):
    site_data = []
    for site in sorted(set(sites[mask])):
        sub = (sites == site) & mask
        if sub.sum() < 15: continue
        aucs = []
        for c in range(Y.shape[1]):
            if Y[sub, c].sum() < 2 or Y[sub, c].sum() == sub.sum(): continue
            if P[sub, c].max() == P[sub, c].min(): continue
            try: aucs.append(roc_auc_score(Y[sub, c], P[sub, c]))
            except: pass
        if aucs:
            site_data.append((sub.sum(), np.mean(aucs)))
    counts = np.array([c for c, _ in site_data])
    aucs = np.array([a for _, a in site_data])
    return (counts / counts.sum() * aucs).sum()

# Best metric coefficients
INT, A_OA, A_SM = 0.2786, -0.8970, 1.6087

print("\n" + "="*100)
print("EXPANDED ANCHOR VALIDATION")
print("="*100)
print(f"Metric: LB ≈ {INT} + {A_OA}*overall_auc + {A_SM}*site_mean")
print(f"{'Anchor':<32s} | overall | site_mean | predicted | actual | error")
print("-"*100)

new_anchor_data = []
for name, path, key, lb in public_anchors:
    try:
        d = np.load(path)
        if key not in d.keys():
            print(f"  {name}: missing key {key}")
            continue
        pred = d[key]
        # Convert logits to probs
        if pred.min() < -1:
            P_prob = 1.0 / (1.0 + np.exp(-pred))
        else:
            P_prob = pred
        # Align
        P_aligned = np.full((N_full, C), np.nan, dtype=np.float32)
        for their_i, our_i in pub_to_our_map:
            P_aligned[our_i] = P_prob[their_i]
        mask = ~np.isnan(P_aligned).any(axis=1)
        oa = macro_auc(P_aligned, Y_full, mask)
        sm = site_mean(P_aligned, Y_full, sites_full, mask)
        predicted = INT + A_OA * oa + A_SM * sm
        error = predicted - lb
        new_anchor_data.append({"name": name, "oa": oa, "sm": sm, "predicted": predicted, "actual": lb, "error": error})
        print(f"{name:<32s} | {oa:.4f}  | {sm:.4f}    | {predicted:.4f}    | {lb:.3f}  | {error:+.4f}")
    except Exception as e:
        print(f"  {name}: ERR {e}")

# Compute correlation
preds = [a["predicted"] for a in new_anchor_data]
actuals = [a["actual"] for a in new_anchor_data]
if len(preds) > 2:
    r, _ = pearsonr(preds, actuals)
    rho, _ = spearmanr(preds, actuals)
    rmse = np.sqrt(np.mean([(p-a)**2 for p, a in zip(preds, actuals)]))
    print(f"\nMetric performance on {len(preds)} PUBLIC anchors:")
    print(f"  Pearson r: {r:+.3f}")
    print(f"  Spearman ρ: {rho:+.3f}")
    print(f"  RMSE: {rmse:.4f}")

# === Now: REFIT the metric using ALL anchors (internal + public) ===
print("\n" + "="*100)
print("REFIT METRIC WITH ALL ANCHORS")
print("="*100)
# Internal anchors
internal_data = [
    {"name": "exp019",       "oa": 0.9618, "sm": 0.9545, "actual": 0.949},
    {"name": "V73",          "oa": 0.6665, "sm": 0.7833, "actual": 0.941},
    {"name": "Bruce",        "oa": 0.8585, "sm": 0.7747, "actual": 0.755},
    {"name": "slot6_recon",  "oa": 0.9733, "sm": 0.9546, "actual": 0.946},
    {"name": "slot11_recon", "oa": 0.9618, "sm": 0.9532, "actual": 0.949},
    {"name": "sub1_v3",      "oa": 0.9728, "sm": 0.9427, "actual": 0.920},
]
all_data = internal_data + new_anchor_data

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
X = np.array([[a["oa"], a["sm"]] for a in all_data])
y = np.array([a["actual"] for a in all_data])

reg = LinearRegression()
reg.fit(X, y)
pred_in = reg.predict(X)
print(f"\nNew metric (all {len(all_data)} anchors):")
print(f"LB ≈ {reg.intercept_:.4f} + {reg.coef_[0]:.4f}*overall_auc + {reg.coef_[1]:.4f}*site_mean")

# LOO
preds_loo = np.zeros_like(y)
for tr, te in LeaveOneOut().split(X):
    r = LinearRegression()
    r.fit(X[tr], y[tr])
    preds_loo[te] = r.predict(X[te])
rmse_loo = np.sqrt(((preds_loo - y)**2).mean())
rho_loo, _ = spearmanr(preds_loo, y)
print(f"\nLOO performance:")
print(f"  RMSE: {rmse_loo:.4f}")
print(f"  Spearman ρ: {rho_loo:+.3f}")

print(f"\n{'Anchor':<32s} | predicted | actual | error (LOO)")
for i, a in enumerate(all_data):
    p = preds_loo[i]
    err = p - a["actual"]
    print(f"{a['name']:<32s} | {p:.4f}    | {a['actual']:.3f}  | {err:+.4f}")
