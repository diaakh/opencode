"""v4 pseudo-labels: same combined prior, but CENTERED log-shift so
absolute Bruce confidence is preserved while still applying per-class
re-ranking from the prior.

Formula change from v3:
  v3:  new_logit = base_logit + w * log(P(c|h))           # mostly NEGATIVE shifts
  v4:  new_logit = base_logit + w * log(P(c|h) * N_CLS)   # CENTERED at 0 for uniform baseline

Result expected: Bruce-confident predictions at typical hours get
MORE confident; same predictions at unusual hours get slightly less
confident; uniform-baseline rows stay unchanged.
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/context_label"
EXT = f"{REPO}/analysis/external_priors"

EPS = 1e-6
W_HOUR = 1.0

# Load
data = np.load(f"{OUT}/bruce_on_unlabeled.npz", allow_pickle=True)
P_bruce = data["P_bruce"]
P_perch = data["perch_soft"]
classes = list(data["classes"])
N_PSEUDO, N_CLS = P_bruce.shape
meta = pd.DataFrame({
    "row_id": data["row_id"], "filename": data["filename"],
    "site": data["site"], "hour": data["hour_utc"].astype(int),
    "year": data["year"].astype(int),
})
hours = meta["hour"].values

prior = pd.read_csv(f"{EXT}/combined_hour_prior.csv").set_index("hour")
prior_arr = np.zeros((24, N_CLS), dtype=np.float32)
for h in range(24):
    if h in prior.index:
        for ci, c in enumerate(classes):
            if c in prior.columns:
                prior_arr[h, ci] = prior.loc[h, c]
prior_per_row = prior_arr[hours]  # (N, C)

# CENTERED log-shift
def logit(p): return np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

# Multiply by N_CLS to center: log(P*N) = 0 at uniform baseline
prior_lift = prior_per_row * N_CLS
prior_lift = np.clip(prior_lift, EPS, None)  # avoid log(0)
prior_logit_centered = W_HOUR * np.log(prior_lift)

print(f"Prior shift (CENTERED) stats:")
print(f"  range: [{prior_logit_centered.min():.3f}, {prior_logit_centered.max():.3f}]")
print(f"  mean : {prior_logit_centered.mean():.4f}")
print(f"  median: {np.median(prior_logit_centered):.4f}")
print(f"  std  : {prior_logit_centered.std():.4f}")

# Compare to UNCENTERED (v3) shift
prior_logit_uncentered = W_HOUR * np.log(np.clip(prior_per_row, EPS, 1.0))
print(f"\nUncentered shift (v3) for reference:")
print(f"  range: [{prior_logit_uncentered.min():.3f}, {prior_logit_uncentered.max():.3f}]")
print(f"  mean : {prior_logit_uncentered.mean():.4f}")

bruce_logit = logit(P_bruce)
P_bruce_shifted_v4 = sigmoid(bruce_logit + prior_logit_centered)
P_perch_shifted_v4 = sigmoid(logit(P_perch) + prior_logit_centered)

print(f"\nBruce raw: range [{P_bruce.min():.4f}, {P_bruce.max():.4f}], mean {P_bruce.mean():.4f}")
print(f"Bruce v4 shifted: range [{P_bruce_shifted_v4.min():.4f}, {P_bruce_shifted_v4.max():.4f}], "
      f"mean {P_bruce_shifted_v4.mean():.4f}")


# ---------- Validate on labeled OOF: does v4 lift Bruce AUC too? ----------
print("\n=== Validating: macro-AUC lift on LABELED OOF with v4 shift ===")
br_lab = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
Y_lab = br_lab["Y"]
P_bruce_lab = br_lab["P_bruce"]
hours_lab = br_lab["row_hour"].astype(int)

# Apply v4 centered shift to labeled
prior_per_row_lab = prior_arr[hours_lab]
prior_lift_lab = np.clip(prior_per_row_lab * N_CLS, EPS, None)
P_bruce_lab_v3 = sigmoid(logit(P_bruce_lab) + W_HOUR * np.log(np.clip(prior_per_row_lab, EPS, 1.0)))
P_bruce_lab_v4 = sigmoid(logit(P_bruce_lab) + W_HOUR * np.log(prior_lift_lab))


def macro_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0 or y[:, c].sum() == len(y):
            continue
        col = p[:, c]
        mask = ~np.isnan(col)
        if mask.sum() < 5 or y[mask, c].sum() == 0 or y[mask, c].sum() == mask.sum():
            continue
        try:
            aucs.append(roc_auc_score(y[mask, c], col[mask]))
        except:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


print(f"  Bruce alone:       {macro_auc(Y_lab, P_bruce_lab):.4f}")
print(f"  Bruce + v3 shift:  {macro_auc(Y_lab, P_bruce_lab_v3):.4f}  (uncentered, mostly -shift)")
print(f"  Bruce + v4 shift:  {macro_auc(Y_lab, P_bruce_lab_v4):.4f}  (centered, mean ≈ 0)")
print("If v4 ≈ v3 macro-AUC, the rank-ordering is identical (just confidence calibration differs).")


# ---------- Per-class calibration tiers ----------
bruce_auc = np.full(N_CLS, np.nan)
for c in range(N_CLS):
    n_pos = int(Y_lab[:, c].sum())
    if n_pos == 0 or n_pos == len(Y_lab):
        continue
    col = P_bruce_lab[:, c]
    mask = ~np.isnan(col)
    if mask.sum() < 5 or Y_lab[mask, c].sum() == 0 or Y_lab[mask, c].sum() == mask.sum():
        continue
    try:
        bruce_auc[c] = roc_auc_score(Y_lab[mask, c], col[mask])
    except:
        pass


def tier_for(c):
    a = bruce_auc[c]
    if np.isnan(a): return ("untested", 0.95, True, 80)
    if a >= 0.95: return ("high_trust", 0.85, False, 500)
    if a >= 0.85: return ("med_trust", 0.90, False, 300)
    if a >= 0.70: return ("low_trust", 0.95, False, 150)
    return ("untrusted", 0.95, True, 50)


# ---------- Build v4 pseudo-labels ----------
rows_out = []
for c_idx in range(N_CLS):
    tier, thresh, require_consensus, cap = tier_for(c_idx)
    bruce_shifted = P_bruce_shifted_v4[:, c_idx]
    bruce_raw = P_bruce[:, c_idx]
    perch_col = P_perch[:, c_idx]

    mask = (~np.isnan(bruce_shifted)) & (bruce_shifted >= thresh)
    if require_consensus:
        mask &= (~np.isnan(perch_col)) & (perch_col >= 0.5)
    qual_idxs = np.where(mask)[0]
    qual_idxs = qual_idxs[np.argsort(bruce_shifted[qual_idxs])[::-1]][:cap]

    for i in qual_idxs:
        rows_out.append({
            "row_id": meta["row_id"].iloc[i],
            "filename": meta["filename"].iloc[i],
            "site": meta["site"].iloc[i],
            "hour": int(meta["hour"].iloc[i]),
            "year": int(meta["year"].iloc[i]) if not pd.isna(meta["year"].iloc[i]) else -1,
            "class": classes[c_idx],
            "bruce_raw": float(bruce_raw[i]),
            "bruce_shifted_v4": float(bruce_shifted[i]),
            "perch_raw": float(perch_col[i]) if not np.isnan(perch_col[i]) else float("nan"),
            "prior_centered_shift": float(prior_logit_centered[i, c_idx]),
            "tier": tier,
            "bruce_labeled_auc": float(bruce_auc[c_idx]) if not np.isnan(bruce_auc[c_idx]) else float("nan"),
            "threshold": thresh,
        })

v4 = pd.DataFrame(rows_out)
print(f"\n=== v4 pseudo-labels ===")
print(f"  Total: {len(v4)} rows")
print(f"  Unique files: {v4['filename'].nunique()}")
print(f"  Unique classes: {v4['class'].nunique()}")
print(f"  Mean prior shift (centered): {v4['prior_centered_shift'].mean():.3f}")
print(f"  Labels CREATED by prior (raw<thresh, shifted>=thresh):")
n_created = ((v4["bruce_raw"] < v4["threshold"]) & (v4["bruce_shifted_v4"] >= v4["threshold"])).sum()
n_kept = ((v4["bruce_raw"] >= v4["threshold"]) & (v4["bruce_shifted_v4"] >= v4["threshold"])).sum()
print(f"    {n_created} created by prior boost ({n_created/len(v4)*100:.1f}%)")
print(f"    {n_kept} kept from already-confident Bruce ({n_kept/len(v4)*100:.1f}%)")

v4.to_parquet(f"{OUT}/pseudo_labels_v4.parquet", index=False)
print(f"\nSaved {OUT}/pseudo_labels_v4.parquet")

# Compare v2/v3/v4
print("\n=== v2 / v3 / v4 comparison ===")
v2 = pd.read_parquet(f"{OUT}/pseudo_labels_v2.parquet")
v3 = pd.read_parquet(f"{OUT}/pseudo_labels_v3.parquet")
for n, df in [("v2 (no prior)", v2), ("v3 (uncentered prior)", v3), ("v4 (centered prior)", v4)]:
    print(f"  {n:30s}: rows={len(df):>5d}  files={df['filename'].nunique():>4d}  classes={df['class'].nunique():>3d}")
