"""Apply the combined hour prior (pseudo+iNat) to Bruce+Perch predictions on
unlabeled, then extract high-confidence pseudo-labels.

Same shape as v2 but uses:
  - The honest w=1.0 prior weight measured on labeled OOF (NOT the corpus's 0.05)
  - The combined hour prior (pseudo for 0-10+17-23, iNat for 11-16)
  - Per-class confidence calibration tiers from labeled per-class AUC

Output: pseudo_labels_v3.parquet
"""
import os, re, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/context_label"
EXT = f"{REPO}/analysis/external_priors"

EPS = 1e-6
W_HOUR = 1.0  # honest weight measured on Bruce labeled OOF (0.867 → 0.974)
HIGH_CONF_THRESHOLD = 0.85  # absolute Bruce prob

# Load Bruce + Perch predictions on unlabeled
data = np.load(f"{OUT}/bruce_on_unlabeled.npz", allow_pickle=True)
P_bruce = data["P_bruce"]  # (127k, 234)
P_perch = data["perch_soft"]
classes = list(data["classes"])
N_PSEUDO, N_CLS = P_bruce.shape
meta = pd.DataFrame({
    "row_id": data["row_id"],
    "filename": data["filename"],
    "site": data["site"],
    "hour": data["hour_utc"].astype(int),
    "year": data["year"].astype(int),
})
hours = meta["hour"].values
print(f"Unlabeled: {N_PSEUDO} windows × {N_CLS} classes")
print(f"Hour distribution: {pd.Series(hours).value_counts().sort_index().to_dict()}")

# Load combined hour prior (drop-in for pseudo_hour_priors with iNat fill at 11-16)
prior = pd.read_csv(f"{EXT}/combined_hour_prior.csv").set_index("hour")
# Build (24, N_CLS) array in taxonomy order
prior_arr = np.zeros((24, N_CLS), dtype=np.float32)
for h in range(24):
    if h in prior.index:
        for ci, c in enumerate(classes):
            if c in prior.columns:
                prior_arr[h, ci] = prior.loc[h, c]
print(f"Combined prior loaded: {prior_arr.shape}, per-row sums: "
      f"{prior_arr.sum(axis=1).round(3).tolist()}")

# Sanity: every row should sum to ~1.0
assert all(0.9 < prior_arr[h].sum() < 1.1 for h in range(24)), "combined prior rows don't sum to 1.0"

# Per-class hour prior column extracted for the rows we have
# prior_per_row[i, c] = prior_arr[hours[i], c]
print("\nApplying combined prior shift at w=1.0...")
prior_per_row = prior_arr[hours]  # (N_PSEUDO, N_CLS)


# Apply log-shift on Bruce predictions
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


bruce_logit = logit(P_bruce)
prior_logit = W_HOUR * np.log(np.clip(prior_per_row, EPS, 1.0))
P_bruce_shifted = sigmoid(bruce_logit + prior_logit)

# Same for Perch
perch_logit = logit(P_perch)
P_perch_shifted = sigmoid(perch_logit + prior_logit)

print(f"Bruce raw: range [{P_bruce.min():.4f}, {P_bruce.max():.4f}], mean {P_bruce.mean():.4f}")
print(f"Bruce shifted: range [{P_bruce_shifted.min():.4f}, {P_bruce_shifted.max():.4f}], mean {P_bruce_shifted.mean():.4f}")


# ---------- Per-class calibration tiers (same as v2) ----------
br_lab = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
Y = br_lab["Y"]
P_bruce_lab = br_lab["P_bruce"]
bruce_auc = np.full(N_CLS, np.nan)
for c in range(N_CLS):
    n_pos = int(Y[:, c].sum())
    if n_pos == 0 or n_pos == len(Y):
        continue
    col = P_bruce_lab[:, c]
    mask = ~np.isnan(col)
    if mask.sum() < 5 or Y[mask, c].sum() == 0 or Y[mask, c].sum() == mask.sum():
        continue
    try:
        bruce_auc[c] = roc_auc_score(Y[mask, c], col[mask])
    except Exception:
        pass


def tier_for_class(c):
    auc = bruce_auc[c]
    if np.isnan(auc):
        return ("untested", 0.95, True, 80)
    if auc >= 0.95:
        return ("high_trust", 0.85, False, 500)
    if auc >= 0.85:
        return ("med_trust", 0.90, False, 300)
    if auc >= 0.70:
        return ("low_trust", 0.95, False, 150)
    return ("untrusted", 0.95, True, 50)


# ---------- Build v3 pseudo-labels ----------
rows_out = []
hours_arr = meta["hour"].values
for c_idx in range(N_CLS):
    tier, thresh, require_consensus, cap = tier_for_class(c_idx)
    # Use the shifted Bruce as primary confidence
    bruce_shifted_col = P_bruce_shifted[:, c_idx]
    bruce_raw_col = P_bruce[:, c_idx]
    perch_col = P_perch[:, c_idx]
    perch_shifted_col = P_perch_shifted[:, c_idx]

    # Require BOTH raw AND shifted Bruce >= threshold
    # (prevents priors alone from creating false positives)
    mask = (~np.isnan(bruce_shifted_col)) & (bruce_shifted_col >= thresh) & (bruce_raw_col >= 0.5)
    if require_consensus:
        mask &= (~np.isnan(perch_col)) & (perch_col >= 0.5)
    qual_idxs = np.where(mask)[0]
    qual_idxs = qual_idxs[np.argsort(bruce_shifted_col[qual_idxs])[::-1]]
    qual_idxs = qual_idxs[:cap]

    for i in qual_idxs:
        rows_out.append({
            "row_id": meta["row_id"].iloc[i],
            "filename": meta["filename"].iloc[i],
            "site": meta["site"].iloc[i],
            "hour": int(meta["hour"].iloc[i]),
            "year": int(meta["year"].iloc[i]) if not pd.isna(meta["year"].iloc[i]) else -1,
            "class": classes[c_idx],
            "bruce_raw": float(bruce_raw_col[i]),
            "bruce_shifted": float(bruce_shifted_col[i]),
            "perch_raw": float(perch_col[i]) if not np.isnan(perch_col[i]) else float("nan"),
            "perch_shifted": float(perch_shifted_col[i]) if not np.isnan(perch_shifted_col[i]) else float("nan"),
            "prior_logit_shift": float(prior_logit[i, c_idx]),
            "tier": tier,
            "bruce_labeled_auc": float(bruce_auc[c_idx]) if not np.isnan(bruce_auc[c_idx]) else float("nan"),
            "threshold": thresh,
            "consensus_required": require_consensus,
        })


labels_v3 = pd.DataFrame(rows_out)
print(f"\n=== v3 pseudo-labels ===")
print(f"Total: {len(labels_v3)} rows")
print(f"Unique files: {labels_v3['filename'].nunique()}")
print(f"Unique classes: {labels_v3['class'].nunique()}")
print(f"Per tier:")
print(labels_v3.groupby("tier").agg(
    n=("row_id", "count"),
    n_files=("filename", "nunique"),
    n_classes=("class", "nunique"),
    mean_bruce_raw=("bruce_raw", "mean"),
    mean_bruce_shifted=("bruce_shifted", "mean"),
).round(3).to_string())

# Hour distribution of v3 pseudo-labels
print(f"\nHour distribution of pseudo-labels:")
print(labels_v3["hour"].value_counts().sort_index().to_string())

# Comparison v1 vs v2 vs v3
try:
    v1 = pd.read_parquet(f"{OUT}/pseudo_labels.parquet")
    v2 = pd.read_parquet(f"{OUT}/pseudo_labels_v2.parquet")
    print(f"\n=== v1 vs v2 vs v3 ===")
    for name, df in [("v1", v1), ("v2", v2), ("v3", labels_v3)]:
        print(f"  {name}: rows={len(df)} files={df['filename'].nunique()} classes={df['class'].nunique()}")
except Exception as e:
    print(f"compare skipped: {e}")

labels_v3.to_parquet(f"{OUT}/pseudo_labels_v3.parquet", index=False)
print(f"\nSaved {OUT}/pseudo_labels_v3.parquet")


# ---------- Per-(site, hour) summary ----------
sh = labels_v3.groupby(["site", "hour"]).size().sort_values(ascending=False).head(20)
print(f"\nTop (site, hour) cells:")
print(sh.to_string())

# Also save per-row delta: bruce_shifted - bruce_raw
labels_v3["delta_from_prior"] = labels_v3["bruce_shifted"] - labels_v3["bruce_raw"]
print(f"\nPrior contribution (delta_from_prior) summary:")
print(labels_v3["delta_from_prior"].describe().round(4).to_string())
# How many labels exist BECAUSE of the prior (i.e., bruce_raw < threshold but bruce_shifted >= threshold)?
n_prior_only = ((labels_v3["bruce_raw"] < labels_v3["threshold"]) & (labels_v3["bruce_shifted"] >= labels_v3["threshold"])).sum()
print(f"\nLabels that exist BECAUSE of prior shift: {n_prior_only} ({n_prior_only/len(labels_v3)*100:.1f}%)")
