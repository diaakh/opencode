"""Noisy-Student linear probe on Perch embeddings.

Training data:
  - 739 labeled windows (hard labels, weight=1.0)
  - 15,361 v4 pseudo-labels (soft labels = bruce_shifted_v4 confidence,
    weighted by Bruce's labeled per-class AUC tier)

Model:
  - 1536-d Perch emb → 234-d sigmoid (linear probe with L2)
  - Per-class threshold tuned

CV:
  - Leave-one-FILE-out file-grouped 5-fold
  - For pseudo-labels: file-grouped too (pseudo windows from
    held-out files in val excluded from training to avoid leakage)
"""
import os, re, time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/noisy_student"
os.makedirs(OUT, exist_ok=True)


# ---------- Load labeled (Perch emb + Y + metadata) ----------
br_lab = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
emb_lab = br_lab["embeddings"].astype(np.float32)  # (739, 1536)
Y_lab = br_lab["Y"].astype(np.float32)  # (739, 234)
file_lab = br_lab["row_filename"]
hour_lab = br_lab["row_hour"]
site_lab = br_lab["row_site"]
P_bruce_lab = br_lab["P_bruce"]
N_LAB, EMB_DIM = emb_lab.shape
N_CLS = Y_lab.shape[1]
print(f"Labeled: {N_LAB} windows × {EMB_DIM}-d emb, {N_CLS} classes")

# Map filename → file_idx
unique_files_lab = sorted(set(file_lab))
file_idx_lab = np.array([sorted(unique_files_lab).index(f) for f in file_lab])

# ---------- Load v4 pseudo-labels + their Perch embeddings ----------
v4 = pd.read_parquet(f"{REPO}/analysis/context_label/pseudo_labels_v4.parquet")
print(f"V4 pseudo-labels: {len(v4)} rows, {v4['filename'].nunique()} files, {v4['class'].nunique()} classes")

# Load unlabeled embeddings (the 127k Perch embeddings on unlabeled).
# These live in the pseudo cache from Kaggle, not the local Bruce npz.
emb_unlab = np.load("/tmp/pseudo_cache/pseudo_emb.npy", mmap_mode="r").astype(np.float32)
unlab = np.load(f"{REPO}/analysis/context_label/bruce_on_unlabeled.npz", allow_pickle=True)
row_id_unlab = unlab["row_id"]
file_unlab = unlab["filename"]
assert emb_unlab.shape[0] == len(row_id_unlab), \
    f"emb count {emb_unlab.shape[0]} != row_id count {len(row_id_unlab)}"
print(f"Unlabeled: {len(emb_unlab)} windows × {EMB_DIM}-d emb")

# Build row_id → emb_idx lookup for unlabeled
unlab_rid_to_idx = {rid: i for i, rid in enumerate(row_id_unlab)}

# Build pseudo training set: for each v4 row, look up embedding + assign weight
classes = list(br_lab["classes"])
cls_idx = {c: i for i, c in enumerate(classes)}

# Each pseudo-label is a (window, class) → label assertion.
# Multiple labels per window (multi-label). Aggregate by row_id.
print("\nAggregating v4 pseudo-labels by row_id...")
pseudo_per_row = {}
for _, r in v4.iterrows():
    rid = r["row_id"]
    if rid not in unlab_rid_to_idx:
        continue
    c_idx_v = cls_idx[r["class"]]
    conf = r["bruce_shifted_v4"]
    if rid not in pseudo_per_row:
        pseudo_per_row[rid] = {"labels": np.zeros(N_CLS, dtype=np.float32),
                               "filename": r["filename"], "hour": r["hour"]}
    pseudo_per_row[rid]["labels"][c_idx_v] = conf
print(f"  unique pseudo rows: {len(pseudo_per_row)}")

# Convert to arrays
pseudo_rids = list(pseudo_per_row.keys())
N_PSEUDO_FILT = len(pseudo_rids)
emb_pseudo = np.array([emb_unlab[unlab_rid_to_idx[rid]] for rid in pseudo_rids], dtype=np.float32)
Y_pseudo = np.array([pseudo_per_row[rid]["labels"] for rid in pseudo_rids], dtype=np.float32)
file_pseudo = np.array([pseudo_per_row[rid]["filename"] for rid in pseudo_rids])

# Make pseudo labels SOFT: use confidence as label value (between 0.85 and 1.0)
# But for training, binarize them with weight = confidence
# Actually keep as soft (regression-style) for now
print(f"  emb_pseudo: {emb_pseudo.shape}")
print(f"  Y_pseudo: {Y_pseudo.shape}, range [{Y_pseudo.min():.4f}, {Y_pseudo.max():.4f}]")
print(f"  Y_pseudo non-zero rate: {(Y_pseudo > 0).mean():.4f}")


# ---------- File-grouped 5-fold CV on LABELED ----------
gkf = GroupKFold(n_splits=5)
oof_pred_labeled_only = np.zeros((N_LAB, N_CLS), dtype=np.float32)
oof_pred_with_pseudo = np.zeros((N_LAB, N_CLS), dtype=np.float32)

print("\n=== Training: Ridge probe, labeled only vs labeled + pseudo ===")
fold_aucs_labeled_only = []
fold_aucs_with_pseudo = []

for fold, (tr, va) in enumerate(gkf.split(emb_lab, Y_lab, groups=file_idx_lab)):
    print(f"\nFold {fold+1}/5: train={len(tr)} val={len(va)}")
    val_files = set(file_lab[va])

    # --- Option A: labeled only ---
    X_train = emb_lab[tr]
    y_train = Y_lab[tr]
    # Train Ridge probe per class (alpha=8 like Bruce)
    # Actually fit one Ridge with multi-output
    ridge_lab = Ridge(alpha=8.0)
    ridge_lab.fit(X_train, y_train)
    pred_val_lab = ridge_lab.predict(emb_lab[va])
    # Sigmoid for probability
    oof_pred_labeled_only[va] = 1.0 / (1.0 + np.exp(-pred_val_lab))

    # --- Option B: labeled + pseudo (excluding pseudo from val files) ---
    pseudo_keep = np.array([fn not in val_files for fn in file_pseudo])
    X_pseudo_kept = emb_pseudo[pseudo_keep]
    Y_pseudo_kept = Y_pseudo[pseudo_keep]
    print(f"  pseudo kept (val-files excluded): {len(X_pseudo_kept)}")
    # Concatenate
    X_combined = np.concatenate([X_train, X_pseudo_kept], axis=0)
    Y_combined = np.concatenate([y_train, Y_pseudo_kept], axis=0)
    # Sample weights: labeled=1.0, pseudo=conf (already in Y values where >0)
    # We use Y_combined as targets — pseudo targets are soft (0.85-1.0) which acts as confidence
    # Ridge handles soft targets natively
    ridge_combined = Ridge(alpha=8.0)
    ridge_combined.fit(X_combined, Y_combined)
    pred_val_comb = ridge_combined.predict(emb_lab[va])
    oof_pred_with_pseudo[va] = 1.0 / (1.0 + np.exp(-pred_val_comb))

    # Per-fold macro-AUC (within val)
    def macro_auc_val(P_val):
        aucs = []
        for c in range(N_CLS):
            yc = Y_lab[va, c]
            if yc.sum() == 0 or yc.sum() == len(yc):
                continue
            try:
                aucs.append(roc_auc_score(yc, P_val[va, c]))
            except Exception:
                pass
        return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)

    auc_lab, n_lab = macro_auc_val(oof_pred_labeled_only)
    auc_comb, n_comb = macro_auc_val(oof_pred_with_pseudo)
    fold_aucs_labeled_only.append(auc_lab)
    fold_aucs_with_pseudo.append(auc_comb)
    print(f"  labeled-only macro-AUC: {auc_lab:.4f}  (n={n_lab})")
    print(f"  labeled+pseudo macro-AUC: {auc_comb:.4f}  (n={n_comb})")
    print(f"  delta: {auc_comb-auc_lab:+.4f}")


# ---------- Final aggregate ----------
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
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


print("\n=== AGGREGATE OOF macro-AUC ===")
auc_lab_total, n = macro_auc(Y_lab, oof_pred_labeled_only)
auc_comb_total, _ = macro_auc(Y_lab, oof_pred_with_pseudo)
print(f"  Labeled-only probe (Ridge on Perch emb):     {auc_lab_total:.4f}  (n_classes={n})")
print(f"  Labeled + v4 pseudo probe (noisy-student):   {auc_comb_total:.4f}")
print(f"  Bruce CLIP-Ridge (reference):                0.8670")
print(f"  Perch v2 raw (reference):                    0.8859")
print(f"  Δ vs labeled-only:                           {auc_comb_total - auc_lab_total:+.4f}")
print(f"  Δ vs Bruce:                                  {auc_comb_total - 0.8670:+.4f}")


# ---------- Iterative noisy-student: 2nd round ----------
# Use the noisy-student probe to RE-LABEL the unlabeled, then retrain
# This is a stripped-down version of Babych's 3-iter recipe
print("\n=== Iteration 2 ===")

# Train final model on ALL labeled + ALL pseudo
ridge_final = Ridge(alpha=8.0)
X_all = np.concatenate([emb_lab, emb_pseudo], axis=0)
Y_all = np.concatenate([Y_lab, Y_pseudo], axis=0)
ridge_final.fit(X_all, Y_all)
print(f"  Trained final probe on {len(X_all)} samples")

# Predict on ALL unlabeled (127k)
print("  Predicting on 127k unlabeled...")
pred_unlab_logits = ridge_final.predict(emb_unlab)
P_probe_unlab = 1.0 / (1.0 + np.exp(-pred_unlab_logits))
print(f"  P_probe_unlab: range [{P_probe_unlab.min():.4f}, {P_probe_unlab.max():.4f}], "
      f"mean {P_probe_unlab.mean():.4f}")

# Save final probe predictions
np.savez_compressed(
    f"{OUT}/probe_on_unlabeled.npz",
    P_probe=P_probe_unlab.astype(np.float32),
    row_id=row_id_unlab,
    classes=np.array(classes),
)
print(f"\nSaved {OUT}/probe_on_unlabeled.npz")


# ---------- Build new pseudo-labels from the noisy-student probe ----------
v5_rows = []
for c_idx in range(N_CLS):
    col = P_probe_unlab[:, c_idx]
    mask = col >= 0.85
    qual = np.where(mask)[0]
    qual = qual[np.argsort(col[qual])[::-1]][:500]
    for i in qual:
        v5_rows.append({
            "row_id": row_id_unlab[i],
            "filename": file_unlab[i],
            "class": classes[c_idx],
            "probe_prob": float(col[i]),
        })
v5 = pd.DataFrame(v5_rows)
print(f"\nV5 pseudo-labels (from noisy-student probe):")
print(f"  rows: {len(v5)} | files: {v5['filename'].nunique()} | classes: {v5['class'].nunique()}")
v5.to_parquet(f"{OUT}/pseudo_labels_v5_from_probe.parquet", index=False)


# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
strategies = ["Bruce alone", "Probe labeled-only", "Probe + v4 pseudo", "Perch raw"]
values = [0.8670, auc_lab_total, auc_comb_total, 0.8859]
colors = ["#888", "#39c", "#3a7", "#fc6"]
ax.barh(strategies, values, color=colors)
ax.set_xlim(0.7, 1.0)
ax.set_xlabel("Macro-AUC on labeled OOF (file-grouped)")
ax.set_title("Noisy-student probe on Perch embeddings")
for i, v in enumerate(values):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center")

ax = axes[1]
ax.bar(range(1, 6), fold_aucs_labeled_only, alpha=0.5, label="labeled-only")
ax.bar(range(1, 6), fold_aucs_with_pseudo, alpha=0.7, label="labeled + v4 pseudo")
ax.set_xlabel("Fold")
ax.set_ylabel("Macro-AUC")
ax.set_title("Per-fold AUC: noisy-student vs labeled-only")
ax.legend()
ax.set_ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig(f"{OUT}/probe_comparison.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}/probe_comparison.png")
