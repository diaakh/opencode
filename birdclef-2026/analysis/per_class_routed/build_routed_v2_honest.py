"""HONEST per-class softmax-weighted blend across 11 models.

For each fold:
  1. Compute per-class AUC on TRAIN files only
  2. Build per-class softmax weights from those AUCs
  3. Apply weighted rank-blend on VAL files
  4. Aggregate → honest macro-AUC

This is the file-grouped version of the earlier 0.913 result that used
the full labeled set for both AUC computation and evaluation.
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import rankdata

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/per_class_routed"

# Load all models on labeled
br = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
Y = br["Y"]
file_lab = br["row_filename"]
N, C = Y.shape
emb_lab = br["embeddings"].astype(np.float32)

M = {
    "bruce": br["P_bruce"],
    "perch": 1.0 / (1.0 + np.exp(-br["P_perch_logits"])),
    "alexander": np.load(f"{REPO}/meta_analysis/alexander_labeled_predictions.npz")["P"],
}
b = np.load(f"{REPO}/meta_analysis/baiyuby_labeled_predictions.npz")
M["baiyuby_flipped"] = -b["P_max"]
M["baiyuby_att"] = b["P_att"]
M["long_convnextv2"] = np.load(f"{REPO}/meta_analysis/long_convnext_predictions.npz")["P"]
M["mauricio"] = np.load(f"{REPO}/meta_analysis/mauricio_labeled_predictions.npz")["P"]
M["snowflake_convnext"] = np.load(f"{REPO}/meta_analysis/snowflake_convnext_predictions.npz")["P"]
M["snowflake_efnetv2m"] = np.load(f"{REPO}/meta_analysis/snowflake_efnetv2m_predictions.npz")["P"]

# Load probe + KNN OOF predictions (already file-grouped)
probe = np.load(f"{REPO}/analysis/gbm_stacker/oof_predictions_v2.npz", allow_pickle=True)
M["probe"] = probe["P_probe"]
M["knn"] = probe["P_knn"]

print(f"Models: {list(M.keys())}")
print(f"N={N}, C={C}")


def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(P.shape[1]):
        col = P[:, c]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            R[valid, c] = rankdata(col[valid]) / valid.sum()
            R[~valid, c] = 0.5
        else:
            R[:, c] = 0.5
    return R


def macro_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0 or y[:, c].sum() == len(y):
            continue
        col = p[:, c]
        mask = ~np.isnan(col)
        if mask.sum() < 5 or y[mask, c].sum() == 0 or y[mask, c].sum() == mask.sum():
            continue
        try: aucs.append(roc_auc_score(y[mask, c], col[mask]))
        except: pass
    return float(np.mean(aucs)) if aucs else float("nan")


# Rank-normalize all
R = {n: rank_norm(P) for n, P in M.items()}

# File-grouped CV
file_idx = pd.Categorical(file_lab).codes
gkf = GroupKFold(n_splits=5)
P_routed_oof = np.zeros((N, C), dtype=np.float32)

print("\n=== File-grouped per-class softmax-weighted blend ===")
for fold, (tr, va) in enumerate(gkf.split(np.zeros(N), Y, groups=file_idx)):
    # Compute per-class per-model AUC on TR only
    n_models = len(M)
    cauc = np.full((n_models, C), np.nan)
    for mi, (name, P) in enumerate(M.items()):
        for c in range(C):
            if Y[tr, c].sum() == 0 or Y[tr, c].sum() == len(tr):
                continue
            col = P[tr, c]
            mask = ~np.isnan(col)
            if mask.sum() < 5 or Y[tr[mask], c].sum() == 0 or Y[tr[mask], c].sum() == mask.sum():
                continue
            try: cauc[mi, c] = roc_auc_score(Y[tr[mask], c], col[mask])
            except: pass

    # Per-class softmax weights (sharpness=5)
    for c in range(C):
        col = cauc[:, c]
        valid = ~np.isnan(col)
        if not valid.any():
            P_routed_oof[va, c] = 0.5
            continue
        a = col[valid] - col[valid].max()
        w = np.exp(5.0 * a)
        w = w / w.sum()
        # Weighted blend on VA
        contrib = np.zeros(len(va), dtype=np.float32)
        for k, mi in enumerate(np.where(valid)[0]):
            name = list(M.keys())[mi]
            contrib += w[k] * R[name][va, c]
        P_routed_oof[va, c] = contrib
    fold_auc = macro_auc(Y[va], P_routed_oof[va])
    print(f"  Fold {fold+1}: macro-AUC = {fold_auc:.4f}  (val={len(va)})")

honest_auc = macro_auc(Y, P_routed_oof)
print(f"\n=== Honest per-class softmax blend macro-AUC: {honest_auc:.4f} ===")
print(f"Reference (Bruce+KNN simple blend): 0.8874")
print(f"Reference (Bruce alone):            0.8670")
print(f"Reference (per-class oracle):       0.9132")
print(f"Reference (per-class oracle in-fold leak): 0.9128")

np.savez_compressed(f"{OUT}/honest_perclass_softmax.npz", P=P_routed_oof, Y=Y)
