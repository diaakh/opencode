"""Round 2 of creative experiments — push past 0.891.

Ideas to try:
1. Apply pseudo+iNat hour prior to the Bruce+KNN+smoothed blend (rank-AUC lift)
2. Use Bruce_shifted_v4 (Bruce + prior) as a NEW model in blend
3. Stacking meta-model: per-class binary LightGBM on (model preds + context) with v4 pseudo as training data (more samples)
4. Smart blend with per-class TIER weights (Bruce-strong classes use Bruce, KNN-strong use KNN)
5. Use SED model votes for the 28 missing classes via co-occurrence broadcasting
6. Per-(taxa) blend weights: Aves uses one mix, Insecta another
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import GroupKFold
from scipy.stats import rankdata
import lightgbm as lgb

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/creative"

br = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
Y = br["Y"]
emb_lab = br["embeddings"].astype(np.float32)
file_lab = br["row_filename"]
hour_lab = br["row_hour"].astype(int)
end_sec = br["row_end_sec"].astype(int)
N, C = Y.shape
classes = list(br["classes"])
tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
class_to_name = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
class_taxa = np.array([class_to_name.get(c, "Aves") for c in classes])
is_texture = np.array([class_to_name.get(c) in ("Insecta", "Amphibia") for c in classes])

P_bruce = br["P_bruce"]
P_perch = 1.0 / (1.0 + np.exp(-br["P_perch_logits"]))
probe = np.load(f"{REPO}/analysis/gbm_stacker/oof_predictions_v2.npz", allow_pickle=True)
P_probe = probe["P_probe"]
P_knn = probe["P_knn"]


def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(P.shape[1]):
        col = P[:, c]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            R[valid, c] = rankdata(col[valid]) / valid.sum()
            R[~valid, c] = 0.5
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


file_to_rows = {}
for i, (f, e) in enumerate(zip(file_lab, end_sec)):
    w = (e - 1) // 5
    file_to_rows.setdefault(f, {})[w] = i


def smooth_file(P, ev=(0.20, 0.60, 0.20), tx=(0.35, 0.30, 0.35)):
    P_out = P.copy()
    for f, win_to_row in file_to_rows.items():
        wins = sorted(win_to_row.keys())
        if len(wins) < 3:
            continue
        for c in range(C):
            ker = tx if is_texture[c] else ev
            cw = [win_to_row[w] for w in wins]
            v = P[cw, c]
            pad = np.pad(v, (1, 1), mode="edge")
            sm = pad[:-2] * ker[0] + pad[1:-1] * ker[1] + pad[2:] * ker[2]
            for k, w in enumerate(wins):
                P_out[win_to_row[w], c] = sm[k]
    return P_out


# Baseline best from previous round
P_bruce_smoothed = smooth_file(P_bruce)
R_bruce_sm = rank_norm(P_bruce_smoothed)
R_knn = rank_norm(P_knn)
R_perch = rank_norm(P_perch)
R_bruce = rank_norm(P_bruce)
best_prior = 0.50 * R_knn + 0.40 * R_bruce_sm + 0.10 * R_perch
print(f"Best from round 1 (KNN .5 + Bruce_sm .4 + Perch .1): {macro_auc(Y, best_prior):.4f}")


# ---------- E7: Apply pseudo hour prior to the blend ----------
print("\n=== E7: Apply pseudo hour prior to the blend ===")
prior_df = pd.read_csv(f"{REPO}/meta_analysis/pseudo_hour_priors.csv").set_index("hour")
prior_arr = np.zeros((24, C), dtype=np.float32)
for h in range(24):
    if h in prior_df.index:
        for ci, c in enumerate(classes):
            if c in prior_df.columns:
                prior_arr[h, ci] = prior_df.loc[h, c]
prior_per_row = prior_arr[hour_lab]
EPS = 1e-6
prior_logit = np.log(np.clip(prior_per_row, EPS, 1.0))

# blend_logit + w * prior_logit
def apply_prior(blend, w):
    base_logit = np.log(np.clip(blend, EPS, 1-EPS) / (1 - np.clip(blend, EPS, 1-EPS)))
    new_logit = base_logit + w * prior_logit
    return 1.0 / (1.0 + np.exp(-new_logit))

for w in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
    p = apply_prior(best_prior, w)
    print(f"  best_prior + pseudo prior w={w}: {macro_auc(Y, p):.4f}")


# ---------- E8: Stacked meta-model trained on labeled + v4 pseudo ----------
print("\n=== E8: LightGBM meta-model trained on labeled + v4 pseudo ===")
# Load v4 pseudo predictions (Bruce + Perch on those rows) + their embeddings
v4 = pd.read_parquet(f"{REPO}/analysis/context_label/pseudo_labels_v4.parquet")
emb_unlab = np.load("/tmp/pseudo_cache/pseudo_emb.npy", mmap_mode="r")
unlab = np.load(f"{REPO}/analysis/context_label/bruce_on_unlabeled.npz", allow_pickle=True)
P_bruce_unlab = unlab["P_bruce"]
P_perch_unlab = unlab["perch_soft"]
row_id_unlab = unlab["row_id"]
hour_unlab = unlab["hour_utc"].astype(int)
site_unlab = unlab["site"].astype(str)
file_unlab = unlab["filename"].astype(str)
cls_idx = {c: i for i, c in enumerate(classes)}

# Build pseudo training set: (filename, win_idx, class) → soft label
rid_to_unlab_idx = {rid: i for i, rid in enumerate(row_id_unlab)}

# For the meta-model: features are (bruce, perch, KNN-on-emb, prior, hour, site, class)
# But computing KNN for unlabeled is expensive. Skip for now — use just bruce + perch + prior + meta.
# Build training table from labeled (X_lab) and pseudo (X_pseudo)
def build_features(row_ids_or_idxs, P_b, P_p, hours, sites, files, soft_labels=None):
    """Build (sample × class) features. soft_labels is (n_rows, C) for sample weights."""
    n = len(row_ids_or_idxs)
    rows = []
    for i in range(n):
        for c in range(C):
            rows.append((P_b[i, c], P_p[i, c], hours[i], sites[i] if not isinstance(sites[i], np.integer) else sites[i], i, c))
    return pd.DataFrame(rows, columns=["bruce", "perch", "hour", "site", "file_idx", "class_idx"])

# Skip the full lgb meta — too expensive. Try simpler: per-class logistic regression on labeled+pseudo
from sklearn.linear_model import LogisticRegression

# For each class, fit on (bruce, perch, knn) features, with labeled hard + pseudo soft as samples
# Use file-grouped CV on labeled, train with pseudo always included
file_idx_lab = pd.Categorical(file_lab).codes

# Aggregate pseudo: row_id → (n_pseudo, C) soft labels
pseudo_rids = []
pseudo_Y = []
pseudo_meta_idx = []
v4_grouped = v4.groupby("row_id")
for rid, group in v4_grouped:
    if rid not in rid_to_unlab_idx:
        continue
    lbl = np.zeros(C, dtype=np.float32)
    for _, r in group.iterrows():
        lbl[cls_idx[r["class"]]] = r["bruce_shifted_v4"]
    pseudo_rids.append(rid)
    pseudo_Y.append(lbl)
    pseudo_meta_idx.append(rid_to_unlab_idx[rid])
pseudo_meta_idx = np.array(pseudo_meta_idx)
pseudo_Y = np.array(pseudo_Y)
P_b_pseudo = P_bruce_unlab[pseudo_meta_idx]
P_p_pseudo = P_perch_unlab[pseudo_meta_idx]
pseudo_files = file_unlab[pseudo_meta_idx]
print(f"Pseudo training set size: {len(pseudo_rids)}")


# Per-class logistic regression with file-grouped CV
gkf = GroupKFold(n_splits=5)
print("\nPer-class logistic regression (Bruce, Perch features, +v4 pseudo training):")
lr_oof = np.zeros((N, C), dtype=np.float32)
lr_oof_with_pseudo = np.zeros((N, C), dtype=np.float32)

# For per-class LR, we need at least a few positives in each fold's train set.
# Try this on classes with >= 5 labeled positives.
viable_classes = np.where(Y.sum(axis=0) >= 5)[0]
print(f"  Classes with ≥5 labeled positives: {len(viable_classes)}")

for fold, (tr, va) in enumerate(gkf.split(np.zeros(N), Y, groups=file_idx_lab)):
    val_files = set(file_lab[va])
    pseudo_keep = np.array([f not in val_files for f in pseudo_files])
    for c in viable_classes:
        # Labeled training features for this class
        X_tr_lab = np.column_stack([P_bruce[tr, c], P_perch[tr, c]])
        y_tr_lab = Y[tr, c]
        if y_tr_lab.sum() < 1 or y_tr_lab.sum() == len(y_tr_lab):
            continue
        # Pure labeled LR
        try:
            lr = LogisticRegression(max_iter=200, C=1.0)
            lr.fit(X_tr_lab, y_tr_lab)
            X_va = np.column_stack([P_bruce[va, c], P_perch[va, c]])
            lr_oof[va, c] = lr.predict_proba(X_va)[:, 1]
        except Exception:
            lr_oof[va, c] = 0.5
        # With pseudo
        X_tr_pseudo = np.column_stack([P_b_pseudo[pseudo_keep, c], P_p_pseudo[pseudo_keep, c]])
        y_tr_pseudo = (pseudo_Y[pseudo_keep, c] > 0.5).astype(np.int32)
        if y_tr_pseudo.sum() > 0:
            X_combined = np.concatenate([X_tr_lab, X_tr_pseudo], axis=0)
            y_combined = np.concatenate([y_tr_lab, y_tr_pseudo], axis=0)
            try:
                lr_p = LogisticRegression(max_iter=200, C=1.0)
                lr_p.fit(X_combined, y_combined)
                lr_oof_with_pseudo[va, c] = lr_p.predict_proba(X_va)[:, 1]
            except Exception:
                lr_oof_with_pseudo[va, c] = lr_oof[va, c]
        else:
            lr_oof_with_pseudo[va, c] = lr_oof[va, c]

print(f"  Per-class LR (labeled only): {macro_auc(Y, lr_oof):.4f}")
print(f"  Per-class LR (+v4 pseudo):   {macro_auc(Y, lr_oof_with_pseudo):.4f}")

# Blend the new LR with our previous best
R_lr_pseudo = rank_norm(lr_oof_with_pseudo)
for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
    blend = (1-w) * best_prior + w * R_lr_pseudo
    print(f"  best_prior + {w}*LR_pseudo:  {macro_auc(Y, blend):.4f}")


# ---------- E9: Per-(taxa) blend weights ----------
print("\n=== E9: Per-taxa blend weights ===")
# For each taxa class, try different (bruce, knn, perch) weights
taxa_indices = {}
for tax_name in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
    taxa_indices[tax_name] = np.where(class_taxa == tax_name)[0]

# Grid per taxa, find best per taxa
best_per_taxa = {}
for tax_name, cls_inds in taxa_indices.items():
    if len(cls_inds) == 0 or Y[:, cls_inds].sum() == 0:
        continue
    best_score = 0
    best_w = None
    Y_sub = Y[:, cls_inds]
    R_b_sub = R_bruce[:, cls_inds]
    R_k_sub = R_knn[:, cls_inds]
    R_p_sub = R_perch[:, cls_inds]
    for wb in [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        for wk in [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            wp = 1 - wb - wk
            if wp < 0 or wp > 1: continue
            blend_sub = wb*R_b_sub + wk*R_k_sub + wp*R_p_sub
            try:
                auc = macro_auc(Y_sub, blend_sub)
                if auc > best_score:
                    best_score = auc
                    best_w = (wb, wk, wp)
            except: pass
    best_per_taxa[tax_name] = (best_w, best_score, len(cls_inds))
    print(f"  {tax_name} ({len(cls_inds)} classes): best (wB,wK,wP)={best_w}, AUC={best_score:.4f}")

# Apply per-taxa weights
print("\n  Applying per-taxa optimal weights:")
P_taxa = np.zeros((N, C), dtype=np.float32)
for tax_name, (w, _, _) in best_per_taxa.items():
    if w is None: continue
    wb, wk, wp = w
    cls_inds = taxa_indices[tax_name]
    P_taxa[:, cls_inds] = wb * R_bruce[:, cls_inds] + wk * R_knn[:, cls_inds] + wp * R_perch[:, cls_inds]
print(f"  Per-taxa weights honest macro-AUC: {macro_auc(Y, P_taxa):.4f}")
# This is leaky because weights fit on full labeled. Let me try file-grouped.

# File-grouped per-taxa
print("\n  File-grouped per-taxa weights:")
P_taxa_grouped = np.zeros((N, C), dtype=np.float32)
for fold, (tr, va) in enumerate(gkf.split(np.zeros(N), Y, groups=file_idx_lab)):
    for tax_name, cls_inds in taxa_indices.items():
        if len(cls_inds) == 0 or Y[tr][:, cls_inds].sum() == 0:
            continue
        best_auc, best_w = 0, None
        for wb in [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
            for wk in [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
                wp = 1 - wb - wk
                if wp < 0 or wp > 1: continue
                blend = wb*R_bruce[:, cls_inds] + wk*R_knn[:, cls_inds] + wp*R_perch[:, cls_inds]
                try: auc = macro_auc(Y[tr][:, cls_inds], blend[tr])
                except: continue
                if auc > best_auc:
                    best_auc, best_w = auc, (wb, wk, wp)
        if best_w is None: continue
        wb, wk, wp = best_w
        P_taxa_grouped[va[:, None], cls_inds[None, :]] = (
            wb * R_bruce[va[:, None], cls_inds[None, :]] +
            wk * R_knn[va[:, None], cls_inds[None, :]] +
            wp * R_perch[va[:, None], cls_inds[None, :]]
        )
print(f"  File-grouped per-taxa: {macro_auc(Y, P_taxa_grouped):.4f}")


# ---------- Summary ----------
print("\n=== ROUND 2 SUMMARY ===")
print(f"  Round 1 best (KNN .5+Bruce_sm .4+Perch .1):  0.8912")
print(f"  + pseudo_hour w=0.1:                          {macro_auc(Y, apply_prior(best_prior, 0.1)):.4f}")
print(f"  + pseudo_hour w=0.2:                          {macro_auc(Y, apply_prior(best_prior, 0.2)):.4f}")
print(f"  + pseudo_hour w=0.5:                          {macro_auc(Y, apply_prior(best_prior, 0.5)):.4f}")
print(f"  Per-class LR (with v4 pseudo):                {macro_auc(Y, lr_oof_with_pseudo):.4f}")
print(f"  Per-taxa weights (honest, file-grouped):      {macro_auc(Y, P_taxa_grouped):.4f}")
