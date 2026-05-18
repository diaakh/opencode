"""LightGBM stacker v2 — add the noisy-student probe + KNN retrieval prior
as new features. See if non-linear interactions among them lift macro-AUC.

New features:
  - noisy_student_probe: prediction from labeled+pseudo-trained Ridge
  - knn_topK: max similarity-weighted label from top-K neighbors in emb space
  - probe_minus_bruce: feature engineering — disagreement between probe and Bruce
"""
import os, re, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/gbm_stacker"

# ---------- Reuse the v1 GBT setup ----------
labels = pd.read_csv("/tmp/bc26/train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)

def parse(fn):
    m = re.match(r"BC2026_Train_\d+_(S\d+)_(\d{8})_(\d{2})(\d{2})(\d{2})", fn)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (None, None, None)

labels[["site", "year_int", "hour"]] = labels["filename"].apply(lambda fn: pd.Series(parse(fn)))
labels["year"] = labels["year_int"].astype(int).apply(lambda d: d // 10000)

tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
cls_idx = {c: i for i, c in enumerate(classes)}
class_to_name = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
N, C = len(labels), len(classes)

br = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
emb_lab = br["embeddings"].astype(np.float32)
Y = br["Y"]
P_bruce = br["P_bruce"]
P_perch = 1.0 / (1.0 + np.exp(-br["P_perch_logits"]))

M = {"bruce": P_bruce, "perch": P_perch,
     "alexander": np.load(f"{REPO}/meta_analysis/alexander_labeled_predictions.npz")["P"]}
b = np.load(f"{REPO}/meta_analysis/baiyuby_labeled_predictions.npz")
M["baiyuby_flipped"] = -b["P_max"]
M["baiyuby_att"] = b["P_att"]
M["long_convnextv2"] = np.load(f"{REPO}/meta_analysis/long_convnext_predictions.npz")["P"]
M["mauricio"] = np.load(f"{REPO}/meta_analysis/mauricio_labeled_predictions.npz")["P"]
M["snowflake_convnext"] = np.load(f"{REPO}/meta_analysis/snowflake_convnext_predictions.npz")["P"]
M["snowflake_efnetv2m"] = np.load(f"{REPO}/meta_analysis/snowflake_efnetv2m_predictions.npz")["P"]


# ---------- NEW: train per-fold noisy-student probe and compute OOF probe predictions ----------
print("=== Building OOF noisy-student probe predictions ===")
file_lab = br["row_filename"]
unique_files_lab = sorted(set(file_lab))
file_idx_lab = np.array([sorted(unique_files_lab).index(f) for f in file_lab])

# Load pseudo labels + their embeddings (cache once)
v4 = pd.read_parquet(f"{REPO}/analysis/context_label/pseudo_labels_v4.parquet")
emb_unlab = np.load("/tmp/pseudo_cache/pseudo_emb.npy", mmap_mode="r")
unlab = np.load(f"{REPO}/analysis/context_label/bruce_on_unlabeled.npz", allow_pickle=True)
row_id_unlab = unlab["row_id"]
file_unlab = unlab["filename"]
rid_to_uidx = {rid: i for i, rid in enumerate(row_id_unlab)}

# Aggregate pseudo per row_id
pseudo_per_row = {}
for _, r in v4.iterrows():
    rid = r["row_id"]
    if rid not in rid_to_uidx:
        continue
    if rid not in pseudo_per_row:
        pseudo_per_row[rid] = {"labels": np.zeros(C, dtype=np.float32), "filename": r["filename"]}
    pseudo_per_row[rid]["labels"][cls_idx[r["class"]]] = r["bruce_shifted_v4"]
pseudo_rids = list(pseudo_per_row.keys())
emb_pseudo = np.array([emb_unlab[rid_to_uidx[rid]] for rid in pseudo_rids], dtype=np.float32)
Y_pseudo = np.array([pseudo_per_row[rid]["labels"] for rid in pseudo_rids], dtype=np.float32)
file_pseudo = np.array([pseudo_per_row[rid]["filename"] for rid in pseudo_rids])
print(f"  Pseudo train set: {emb_pseudo.shape}")

# Per-fold: train probe on labeled-tr + pseudo (excluding val-files), predict val
gkf = GroupKFold(n_splits=5)
probe_oof = np.zeros((N, C), dtype=np.float32)
for fold, (tr, va) in enumerate(gkf.split(emb_lab, Y, groups=file_idx_lab)):
    val_files = set(file_lab[va])
    pseudo_keep = np.array([fn not in val_files for fn in file_pseudo])
    X_train = np.concatenate([emb_lab[tr], emb_pseudo[pseudo_keep]], axis=0)
    y_train = np.concatenate([Y[tr], Y_pseudo[pseudo_keep]], axis=0)
    ridge = Ridge(alpha=8.0)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(emb_lab[va])
    probe_oof[va] = 1.0 / (1.0 + np.exp(-pred))
    print(f"  Fold {fold+1}/5: probe trained on {len(X_train)} samples")

M["noisy_student_probe"] = probe_oof
print(f"\nProbe macro-AUC alone:")
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

probe_auc, n_probe = macro_auc(Y, probe_oof)
print(f"  {probe_auc:.4f}  (n_classes={n_probe})")


# ---------- NEW: KNN retrieval prior ----------
# For each labeled window, find top-K nearest neighbors in emb_lab itself (LOFO)
# and use their labels as a prior. Cross-validate via file-grouped.
print("\n=== KNN retrieval prior (cosine sim in Perch emb space) ===")
from sklearn.preprocessing import normalize
emb_lab_n = normalize(emb_lab, axis=1)
K = 20
knn_pred = np.zeros((N, C), dtype=np.float32)
sim_matrix = emb_lab_n @ emb_lab_n.T  # (N, N) cosine sim
# Mask out same-file neighbors (within-file leakage)
file_arr = file_lab
same_file = np.array([[file_arr[i] == file_arr[j] for j in range(N)] for i in range(N)])
sim_matrix[same_file] = -np.inf
for i in range(N):
    top_k = np.argsort(sim_matrix[i])[::-1][:K]
    weights = sim_matrix[i, top_k]
    weights = np.maximum(weights, 0)
    weights /= weights.sum() if weights.sum() > 0 else 1.0
    knn_pred[i] = weights @ Y[top_k]
M["knn_retrieval"] = knn_pred
knn_auc, _ = macro_auc(Y, knn_pred)
print(f"  KNN-prior macro-AUC: {knn_auc:.4f}")


# ---------- Hour prior feature ----------
prior_df = pd.read_csv(f"{REPO}/analysis/external_priors/combined_hour_prior.csv").set_index("hour")
prior_arr = np.zeros((24, C), dtype=np.float32)
for h in range(24):
    if h in prior_df.index:
        for ci, c in enumerate(classes):
            if c in prior_df.columns:
                prior_arr[h, ci] = prior_df.loc[h, c]

hours_arr = labels["hour"].values.astype(int)
sites_arr = pd.Categorical(labels["site"]).codes
years_arr = labels["year"].values.astype(int)
filenames = labels["filename"].values
unique_files = sorted(set(filenames))
file_to_idx = {f: i for i, f in enumerate(unique_files)}
file_idx = np.array([file_to_idx[f] for f in filenames])


# ---------- Build feature matrix ----------
print(f"\nBuilding feature matrix...")
features = {}
for name, P in M.items():
    flat = P.flatten()
    flat = np.where(np.isnan(flat), 0.5, flat).astype(np.float32)
    features[f"pred_{name}"] = flat

# Engineered: probe vs bruce delta
features["probe_minus_bruce"] = features["pred_noisy_student_probe"] - features["pred_bruce"]
features["probe_x_bruce"] = features["pred_noisy_student_probe"] * features["pred_bruce"]
features["knn_x_bruce"] = features["pred_knn_retrieval"] * features["pred_bruce"]

# Categorical
features["class_idx"] = np.tile(np.arange(C, dtype=np.int32), N)
features["hour"] = np.repeat(hours_arr.astype(np.int32), C)
features["site"] = np.repeat(sites_arr.astype(np.int32), C)
features["year"] = np.repeat(years_arr.astype(np.int32), C)

# Prior
prior_per_row = prior_arr[hours_arr]
features["prior_combined"] = prior_per_row.flatten().astype(np.float32)
features["prior_combined_log"] = np.log(np.clip(prior_per_row.flatten(), 1e-6, None)).astype(np.float32)

# Class metadata
bruce_per_class_auc = np.full(C, np.nan)
for c in range(C):
    if 0 < Y[:, c].sum() < N:
        col = P_bruce[:, c]
        mask = ~np.isnan(col)
        if mask.sum() > 5 and 0 < Y[mask, c].sum() < mask.sum():
            try:
                bruce_per_class_auc[c] = roc_auc_score(Y[mask, c], col[mask])
            except Exception:
                pass
features["bruce_class_auc"] = np.tile(np.where(np.isnan(bruce_per_class_auc), 0.5, bruce_per_class_auc).astype(np.float32), N)

class_taxa_map = {"Aves": 0, "Amphibia": 1, "Insecta": 2, "Mammalia": 3, "Reptilia": 4}
class_taxa = np.array([class_taxa_map.get(class_to_name.get(c), -1) for c in classes], dtype=np.int32)
features["class_taxa"] = np.tile(class_taxa, N)

X = pd.DataFrame(features)
y = Y.flatten().astype(np.int32)
file_idx_per_sample = np.repeat(file_idx, C)
valid_class_mask = np.tile(Y.sum(axis=0) > 0, N)
X = X[valid_class_mask].reset_index(drop=True)
y = y[valid_class_mask]
file_idx_per_sample = file_idx_per_sample[valid_class_mask]
print(f"X shape: {X.shape}, pos rate: {y.mean():.4f}")


# ---------- Train ----------
cat_features = ["class_idx", "hour", "site", "year", "class_taxa"]
for c in cat_features:
    X[c] = X[c].astype("category")

params = dict(objective="binary", metric="auc", learning_rate=0.05,
              num_leaves=63, max_depth=-1, min_child_samples=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
              is_unbalance=True, verbose=-1)

print("\n=== Training LightGBM v2 (with probe + KNN) ===")
oof_pred = np.zeros(len(y), dtype=np.float32)
fold_aucs = []
last_fi = None
for fold, (tr, va) in enumerate(gkf.split(X, y, groups=file_idx_per_sample)):
    train_data = lgb.Dataset(X.iloc[tr], label=y[tr], categorical_feature=cat_features)
    val_data = lgb.Dataset(X.iloc[va], label=y[va], categorical_feature=cat_features, reference=train_data)
    model = lgb.train(params, train_data, num_boost_round=600, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)])
    oof_pred[va] = model.predict(X.iloc[va], num_iteration=model.best_iteration)
    fold_aucs.append(roc_auc_score(y[va], oof_pred[va]))
    last_fi = pd.DataFrame({"feature": X.columns, "gain": model.feature_importance(importance_type="gain")}).sort_values("gain", ascending=False)
    print(f"  Fold {fold+1}: AUC={fold_aucs[-1]:.4f}")

print(f"\nMean fold-AUC: {np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}")
print(f"\nTop 15 features (last fold):")
print(last_fi.head(15).to_string())

# Per-class macro-AUC
oof_2d = np.full(N * C, 0.5, dtype=np.float32)
oof_2d[valid_class_mask] = oof_pred
oof_2d = oof_2d.reshape(N, C)
lgb_macro, _ = macro_auc(Y, oof_2d)
print(f"\n=== Honest macro-AUC comparison ===")
print(f"  Bruce alone:                   0.8670")
print(f"  Perch raw (46 classes):        0.8859")
print(f"  Noisy-student probe alone:     {probe_auc:.4f}")
print(f"  KNN retrieval alone:           {knn_auc:.4f}")
print(f"  LGBM v1 stacker:               0.8639")
print(f"  LGBM v2 (+probe+knn):          {lgb_macro:.4f}")
print(f"  Δ v2 over v1:                  {lgb_macro - 0.8639:+.4f}")

# Save
np.savez_compressed(f"{OUT}/oof_predictions_v2.npz", P_lgb=oof_2d, Y=Y,
                    P_probe=probe_oof, P_knn=knn_pred)
import json
with open(f"{OUT}/summary_v2.json", "w") as f:
    json.dump({
        "fold_aucs": [float(x) for x in fold_aucs],
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "lgb_macro_auc": float(lgb_macro),
        "probe_macro_auc": float(probe_auc),
        "knn_macro_auc": float(knn_auc),
    }, f, indent=2)
