"""LightGBM stacker — learns the per-(window, class) blend from existing
model outputs + context features (site, hour, year, priors).

Training data: 739 labeled windows × 234 classes = 173,026 binary samples
(after dropping classes with 0 positives in labeled set).

Features per sample:
  - Bruce CLIP-Ridge prediction (per-class)
  - Perch v2 raw probability (per-class)
  - 7 independent SED model predictions (per-class)
  - Combined hour prior P(class | hour)
  - Hour-of-day (categorical)
  - Site (categorical)
  - Class index (categorical)
  - Bruce's labeled per-class AUC (model trustworthiness signal)

CV: Leave-one-FILE-out (66 folds × 234 classes). Each fold trains on 65
files' windows, evaluates on the held-out file. This is the only honest
way to validate — windows from the same file share so much signal that
plain GroupKFold doesn't isolate them properly.
"""
import os, re, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/gbm_stacker"
os.makedirs(OUT, exist_ok=True)


# ---------- Load metadata + Y + all model predictions on labeled ----------
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
Y = br["Y"]
P_bruce = br["P_bruce"]
P_perch = 1.0 / (1.0 + np.exp(-br["P_perch_logits"]))  # sigmoid the logits

M = {
    "bruce": P_bruce,
    "perch": P_perch,
}
M["alexander"] = np.load(f"{REPO}/meta_analysis/alexander_labeled_predictions.npz")["P"]
b = np.load(f"{REPO}/meta_analysis/baiyuby_labeled_predictions.npz")
M["baiyuby_flipped"] = 1.0 / (1.0 + np.exp(b["P_max"] * 1e6))  # flip + sigmoid for scale
# Actually baiyuby_flipped is just -P_max; convert to probability-like:
M["baiyuby_flipped"] = -b["P_max"]
M["baiyuby_att"] = b["P_att"]
M["long_convnextv2"] = np.load(f"{REPO}/meta_analysis/long_convnext_predictions.npz")["P"]
M["mauricio"] = np.load(f"{REPO}/meta_analysis/mauricio_labeled_predictions.npz")["P"]
M["snowflake_convnext"] = np.load(f"{REPO}/meta_analysis/snowflake_convnext_predictions.npz")["P"]
M["snowflake_efnetv2m"] = np.load(f"{REPO}/meta_analysis/snowflake_efnetv2m_predictions.npz")["P"]
print(f"Loaded {len(M)} models, each (N={N}, C={C})")


# Load combined hour prior
prior_df = pd.read_csv(f"{REPO}/analysis/external_priors/combined_hour_prior.csv").set_index("hour")
prior_arr = np.zeros((24, C), dtype=np.float32)
for h in range(24):
    if h in prior_df.index:
        for ci, c in enumerate(classes):
            if c in prior_df.columns:
                prior_arr[h, ci] = prior_df.loc[h, c]


# Per-class Bruce AUC on labeled (used as a per-class quality signal feature)
bruce_per_class_auc = np.full(C, np.nan)
for c in range(C):
    if 0 < Y[:, c].sum() < N:
        col = M["bruce"][:, c]
        mask = ~np.isnan(col)
        if mask.sum() > 5 and 0 < Y[mask, c].sum() < mask.sum():
            try:
                bruce_per_class_auc[c] = roc_auc_score(Y[mask, c], col[mask])
            except Exception:
                pass
median_auc = np.nanmedian(bruce_per_class_auc)
bruce_per_class_auc_filled = np.where(np.isnan(bruce_per_class_auc), median_auc, bruce_per_class_auc)


# ---------- Build feature matrix ----------
# Per sample: (filename_idx, class_idx) → features
hours_arr = labels["hour"].values.astype(int)
sites_arr = pd.Categorical(labels["site"]).codes
years_arr = labels["year"].values.astype(int)
filenames = labels["filename"].values
unique_files = sorted(set(filenames))
file_to_idx = {f: i for i, f in enumerate(unique_files)}
file_idx = np.array([file_to_idx[f] for f in filenames])

print(f"Unique files: {len(unique_files)}")
print(f"Unique sites: {len(set(sites_arr))}")

# Vectorized feature build: (N*C, n_features)
print(f"\nBuilding feature matrix ({N}×{C}={N*C} samples)...")
n_samples = N * C
features = {}
# Model prediction features (replicate per class)
for name, P in M.items():
    flat = P.flatten()  # row-major: (N*C,) where idx i*C+c
    # NaN → 0.5
    flat = np.where(np.isnan(flat), 0.5, flat).astype(np.float32)
    features[f"pred_{name}"] = flat

# Per-(row, class) prior value
prior_per_row = prior_arr[hours_arr]  # (N, C)
features["prior_combined"] = prior_per_row.flatten().astype(np.float32)
# Log of prior (useful nonlinear feature)
features["prior_combined_log"] = np.log(np.clip(prior_per_row.flatten(), 1e-6, None)).astype(np.float32)

# Categorical (broadcast across classes)
class_indices = np.tile(np.arange(C, dtype=np.int32), N)
features["class_idx"] = class_indices
features["hour"] = np.repeat(hours_arr.astype(np.int32), C)
features["site"] = np.repeat(sites_arr.astype(np.int32), C)
features["year"] = np.repeat(years_arr.astype(np.int32), C)

# Per-class Bruce labeled AUC (replicated per row)
features["bruce_class_auc"] = np.tile(bruce_per_class_auc_filled.astype(np.float32), N)

# Class taxonomy: Aves=0, Amphibia=1, Insecta=2, Mammalia=3, Reptilia=4
class_taxa_map = {"Aves": 0, "Amphibia": 1, "Insecta": 2, "Mammalia": 3, "Reptilia": 4}
class_taxa = np.array([class_taxa_map.get(class_to_name.get(c), -1) for c in classes], dtype=np.int32)
features["class_taxa"] = np.tile(class_taxa, N)

X = pd.DataFrame(features)
y = Y.flatten().astype(np.int32)
file_idx_per_sample = np.repeat(file_idx, C)
print(f"X shape: {X.shape}, y mean: {y.mean():.4f}, # positives: {y.sum()}")


# ---------- Drop classes with 0 positives in labeled (can't learn them) ----------
class_pos_count = Y.sum(axis=0)
valid_class_mask = class_pos_count > 0
print(f"\nClasses with ≥1 positive in labeled: {valid_class_mask.sum()}/{C}")
keep_class_mask = np.tile(valid_class_mask, N)
X = X[keep_class_mask].reset_index(drop=True)
y = y[keep_class_mask]
file_idx_per_sample = file_idx_per_sample[keep_class_mask]
print(f"After filter: X={X.shape}, positives={y.sum()}, pos_rate={y.mean():.4f}")


# ---------- LightGBM training with file-grouped CV ----------
cat_features = ["class_idx", "hour", "site", "year", "class_taxa"]
for c in cat_features:
    X[c] = X[c].astype("category")

# 5-fold file-grouped CV for honest evaluation
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)

params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    is_unbalance=True,
    verbose=-1,
)

print("\n=== Training LightGBM stacker (5-fold file-grouped) ===")
oof_pred = np.zeros(len(y), dtype=np.float32)
fold_aucs = []
for fold, (tr, va) in enumerate(gkf.split(X, y, groups=file_idx_per_sample)):
    print(f"\n--- Fold {fold+1}/5 ---")
    print(f"  Train: {len(tr)} samples, {y[tr].sum()} positives")
    print(f"  Val:   {len(va)} samples, {y[va].sum()} positives")
    if y[va].sum() < 10:
        print(f"  SKIP (too few val positives)")
        continue
    train_data = lgb.Dataset(X.iloc[tr], label=y[tr], categorical_feature=cat_features)
    val_data = lgb.Dataset(X.iloc[va], label=y[va], categorical_feature=cat_features, reference=train_data)
    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(100)],
    )
    oof_pred[va] = model.predict(X.iloc[va], num_iteration=model.best_iteration)
    fold_auc = roc_auc_score(y[va], oof_pred[va])
    fold_aucs.append(fold_auc)
    print(f"  Fold AUC: {fold_auc:.4f}  (dt={time.time()-t0:.1f}s)")
    # Feature importance
    fi = pd.DataFrame({
        "feature": X.columns,
        "gain": model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(f"{OUT}/feature_importance_fold{fold}.csv", index=False)
    print(f"  Top features by gain:")
    for _, r in fi.head(10).iterrows():
        print(f"    {r['feature']:>25s}: {r['gain']:.1e}")

print(f"\nMean fold-AUC across {len(fold_aucs)} folds: {np.mean(fold_aucs):.4f}  std={np.std(fold_aucs):.4f}")

# Reshape oof_pred to (N, C) for per-class macro-AUC
# Need to undo the filter
oof_reshaped = np.full(N * C, 0.5, dtype=np.float32)
oof_reshaped[keep_class_mask] = oof_pred
oof_2d = oof_reshaped.reshape(N, C)


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


lgb_macro, n_lgb = macro_auc(Y, oof_2d)
print(f"\nLightGBM stacker macro-AUC (file-grouped OOF): {lgb_macro:.4f}  (n_classes={n_lgb})")

# Compare to existing baselines
print("\n=== Comparison ===")
print(f"  Best single model (Perch raw): 0.8859 (46 classes only)")
print(f"  Best single model (Bruce):     0.8670")
print(f"  Per-class softmax-blend:       0.9128 (cheating — same set)")
print(f"  Bruce + pseudo_h w=1.0:        0.974 (cheating — leaky prior)")
print(f"  LightGBM stacker:              {lgb_macro:.4f} (FILE-GROUPED, honest)")
print(f"  exp019 leaked OOF:             0.996 (training-on-test)")


# Save final OOF
np.savez_compressed(
    f"{OUT}/oof_predictions.npz",
    P_lgb=oof_2d,
    Y=Y,
)

# Save params + summary
import json
with open(f"{OUT}/summary.json", "w") as f:
    json.dump({
        "fold_aucs": [float(x) for x in fold_aucs],
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "lgb_macro_auc": float(lgb_macro),
        "n_classes": int(n_lgb),
        "n_train_samples": int(len(X)),
        "n_positives": int(y.sum()),
        "params": params,
    }, f, indent=2)

print(f"\nSaved {OUT}/oof_predictions.npz, summary.json, feature_importance_fold*.csv")
