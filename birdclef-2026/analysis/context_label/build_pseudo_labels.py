"""Context-aware pseudo-labeler for the 10,592 unlabeled train_soundscapes.

Step 1: Run Bruce CLIP-Ridge on the 127k Perch embeddings (local, no GPU).
Step 2: Compute per-(site, hour) per-model AUC from labeled OOF to derive
        a routing function (model_blend_weights | site, hour).
Step 3: Apply per-context blend to unlabeled predictions.
Step 4: Save pseudo_labels.parquet with confidence scores.
"""
import os, re, pickle, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/context_label"
os.makedirs(OUT, exist_ok=True)

PSEUDO = Path("/tmp/pseudo_cache")
BRUCE_PKL = Path("/tmp/bruce_bundle/clip_student_bundle.pkl")

# ---------- 1. Load taxonomy + labeled metadata ----------
tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
cls_idx = {c: i for i, c in enumerate(classes)}
N_CLS = len(classes)

labels = pd.read_csv("/tmp/bc26/train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)


def parse_filename(fn):
    m = re.match(r"BC2026_Train_\d+_(S\d+)_(\d{8})_(\d{2})(\d{2})(\d{2})", fn)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)


labels[["site", "date", "hour"]] = labels["filename"].apply(
    lambda fn: pd.Series(parse_filename(fn))
)


# ---------- 2. Load pseudo cache (Perch on 127k unlabeled windows) ----------
print("Loading pseudo cache...")
emb = np.load(PSEUDO / "pseudo_emb.npy")  # (127104, 1536) fp16
scores = np.load(PSEUDO / "pseudo_scores.npy")  # (127104, 234) fp16, raw Perch logits in BC2026 order
soft = np.load(PSEUDO / "pseudo_soft.npy")  # (127104, 234) fp16, sigmoid'd
meta = pd.read_parquet(PSEUDO / "pseudo_meta.parquet")
N_PSEUDO = len(meta)
print(f"  pseudo cache: {N_PSEUDO} windows, {emb.shape[1]}-d emb, {scores.shape[1]} classes")


# Parse year + date from filename for additional context
def parse_year(fn):
    m = re.match(r"BC2026_Train_\d+_S\d+_(\d{4})", fn)
    return int(m.group(1)) if m else None


meta["year"] = meta["filename"].apply(parse_year)


# ---------- 3. Run Bruce CLIP-Ridge on the 127k embeddings ----------
print("\nLoading Bruce bundle...")
with open(BRUCE_PKL, "rb") as f:
    bundle = pickle.load(f)
clip_bundle = bundle["clip_bundle"]
emb_scaler = clip_bundle["emb_scaler"]
pca = clip_bundle["pca"]
feat_scaler = clip_bundle["feature_scaler"]
ridge = clip_bundle["model"]
primary_labels_bundle = list(bundle["primary_labels"])
class_to_bc = bundle["class_to_bc"]
print(f"  Bundle PCA dim: {pca.n_components_}")
print(f"  Bundle primary_labels: {len(primary_labels_bundle)}")
print(f"  Bundle covers Perch-mapped classes: "
      f"{sum(1 for c in primary_labels_bundle if not (isinstance(class_to_bc.get(str(c)), float) and np.isnan(class_to_bc[str(c)])))}/{len(primary_labels_bundle)}")


# Build BC2026 class order alignment between pseudo_scores and primary_labels_bundle
# (pseudo_scores should be in BC2026 taxonomy order — verify)
# Build src → tax mapping
bundle_to_tax = np.array([cls_idx.get(str(c), -1) for c in primary_labels_bundle])
print(f"  bundle → taxonomy mapping: {(bundle_to_tax >= 0).sum()}/{len(primary_labels_bundle)} matched")

# The pseudo_scores has 234 cols. If they're in taxonomy order, we re-order to
# bundle order before feeding Bruce. Otherwise leave as-is.
# We need to know which order — let's assume taxonomy order (matches our other npz files).
# Then bundle wants its own order, so we map: scores[:, bundle_to_tax]
scores_in_bundle_order = scores[:, bundle_to_tax].astype(np.float32)
print(f"  scores_in_bundle_order: {scores_in_bundle_order.shape}")


# Apply Bruce pipeline
print("\nApplying Bruce pipeline...")
t0 = time.time()

# 3a. Scale + PCA the embeddings
emb_f32 = emb.astype(np.float32)
emb_s = emb_scaler.transform(emb_f32)
print(f"  emb_scaler done ({time.time()-t0:.1f}s)")
emb_pca = pca.transform(emb_s)
print(f"  pca done: shape={emb_pca.shape} ({time.time()-t0:.1f}s)")

# 3b. Concat with bundle-ordered raw Perch logits
features = np.concatenate([emb_pca, scores_in_bundle_order], axis=1)
features = feat_scaler.transform(features)
print(f"  features built: shape={features.shape} ({time.time()-t0:.1f}s)")

# 3c. Run ridge (in batches to be safe)
BATCH = 8192
bruce_logits = np.empty((N_PSEUDO, len(primary_labels_bundle)), dtype=np.float32)
for i in range(0, N_PSEUDO, BATCH):
    bruce_logits[i : i + BATCH] = ridge.predict(features[i : i + BATCH])
print(f"  ridge done ({time.time()-t0:.1f}s)")

# 3d. Reorder bundle→taxonomy order, sigmoid
bruce_logits_tax = np.full((N_PSEUDO, N_CLS), np.nan, dtype=np.float32)
for src_idx, tax_idx in enumerate(bundle_to_tax):
    if tax_idx >= 0:
        bruce_logits_tax[:, tax_idx] = bruce_logits[:, src_idx]
P_bruce = 1.0 / (1.0 + np.exp(-bruce_logits_tax))
print(f"  Bruce probabilities: shape={P_bruce.shape}, range=[{np.nanmin(P_bruce):.4f}, {np.nanmax(P_bruce):.4f}]")
print(f"Total Bruce inference time: {time.time()-t0:.1f}s")

# Save Bruce predictions on unlabeled
np.savez_compressed(
    f"{OUT}/bruce_on_unlabeled.npz",
    P_bruce=P_bruce.astype(np.float32),
    perch_soft=soft.astype(np.float32),
    perch_scores=scores.astype(np.float32),
    row_id=meta["row_id"].values,
    filename=meta["filename"].values,
    site=meta["site"].values,
    hour_utc=meta["hour_utc"].values.astype(np.int8),
    year=meta["year"].values.astype(np.int16),
    classes=np.array(classes),
)
print(f"\nSaved {OUT}/bruce_on_unlabeled.npz")


# ---------- 4. Build per-(site, hour) per-model AUC table from labeled OOF ----------
print("\n=== Building per-(site, hour) per-model AUC routing table ===")

# Reload labels for Y
labels["end_sec"] = labels["end"].apply(
    lambda t: int(str(t).split(":")[0]) * 3600 + int(str(t).split(":")[1]) * 60 + int(str(t).split(":")[2])
)
labels["win_idx"] = labels["end_sec"].apply(lambda s: (s - 1) // 5)
hours_lab = labels["hour"].to_numpy()
sites_lab = labels["site"].to_numpy()
N_LAB = len(labels)
Y = np.zeros((N_LAB, N_CLS), dtype=np.float32)
for i, lab in enumerate(labels["primary_label"]):
    for c in str(lab).split(";"):
        if c in cls_idx:
            Y[i, cls_idx[c]] = 1.0

br_lab = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
M_lab = {
    "perch_soft": 1.0 / (1.0 + np.exp(-br_lab["P_perch_logits"])),  # sigmoid Perch logits
    "bruce": br_lab["P_bruce"],
    "alexander_b0": np.load(f"{REPO}/meta_analysis/alexander_labeled_predictions.npz")["P"],
}
b = np.load(f"{REPO}/meta_analysis/baiyuby_labeled_predictions.npz")
M_lab["baiyuby_flipped"] = -b["P_max"]
M_lab["baiyuby_att"] = b["P_att"]
M_lab["long_convnextv2"] = np.load(f"{REPO}/meta_analysis/long_convnext_predictions.npz")["P"]
M_lab["mauricio_b0"] = np.load(f"{REPO}/meta_analysis/mauricio_labeled_predictions.npz")["P"]


def per_class_auc(y, p):
    out = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0 or y[:, c].sum() == len(y):
            continue
        col = p[:, c]
        mask = ~np.isnan(col)
        if mask.sum() < 5 or y[mask, c].sum() == 0 or y[mask, c].sum() == mask.sum():
            continue
        try:
            out[c] = roc_auc_score(y[mask, c], col[mask])
        except Exception:
            pass
    return out


# Global per-class AUC by model (used as default in unseen contexts)
global_cauc = {n: per_class_auc(Y, P) for n, P in M_lab.items()}

# Per-(site, hour) per-class AUCs (where we have data)
print(f"  Building (site,hour) per-class AUCs for {len(M_lab)} models...")
cell_cauc = {}  # (site, hour) -> {model -> per_class_auc_array}
for s in sorted(set(sites_lab)):
    for h in sorted(set(hours_lab)):
        mask = (sites_lab == s) & (hours_lab == h)
        if mask.sum() < 6:
            continue
        cell_cauc[(s, h)] = {}
        for name, P in M_lab.items():
            cell_cauc[(s, h)][name] = per_class_auc(Y[mask], P[mask])
print(f"  populated cells: {len(cell_cauc)}")


# ---------- 5. Build routing function ----------
# Strategy: for (site, hour) cell with data, use per-class winner from that cell.
# For uncovered cells, fall back to (hour-only) winner, then global winner.
# Build a (n_models, n_classes) weight matrix per cell.

# Hour-only fallback
hour_cauc = {}
for h in sorted(set(hours_lab)):
    mask = hours_lab == h
    if mask.sum() < 10:
        continue
    hour_cauc[h] = {n: per_class_auc(Y[mask], P[mask]) for n, P in M_lab.items()}

# Site-only fallback
site_cauc = {}
for s in sorted(set(sites_lab)):
    mask = sites_lab == s
    if mask.sum() < 20:
        continue
    site_cauc[s] = {n: per_class_auc(Y[mask], P[mask]) for n, P in M_lab.items()}


def auc_to_weights(auc_dict, sharpness=5.0):
    """Convert per-class AUCs (model→array) to per-class softmax weights."""
    model_names = list(auc_dict.keys())
    # auc_arr: (n_models, n_classes)
    auc_arr = np.array([auc_dict[m] for m in model_names])
    weights = np.full(auc_arr.shape, 1.0 / len(model_names), dtype=np.float32)
    for c in range(auc_arr.shape[1]):
        col = auc_arr[:, c]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        a = col[valid] - col[valid].max()
        w = np.exp(sharpness * a)
        w /= w.sum()
        weights[valid, c] = w
        weights[~valid, c] = 0.0
    return model_names, weights


# Decide routing per cell, with fallback chain
def get_routing(site, hour):
    """Return (model_names_list, (n_models, n_classes) weight matrix, source_label)."""
    if (site, hour) in cell_cauc:
        names, weights = auc_to_weights(cell_cauc[(site, hour)])
        return names, weights, f"site_hour({site},{hour})"
    if hour in hour_cauc:
        names, weights = auc_to_weights(hour_cauc[hour])
        return names, weights, f"hour({hour})"
    if site in site_cauc:
        names, weights = auc_to_weights(site_cauc[site])
        return names, weights, f"site({site})"
    names, weights = auc_to_weights(global_cauc)
    return names, weights, "global"


print("\n=== Routing decisions per unique unlabeled cell ===")
print("(showing first 20)")
unique_unlabeled_cells = sorted(set(zip(meta["site"], meta["hour_utc"])))
routing_summary = []
for s, h in unique_unlabeled_cells[:20]:
    names, weights, src = get_routing(s, h)
    routing_summary.append({"site": s, "hour": h, "source": src})
    print(f"  ({s}, h={h}): routing from {src}")
print(f"  ... + {len(unique_unlabeled_cells)-20} more cells")


# ---------- 6. Apply context-aware blend to unlabeled ----------
print("\n=== Applying per-context blend to 127k unlabeled windows ===")

# We have local predictions for: perch_soft, bruce. We DON'T have alexander/baiyuby/etc
# applied to unlabeled (would require running them). For now, blend only what we have.
# If alex/baiyuby/long_convnext/mauricio aren't available, drop them from weights.

available_models_on_unlabeled = {"perch_soft", "bruce"}

P_routed = np.zeros((N_PSEUDO, N_CLS), dtype=np.float32)
# Need (N_PSEUDO, 234) per model
P_unlabeled = {
    "perch_soft": soft.astype(np.float32),
    "bruce": P_bruce,
}

# Pre-compute per-class rank-norm for available models (light memoization)
print("  Rank-normalizing predictions per class...")
from scipy.stats import rankdata
R_unlabeled = {}
for name, P in P_unlabeled.items():
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(N_CLS):
        col = P[:, c]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            R[valid, c] = rankdata(col[valid]) / valid.sum()
            R[~valid, c] = 0.5
        else:
            R[:, c] = 0.5
    R_unlabeled[name] = R

# Per-context blend
print("  Applying blend per row...")
row_source = np.empty(N_PSEUDO, dtype=object)
for ci, (s, h) in enumerate(unique_unlabeled_cells):
    names, weights, src = get_routing(s, h)
    row_mask = (meta["site"].values == s) & (meta["hour_utc"].values == h)
    if not row_mask.any():
        continue
    # Filter weights to models we have on unlabeled
    model_idxs = [i for i, n in enumerate(names) if n in available_models_on_unlabeled]
    if not model_idxs:
        continue
    used_names = [names[i] for i in model_idxs]
    # Re-normalize weights over available models
    w_subset = weights[model_idxs]  # (n_used, 234)
    w_subset_sum = w_subset.sum(axis=0, keepdims=True)
    w_subset_sum = np.where(w_subset_sum > 0, w_subset_sum, 1.0)
    w_subset = w_subset / w_subset_sum
    # Blend
    contrib = np.zeros((row_mask.sum(), N_CLS), dtype=np.float32)
    for i, name in enumerate(used_names):
        contrib += w_subset[i] * R_unlabeled[name][row_mask]
    P_routed[row_mask] = contrib
    row_source[row_mask] = src
print(f"  Done. Source distribution:")
for src, count in pd.Series(row_source).value_counts().head(15).items():
    print(f"    {src}: {count}")


# ---------- 7. Confidence-based filtering + save ----------
# Per-class top-K with prob >= threshold (samuelzxu approach)
print("\n=== Building filtered pseudo-labels ===")
TOP_K_PER_CLASS = 100
MIN_PROB = 0.50  # since P_routed is rank-normalized, this picks top-50%
MIN_PROB_FOR_LABEL = 0.85  # high-confidence threshold for "pseudo-label"

# Build per-class confident labels
rows_out = []
for c_idx in range(N_CLS):
    col = P_routed[:, c_idx]
    # top-K
    top_k_idx = np.argsort(col)[::-1][:TOP_K_PER_CLASS]
    for i in top_k_idx:
        if col[i] < MIN_PROB:
            break
        rows_out.append(
            {
                "row_id": meta["row_id"].iloc[i],
                "filename": meta["filename"].iloc[i],
                "site": meta["site"].iloc[i],
                "hour": int(meta["hour_utc"].iloc[i]),
                "year": int(meta["year"].iloc[i]) if not pd.isna(meta["year"].iloc[i]) else -1,
                "class": classes[c_idx],
                "prob_routed": float(col[i]),
                "perch_soft": float(soft[i, c_idx]),
                "bruce": float(P_bruce[i, c_idx]) if not np.isnan(P_bruce[i, c_idx]) else float("nan"),
                "routing_source": row_source[i],
                "high_confidence": col[i] >= MIN_PROB_FOR_LABEL,
            }
        )

labels_df = pd.DataFrame(rows_out)
print(f"  built {len(labels_df)} per-class pseudo-labels")
print(f"  unique files: {labels_df['filename'].nunique()}")
print(f"  unique classes: {labels_df['class'].nunique()}")
print(f"  high-confidence: {labels_df['high_confidence'].sum()}")

# Per-class summary
class_summary = labels_df.groupby("class").agg(
    n_labels=("row_id", "count"),
    n_high_conf=("high_confidence", "sum"),
    mean_prob=("prob_routed", "mean"),
    n_files=("filename", "nunique"),
).sort_values("n_labels", ascending=False)
print(f"\n  Top-15 classes by pseudo-label count:")
print(class_summary.head(15).to_string())

labels_df.to_parquet(f"{OUT}/pseudo_labels.parquet", index=False)
class_summary.to_csv(f"{OUT}/per_class_summary.csv")
print(f"\nSaved {OUT}/pseudo_labels.parquet")
print(f"Saved {OUT}/per_class_summary.csv")


# ---------- 8. Sanity-check: per-(site, hour) coverage of unlabeled vs labeled ----------
print("\n=== Coverage of unlabeled vs labeled by (site, hour) ===")
unlabeled_cells = pd.Series(list(unique_unlabeled_cells)).value_counts()
labeled_cells = set(cell_cauc.keys())
direct_hit = sum(1 for c in unique_unlabeled_cells if c in labeled_cells)
hour_hit = sum(1 for c in unique_unlabeled_cells if c not in labeled_cells and c[1] in hour_cauc)
site_hit = sum(1 for c in unique_unlabeled_cells if c not in labeled_cells and c[1] not in hour_cauc and c[0] in site_cauc)
global_hit = len(unique_unlabeled_cells) - direct_hit - hour_hit - site_hit
print(f"  Total unique unlabeled cells: {len(unique_unlabeled_cells)}")
print(f"    Direct (site,hour) match in labeled: {direct_hit}")
print(f"    Fallback to hour-only: {hour_hit}")
print(f"    Fallback to site-only: {site_hit}")
print(f"    Fallback to global: {global_hit}")
