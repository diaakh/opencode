"""Integrate exp019 (ProtoSSM, SED, Model_7 blend) predictions on labeled OOF
into the existing audit."""
import os, re, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/model_audit"

# ---------- Reload labels + metadata ----------
labels = pd.read_csv("/tmp/bc26/train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)


def parse(fn):
    m = re.match(r"BC2026_Train_\d+_(S\d+)_(\d{8})_(\d{2})(\d{2})(\d{2})", fn)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)


labels[["site", "date", "hour"]] = labels["filename"].apply(
    lambda fn: pd.Series(parse(fn))
)
hours = labels["hour"].to_numpy()


def parse_time(t):
    try:
        h, m, s = str(t).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None


labels["end_sec"] = labels["end"].apply(parse_time)
labels["win_idx"] = labels["end_sec"].apply(lambda s: (s - 1) // 5 if s else None)

tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
cls_idx = {c: i for i, c in enumerate(classes)}
N, C = len(labels), len(classes)

# Y matrix
Y = np.load(f"{REPO}/meta_analysis/alexander_labeled_predictions.npz")["Y"]
print(f"Y shape: {Y.shape}")


# ---------- Build row_id → row_index lookup ----------
def make_row_id(filename, win_idx):
    # exp019 outputs use stem (no .ogg) + "_<end_sec>"
    stem = filename.replace(".ogg", "")
    end_sec = (win_idx + 1) * 5
    return f"{stem}_{end_sec}"


labels["row_id"] = [
    make_row_id(row["filename"], int(row["win_idx"]))
    for _, row in labels.iterrows()
]
row_id_to_idx = {rid: i for i, rid in enumerate(labels["row_id"])}
print(f"Built row_id index: {len(row_id_to_idx)} entries")
print(f"Sample row_ids: {labels['row_id'].iloc[:3].tolist()}")


# ---------- Load exp019 outputs ----------
def load_csv_predictions(path, name):
    """Load a 792×235 submission CSV and align to our row order."""
    df = pd.read_csv(path)
    print(f"\n{name}: {df.shape}")
    if "row_id" not in df.columns:
        print(f"  no row_id, skip")
        return None
    df["row_id"] = df["row_id"].astype(str)
    # Sample exp019 row_id format
    print(f"  sample csv row_id: {df['row_id'].iloc[0]}")
    print(f"  our row_id format: {labels['row_id'].iloc[0]}")
    # Build P aligned to our Y order
    P = np.full((N, C), np.nan, dtype=np.float32)
    matched = 0
    for _, csv_row in df.iterrows():
        rid = csv_row["row_id"]
        # exp019 row_id has "BC2026_Train_0001_S08_20250606_030007_5" — same as ours? Check
        if rid in row_id_to_idx:
            i = row_id_to_idx[rid]
            for ci, c in enumerate(classes):
                if c in df.columns:
                    P[i, ci] = csv_row[c]
            matched += 1
    print(f"  matched {matched}/{len(df)} rows to Y")
    return P


def load_npz_predictions(path, name):
    z = np.load(path, allow_pickle=True)
    P_raw = z["P"]
    row_ids = z["row_ids"]
    classes_in = z["classes"]
    print(f"\n{name}: P shape={P_raw.shape}, row_ids={len(row_ids)}, classes={len(classes_in)}")
    print(f"  sample npz row_id: {row_ids[0]}")
    # Build (class_in -> taxonomy_idx) mapping
    class_in_to_tax = np.array([cls_idx.get(str(c), -1) for c in classes_in])
    P = np.full((N, C), np.nan, dtype=np.float32)
    matched_rows = 0
    for k, rid in enumerate(row_ids):
        rid_str = str(rid)
        if rid_str in row_id_to_idx:
            i = row_id_to_idx[rid_str]
            for src_idx in range(len(classes_in)):
                tax_idx = class_in_to_tax[src_idx]
                if tax_idx >= 0:
                    P[i, tax_idx] = P_raw[k, src_idx]
            matched_rows += 1
    print(f"  matched {matched_rows}/{len(row_ids)} rows")
    return P


P_protossm = load_csv_predictions("/tmp/exp019_out/submission_protossm.csv", "ProtoSSM")
P_sed = load_csv_predictions("/tmp/exp019_out/submission_sed.csv", "Distilled-SED")
P_model7 = load_npz_predictions("/tmp/exp019_out/labeled_oof_model7_pre_align.npz", "Model_7 blend")


# Also load Bruce (still imperfect — only first 234 of Perch's 14795 — but include for completeness)
bruce = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
P_bruce = bruce["P_bruce"]
P_perch_raw = bruce["P_perch_logits"]
print(f"\nBruce: P_bruce={P_bruce.shape}, P_perch_raw={P_perch_raw.shape}")


# ---------- Compute macro-AUC ----------
def macro_auc(y, p):
    if p is None:
        return float("nan"), 0
    aucs = []
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0 or y[:, c].sum() == len(y):
            continue
        col = p[:, c]
        mask = ~np.isnan(col)
        if mask.sum() < 5:
            continue
        if y[mask, c].sum() == 0 or y[mask, c].sum() == mask.sum():
            continue
        try:
            aucs.append(roc_auc_score(y[mask, c], col[mask]))
        except Exception:
            pass
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


new_models = {
    "exp019_Model_7_blend": P_model7,
    "exp019_ProtoSSM": P_protossm,
    "exp019_Distilled_SED": P_sed,
    "bruce_clip_ridge_buggy": P_bruce,  # buggy mapping
    "perch_raw_logits_buggy": P_perch_raw,  # buggy mapping
}

print("\n=== Macro-AUC of new models on labeled OOF ===")
for name, P in new_models.items():
    auc, n = macro_auc(Y, P)
    print(f"  {name:30s}: {auc:.4f}  (n_classes={n})")


# ---------- Load all existing models for comparison ----------
M = {}
M["alexander_b0"] = np.load(f"{REPO}/meta_analysis/alexander_labeled_predictions.npz")["P"]
b = np.load(f"{REPO}/meta_analysis/baiyuby_labeled_predictions.npz")
M["baiyuby_v2s_max_FLIPPED"] = -b["P_max"]
M["baiyuby_v2s_att"] = b["P_att"]
M["long_convnextv2"] = np.load(f"{REPO}/meta_analysis/long_convnext_predictions.npz")["P"]
M["mauricio_b0_scratch"] = np.load(f"{REPO}/meta_analysis/mauricio_labeled_predictions.npz")["P"]
M["snowflake_convnext"] = np.load(f"{REPO}/meta_analysis/snowflake_convnext_predictions.npz")["P"]
M["snowflake_efnetv2m"] = np.load(f"{REPO}/meta_analysis/snowflake_efnetv2m_predictions.npz")["P"]
# Add the new ones
M.update({k: v for k, v in new_models.items() if v is not None})

# Re-compute overall AUC across all models
print("\n=== Overall macro-AUC for ALL models (combined) ===")
all_auc = {}
for name, P in M.items():
    auc, n = macro_auc(Y, P)
    all_auc[name] = auc
    print(f"  {name:35s}: {auc:.4f}  (n_classes={n})")


# ---------- Per-hour breakdown for new models ----------
unique_hours = sorted(set(hours.tolist()))
print(f"\n=== Per-hour macro-AUC (new models only) ===")
hour_df = pd.DataFrame(index=list(new_models.keys()), columns=unique_hours, dtype=float)
for name, P in new_models.items():
    if P is None:
        continue
    for h in unique_hours:
        mask = hours == h
        if mask.sum() < 5:
            continue
        aucs = []
        for c in range(C):
            yc = Y[mask, c]
            pc = P[mask, c]
            valid = ~np.isnan(pc)
            if valid.sum() < 3:
                continue
            if yc[valid].sum() == 0 or yc[valid].sum() == valid.sum():
                continue
            try:
                aucs.append(roc_auc_score(yc[valid], pc[valid]))
            except Exception:
                pass
        if aucs:
            hour_df.loc[name, h] = np.mean(aucs)
print(hour_df.round(3))
hour_df.to_csv(f"{OUT}/per_hour_macro_auc_new_models.csv")


# ---------- Combined per-hour heatmap (all models) ----------
hour_all = pd.DataFrame(index=list(M.keys()), columns=unique_hours, dtype=float)
for name, P in M.items():
    for h in unique_hours:
        mask = hours == h
        if mask.sum() < 5:
            continue
        aucs = []
        for c in range(C):
            yc = Y[mask, c]
            pc = P[mask, c]
            valid = ~np.isnan(pc)
            if valid.sum() < 3:
                continue
            if yc[valid].sum() == 0 or yc[valid].sum() == valid.sum():
                continue
            try:
                aucs.append(roc_auc_score(yc[valid], pc[valid]))
            except Exception:
                pass
        if aucs:
            hour_all.loc[name, h] = np.mean(aucs)

# Sort by mean AUC for cleaner display
hour_all = hour_all.reindex(sorted(hour_all.index, key=lambda n: -hour_all.loc[n].mean()))
hour_all.to_csv(f"{OUT}/per_hour_all_models.csv")


# ---------- Plot ----------
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(13, 8))
im = ax.imshow(hour_all.astype(float).values, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(hour_all.columns)))
ax.set_xticklabels([str(h) for h in hour_all.columns])
ax.set_yticks(range(len(hour_all.index)))
ax.set_yticklabels(hour_all.index, fontsize=9)
ax.set_xlabel("Hour of day (UTC)")
for i in range(hour_all.shape[0]):
    for j in range(hour_all.shape[1]):
        v = hour_all.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.78 else "black", fontsize=6)
plt.colorbar(im, ax=ax, label="Macro-AUC")
ax.set_title("Per-(model, hour) macro-AUC — ALL 12 models\n(sorted by overall AUC; exp019 stack now included)")
plt.savefig(f"{OUT}/07_per_hour_heatmap_with_exp019.png")
plt.close()

# Overall ranking bar
fig, ax = plt.subplots(figsize=(12, 8))
sorted_all = sorted(all_auc.items(), key=lambda x: x[1])
names = [n for n, _ in sorted_all]
vals = [v for _, v in sorted_all]
colors = []
for n in names:
    if "exp019" in n:
        colors.append("#c44")
    elif "bruce" in n or "perch" in n:
        colors.append("#fc6")
    else:
        colors.append("#888")
ax.barh(names, vals, color=colors)
ax.set_xlim(0.3, 1.0)
ax.set_xlabel("Overall macro-AUC on labeled OOF")
ax.axvline(0.7, ls=":", color="grey", alpha=0.4)
ax.axvline(0.9, ls=":", color="grey", alpha=0.4)
for i, v in enumerate(vals):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_title("Overall macro-AUC — all available models on 739 labeled windows\n"
             "Red = exp019 stack pulled from Kaggle today; Yellow = Bruce/Perch (mapping bug)")
plt.savefig(f"{OUT}/08_overall_ranking_full.png")
plt.close()

print("\nDone. Plots:")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print(f"  {f}")
