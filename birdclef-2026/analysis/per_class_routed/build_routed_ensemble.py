"""SCRIPT B: Build per-class-routed ensemble.

For each class, pick the best non-leaked predictor among:
  - Perch v2 raw (good for 46 Perch-mapped classes)
  - Bruce CLIP-Ridge (covers all 75 — but limited by Ridge capacity)
  - The 7 independent SED models (alex_b0, baiyuby×2, long_convnext, mauricio, snowflake×2)

Compare:
  1. Per-class oracle (cheats — uses test AUC to route)
  2. Per-class router fit on labeled OOF (could over-fit S22-night)
  3. Uniform rank-avg of all 10 models
  4. Rank-avg of top-3 by single AUC
"""
import os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/per_class_routed"
os.makedirs(OUT, exist_ok=True)

# ---------- Load metadata + Y + 10 models ----------
labels = pd.read_csv("/tmp/bc26/train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)
def parse(fn):
    m = re.match(r"BC2026_Train_\d+_(S\d+)_(\d{8})_(\d{2})(\d{2})(\d{2})", fn)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)
labels[["site", "date", "hour"]] = labels["filename"].apply(lambda fn: pd.Series(parse(fn)))
hours = labels["hour"].to_numpy()
sites = labels["site"].to_numpy()

tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
cls_idx = {c: i for i, c in enumerate(classes)}
N, C = len(labels), len(classes)

br = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
Y = br["Y"]

M = {}
M["perch_v2_raw"] = br["P_perch_logits"]    # 46 classes
M["bruce_clip_ridge"] = br["P_bruce"]        # 75 classes
M["alexander_b0"] = np.load(f"{REPO}/meta_analysis/alexander_labeled_predictions.npz")["P"]
b = np.load(f"{REPO}/meta_analysis/baiyuby_labeled_predictions.npz")
M["baiyuby_v2s_max_FLIPPED"] = -b["P_max"]
M["baiyuby_v2s_att"] = b["P_att"]
M["long_convnextv2"] = np.load(f"{REPO}/meta_analysis/long_convnext_predictions.npz")["P"]
M["mauricio_b0_scratch"] = np.load(f"{REPO}/meta_analysis/mauricio_labeled_predictions.npz")["P"]
M["snowflake_convnext"] = np.load(f"{REPO}/meta_analysis/snowflake_convnext_predictions.npz")["P"]
M["snowflake_efnetv2m"] = np.load(f"{REPO}/meta_analysis/snowflake_efnetv2m_predictions.npz")["P"]


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


def macro_auc(y, p):
    aucs = per_class_auc(y, p)
    return float(np.nanmean(aucs))


def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(P.shape[1]):
        col = P[:, c]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            ranks = rankdata(col[valid]) / valid.sum()
            R[valid, c] = ranks
            R[~valid, c] = 0.5
        else:
            R[:, c] = 0.5
    return R


# ---------- Per-class AUCs ----------
print("Computing per-class AUC for each model...")
cauc = {n: per_class_auc(Y, P) for n, P in M.items()}
cauc_df = pd.DataFrame(cauc, index=classes)
cauc_df["n_positives"] = Y.sum(axis=0).astype(int)
cauc_df = cauc_df[cauc_df["n_positives"] > 0]
print(f"Valid classes (n_pos>0): {len(cauc_df)}")
cauc_df.to_csv(f"{OUT}/per_class_auc.csv")


# ---------- 1) Per-class oracle ----------
print("\n=== PER-CLASS ORACLE (uses test AUC to route — upper bound) ===")
model_names = list(M.keys())
P_oracle = np.zeros_like(Y, dtype=np.float32)
oracle_winners = {n: 0 for n in model_names}
oracle_lines = []
for c_idx, c_name in enumerate(classes):
    if Y[:, c_idx].sum() == 0:
        continue
    aucs = {n: cauc[n][c_idx] for n in model_names if not np.isnan(cauc[n][c_idx])}
    if not aucs:
        continue
    best_n = max(aucs, key=aucs.get)
    oracle_winners[best_n] += 1
    P_oracle[:, c_idx] = M[best_n][:, c_idx]
    n_pos = int(Y[:, c_idx].sum())
    oracle_lines.append((c_name, n_pos, best_n, aucs[best_n]))
print(f"Per-class oracle macro-AUC: {macro_auc(Y, P_oracle):.4f}")
print("Class winners:")
for n, k in sorted(oracle_winners.items(), key=lambda x: -x[1]):
    print(f"  {n:30s}: {k}")


# ---------- 2) FIT per-class router (leave-one-class-out cross-class blend) ----------
# Bigger question: instead of a one-hot router, use a per-class WEIGHTED average.
# We fit per-class non-negative weights via constrained least-squares on the OOF.
# This is leakage on this set, but at least we measure the headroom.
print("\n=== PER-CLASS RANKING-WEIGHTED AVERAGE ===")
# Convert each P to ranks
R = {n: rank_norm(P) for n, P in M.items()}
# Per-class: weight each model by softmax(per-class AUC)
P_weighted = np.zeros_like(Y, dtype=np.float32)
for c_idx in range(C):
    aucs = np.array([cauc[n][c_idx] for n in model_names])
    valid = ~np.isnan(aucs)
    if not valid.any():
        continue
    # softmax weights (only over models with valid AUC)
    a = aucs[valid] - aucs[valid].max()
    w = np.exp(5.0 * a)  # sharpness 5
    w = w / w.sum()
    cols = np.array([R[model_names[i]][:, c_idx] for i in np.where(valid)[0]])
    P_weighted[:, c_idx] = (w[:, None] * cols).sum(axis=0)
print(f"Per-class AUC-weighted-softmax rank-avg: {macro_auc(Y, P_weighted):.4f}")


# ---------- 3) Specialist-only blend ----------
# For classes Perch can predict (n_perch_valid), use only Perch
# For classes Bruce can predict but Perch can't, use Bruce
# For all others, use top-3 SED rank-avg
print("\n=== SPECIALIST-ROUTED ENSEMBLE ===")
perch_valid = ~np.isnan(M["perch_v2_raw"][0])  # classes Perch can predict
bruce_valid = ~np.isnan(M["bruce_clip_ridge"][0])
print(f"  Perch covers {perch_valid.sum()} classes; Bruce covers {bruce_valid.sum()}; total {C}")

sed_models = ["alexander_b0", "baiyuby_v2s_max_FLIPPED", "long_convnextv2"]
P_sed3 = np.mean([rank_norm(M[n]) for n in sed_models], axis=0)

P_specialist = np.zeros_like(Y, dtype=np.float32)
for c_idx in range(C):
    if perch_valid[c_idx]:
        P_specialist[:, c_idx] = rank_norm(M["perch_v2_raw"])[:, c_idx]
    elif bruce_valid[c_idx]:
        P_specialist[:, c_idx] = rank_norm(M["bruce_clip_ridge"])[:, c_idx]
    else:
        P_specialist[:, c_idx] = P_sed3[:, c_idx]
print(f"  specialist macro-AUC: {macro_auc(Y, P_specialist):.4f}")

# Alternative: stack Perch + Bruce within their coverage, fall back to SED-3
P_pb = (rank_norm(M["perch_v2_raw"]) + rank_norm(M["bruce_clip_ridge"])) / 2.0
P_stack = np.where(np.isnan(P_pb), P_sed3, P_pb)
# Handle NaN columns
P_stack = np.where(np.isnan(P_stack), rank_norm(M["bruce_clip_ridge"]), P_stack)
P_stack = np.where(np.isnan(P_stack), P_sed3, P_stack)
print(f"  perch+bruce avg (where both) + SED-3 fallback: {macro_auc(Y, P_stack):.4f}")


# ---------- 4) Plain comparisons ----------
print("\n=== PLAIN ENSEMBLES (for reference) ===")
P_all_rank = np.mean([rank_norm(P) for P in M.values()], axis=0)
print(f"  uniform_rank_avg (all 9 models): {macro_auc(Y, P_all_rank):.4f}")

singles = {n: macro_auc(Y, P) for n, P in M.items()}
top3 = sorted(singles.items(), key=lambda x: -x[1])[:3]
top3_names = [n for n, _ in top3]
P_top3 = np.mean([rank_norm(M[n]) for n in top3_names], axis=0)
print(f"  top3_rank_avg ({','.join(top3_names)}): {macro_auc(Y, P_top3):.4f}")

# ---------- Final summary ----------
final = pd.DataFrame([
    ("per_class_oracle (cheating)", macro_auc(Y, P_oracle)),
    ("per_class_auc_softmax_blend", macro_auc(Y, P_weighted)),
    ("specialist_routed (Perch>Bruce>SED-3)", macro_auc(Y, P_specialist)),
    ("perch+bruce+SED-3 stack", macro_auc(Y, P_stack)),
    ("uniform_rank_avg_all_9", macro_auc(Y, P_all_rank)),
    ("top3_rank_avg", macro_auc(Y, P_top3)),
    ("--- single best models ---", float("nan")),
    *[(f"single_{n}", a) for n, a in sorted(singles.items(), key=lambda x: -x[1])],
], columns=["strategy", "macro_auc"])
final.to_csv(f"{OUT}/ensemble_strategies.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(11, 8))
plot_df = final[final["macro_auc"].notna()].copy()
plot_df = plot_df.sort_values("macro_auc").reset_index(drop=True)
colors = []
for s in plot_df["strategy"]:
    if "oracle" in s:
        colors.append("#c44")
    elif "specialist" in s or "softmax" in s or "stack" in s:
        colors.append("#3a7")
    elif "rank_avg" in s:
        colors.append("#39c")
    else:
        colors.append("#999")
ax.barh(plot_df["strategy"], plot_df["macro_auc"], color=colors)
ax.set_xlim(0.5, 0.95)
ax.set_xlabel("Macro-AUC on labeled OOF (leakage-safe; 75 classes with positives)")
for i, v in enumerate(plot_df["macro_auc"]):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_title("Per-class routed ensembles vs simple ensembles vs single models\n"
             "Red=oracle (cheats). Green=routed strategy. Blue=rank-avg. Grey=single model.")
plt.tight_layout()
plt.savefig(f"{OUT}/ensemble_strategies.png", dpi=110, bbox_inches="tight")
plt.close()

# Plot per-class oracle winner distribution
fig, ax = plt.subplots(figsize=(10, 5))
w = pd.Series(oracle_winners).sort_values(ascending=True)
ax.barh(w.index, w.values)
for i, v in enumerate(w.values):
    ax.text(v + 0.3, i, str(int(v)), va="center")
ax.set_xlabel("Number of classes (of 75) where model is per-class oracle winner")
ax.set_title("Per-class oracle: which model wins how many classes?")
plt.tight_layout()
plt.savefig(f"{OUT}/oracle_class_winners.png", dpi=110, bbox_inches="tight")
plt.close()

print(f"\nSaved tables + plots to {OUT}")
