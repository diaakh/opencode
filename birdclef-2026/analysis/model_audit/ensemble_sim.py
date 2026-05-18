"""Build candidate ensembles from the per-hour winners + test naive averages."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/model_audit"

# --- load metadata from main script ---
import re
labels = pd.read_csv("/tmp/bc26/train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)
hours = np.array([
    int(re.match(r"BC2026_Train_\d+_S\d+_\d{8}_(\d{2})", fn).group(1))
    for fn in labels["filename"]
])

tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
classes = tax["primary_label"].tolist()
cls_idx = {c: i for i, c in enumerate(classes)}

Y = np.load(f"{REPO}/meta_analysis/alexander_labeled_predictions.npz")["Y"]

# Load all model P matrices — apply baiyuby sign fix
M = {}
M["alexander_b0"] = np.load(f"{REPO}/meta_analysis/alexander_labeled_predictions.npz")["P"]
b = np.load(f"{REPO}/meta_analysis/baiyuby_labeled_predictions.npz")
M["baiyuby_v2s_max_FLIPPED"] = -b["P_max"]  # rank-inverted in source
M["baiyuby_v2s_att"] = b["P_att"]
M["long_convnextv2"] = np.load(f"{REPO}/meta_analysis/long_convnext_predictions.npz")["P"]
M["mauricio_b0_scratch"] = np.load(f"{REPO}/meta_analysis/mauricio_labeled_predictions.npz")["P"]
M["snowflake_convnext"] = np.load(f"{REPO}/meta_analysis/snowflake_convnext_predictions.npz")["P"]
M["snowflake_efnetv2m"] = np.load(f"{REPO}/meta_analysis/snowflake_efnetv2m_predictions.npz")["P"]


def macro_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        if 0 < y[:, c].sum() < y.shape[0]:
            try:
                aucs.append(roc_auc_score(y[:, c], p[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")


def rank_normalize(P):
    """Within-column rank-normalize so each class is in [0,1]."""
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(P.shape[1]):
        R[:, c] = rankdata(P[:, c], method="average") / P.shape[0]
    return R


# ---------- Baselines ----------
print("=== Single-model baselines ===")
single = {n: macro_auc(Y, p) for n, p in M.items()}
for n, a in sorted(single.items(), key=lambda x: -x[1]):
    print(f"  {n:30s}: {a:.4f}")

# ---------- Naive averages ----------
print("\n=== Naive ensembles ===")
names = list(M.keys())
P_avg_prob = np.mean([M[n] for n in names], axis=0)
P_avg_rank = np.mean([rank_normalize(M[n]) for n in names], axis=0)
print(f"  uniform_prob_avg (7 models)  : {macro_auc(Y, P_avg_prob):.4f}")
print(f"  uniform_rank_avg (7 models)  : {macro_auc(Y, P_avg_rank):.4f}")

# Without the flipped baiyuby
names_drop = [n for n in names if n != "baiyuby_v2s_max_FLIPPED"]
P_avg_rank_drop = np.mean([rank_normalize(M[n]) for n in names_drop], axis=0)
print(f"  uniform_rank_avg (6, no flip): {macro_auc(Y, P_avg_rank_drop):.4f}")

# Top-3 by single AUC
top3 = sorted(single.items(), key=lambda x: -x[1])[:3]
top3_names = [n for n, _ in top3]
P_top3 = np.mean([rank_normalize(M[n]) for n in top3_names], axis=0)
print(f"  top3_rank_avg ({','.join(top3_names)}): {macro_auc(Y, P_top3):.4f}")


# ---------- Per-hour oracle (cheating) ----------
print("\n=== Per-hour winner oracle (best possible ensemble) ===")

per_hour_winner = {}
P_oracle_h = np.zeros_like(Y, dtype=np.float32)
for h in sorted(set(hours.tolist())):
    mask = hours == h
    if mask.sum() < 5:
        continue
    best_name, best_auc = None, -1
    for n, p in M.items():
        a = macro_auc(Y[mask], p[mask])
        if not np.isnan(a) and a > best_auc:
            best_auc = a
            best_name = n
    per_hour_winner[h] = (best_name, best_auc)
    P_oracle_h[mask] = M[best_name][mask]
    print(f"  h={h:2d} (n={mask.sum():3d}): {best_name:30s} ({best_auc:.4f})")
print(f"  oracle_hour macro-AUC      : {macro_auc(Y, P_oracle_h):.4f}")


# ---------- Per-class oracle ----------
print("\n=== Per-class winner oracle ===")
per_class_winner = {}
class_aucs_all = {n: np.array([
    roc_auc_score(Y[:, c], p[:, c]) if 0 < Y[:, c].sum() < Y.shape[0] else np.nan
    for c in range(Y.shape[1])
]) for n, p in M.items()}

P_oracle_c = np.zeros_like(Y, dtype=np.float32)
n_winners = {n: 0 for n in M}
for c in range(Y.shape[1]):
    if not (0 < Y[:, c].sum() < Y.shape[0]):
        continue
    cs = {n: class_aucs_all[n][c] for n in M}
    best_n = max(cs, key=cs.get)
    n_winners[best_n] += 1
    P_oracle_c[:, c] = M[best_n][:, c]
print("  per-class oracle macro-AUC :", f"{macro_auc(Y, P_oracle_c):.4f}")
print("  classes won per model      :")
for n, c in sorted(n_winners.items(), key=lambda x: -x[1]):
    print(f"    {n:30s}: {c}")


# ---------- Weighted ensemble: per-hour optimal-weights via grid ----------
# For each hour, find blend weight that maximizes macro-AUC on that hour
# using rank-normalized predictions, restricted to top-3 models
print("\n=== Per-hour weighted blend (grid search) ===")
P_grid_h = np.zeros_like(Y, dtype=np.float32)
# rank-normalize once
R = {n: rank_normalize(M[n]) for n in M}
for h in sorted(set(hours.tolist())):
    mask = hours == h
    if mask.sum() < 5:
        continue
    # Search uniform blend of top-3 vs hour-winner vs avg-of-3
    best_a, best_label, best_pred = -1, None, None
    candidates = [
        ("top3_rank_avg", P_top3[mask]),
        ("hour_winner", M[per_hour_winner[h][0]][mask]),
        ("uniform_rank_all", P_avg_rank[mask]),
        ("uniform_rank_no_flip", P_avg_rank_drop[mask]),
    ]
    for lbl, p in candidates:
        a = macro_auc(Y[mask], p)
        if not np.isnan(a) and a > best_a:
            best_a, best_label, best_pred = a, lbl, p
    P_grid_h[mask] = best_pred
    print(f"  h={h:2d}: {best_label:25s} ({best_a:.4f})")
print(f"  best-of-candidates oracle  : {macro_auc(Y, P_grid_h):.4f}")


# ---------- Plot ----------
results = {
    **single,
    "uniform_prob_avg_7": macro_auc(Y, P_avg_prob),
    "uniform_rank_avg_7": macro_auc(Y, P_avg_rank),
    "uniform_rank_avg_6_noflip": macro_auc(Y, P_avg_rank_drop),
    "top3_rank_avg": macro_auc(Y, P_top3),
    "oracle_per_hour": macro_auc(Y, P_oracle_h),
    "oracle_per_class": macro_auc(Y, P_oracle_c),
    "oracle_per_hour_blends": macro_auc(Y, P_grid_h),
}
df = pd.DataFrame(sorted(results.items(), key=lambda x: x[1]), columns=["strategy", "macro_auc"])
df.to_csv(f"{OUT}/ensemble_results.csv", index=False)
fig, ax = plt.subplots(figsize=(11, 7))
colors = ["#888"] * len(df)
for i, (n, v) in enumerate(zip(df["strategy"], df["macro_auc"])):
    if "oracle" in n:
        colors[i] = "#c44"
    elif "rank_avg" in n or "prob_avg" in n or "top3" in n:
        colors[i] = "#3a7"
ax.barh(df["strategy"], df["macro_auc"], color=colors)
ax.set_xlim(0.3, 1.0)
ax.set_xlabel("Macro-AUC on labeled OOF (75 valid classes)")
ax.set_title("Single models vs naive ensembles vs oracle bounds\n"
             "Green = realistic ensemble; Red = oracle (upper bound)")
ax.axvline(0.6344, ls=":", color="black", alpha=0.4, label="best single (alex_b0)")
for i, (n, v) in enumerate(zip(df["strategy"], df["macro_auc"])):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/06_ensemble_comparison.png", dpi=110)
plt.close()
print(f"\nSaved {OUT}/06_ensemble_comparison.png and ensemble_results.csv")
