"""Run ensemble of available models on labeled OOF, slice by hour/class."""
import os, re, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/model_audit"
os.makedirs(OUT, exist_ok=True)


# ---------- 1. Load labels + metadata ----------
labels = pd.read_csv("/tmp/bc26/train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)


def parse(fn):
    m = re.match(r"BC2026_Train_\d+_(S\d+)_(\d{8})_(\d{2})(\d{2})(\d{2})", fn)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)


labels[["site", "date", "hour"]] = labels["filename"].apply(
    lambda fn: pd.Series(parse(fn))
)
hours = labels["hour"].to_numpy()
sites = labels["site"].to_numpy()

# Build Y matrix in taxonomy order
tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
classes = tax["primary_label"].tolist()
cls_idx = {c: i for i, c in enumerate(classes)}
N, C = len(labels), len(classes)
Y = np.zeros((N, C), dtype=np.float32)
for i, lab in enumerate(labels["primary_label"]):
    for c in str(lab).split(";"):
        if c in cls_idx:
            Y[i, cls_idx[c]] = 1.0


# ---------- 2. Load model predictions ----------
def load(name, file, key="P"):
    z = np.load(f"{REPO}/meta_analysis/{file}")
    return name, z[key]


models = [
    load("alexander_b0", "alexander_labeled_predictions.npz"),
    load("baiyuby_v2s_max", "baiyuby_labeled_predictions.npz", "P_max"),
    load("baiyuby_v2s_att", "baiyuby_labeled_predictions.npz", "P_att"),
    load("long_convnextv2", "long_convnext_predictions.npz"),
    load("mauricio_b0_scratch", "mauricio_labeled_predictions.npz"),
    load("snowflake_convnext", "snowflake_convnext_predictions.npz"),
    load("snowflake_efnetv2m", "snowflake_efnetv2m_predictions.npz"),
]


# ---------- 3. Add saved priors as "models" ----------
def prior_predictor(prior_path, hour_col_name="hour"):
    """Apply a saved hour-prior table as a constant prediction per row."""
    df = pd.read_csv(prior_path)
    df = df.set_index(hour_col_name)
    cls_in_prior = [c for c in df.columns if c in cls_idx]
    P = np.zeros((N, C), dtype=np.float32)
    for i, h in enumerate(hours):
        if h in df.index:
            for cn in cls_in_prior:
                P[i, cls_idx[cn]] = df.loc[h, cn]
    return P


try:
    P_pseudo_h = prior_predictor(f"{REPO}/meta_analysis/pseudo_hour_priors.csv")
    models.append(("pseudo_hour_prior", P_pseudo_h))
except Exception as e:
    print("pseudo_hour_priors load failed:", e)

try:
    P_lab_h = prior_predictor(f"{REPO}/meta_analysis/hourly_species_priors.csv")
    models.append(("labeled_hour_prior", P_lab_h))
except Exception as e:
    print("labeled_hour_priors load failed:", e)


# ---------- 4. Compute metrics ----------
def macro_auc(y, p, classes_subset=None):
    if classes_subset is None:
        classes_subset = range(y.shape[1])
    aucs = []
    for c in classes_subset:
        if y[:, c].sum() == 0 or y[:, c].sum() == y.shape[0]:
            continue
        try:
            aucs.append(roc_auc_score(y[:, c], p[:, c]))
        except Exception:
            pass
    return np.mean(aucs), len(aucs)


def per_class_auc(y, p):
    aucs = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        pos = y[:, c].sum()
        if pos == 0 or pos == y.shape[0]:
            continue
        try:
            aucs[c] = roc_auc_score(y[:, c], p[:, c])
        except Exception:
            pass
    return aucs


# Overall macro-AUC per model
print("\n=== Overall macro-AUC (skip-zero-positive classes) ===")
overall = {}
for name, P in models:
    auc, n = macro_auc(Y, P)
    overall[name] = auc
    print(f"  {name:25s}: {auc:.4f}  (n_classes={n})")


# Per-hour macro-AUC
unique_hours = sorted(set(hours.tolist()))
print(f"\n=== Per-hour macro-AUC ({len(unique_hours)} hours) ===")
hour_auc = {}
for name, P in models:
    by_h = {}
    for h in unique_hours:
        mask = hours == h
        if mask.sum() < 5:
            continue
        # macro-AUC over classes that have ≥1 positive AND ≥1 negative in this slice
        aucs = []
        for c in range(C):
            yc = Y[mask, c]
            pc = P[mask, c]
            if yc.sum() == 0 or yc.sum() == len(yc):
                continue
            try:
                aucs.append(roc_auc_score(yc, pc))
            except Exception:
                pass
        if aucs:
            by_h[h] = (np.mean(aucs), len(aucs), mask.sum())
    hour_auc[name] = by_h


# Per-class AUC (overall)
class_aucs = {name: per_class_auc(Y, P) for name, P in models}


# ---------- 5. Save tables ----------
# Overall AUC table
overall_df = pd.DataFrame(
    [(k, v) for k, v in sorted(overall.items(), key=lambda x: -x[1])],
    columns=["model", "macro_auc"],
)
overall_df.to_csv(f"{OUT}/overall_auc.csv", index=False)

# Per-hour table: rows=model, cols=hour
hour_df = pd.DataFrame(index=[n for n, _ in models], columns=unique_hours, dtype=float)
hour_n_df = pd.DataFrame(index=[n for n, _ in models], columns=unique_hours, dtype=int)
for name, by_h in hour_auc.items():
    for h, (auc, n_cls, n_rows) in by_h.items():
        hour_df.loc[name, h] = auc
        hour_n_df.loc[name, h] = n_rows
hour_df.to_csv(f"{OUT}/per_hour_macro_auc.csv")
hour_n_df.to_csv(f"{OUT}/per_hour_row_counts.csv")

# Per-class AUC table: rows=class, cols=model
class_df = pd.DataFrame(class_aucs, index=classes)
class_df["n_positives"] = Y.sum(axis=0).astype(int)
class_df = class_df[class_df["n_positives"] > 0].sort_values(
    "n_positives", ascending=False
)
class_df.to_csv(f"{OUT}/per_class_auc.csv")
print(f"\nClass AUCs computed for {len(class_df)} classes (with ≥1 positive)")


# ---------- 6. Plots ----------
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# Plot 1: per-hour macro-AUC line chart
fig, ax = plt.subplots(figsize=(13, 7))
for name in hour_df.index:
    series = hour_df.loc[name].dropna().sort_index()
    if len(series) > 1:
        ax.plot(series.index.astype(int), series.values, marker="o", label=name)
ax.set_xlabel("Hour of day (UTC)")
ax.set_ylabel("Macro-AUC on labeled OOF")
ax.set_title("Model performance vs hour\n(Labeled set; classes with ≥1 positive in each hour slice)")
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=8)
plt.savefig(f"{OUT}/01_per_hour_macro_auc.png")
plt.close()

# Plot 2: per-hour heatmap
fig, ax = plt.subplots(figsize=(13, 6))
data = hour_df.reindex(
    sorted(hour_df.index, key=lambda n: -hour_df.loc[n].mean())
).astype(float)
im = ax.imshow(data.values, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(data.columns)))
ax.set_xticklabels([str(h) for h in data.columns])
ax.set_yticks(range(len(data.index)))
ax.set_yticklabels(data.index)
ax.set_xlabel("Hour of day (UTC)")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.78 else "black", fontsize=7)
plt.colorbar(im, ax=ax, label="Macro-AUC")
ax.set_title("Per-(model, hour) macro-AUC\n(white→black text threshold = 0.78)")
plt.savefig(f"{OUT}/02_per_hour_heatmap.png")
plt.close()

# Plot 3: per-class AUC heatmap (top 30 most populous classes)
top30 = class_df.head(30)
model_cols = [c for c in top30.columns if c != "n_positives"]
fig, ax = plt.subplots(figsize=(14, 10))
data = top30[model_cols].T.values  # rows=models, cols=classes
im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(top30)))
ax.set_xticklabels(
    [f"{c}\n(n={top30.loc[c,'n_positives']})" for c in top30.index],
    rotation=70, ha="right", fontsize=7,
)
ax.set_yticks(range(len(model_cols)))
ax.set_yticklabels(model_cols)
for i in range(len(model_cols)):
    for j in range(len(top30)):
        v = data[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.78 else "black", fontsize=6)
plt.colorbar(im, ax=ax, label="AUC")
ax.set_title("Per-(model, class) AUC — top 30 most-positive classes")
plt.savefig(f"{OUT}/03_per_class_heatmap_top30.png")
plt.close()

# Plot 4: per-hour, count winners per hour
winners_by_hour = {}
for h in unique_hours:
    col = hour_df[h].dropna()
    if len(col) > 0:
        winner = col.idxmax()
        winners_by_hour[h] = (winner, col.max())
print("\n=== Per-hour winner ===")
for h, (w, a) in sorted(winners_by_hour.items()):
    print(f"  h={h:2d}: {w:25s} ({a:.4f})")

# Plot 5: which model wins per-class (top 50 most populous)
top50 = class_df.head(50)
class_winners = top50[model_cols].idxmax(axis=1).value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(class_winners.index[::-1], class_winners.values[::-1])
ax.set_xlabel("Number of (top-50 populous) classes where model wins")
ax.set_title("Per-class AUC winners — top 50 most-positive classes")
for i, v in enumerate(class_winners.values[::-1]):
    ax.text(v + 0.1, i, str(int(v)), va="center")
plt.savefig(f"{OUT}/04_class_winners.png")
plt.close()

# Plot 6: overall ranking bar
fig, ax = plt.subplots(figsize=(10, 5))
sorted_overall = sorted(overall.items(), key=lambda x: x[1])
names = [n for n, _ in sorted_overall]
vals = [v for _, v in sorted_overall]
ax.barh(names, vals)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("Overall macro-AUC on 739 labeled windows")
ax.axvline(0.7, ls=":", color="grey", alpha=0.5)
ax.axvline(0.9, ls=":", color="grey", alpha=0.5)
for i, v in enumerate(vals):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_title("Overall macro-AUC by model (labeled OOF, 739 windows)")
plt.savefig(f"{OUT}/05_overall_ranking.png")
plt.close()

# Save a structured summary
summary = {
    "n_rows": int(N),
    "n_classes_with_positives": int((Y.sum(axis=0) > 0).sum()),
    "unique_hours": unique_hours,
    "site_distribution": pd.Series(sites).value_counts().to_dict(),
    "overall_auc": {k: float(v) for k, v in overall.items()},
    "per_hour_winner": {
        int(h): {"model": w, "auc": float(a)} for h, (w, a) in winners_by_hour.items()
    },
    "per_class_winner_counts": class_winners.to_dict(),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nDone. Plots + tables in {OUT}/")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
