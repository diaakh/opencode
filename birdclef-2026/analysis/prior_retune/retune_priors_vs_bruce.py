"""SCRIPT A: Re-tune v4 prior weights against Bruce CLIP-Ridge (leakage-safe base).

The corpus's v3/v4 scaffolding tuned weights against exp019's leaked OOF.
With a proper non-leaked base (Bruce 0.867 AUC), we get an HONEST measurement
of how much the hour-prior + site-shrinkage post-proc actually helps.
"""
import os, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/prior_retune"
os.makedirs(OUT, exist_ok=True)

# ---------- Load metadata ----------
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
P_bruce = br["P_bruce"]  # already sigmoid'd; 0.867 macro-AUC


# ---------- Load priors ----------
def load_prior(csv_path, hour_or_site_or_sitehr="hour"):
    df = pd.read_csv(csv_path)
    # Identify key column
    if hour_or_site_or_sitehr == "hour":
        key = "hour" if "hour" in df.columns else df.columns[0]
    elif hour_or_site_or_sitehr == "site":
        key = "site" if "site" in df.columns else df.columns[0]
    else:
        key = "site_hour" if "site_hour" in df.columns else df.columns[0]
    df = df.set_index(key)
    # Restrict to known taxonomy classes
    valid_cols = [c for c in df.columns if c in cls_idx]
    return df[valid_cols], valid_cols


pseudo_hour, cls_pseudo_hour = load_prior(f"{REPO}/meta_analysis/pseudo_hour_priors.csv", "hour")
pseudo_site, cls_pseudo_site = load_prior(f"{REPO}/meta_analysis/pseudo_site_priors.csv", "site")
labeled_hour, cls_lab_hour = load_prior(f"{REPO}/meta_analysis/hourly_species_priors.csv", "hour")
print(f"pseudo_hour: {pseudo_hour.shape}")
print(f"pseudo_site: {pseudo_site.shape}")
print(f"labeled_hour: {labeled_hour.shape}")


def prior_to_row_array(prior_df, valid_cls, lookups, default=None):
    """For each row, look up the prior value in the prior table."""
    P = np.full((N, C), np.nan, dtype=np.float32)
    for i, key in enumerate(lookups):
        if key in prior_df.index:
            for cn in valid_cls:
                P[i, cls_idx[cn]] = prior_df.loc[key, cn]
    return P


P_pseudo_h = prior_to_row_array(pseudo_hour, cls_pseudo_hour, hours)
P_pseudo_s = prior_to_row_array(pseudo_site, cls_pseudo_site, sites)
P_labeled_h = prior_to_row_array(labeled_hour, cls_lab_hour, hours)


# ---------- Eval helper ----------
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
    return float(np.mean(aucs)) if aucs else float("nan")


EPS = 1e-6


def apply_prior(base_prob, hour_prior, w):
    """logit(base) + w * log(prior). Returns sigmoid."""
    base_logit = np.log(np.clip(base_prob, EPS, 1 - EPS) / (1 - np.clip(base_prob, EPS, 1 - EPS)))
    prior_logit = w * np.log(np.clip(hour_prior, EPS, 1.0))
    # Handle NaNs (rows where prior is missing): leave unchanged
    valid = ~np.isnan(prior_logit)
    out_logit = base_logit.copy()
    out_logit[valid] = base_logit[valid] + prior_logit[valid]
    return 1.0 / (1.0 + np.exp(-out_logit))


def apply_site_shrinkage(base_prob, site_prior, w):
    """(base + w * site_prior) / (1 + w)."""
    out = base_prob.copy()
    valid = ~np.isnan(site_prior)
    out[valid] = (base_prob[valid] + w * site_prior[valid]) / (1.0 + w)
    return out


# ---------- Baseline ----------
print(f"\nBruce alone macro-AUC: {macro_auc(Y, P_bruce):.4f}")

# ---------- 1D sweep: hour_prior_weight (pseudo) ----------
print("\n=== 1D sweep: pseudo_hour_prior weight ===")
results_h = {}
for w in [0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]:
    pred = apply_prior(P_bruce, P_pseudo_h, w)
    auc = macro_auc(Y, pred)
    results_h[w] = auc
    print(f"  w={w:6.3f}: macro-AUC={auc:.4f}")
best_w_h_pseudo = max(results_h, key=results_h.get)
print(f"  BEST: w={best_w_h_pseudo} → {results_h[best_w_h_pseudo]:.4f}")

# ---------- 1D sweep: labeled_hour weight ----------
print("\n=== 1D sweep: labeled_hour_prior weight ===")
results_lh = {}
for w in [0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
    pred = apply_prior(P_bruce, P_labeled_h, w)
    auc = macro_auc(Y, pred)
    results_lh[w] = auc
    print(f"  w={w:6.3f}: macro-AUC={auc:.4f}")
best_w_lh = max(results_lh, key=results_lh.get)
print(f"  BEST: w={best_w_lh} → {results_lh[best_w_lh]:.4f}")

# ---------- 1D sweep: site_shrinkage ----------
print("\n=== 1D sweep: pseudo_site_shrinkage weight ===")
results_s = {}
for w in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
    pred = apply_site_shrinkage(P_bruce, P_pseudo_s, w)
    auc = macro_auc(Y, pred)
    results_s[w] = auc
    print(f"  w={w:5.2f}: macro-AUC={auc:.4f}")
best_w_s = max(results_s, key=results_s.get)
print(f"  BEST: w={best_w_s} → {results_s[best_w_s]:.4f}")

# ---------- 2D sweep: hour × site, both pseudo ----------
print("\n=== 2D sweep: pseudo_hour × pseudo_site_shrinkage ===")
hour_ws = [0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
site_ws = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
sweep = pd.DataFrame(index=hour_ws, columns=site_ws, dtype=float)
for wh in hour_ws:
    for ws in site_ws:
        pred = apply_prior(P_bruce, P_pseudo_h, wh)
        pred = apply_site_shrinkage(pred, P_pseudo_s, ws)
        sweep.loc[wh, ws] = macro_auc(Y, pred)
print(sweep.round(4))
best_idx = np.unravel_index(np.nanargmax(sweep.values), sweep.shape)
best_wh, best_ws = hour_ws[best_idx[0]], site_ws[best_idx[1]]
print(f"  BEST: w_hour={best_wh}, w_site={best_ws} → {sweep.values[best_idx]:.4f}")
sweep.to_csv(f"{OUT}/sweep_2d_pseudo.csv")

# ---------- 2D sweep: labeled_hour × pseudo_site (v4-style hybrid) ----------
print("\n=== 2D sweep: labeled_hour × pseudo_site (v4-style hybrid) ===")
sweep_v4 = pd.DataFrame(index=hour_ws, columns=site_ws, dtype=float)
for wh in hour_ws:
    for ws in site_ws:
        pred = apply_prior(P_bruce, P_labeled_h, wh)
        pred = apply_site_shrinkage(pred, P_pseudo_s, ws)
        sweep_v4.loc[wh, ws] = macro_auc(Y, pred)
print(sweep_v4.round(4))
best_idx = np.unravel_index(np.nanargmax(sweep_v4.values), sweep_v4.shape)
print(f"  BEST: w_hour={hour_ws[best_idx[0]]}, w_site={site_ws[best_idx[1]]} → {sweep_v4.values[best_idx]:.4f}")
sweep_v4.to_csv(f"{OUT}/sweep_2d_hybrid.csv")

# ---------- 2D sweep: pseudo_hour × LABELED_hour (both hour signals at once) ----------
print("\n=== 2D: pseudo_hour × labeled_hour (test if hybrid > pure pseudo) ===")
sweep_hh = pd.DataFrame(index=hour_ws, columns=hour_ws, dtype=float)
for wp in hour_ws:
    for wl in hour_ws:
        pred = apply_prior(P_bruce, P_pseudo_h, wp)
        pred = apply_prior(pred, P_labeled_h, wl)
        sweep_hh.loc[wp, wl] = macro_auc(Y, pred)
print(sweep_hh.round(4))
sweep_hh.to_csv(f"{OUT}/sweep_2d_hour_hybrid.csv")
best_idx = np.unravel_index(np.nanargmax(sweep_hh.values), sweep_hh.shape)
print(f"  BEST: w_pseudo_h={hour_ws[best_idx[0]]}, w_labeled_h={hour_ws[best_idx[1]]} → {sweep_hh.values[best_idx]:.4f}")

# ---------- Plot the 2D sweep heatmap ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, df, title in zip(axes, [sweep, sweep_v4], ["pseudo_hour × pseudo_site", "labeled_hour × pseudo_site"]):
    im = ax.imshow(df.astype(float).values, aspect="auto", cmap="viridis", vmin=0.85, vmax=0.95)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([f"{w:.2g}" for w in df.columns])
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([f"{w:.2g}" for w in df.index])
    ax.set_xlabel("w_site_shrinkage")
    ax.set_ylabel("w_hour")
    ax.set_title(f"{title}\nbase=Bruce 0.867 (leakage-safe)")
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < 0.91 else "black", fontsize=7)
    plt.colorbar(im, ax=ax, label="macro-AUC")
plt.suptitle("HONEST prior-weight tuning on Bruce (leakage-safe base)\n"
             "Compare against CONTEXT.md v3 best 0.9585 / v4 best 0.9691 (both LEAKED)")
plt.tight_layout()
plt.savefig(f"{OUT}/heatmap_2d.png", dpi=110, bbox_inches="tight")
plt.close()

print(f"\nSaved plots + tables to {OUT}")
