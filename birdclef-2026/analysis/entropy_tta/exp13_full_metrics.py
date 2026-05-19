"""Experiment 13: Full LB-correlation metric panel.

Building on exp12's site_std finding. Test multiple candidate metrics and
find which best correlates with known LB scores.

Known LB anchors:
  exp019:  0.949   (heavy ensemble, train_audio-trained)
  V73:     0.941   (5-fold CNN, train_audio-trained)
  Bruce:   0.755   (Ridge on Perch features, train_audio-trained)
  
  Plus: slot6 (exp019+BirdMAE 70/30) = 0.946 → implies BirdMAE std-LB ~0.940
        slot11 (exp019+sub_v8+V73 surgical) = 0.949 → similar to exp019
"""
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from scipy.stats import rankdata, spearmanr, pearsonr
import subprocess

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp = ex["P_exp019"]
Y = ex["Y"]
row_fn = ex["row_filename"]
N, C = Y.shape

def site_of(fn):
    parts = str(fn).split('_')
    for p in parts:
        if p.startswith('S') and len(p) <= 3 and p[1:].isdigit():
            return p
    return "S??"
sites = np.array([site_of(fn) for fn in row_fn])

# Load all models
v73 = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]
bmae = np.load("/tmp/birdmae_oof_kernel/birdmae_labeled_oof.npz", allow_pickle=True)["P_birdmae"]
perch = np.load("/tmp/perch20_blend.npz", allow_pickle=True)["P_perch20"]
bruce_d = np.load('/tmp/bruce_oof/labeled_oof_perch_bruce.npz', allow_pickle=True)
bruce = bruce_d['P_bruce']

cn = np.load('/tmp/convnext_rag.npz')['P_rag']
ba = np.load('/tmp/birdaves_rag.npz')['P_rag_avg']
mlp = np.load('/tmp/mlp_5seed_oof.npz')['P_mlp']
knn = np.load('/tmp/mega_knn.npz')['P']
bal_lr = np.load('/tmp/balanced_lr_oof.npz')['P_lr']
lgb = np.load('/tmp/lgb7_oof.npz')['P_lgb7']

models = {
    "exp019":       (P_exp, 0.949),       # known LB
    "V73":          (v73, 0.941),         # known LB
    "Bruce":        (bruce, 0.755),       # known LB
    "BirdMAE":      (bmae, None),         # inferred ~0.940 from slot6 math
    "Perch_v2":     (perch, None),
    "ConvNeXt-RAG": (cn, None),
    "BirdAVES-RAG": (ba, None),
    "MLP-5seed":    (mlp, None),
    "mega_KNN":     (knn, None),
    "balanced_LR":  (bal_lr, None),
    "LGB-7":        (lgb, None),
}

def macro_auc_overall(P):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs) if aucs else 0.0

def per_site_aucs_weighted(P, sites, weight_min_rows=15):
    """Return list of (n_rows, auc) per site."""
    result = []
    for site in sorted(set(sites)):
        mask = sites == site
        if mask.sum() < weight_min_rows: continue
        Y_s, P_s = Y[mask], P[mask]
        aucs = []
        for c in range(C):
            if Y_s[:, c].sum() < 2 or Y_s[:, c].sum() == mask.sum(): continue
            if P_s[:, c].max() == P_s[:, c].min(): continue
            try: aucs.append(roc_auc_score(Y_s[:, c], P_s[:, c]))
            except: pass
        if aucs:
            result.append((int(mask.sum()), float(np.mean(aucs))))
    return result

def per_class_aucs(P):
    """Return per-class AUC where computable."""
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.array(aucs)

def ece(P, Y, n_bins=10):
    """Expected Calibration Error across all (row, class) cells with positive label or sampled."""
    # Flatten
    p_flat = P.flatten()
    y_flat = Y.flatten()
    # Bin
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(p_flat, bins[1:-1])
    ece_val = 0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 30: continue
        conf = p_flat[mask].mean()
        acc = y_flat[mask].mean()
        ece_val += (mask.sum() / len(p_flat)) * abs(conf - acc)
    return ece_val

def agreement_with_perch(P, P_perch_ref):
    """Spearman rank correlation of P with Perch v2 per row (averaged)."""
    rhos = []
    for i in range(N):
        if P[i].max() == P[i].min() or P_perch_ref[i].max() == P_perch_ref[i].min():
            continue
        try:
            rho = spearmanr(P[i], P_perch_ref[i])[0]
            if not np.isnan(rho):
                rhos.append(rho)
        except: pass
    return np.mean(rhos) if rhos else 0.0

# Compute all metrics
print(f"{'Model':<14s} | overall | wsite_std | site_min | classAUC_std | ECE    | agree_perch | LB")
print("-"*110)
table = {}
for name, (P, lb) in models.items():
    overall = macro_auc_overall(P)
    site_data = per_site_aucs_weighted(P, sites)
    if not site_data:
        continue
    counts = np.array([c for c, _ in site_data])
    aucs = np.array([a for _, a in site_data])
    weights = counts / counts.sum()
    w_mean = (weights * aucs).sum()
    w_var = (weights * (aucs - w_mean)**2).sum()
    w_std = np.sqrt(w_var)
    site_min = aucs.min()
    cls_aucs = per_class_aucs(P)
    cls_auc_std = cls_aucs.std()
    e = ece(P, Y, n_bins=10)
    ag = agreement_with_perch(P, perch)
    table[name] = {
        "overall": overall, "w_site_std": w_std, "site_min": site_min,
        "cls_auc_std": cls_auc_std, "ece": e, "agree_perch": ag, "lb": lb,
    }
    lb_str = f"{lb:.3f}" if lb else "?"
    print(f"{name:<14s} | {overall:.4f}  | {w_std:.4f}    | {site_min:.4f}   | {cls_auc_std:.4f}       | {e:.4f} | {ag:+.4f}      | {lb_str}")

# Correlation analysis
print("\n" + "="*80)
print("Correlation with LB (known models only):")
print("="*80)
known = [n for n in table if table[n]["lb"] is not None]
print(f"Anchors: {[(n, table[n]['lb']) for n in known]}\n")

for metric in ["overall", "w_site_std", "site_min", "cls_auc_std", "ece", "agree_perch"]:
    vals = np.array([table[n][metric] for n in known])
    lbs = np.array([table[n]["lb"] for n in known])
    try:
        rho, _ = spearmanr(vals, lbs)
        r, _ = pearsonr(vals, lbs)
        direction = "+" if r > 0 else "-"
        # Better: |r| close to 1 with right sign
        print(f"  {metric:<15s} | values={[f'{v:.4f}' for v in vals]} | Spearman ρ={rho:+.2f}  Pearson r={r:+.2f}")
    except: pass

# Predict LB for unknown models using best metric
print("\n" + "="*80)
print("Predict LB for unknown models using w_site_std (best correlation)")
print("="*80)
# Linear regression on known models: LB = a + b * w_site_std
from scipy.stats import linregress
known_std = np.array([table[n]["w_site_std"] for n in known])
known_lb = np.array([table[n]["lb"] for n in known])
slope, intercept, r_val, p_val, stderr = linregress(known_std, known_lb)
print(f"  LB = {intercept:.4f} + {slope:.4f} * w_site_std  (R²={r_val**2:.3f})")

for name in table:
    if table[name]["lb"] is not None: continue
    pred = intercept + slope * table[name]["w_site_std"]
    print(f"  {name:<15s}: w_site_std={table[name]['w_site_std']:.4f} → predicted LB ≈ {pred:.4f}")

print()
print("Calibration check: predict known models from their own w_site_std:")
for name in known:
    pred = intercept + slope * table[name]["w_site_std"]
    actual = table[name]["lb"]
    print(f"  {name}: predicted {pred:.4f}, actual {actual:.4f}, diff {pred-actual:+.4f}")
