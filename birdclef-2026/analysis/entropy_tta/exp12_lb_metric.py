"""Experiment 12: Find a metric correlated with LB instead of validation AUC.

Hypothesis: labeled OOF (train_soundscapes_labels) is biased because labels are
recorded at SAME sites as train_audio. Models trained on train_audio learn site
biases → labeled OOF inflated.

Candidate LB-correlated metrics:
1. Per-site AUC variance: high variance = site-dependent = poor LB transfer
2. Min-site AUC: worst-case site performance proxies generalization floor
3. Agreement with Perch v2: external arbiter (zero train_audio exposure)
4. Calibration ECE: well-calibrated → generalizes
5. Per-class AUC variance: consistent across classes vs unstable

Known LB scores:
  exp019:  0.949
  V73:     0.941
  Bruce:   0.755  (huge LB drop from OOF)
  slot6 (exp019+BirdMAE 70/30): 0.946 (BirdMAE drags -0.003)
  slot11:  0.949 (surgical)
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr
import re

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"

ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp = ex["P_exp019"]
Y = ex["Y"]
row_filename = ex["row_filename"]
N, C = Y.shape

# Parse site from filename: BC2026_Train_NNNN_SXX_YYYYMMDD_HHMMSS.ogg → SXX
def site_of(fn):
    parts = str(fn).split('_')
    for p in parts:
        if p.startswith('S') and len(p) <= 3 and p[1:].isdigit():
            return p
    return "S??"

sites = np.array([site_of(fn) for fn in row_filename])
print(f"Sites: {sorted(set(sites))}")
print(f"Rows per site: {dict(zip(*np.unique(sites, return_counts=True)))}")

# Load all models
v73 = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]
bmae = np.load("/tmp/birdmae_oof_kernel/birdmae_labeled_oof.npz", allow_pickle=True)["P_birdmae"]
perch = np.load("/tmp/perch20_blend.npz", allow_pickle=True)["P_perch20"]

# Get other models from git history
import subprocess
for fn in ['convnext_rag.npz', 'birdaves_rag.npz', 'mlp_5seed_oof.npz', 'mega_knn.npz', 'balanced_lr_oof.npz', 'lgb7_oof.npz']:
    target = f'/tmp/{fn}'
    try:
        with open(target, 'wb') as f:
            subprocess.run(['git', 'show', f'7c984ba:birdclef-2026/analysis/creative/{fn}'], stdout=f, check=False)
    except: pass

cn = np.load('/tmp/convnext_rag.npz')['P_rag']
ba = np.load('/tmp/birdaves_rag.npz')['P_rag_avg']
mlp = np.load('/tmp/mlp_5seed_oof.npz')['P_mlp']
knn = np.load('/tmp/mega_knn.npz')['P']  # mega-KNN raw scores
bal_lr = np.load('/tmp/balanced_lr_oof.npz')['P_lr']
lgb = np.load('/tmp/lgb7_oof.npz')['P_lgb7']

# Bruce we don't have separately — use mega_knn as proxy for Bruce-style model

# Also load Bruce from labeled OOF kernel
bruce_d = np.load('/tmp/bruce_oof/labeled_oof_perch_bruce.npz', allow_pickle=True)
bruce = bruce_d['P_bruce']
perch_raw_alt = bruce_d['P_perch_logits']

models = {
    "exp019": P_exp,
    "V73": v73,
    "BirdMAE": bmae,
    "Bruce": bruce,
    "Perch_v2": perch,
    "ConvNeXt-RAG": cn,
    "BirdAVES-RAG": ba,
    "MLP-5seed": mlp,
    "mega_KNN": knn,
    "balanced_LR": bal_lr,
    "LGB-7": lgb,
}

# Known LB scores
known_lb = {
    "exp019": 0.949,
    "V73": 0.941,
    "Bruce": 0.755,
}

def macro_auc_overall(P, Y):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs)

def per_site_aucs(P, Y, sites):
    """Return dict of site → macro AUC computed only on that site's rows."""
    result = {}
    for site in sorted(set(sites)):
        mask = sites == site
        if mask.sum() < 10: continue  # need enough rows
        Y_s = Y[mask]
        P_s = P[mask]
        aucs = []
        for c in range(C):
            if Y_s[:, c].sum() < 2 or Y_s[:, c].sum() == mask.sum(): continue
            if P_s[:, c].max() == P_s[:, c].min(): continue
            try: aucs.append(roc_auc_score(Y_s[:, c], P_s[:, c]))
            except: pass
        if aucs:
            result[site] = np.mean(aucs)
    return result

# Compute per-site AUCs
print("\n" + "="*75)
print(f"{'Model':<15s} | Overall | Per-site AUCs (mean/std/min)            | LB")
print("="*75)

metrics_table = {}
for name, P in models.items():
    overall = macro_auc_overall(P, Y)
    site_aucs = per_site_aucs(P, Y, sites)
    if not site_aucs:
        continue
    sa = np.array(list(site_aucs.values()))
    metrics_table[name] = {
        "overall_auc": overall,
        "site_mean": sa.mean(),
        "site_std": sa.std(),
        "site_min": sa.min(),
        "site_max": sa.max(),
        "site_range": sa.max() - sa.min(),
        "n_sites": len(site_aucs),
        "lb": known_lb.get(name, None),
    }
    lb_str = f"{known_lb[name]:.3f}" if name in known_lb else "?"
    print(f"{name:<15s} | {overall:.4f}  | mean={sa.mean():.3f} std={sa.std():.3f} min={sa.min():.3f} ({len(site_aucs)} sites) | {lb_str}")

# Now: correlation between per-site std and LB (for known-LB models)
print("\n" + "="*75)
print("Correlation analysis: which metric predicts LB?")
print("="*75)

known_models = [n for n in metrics_table if metrics_table[n]["lb"] is not None]
print(f"Known LB models: {known_models}")

# Build table
for metric in ["overall_auc", "site_mean", "site_std", "site_min", "site_max", "site_range"]:
    vals = [metrics_table[n][metric] for n in known_models]
    lbs = [metrics_table[n]["lb"] for n in known_models]
    if len(vals) < 2: continue
    try:
        rho, p = spearmanr(vals, lbs)
        # Pearson too
        r = np.corrcoef(vals, lbs)[0, 1]
        # Direction
        print(f"  {metric:<20s} | values={[f'{v:.3f}' for v in vals]} | LBs={lbs} | Spearman ρ={rho:+.2f} Pearson r={r:+.2f}")
    except: pass

# Hypothesis test: low site_std → high LB
print("\nKey insight per model:")
for n in known_models:
    m = metrics_table[n]
    print(f"  {n}: overall_auc={m['overall_auc']:.4f}, site_std={m['site_std']:.4f}, LB={m['lb']}")
    print(f"    OOF→LB gap: {m['lb'] - m['overall_auc']:+.4f}")
