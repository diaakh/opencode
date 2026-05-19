"""Experiment 21: Build a 'Perch v2 + trained head' approximation as new LB anchor.

Public kernels: Perch v2 + trained classifier head → LB 0.905-0.912
We have: Perch v2 embeddings (1536-dim) on labeled train_soundscapes

Approach:
1. Train K-fold CV MLP on Perch v2 embeddings to predict Y
2. Get CV-OOF predictions (no leakage)
3. Compute our metric on these predictions  
4. Compare predicted LB to known public tier (0.91)
5. Validates metric's calibration at the LB 0.91 tier
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import re

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
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

# Load Perch v2 embeddings + raw OOF
pe = np.load("/tmp/perch20_blend.npz", allow_pickle=True)
emb = pe["emb_perch20"]  # (739, 1536)
P_perch_raw = pe["P_perch20"]
print(f"Perch v2 embeddings: {emb.shape}, range=[{emb.min():.3f}, {emb.max():.3f}]")

# Build site-aware K-fold (group by site to avoid trivial within-site leak)
unique_sites = np.unique(sites)
print(f"Sites: {len(unique_sites)} → group K-fold")

# Train CV-OOF predictions using both LR (matches public starter) and MLP
def cv_predict(emb, Y, sites, model_type="lr", n_splits=5):
    """Group K-fold OOF predictions."""
    P_oof = np.zeros_like(Y, dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(emb, Y[:, 0], groups=sites)):
        for c in range(C):
            if Y[tr, c].sum() < 2 or Y[tr, c].sum() == len(tr):
                P_oof[te, c] = Y[tr, c].mean()
                continue
            try:
                if model_type == "lr":
                    clf = LogisticRegression(C=1.0, max_iter=200, solver='lbfgs')
                elif model_type == "mlp":
                    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=50, random_state=42)
                clf.fit(emb[tr], Y[tr, c])
                if hasattr(clf, "predict_proba"):
                    P_oof[te, c] = clf.predict_proba(emb[te])[:, 1]
                else:
                    P_oof[te, c] = clf.decision_function(emb[te])
            except Exception as e:
                P_oof[te, c] = Y[tr, c].mean()
        if fold == 0:
            print(f"  Fold {fold} done, {len(tr)} train, {len(te)} test")
    return P_oof

print("\nTraining LR on Perch v2 embeddings (5-fold group CV by site)...")
P_lr = cv_predict(emb, Y, sites, model_type="lr", n_splits=5)
print("Training MLP on Perch v2 embeddings (5-fold group CV by site)...")
P_mlp_perch = cv_predict(emb, Y, sites, model_type="mlp", n_splits=5)

def macro_auc(P):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs) if aucs else 0

def site_metrics(P):
    site_data = []
    for site in sorted(set(sites)):
        mask = sites == site
        if mask.sum() < 15: continue
        Y_s, P_s = Y[mask], P[mask]
        aucs = []
        for c in range(C):
            if Y_s[:, c].sum() < 2 or Y_s[:, c].sum() == mask.sum(): continue
            if P_s[:, c].max() == P_s[:, c].min(): continue
            try: aucs.append(roc_auc_score(Y_s[:, c], P_s[:, c]))
            except: pass
        if aucs:
            site_data.append((mask.sum(), np.mean(aucs)))
    counts = np.array([c for c, _ in site_data])
    aucs = np.array([a for _, a in site_data])
    w = counts / counts.sum()
    mean = (w * aucs).sum()
    var = (w * (aucs - mean)**2).sum()
    return np.sqrt(var), mean

def predict_lb(P):
    oa = macro_auc(P)
    _, mean = site_metrics(P)
    gap = oa - mean
    return 0.277 + 0.714 * mean - 0.894 * gap, oa, mean, gap

print("\n" + "="*80)
print("NEW ANCHOR: Perch v2 + trained head (CV-OOF) — matches public LB 0.91 tier")
print("="*80)
print(f"\nLR head:")
pred, oa, mean, gap = predict_lb(P_lr)
print(f"  overall_auc: {oa:.4f}")
print(f"  site_mean:   {mean:.4f}")
print(f"  gap:         {gap:+.4f}")
print(f"  PREDICTED LB: {pred:.4f}")
print(f"  EXPECTED LB:  0.905-0.912 (public Perch v2 + LR head kernels)")
print(f"  METRIC ERROR: {pred - 0.908:+.4f}")

print(f"\nMLP head:")
pred, oa, mean, gap = predict_lb(P_mlp_perch)
print(f"  overall_auc: {oa:.4f}")
print(f"  site_mean:   {mean:.4f}")
print(f"  gap:         {gap:+.4f}")
print(f"  PREDICTED LB: {pred:.4f}")
print(f"  EXPECTED LB:  0.905-0.912 (public Perch v2 + MLP head kernels)")

# Save predictions
np.savez(f"{ETT}/perch_v2_with_head_oof.npz", P_lr=P_lr, P_mlp=P_mlp_perch, Y=Y)
print(f"\nSaved: {ETT}/perch_v2_with_head_oof.npz")
