"""Experiment 9: Per-class logistic regression stacking.

Instead of fixed/entropy weights, learn per-class blend weights via LR with
K-fold CV (to avoid using the same OOF for training and eval on same rows).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scipy.stats import rankdata

CR = "/home/user/opencode/birdclef-2026/analysis/creative"
ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"

ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp = ex["P_exp019"]
Y = ex["Y"]
N, C = Y.shape

v73 = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]
cn = np.load(f"{CR}/convnext_rag.npz")["P_rag"]
ba = np.load(f"{CR}/birdaves_rag.npz")["P_rag_avg"]
mlp = np.load(f"{CR}/mlp_5seed_oof.npz")["P_mlp"]

def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 1) / (N - 1)
    return R

# Stack features
R_exp = rank_norm(P_exp)
R_cn = rank_norm(cn)
R_ba = rank_norm(ba)
R_mlp = rank_norm(mlp)
R_v73 = rank_norm(v73)
R_sub_proxy = (R_cn + R_ba + R_mlp) / 3

# Per-class stacking: features = [R_exp_c, R_sub_c, R_v73_c], target = Y_c
def kfold_stack(features_list, Y, n_splits=5):
    """Predict each class's OOF probs via K-fold LR stacking."""
    M = len(features_list)
    X = np.stack(features_list, axis=-1)  # (N, C, M)
    P_out = np.zeros_like(Y, dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for c in range(C):
        if Y[:, c].sum() < 5 or Y[:, c].sum() == N:
            P_out[:, c] = features_list[0][:, c]  # fall back to anchor
            continue
        Xc = X[:, c, :]
        Yc = Y[:, c]
        for tr_idx, te_idx in kf.split(Xc):
            lr = LogisticRegression(C=1.0, max_iter=200, solver='lbfgs')
            try:
                lr.fit(Xc[tr_idx], Yc[tr_idx])
                P_out[te_idx, c] = lr.predict_proba(Xc[te_idx])[:, 1]
            except Exception:
                P_out[te_idx, c] = Xc[te_idx, 0]  # fall back to anchor
    return P_out

def macro_auc(P, Y):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs)

base = macro_auc(P_exp, Y)
print(f"baseline: {base:.4f}\n")

# Stacking with various feature sets
print("="*70)
print("LR stacking (5-fold CV) per class")
print("="*70)

print("\nFeatures: [exp019]")
P_stack = kfold_stack([R_exp], Y)
print(f"  auc: {macro_auc(P_stack, Y):.4f}  Δ={macro_auc(P_stack, Y)-base:+.4f}")

print("\nFeatures: [exp019, sub_v8_proxy]")
P_stack = kfold_stack([R_exp, R_sub_proxy], Y)
print(f"  auc: {macro_auc(P_stack, Y):.4f}  Δ={macro_auc(P_stack, Y)-base:+.4f}")

print("\nFeatures: [exp019, sub_v8_proxy, v73]")
P_stack = kfold_stack([R_exp, R_sub_proxy, R_v73], Y)
print(f"  auc: {macro_auc(P_stack, Y):.4f}  Δ={macro_auc(P_stack, Y)-base:+.4f}")

print("\nFeatures: [exp019, convnext, birdaves, mlp, v73]")
P_stack = kfold_stack([R_exp, R_cn, R_ba, R_mlp, R_v73], Y)
print(f"  auc: {macro_auc(P_stack, Y):.4f}  Δ={macro_auc(P_stack, Y)-base:+.4f}")

print("\nFeatures: [exp019, convnext, birdaves] (no v73, no leaky)")
P_stack = kfold_stack([R_exp, R_cn, R_ba], Y)
print(f"  auc: {macro_auc(P_stack, Y):.4f}  Δ={macro_auc(P_stack, Y)-base:+.4f}")

print("\nFeatures: [exp019, convnext_rag] (cleanest 2-way)")
P_stack = kfold_stack([R_exp, R_cn], Y)
print(f"  auc: {macro_auc(P_stack, Y):.4f}  Δ={macro_auc(P_stack, Y)-base:+.4f}")
