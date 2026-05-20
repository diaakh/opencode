"""Experiment 4: Use RAW, ALIGNED model OOF outputs.

Cleaned up sources:
- exp019 (from labeled_oof_model7_pre_align, aligned)
- V73 raw (from v73_labeled_oof.npz, 3-shift TTA built-in)
- BirdMAE raw (from birdmae_labeled_oof.npz)
- ConvNeXt-RAG (from creative/, may be RAG-smoothed)
- BirdAVES-RAG, MLP-5seed

Goal: find LB-realistic best blend with per-cell entropy gating.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

CR = "/home/user/opencode/birdclef-2026/analysis/creative"
ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"

# Aligned exp019
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp019 = ex["P_exp019"]
Y = ex["Y"]
N, C = Y.shape
classes = ex["classes"]

# Raw V73 from kernel (already includes 3-shift TTA)
v73d = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)
P_v73 = v73d["P_v73"]
classes_v73 = v73d["classes"]
assert np.array_equal(classes, classes_v73), "class mismatch"

# Raw BirdMAE from kernel
bmd = np.load("/tmp/birdmae_oof_kernel/birdmae_labeled_oof.npz", allow_pickle=True)
P_bmae = bmd["P_birdmae"]
classes_bm = bmd["classes"]
assert np.array_equal(classes, classes_bm)

# ConvNeXt-RAG (smoothed but architecturally distinct from V73/BirdMAE)
cnd = np.load(f"{CR}/convnext_rag.npz")
P_cn = cnd["P_rag"]

# BirdAVES-RAG
bad = np.load(f"{CR}/birdaves_rag.npz")
P_ba = bad["P_rag_avg"]

# MLP-5seed
mlpd = np.load(f"{CR}/mlp_5seed_oof.npz")
P_mlp = mlpd["P_mlp"]

models = {
    "exp019": P_exp019,
    "v73_raw": P_v73,
    "birdmae": P_bmae,
    "convnext_rag": P_cn,
    "birdaves_rag": P_ba,
    "mlp_5seed": P_mlp,
}

def macro_auc(P, Y):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs), len(aucs)

def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 1) / (N - 1)
    return R

print("=== Individual AUCs (raw probs) ===")
for n, P in models.items():
    a, _ = macro_auc(P, Y)
    print(f"  {n:15s}: {a:.4f}")
print()

ranks = {n: rank_norm(P) for n, P in models.items()}
auc_base = macro_auc(P_exp019, Y)[0]
print(f"exp019 anchor: {auc_base:.4f}\n")

def entropy_gate_blend(P_anchor_rank, P_anchor_raw, P_other_rank, T=2.0, floor=0.5, ceil=0.95):
    """Per-cell entropy gate using ANCHOR's raw probabilities (not ranks).
    
    floor: minimum anchor weight
    ceil: maximum anchor weight
    """
    eps = 1e-6
    p = np.clip(P_anchor_raw, eps, 1-eps)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
    conf = (1 - H) ** T
    w = np.clip(conf, floor, ceil)
    return w * P_anchor_rank + (1 - w) * P_other_rank

print("="*72)
print("2-way entropy gate vs blanket rank-blend (CLEAN models)")
print("="*72)
print(f"{'helper':<18s} | gate floor=0.5 | gate floor=0.7 | blanket w=0.3 | blanket w=0.5")
for hname in ["v73_raw", "birdmae", "convnext_rag", "birdaves_rag", "mlp_5seed"]:
    R_o = ranks[hname]
    P_g50 = entropy_gate_blend(ranks["exp019"], P_exp019, R_o, T=2.0, floor=0.5)
    P_g70 = entropy_gate_blend(ranks["exp019"], P_exp019, R_o, T=2.0, floor=0.7)
    P_b3 = 0.7 * ranks["exp019"] + 0.3 * R_o
    P_b5 = 0.5 * ranks["exp019"] + 0.5 * R_o
    a50 = macro_auc(P_g50, Y)[0]
    a70 = macro_auc(P_g70, Y)[0]
    ab3 = macro_auc(P_b3, Y)[0]
    ab5 = macro_auc(P_b5, Y)[0]
    print(f"{hname:<18s} |  {a50:.4f} ({a50-auc_base:+.3f}) |  {a70:.4f} ({a70-auc_base:+.3f}) |  {ab3:.4f} ({ab3-auc_base:+.3f}) |  {ab5:.4f} ({ab5-auc_base:+.3f})")

# Multi-helper with entropy gate
print()
print("="*72)
print("Multi-helper entropy gate (exp019 + N, equal split among helpers)")
print("="*72)

def multi_gate(anchor_rank, anchor_raw, helper_ranks, T=2.0, floor=0.5, ceil=0.95):
    eps = 1e-6
    p = np.clip(anchor_raw, eps, 1-eps)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
    conf = (1 - H) ** T
    w = np.clip(conf, floor, ceil)
    out = w * anchor_rank
    nh = len(helper_ranks)
    each = (1 - w) / nh
    for hr in helper_ranks:
        out = out + each * hr
    return out

helper_sets = [
    ["v73_raw"],
    ["v73_raw", "convnext_rag"],
    ["v73_raw", "convnext_rag", "birdaves_rag"],
    ["birdmae"],
    ["birdmae", "v73_raw"],
    ["birdmae", "v73_raw", "convnext_rag"],
    ["birdmae", "convnext_rag"],
    ["convnext_rag", "birdaves_rag"],
    ["v73_raw", "birdmae", "convnext_rag", "birdaves_rag"],
]
for helpers in helper_sets:
    helper_ranks = [ranks[h] for h in helpers]
    print(f"helpers={helpers}")
    for floor in [0.3, 0.4, 0.5, 0.6, 0.7]:
        P_m = multi_gate(ranks["exp019"], P_exp019, helper_ranks, T=2.0, floor=floor)
        a = macro_auc(P_m, Y)[0]
        print(f"  floor={floor:.1f}: {a:.4f}  Δ={a-auc_base:+.4f}")
    print()

# Special: GEOMETRIC entropy gate (multiply rank-space)
print("="*72)
print("Geometric entropy gate (log-space blend)")
print("="*72)
def geom_gate(anchor_rank, anchor_raw, helper_ranks, T=2.0, floor=0.5, ceil=0.95):
    eps = 1e-6
    p = np.clip(anchor_raw, eps, 1-eps)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
    conf = (1 - H) ** T
    w = np.clip(conf, floor, ceil)
    nh = len(helper_ranks)
    each = (1 - w) / nh
    out = w * np.log(anchor_rank + eps)
    for hr in helper_ranks:
        out = out + each * np.log(hr + eps)
    return np.exp(out)

for helpers in [["v73_raw"], ["v73_raw", "convnext_rag"], ["birdmae"], ["birdmae", "v73_raw"]]:
    helper_ranks = [ranks[h] for h in helpers]
    print(f"helpers={helpers}")
    for floor in [0.4, 0.5, 0.6]:
        P_m = geom_gate(ranks["exp019"], P_exp019, helper_ranks, T=2.0, floor=floor)
        a = macro_auc(P_m, Y)[0]
        print(f"  floor={floor:.1f}: {a:.4f}  Δ={a-auc_base:+.4f}")
