"""Experiment 3: Conservative entropy gates + multi-model.

The OOF→LB gap for BirdMAE is +0.02 OOF but -0.003 LB (slot6 blanket).
Test: does per-cell entropy gating transfer better because it only swaps
when exp019 is genuinely uncertain?

Conservative gate: floor on exp019's weight (min 0.6, 0.7, 0.8 even when uncertain).
Multi-model: include exp019 + ConvNeXt-RAG (non-leaking) + BirdMAE (leaking).
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

CR = "/home/user/opencode/birdclef-2026/analysis/creative"
ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"

ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp019 = ex["P_exp019"]
Y = ex["Y"]
N, C = Y.shape

models = {"exp019": P_exp019}
for nm, p, k in [
    ("birdmae", "birdmae_blend.npz", "P_birdmae"),
    ("convnext_rag", "convnext_rag.npz", "P_rag"),
    ("birdaves_rag", "birdaves_rag.npz", "P_rag_avg"),
    ("v73_rag", "v73_rag.npz", "P_rag_v73"),
    ("mlp_5seed", "mlp_5seed_oof.npz", "P_mlp"),
]:
    try:
        d = np.load(f"{CR}/{p}")
        if k in d.keys() and d[k].shape == (N, C):
            models[nm] = d[k].astype(np.float32)
    except: pass
print(f"Models: {list(models.keys())}")

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

ranks = {n: rank_norm(P) for n, P in models.items()}
auc_base, _ = macro_auc(P_exp019, Y)
print(f"\nexp019 anchor: {auc_base:.4f}\n")

def entropy_gate_2way(P_anchor, P_other, T=2.0, floor=0.5, ceil=0.95):
    """Per-cell entropy gate.
    floor: minimum weight for anchor (never drop exp019 below this)
    ceil: maximum weight for anchor
    """
    eps = 1e-6
    p = np.clip(P_anchor, eps, 1-eps)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)  # [0, 1]
    conf = (1 - H) ** T
    w_anchor = np.clip(conf, floor, ceil)
    return w_anchor * P_anchor + (1 - w_anchor) * P_other

print("="*70)
print("Conservative gate: floor=anchor weight minimum (when exp019 uncertain)")
print("="*70)
print(f"{'helper':<15s} | floor=0.50 | floor=0.60 | floor=0.70 | floor=0.80 | floor=0.90")
for hname in ["birdmae", "convnext_rag", "birdaves_rag", "mlp_5seed", "v73_rag"]:
    if hname not in ranks: continue
    R_o = ranks[hname]
    row = f"{hname:<15s} |"
    for floor in [0.50, 0.60, 0.70, 0.80, 0.90]:
        P_b = entropy_gate_2way(ranks["exp019"], R_o, T=2.0, floor=floor, ceil=0.95)
        auc, _ = macro_auc(P_b, Y)
        delta = auc - auc_base
        row += f"  {auc:.4f}({delta:+.3f}) |"
    print(row)

print()
print("="*70)
print("Multi-model entropy gate (exp019 anchor + N helpers, conservative)")
print("="*70)

def multi_entropy_blend(ranks_dict, helpers, anchor="exp019", T_conf=2.0, anchor_floor=0.5, anchor_ceil=0.9):
    """Anchor + helpers, weights are anchor's confidence + softmax over helpers."""
    eps = 1e-6
    R_a = ranks_dict[anchor]
    P_a = ranks_dict[anchor]  # for entropy calc, use rank-norm
    p = np.clip(P_a, eps, 1-eps)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
    conf_a = (1 - H) ** T_conf  # higher = more confident
    w_anchor = np.clip(conf_a, anchor_floor, anchor_ceil)
    # remaining (1-w_anchor) split equally among helpers
    n_help = len(helpers)
    w_help_each = (1 - w_anchor) / n_help
    out = w_anchor * R_a
    for h in helpers:
        out = out + w_help_each * ranks_dict[h]
    return out

helper_sets = [
    ["birdmae"],
    ["birdmae", "convnext_rag"],
    ["birdmae", "convnext_rag", "birdaves_rag"],
    ["convnext_rag", "birdaves_rag"],
    ["convnext_rag"],
    ["birdmae", "mlp_5seed"],
]
for helpers in helper_sets:
    print(f"\nhelpers={helpers}")
    for floor in [0.5, 0.6, 0.7, 0.8]:
        P_m = multi_entropy_blend(ranks, helpers, T_conf=2.0, anchor_floor=floor, anchor_ceil=0.9)
        auc, _ = macro_auc(P_m, Y)
        print(f"  floor={floor}: {auc:.4f}  Δ={auc-auc_base:+.4f}")

print()
print("="*70)
print("Logit-confidence weighted multi-blend (per-cell)")
print("="*70)
def logit_conf_blend(ranks_dict, source_models, T_w=1.0):
    """Each cell, weight ∝ exp(T * |logit(raw_prob)|)."""
    eps = 1e-6
    stack_R = np.stack([ranks_dict[n] for n in source_models], axis=-1)
    # use raw probs (not ranks) for confidence
    stack_conf = np.stack([np.abs(np.log(np.clip(models[n], eps, 1-eps) / np.clip(1 - models[n], eps, 1-eps))) for n in source_models], axis=-1)
    logits = T_w * stack_conf
    logits = logits - logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=-1, keepdims=True)
    return (stack_R * w).sum(axis=-1), w

for combo in [["exp019", "birdmae"], ["exp019", "birdmae", "convnext_rag"], 
              ["exp019", "convnext_rag"], ["exp019", "birdmae", "convnext_rag", "birdaves_rag"]]:
    for T in [0.3, 0.5, 1.0, 2.0]:
        P_b, w = logit_conf_blend(ranks, combo, T_w=T)
        auc, _ = macro_auc(P_b, Y)
        # weight stats per model
        avg_w = w.mean(axis=(0,1))
        wmsg = " ".join([f"{n}={avg_w[i]:.2f}" for i, n in enumerate(combo)])
        print(f"  combo={combo} T={T}: {auc:.4f} Δ={auc-auc_base:+.4f}  weights: {wmsg}")
    print()

# === Final winner search: scan over T, floor, helpers ===
print("\n" + "="*70)
print("Final scan: best config")
print("="*70)
best = (auc_base, "exp019_only")
for helpers in [["birdmae"], ["birdmae", "convnext_rag"], ["convnext_rag"], 
                ["birdmae", "convnext_rag", "birdaves_rag"]]:
    for T in [1.0, 2.0, 3.0]:
        for floor in [0.4, 0.5, 0.6, 0.7, 0.8]:
            for ceil in [0.85, 0.9, 0.95]:
                P_m = multi_entropy_blend(ranks, helpers, T_conf=T, anchor_floor=floor, anchor_ceil=ceil)
                auc, _ = macro_auc(P_m, Y)
                if auc > best[0]:
                    best = (auc, f"helpers={helpers}, T={T}, floor={floor}, ceil={ceil}")
print(f"BEST: {best[0]:.4f}  ({best[1]})")
