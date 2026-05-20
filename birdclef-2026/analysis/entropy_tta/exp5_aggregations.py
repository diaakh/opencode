"""Experiment 5: Compare aggregation strategies more carefully.

What we want to know: when combining exp019 with N helpers, what aggregation
beats arithmetic mean of ranks?

Strategies tested on labeled OOF:
- Arithmetic mean of ranks
- Geometric mean (log-domain)
- Median
- Trimmed mean
- Entropy-weighted (per-cell)
- Logit-space mean
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

v73d = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)
P_v73 = v73d["P_v73"]
bmd = np.load("/tmp/birdmae_oof_kernel/birdmae_labeled_oof.npz", allow_pickle=True)
P_bmae = bmd["P_birdmae"]

cnd = np.load(f"{CR}/convnext_rag.npz")
P_cn = cnd["P_rag"]
bad = np.load(f"{CR}/birdaves_rag.npz")
P_ba = bad["P_rag_avg"]
mlpd = np.load(f"{CR}/mlp_5seed_oof.npz")
P_mlp = mlpd["P_mlp"]

models = {
    "exp019": P_exp019,
    "v73": P_v73,
    "convnext_rag": P_cn,
    "birdaves_rag": P_ba,
    "mlp_5seed": P_mlp,
    # birdmae excluded — known OOF leak
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

ranks = {n: rank_norm(P) for n, P in models.items()}
print(f"models: {list(models.keys())}")
print(f"exp019 baseline: {macro_auc(P_exp019, Y)[0]:.4f}\n")

# Combinations to test
combos_to_test = [
    ["exp019", "v73"],
    ["exp019", "convnext_rag"],
    ["exp019", "v73", "convnext_rag"],
    ["exp019", "v73", "convnext_rag", "birdaves_rag"],
    ["exp019", "v73", "convnext_rag", "birdaves_rag", "mlp_5seed"],
]

def aggregate(stack, method, anchor_idx=0):
    """stack: (N, C, M) rank-normalized. anchor_idx: index of exp019 in stack."""
    N_, C_, M_ = stack.shape
    eps = 1e-7
    if method == "mean":
        return stack.mean(axis=-1)
    elif method == "geom":
        return np.exp(np.log(stack + eps).mean(axis=-1))
    elif method == "median":
        return np.median(stack, axis=-1)
    elif method == "trim":
        if M_ < 3: return stack.mean(axis=-1)
        srt = np.sort(stack, axis=-1)
        return srt[..., 1:-1].mean(axis=-1) if M_ >= 4 else srt[..., 1:].mean(axis=-1)
    elif method == "logit":
        # mean in logit space
        p = np.clip(stack, eps, 1-eps)
        return 1 / (1 + np.exp(-np.log(p / (1-p)).mean(axis=-1)))
    elif method.startswith("anchor_gate"):
        # use anchor (exp019)'s rank confidence (distance from 0.5)
        anchor_rank = stack[..., anchor_idx]
        # rank confidence: |rank - 0.5| * 2 → [0, 1]
        anchor_conf = np.abs(anchor_rank - 0.5) * 2
        floor = float(method.split("_")[-1])
        w_a = np.clip(anchor_conf ** 2, floor, 0.95)
        others = [m for m in range(M_) if m != anchor_idx]
        w_help_each = (1 - w_a) / max(len(others), 1)
        out = w_a * stack[..., anchor_idx]
        for m in others:
            out = out + w_help_each * stack[..., m]
        return out
    elif method.startswith("raw_gate"):
        # entropy gate based on exp019 raw probability
        floor = float(method.split("_")[-1])
        # need access to raw probs — passed separately
        raise NotImplementedError("use raw_gate_external")
    return stack.mean(axis=-1)

def raw_gate_external(stack, raw_anchor, T, floor, ceil=0.95, anchor_idx=0):
    """Entropy gate using ANCHOR's raw prob entropy."""
    p = np.clip(raw_anchor, 1e-6, 1-1e-6)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
    conf = (1 - H) ** T
    w_a = np.clip(conf, floor, ceil)
    M_ = stack.shape[-1]
    others = [m for m in range(M_) if m != anchor_idx]
    w_help_each = (1 - w_a) / max(len(others), 1)
    out = w_a * stack[..., anchor_idx]
    for m in others:
        out = out + w_help_each * stack[..., m]
    return out

print("="*80)
print(f"{'combo':<55s} | mean   | geom   | median | trim   | logit  | gate0.5 | gate0.7")
print("="*80)
for combo in combos_to_test:
    stack = np.stack([ranks[n] for n in combo], axis=-1)
    raw_anchor = models["exp019"]
    s = f"{'+'.join(combo):<55s} |"
    for method in ["mean", "geom", "median", "trim", "logit"]:
        P_a = aggregate(stack, method, anchor_idx=combo.index("exp019"))
        a, _ = macro_auc(P_a, Y)
        s += f" {a:.4f} |"
    for floor in [0.5, 0.7]:
        P_g = raw_gate_external(stack, raw_anchor, T=2.0, floor=floor, anchor_idx=combo.index("exp019"))
        a, _ = macro_auc(P_g, Y)
        s += f" {a:.4f}  |"
    print(s)

# Now scan finer floor values for [exp019, v73, convnext_rag, birdaves_rag]
print()
print("="*80)
print("Floor scan for [exp019, v73, convnext_rag, birdaves_rag]")
print("="*80)
combo = ["exp019", "v73", "convnext_rag", "birdaves_rag"]
stack = np.stack([ranks[n] for n in combo], axis=-1)
for T in [1.0, 2.0, 3.0]:
    for floor in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for ceil in [0.85, 0.90, 0.95]:
            P_g = raw_gate_external(stack, P_exp019, T=T, floor=floor, ceil=ceil, anchor_idx=0)
            a, _ = macro_auc(P_g, Y)
            print(f"  T={T} floor={floor} ceil={ceil}: {a:.4f}  Δ={a-0.9618:+.4f}")
    print()
