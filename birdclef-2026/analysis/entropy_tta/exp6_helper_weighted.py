"""Experiment 6: Weight helpers by their own confidence too (not just anchor's).

Current slot12: anchor weight = f(anchor entropy), helpers split equally.
This experiment: helpers weighted by their OWN confidence too.
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
base = macro_auc(P_exp019, Y)[0]
print(f"baseline: {base:.4f}\n")

def entropy_conf(P, T=2.0):
    p = np.clip(P, 1e-7, 1-1e-7)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
    return (1 - H) ** T

# Method A: helper weights split equally among helpers (slot12)
# Method B: helper weights ∝ helper's own confidence
# Method C: helper weights ∝ softmax(helper_confidence / T_help)

names = ["exp019", "v73", "convnext_rag", "birdaves_rag"]
stack = np.stack([ranks[n] for n in names], axis=-1)  # (N, C, 4)
raw_anchor = P_exp019

def gate_equal(stack, raw_anchor, T=2.0, floor=0.5, ceil=0.95):
    conf_a = entropy_conf(raw_anchor, T)
    w_a = np.clip(conf_a, floor, ceil)
    M_ = stack.shape[-1]
    w_h = (1 - w_a) / (M_ - 1)
    out = w_a * stack[..., 0]
    for m in range(1, M_):
        out = out + w_h * stack[..., m]
    return out

def gate_weighted(stack, raw_anchor, raw_helpers, T_anchor=2.0, T_help=1.0, floor=0.5, ceil=0.95):
    """Helpers weighted by their own confidence."""
    conf_a = entropy_conf(raw_anchor, T_anchor)
    w_a = np.clip(conf_a, floor, ceil)
    # helper confidences
    conf_helpers = np.stack([entropy_conf(rh, T_help) for rh in raw_helpers], axis=-1)  # (N, C, H)
    conf_norm = conf_helpers / conf_helpers.sum(axis=-1, keepdims=True)  # (N, C, H)
    w_helpers = (1 - w_a)[..., None] * conf_norm  # (N, C, H)
    out = w_a * stack[..., 0]
    for m in range(1, stack.shape[-1]):
        out = out + w_helpers[..., m-1] * stack[..., m]
    return out

raw_helpers_list = [P_v73, P_cn, P_ba]
print("="*70)
print(f"{'method':<35s} | floor=0.3 | floor=0.5 | floor=0.7")
print("="*70)

for floor in [0.3, 0.5, 0.7]:
    pass

# Equal
row = f"{'Equal-split helpers (slot12)':<35s} |"
for floor in [0.3, 0.5, 0.7]:
    P_b = gate_equal(stack, raw_anchor, T=2.0, floor=floor)
    a = macro_auc(P_b, Y)[0]
    row += f"  {a:.4f}  |"
print(row)

# Weighted by helper conf
for T_help in [1.0, 2.0]:
    row = f"{'Helper-conf-weighted T_h='+str(T_help):<35s} |"
    for floor in [0.3, 0.5, 0.7]:
        P_b = gate_weighted(stack, raw_anchor, raw_helpers_list, T_anchor=2.0, T_help=T_help, floor=floor)
        a = macro_auc(P_b, Y)[0]
        row += f"  {a:.4f}  |"
    print(row)

# Method D: drop helpers when their confidence is low
def gate_drop_low_help(stack, raw_anchor, raw_helpers, conf_thresh=0.3, T=2.0, floor=0.5, ceil=0.95):
    conf_a = entropy_conf(raw_anchor, T)
    w_a = np.clip(conf_a, floor, ceil)
    # active helpers per cell: those with conf > thresh
    conf_h = np.stack([entropy_conf(rh, 1.0) for rh in raw_helpers], axis=-1)  # (N, C, H)
    active = (conf_h > conf_thresh).astype(np.float32)  # (N, C, H)
    n_active = active.sum(axis=-1, keepdims=True)  # (N, C, 1)
    n_active = np.maximum(n_active, 1.0)
    w_each = (1 - w_a)[..., None] / n_active * active  # (N, C, H)
    out = w_a * stack[..., 0]
    for m in range(1, stack.shape[-1]):
        out = out + w_each[..., m-1] * stack[..., m]
    return out

for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
    row = f"{'Drop helpers conf<'+str(thresh):<35s} |"
    for floor in [0.3, 0.5, 0.7]:
        P_b = gate_drop_low_help(stack, raw_anchor, raw_helpers_list, conf_thresh=thresh, T=2.0, floor=floor)
        a = macro_auc(P_b, Y)[0]
        row += f"  {a:.4f}  |"
    print(row)

# Method E: anchor weight from MULTI-MODEL entropy (use anchor disagreement)
def gate_disagreement(stack, raw_anchor, raw_helpers, T=2.0, floor=0.5, ceil=0.95):
    """Trust anchor more when helpers DISAGREE with it (suggesting helpers unreliable here)."""
    conf_a = entropy_conf(raw_anchor, T)
    # disagreement: |anchor_rank - mean(helper_ranks)|
    helper_mean = stack[..., 1:].mean(axis=-1)  # (N, C)
    disagree = np.abs(stack[..., 0] - helper_mean)  # higher = more disagreement
    # when high disagreement AND anchor confident → trust anchor MORE
    boost = conf_a * (1 + disagree)
    w_a = np.clip(boost, floor, ceil)
    n_help = stack.shape[-1] - 1
    w_h = (1 - w_a) / n_help
    out = w_a * stack[..., 0]
    for m in range(1, stack.shape[-1]):
        out = out + w_h * stack[..., m]
    return out

print()
for floor in [0.3, 0.5, 0.7]:
    P_b = gate_disagreement(stack, raw_anchor, raw_helpers_list, T=2.0, floor=floor)
    a = macro_auc(P_b, Y)[0]
    print(f"  Disagreement-boost floor={floor}: {a:.4f}")
