"""Experiment 1: Entropy-weighted aggregation across diverse models on labeled OOF.

Goal: Find an aggregation method that beats simple mean.
Tests: arithmetic mean, geometric mean, entropy-weighted, max-confidence,
       per-window entropy-gated routing, trimmed mean.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ROOT = "/home/user/opencode/birdclef-2026/analysis/creative"
d = np.load(f"{ROOT}/best_oof_predictions.npz")
Y = d["Y"]
N, C = Y.shape
print(f"Labels: N={N}, C={C}, pos={int(Y.sum())}")

# Load diverse models — architecturally different sources
models = {}

def load(name, path, key, optional=False):
    try:
        d = np.load(f"{ROOT}/{path}")
        if key not in d.keys():
            print(f"  SKIP {name}: missing key {key}, has {list(d.keys())}")
            return
        models[name] = d[key]
        if models[name].shape != (N, C):
            print(f"  SKIP {name}: shape {models[name].shape} != ({N},{C})")
            del models[name]
            return
        print(f"  loaded {name}: shape={models[name].shape}, min={models[name].min():.4f}, max={models[name].max():.4f}")
    except Exception as e:
        if not optional:
            print(f"  ERR {name}: {e}")

load("birdmae", "birdmae_blend.npz", "P_birdmae")
load("convnext_rag", "convnext_rag.npz", "P_rag")
load("birdaves_rag", "birdaves_rag.npz", "P_rag_avg")
load("v73_rag", "v73_rag.npz", "P_rag_v73")
load("mlp_5seed", "mlp_5seed_oof.npz", "P_mlp")
load("blend_baseline", "best_oof_predictions.npz", "P_blend")
load("lgb7", "lgb7_oof.npz", "P_lgb7")
load("balanced_lr", "balanced_lr_oof.npz", "P_lr")
load("mega_knn", "mega_knn.npz", "P")

def macro_auc(P, Y):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N:
            continue
        if P[:, c].max() == P[:, c].min():
            continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs), len(aucs)

print("\n=== Individual model AUCs ===")
for name, P in models.items():
    auc, n = macro_auc(P, Y)
    print(f"  {name:20s} auc={auc:.4f}  ({n} classes)")

def rank_norm(P):
    """Per-class rank normalization in [0, 1]."""
    R = np.zeros_like(P)
    for c in range(C):
        R[:, c] = rankdata(P[:, c]) / N
    return R

def entropy_per_row(P):
    """Bernoulli entropy summed across classes per row."""
    p = np.clip(P, 1e-7, 1-1e-7)
    H = -(p * np.log(p) + (1-p) * np.log(1-p))
    return H.sum(axis=1)  # (N,)

print("\n=== Rank-normalize all models for fair aggregation ===")
ranked = {name: rank_norm(P) for name, P in models.items()}

# Build stack: (N, C, M) tensor
names = list(models.keys())
M = len(names)
print(f"  models: {names}")

stack = np.stack([ranked[n] for n in names], axis=-1)  # (N, C, M)
raw_stack = np.stack([models[n] for n in names], axis=-1)
print(f"  stack shape: {stack.shape}")

# === Aggregation experiments ===
print("\n=== Aggregation experiments ===")

# 1. Arithmetic mean of ranks
P_mean = stack.mean(axis=-1)
auc, n = macro_auc(P_mean, Y)
print(f"  arithmetic mean of ranks    auc={auc:.4f} ({n})")

# 2. Geometric mean of ranks (offset to avoid log 0)
P_geo = np.exp(np.log(stack + 1e-6).mean(axis=-1))
auc, n = macro_auc(P_geo, Y)
print(f"  geometric mean of ranks     auc={auc:.4f} ({n})")

# 3. Median
P_med = np.median(stack, axis=-1)
auc, n = macro_auc(P_med, Y)
print(f"  median of ranks             auc={auc:.4f} ({n})")

# 4. Trimmed mean (drop top and bottom 1)
sorted_stack = np.sort(stack, axis=-1)
if M >= 4:
    P_trim = sorted_stack[..., 1:-1].mean(axis=-1)
    auc, n = macro_auc(P_trim, Y)
    print(f"  trimmed mean (drop hi/lo)   auc={auc:.4f} ({n})")

# 5. Max (use the most confident model per cell)
P_max = stack.max(axis=-1)
auc, n = macro_auc(P_max, Y)
print(f"  max across models           auc={auc:.4f} ({n})")

# 6. Min
P_min = stack.min(axis=-1)
auc, n = macro_auc(P_min, Y)
print(f"  min across models           auc={auc:.4f} ({n})")

# 7. Entropy-weighted (per row, per model — high confidence wins)
# For each model and row, confidence = mean(max(p, 1-p)) across classes
conf_per_row_per_model = np.zeros((N, M))
for i, name in enumerate(names):
    p = np.clip(models[name], 1e-7, 1-1e-7)
    H = -(p * np.log(p) + (1-p) * np.log(1-p))
    # lower entropy = more confident
    avg_H = H.mean(axis=1)  # (N,)
    conf_per_row_per_model[:, i] = 1.0 / (avg_H + 1e-6)

# normalize confidences across models per row
w = conf_per_row_per_model / conf_per_row_per_model.sum(axis=1, keepdims=True)  # (N, M)
# expand to (N, C, M)
w_exp = w[:, None, :]  # (N, 1, M)
P_ent = (stack * w_exp).sum(axis=-1)
auc, n = macro_auc(P_ent, Y)
print(f"  row-entropy weighted        auc={auc:.4f} ({n})")

# 8. Per-cell entropy weighted: weight = max(p, 1-p) per cell
def cell_confidence(P):
    return np.maximum(P, 1 - P)  # (N, C)

cell_conf = np.stack([cell_confidence(models[n]) for n in names], axis=-1)  # (N, C, M)
w_cell = cell_conf / cell_conf.sum(axis=-1, keepdims=True)
P_cellent = (stack * w_cell).sum(axis=-1)
auc, n = macro_auc(P_cellent, Y)
print(f"  per-cell entropy weighted   auc={auc:.4f} ({n})")

# 9. Per-cell entropy weighted with temperature
for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
    # softmax over models with confidence as logit
    logits = T * cell_conf  # (N, C, M)
    logits = logits - logits.max(axis=-1, keepdims=True)
    w_soft = np.exp(logits)
    w_soft /= w_soft.sum(axis=-1, keepdims=True)
    P_soft = (stack * w_soft).sum(axis=-1)
    auc, n = macro_auc(P_soft, Y)
    print(f"  cell-conf softmax T={T:4.1f}  auc={auc:.4f} ({n})")

# 10. Max-confidence routing (per cell, pick the model with highest confidence)
best_model_idx = cell_conf.argmax(axis=-1)  # (N, C)
i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(C), indexing='ij')
P_route = stack[i_idx, j_idx, best_model_idx]
auc, n = macro_auc(P_route, Y)
print(f"  per-cell argmax route       auc={auc:.4f} ({n})")

# 11. Per-cell entropy-gated with floor (entropy weighting only when blend > base)
# More principled: entropy applied to logits, sum log-odds
def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1 - p))

logit_stack = np.stack([logit(models[n]) for n in names], axis=-1)
# weight by 1/abs(logit) ... no, that gives MORE weight to uncertain. Inverse: high |logit| = confident
abs_logit = np.abs(logit_stack)
w_lg = abs_logit / abs_logit.sum(axis=-1, keepdims=True)
P_lgw = (stack * w_lg).sum(axis=-1)
auc, n = macro_auc(P_lgw, Y)
print(f"  abs-logit weighted          auc={auc:.4f} ({n})")

print("\n=== Best model = baseline to beat ===")
best_single = max(models.items(), key=lambda x: macro_auc(x[1], Y)[0])
print(f"  Best single: {best_single[0]} auc={macro_auc(best_single[1], Y)[0]:.4f}")

# Save
out = "/home/user/opencode/birdclef-2026/analysis/entropy_tta/exp1_results.npz"
np.savez(out, P_mean=P_mean, P_geo=P_geo, P_med=P_med, P_ent=P_ent, P_cellent=P_cellent, P_route=P_route, Y=Y, names=np.array(names))
print(f"\nSaved: {out}")
