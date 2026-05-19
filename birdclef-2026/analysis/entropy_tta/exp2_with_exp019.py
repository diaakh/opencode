"""Experiment 2: Entropy-weighted blending of exp019 + diverse models.

exp019 is our LB anchor (0.949). All experiments anchor on exp019.
Test:
  - Per-row entropy gating: trust exp019 when confident, blend when uncertain
  - Per-class entropy weighting
  - Geometric mean variants
  - Trimmed mean
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

CR = "/home/user/opencode/birdclef-2026/analysis/creative"
ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"

# Load aligned exp019
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp019 = ex["P_exp019"]
Y = ex["Y"]
N, C = Y.shape

# Load all other models
models = {"exp019": P_exp019}
def load(name, path, key):
    try:
        d = np.load(f"{CR}/{path}")
        if key in d.keys() and d[key].shape == (N, C):
            models[name] = d[key].astype(np.float32)
    except: pass

load("birdmae", "birdmae_blend.npz", "P_birdmae")
load("convnext_rag", "convnext_rag.npz", "P_rag")
load("birdaves_rag", "birdaves_rag.npz", "P_rag_avg")
load("v73_rag", "v73_rag.npz", "P_rag_v73")
load("mlp_5seed", "mlp_5seed_oof.npz", "P_mlp")
load("blend_baseline", "best_oof_predictions.npz", "P_blend")
load("balanced_lr", "balanced_lr_oof.npz", "P_lr")

print(f"Models: {list(models.keys())}\n")

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

print("=== Individual AUCs ===")
ranks = {}
for name, P in models.items():
    auc, n = macro_auc(P, Y)
    print(f"  {name:18s} auc={auc:.4f} ({n})")
    ranks[name] = rank_norm(P)

print()
print("="*70)
print("Anchor: exp019 OOF =", f"{macro_auc(P_exp019, Y)[0]:.4f}")
print("="*70)

# ===== Test 1: 2-way blends with exp019 (rank-blend) =====
print("\n=== Test 1: 2-way rank blends with exp019 ===")
R_exp = ranks["exp019"]
best_2way = []
for name, R_other in ranks.items():
    if name == "exp019": continue
    for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
        P_blend = (1 - w) * R_exp + w * R_other
        auc, _ = macro_auc(P_blend, Y)
        if auc > 0.9618:
            best_2way.append((name, w, auc))
        if w in [0.2, 0.3]:
            print(f"  exp019 + {name:18s} w={w:.1f}: {auc:.4f}  Δ={auc-0.9618:+.4f}")

best_2way.sort(key=lambda x: -x[2])
print(f"\nTop 2-way wins:")
for name, w, auc in best_2way[:10]:
    print(f"  exp019 + {name:18s} w={w:.2f}: {auc:.4f}  Δ={auc-0.9618:+.4f}")

# ===== Test 2: Per-cell entropy-gated blending =====
print("\n=== Test 2: Per-cell entropy-gated blending (exp019 vs each model) ===")
# When exp019 is confident (close to 0 or 1), trust it more
# When exp019 is uncertain, blend more with other model

def entropy_gate_blend(P_anchor, P_other, gate="cell", T=2.0):
    """Adaptive blend: anchor weight = 1-H(anchor) raised to temperature.
    
    cell: per-cell entropy
    row: per-row entropy
    """
    eps = 1e-6
    P_a = np.clip(P_anchor, eps, 1 - eps)
    H_a = -(P_a * np.log(P_a) + (1-P_a) * np.log(1-P_a))  # in [0, ln(2)]
    H_a_norm = H_a / np.log(2)  # in [0, 1]
    if gate == "row":
        H_a_norm = H_a_norm.mean(axis=1, keepdims=True)
    conf_a = (1 - H_a_norm) ** T  # high when anchor confident
    w_anchor = conf_a / (conf_a + 1 - conf_a + eps)
    # equivalent to conf_a itself, ranges [0, 1]
    w_anchor = np.clip(conf_a, 0.4, 0.95)  # never fully drop either model
    return w_anchor * P_anchor + (1 - w_anchor) * P_other

print("Per-cell entropy gate (T=2.0):")
for name, R_other in ranks.items():
    if name == "exp019": continue
    P_blend = entropy_gate_blend(R_exp, R_other, gate="cell", T=2.0)
    auc, _ = macro_auc(P_blend, Y)
    print(f"  exp019 ⊕ {name:18s}: {auc:.4f}  Δ={auc-0.9618:+.4f}")

print("\nPer-row entropy gate (T=2.0):")
for name, R_other in ranks.items():
    if name == "exp019": continue
    P_blend = entropy_gate_blend(R_exp, R_other, gate="row", T=2.0)
    auc, _ = macro_auc(P_blend, Y)
    print(f"  exp019 ⊕ {name:18s}: {auc:.4f}  Δ={auc-0.9618:+.4f}")

# ===== Test 3: Inverse-entropy weighted multi-blend =====
print("\n=== Test 3: Multi-blend with inverse-entropy weighting ===")
# Pick top candidates (those that helped in 2-way)
helpers = [name for name, w, auc in best_2way[:5] if w >= 0.1]
print(f"Helpers: {helpers}")

eps = 1e-6
def confidence_logit(P):
    """|logit| = high when confident."""
    p = np.clip(P, eps, 1-eps)
    return np.abs(np.log(p / (1-p)))

stack_names = ["exp019"] + helpers
stack_R = np.stack([ranks[n] for n in stack_names], axis=-1)  # (N, C, M)
stack_conf = np.stack([confidence_logit(models[n]) for n in stack_names], axis=-1)

# Various T values
for T in [0.5, 1.0, 2.0, 5.0]:
    logits = T * stack_conf
    logits = logits - logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=-1, keepdims=True)
    P_blend = (stack_R * w).sum(axis=-1)
    auc, _ = macro_auc(P_blend, Y)
    print(f"  Multi-blend logit-conf T={T}: {auc:.4f}  Δ={auc-0.9618:+.4f}")

# Mean baseline
P_mean = stack_R.mean(axis=-1)
auc, _ = macro_auc(P_mean, Y)
print(f"  Plain mean of {stack_names}: {auc:.4f}  Δ={auc-0.9618:+.4f}")

# Geometric mean
P_geo = np.exp(np.log(stack_R + eps).mean(axis=-1))
auc, _ = macro_auc(P_geo, Y)
print(f"  Geom mean of {stack_names}: {auc:.4f}  Δ={auc-0.9618:+.4f}")

# ===== Test 4: Geometric vs arithmetic for fixed exp019 + birdmae =====
print("\n=== Test 4: Geom vs arithmetic for exp019 + birdmae ===")
if "birdmae" in ranks:
    R_b = ranks["birdmae"]
    for w in [0.1, 0.2, 0.3, 0.4]:
        P_arith = (1-w) * R_exp + w * R_b
        # geom: use log-domain
        eps_ = 1e-6
        P_geom = np.exp((1-w) * np.log(R_exp + eps_) + w * np.log(R_b + eps_))
        auc_a, _ = macro_auc(P_arith, Y)
        auc_g, _ = macro_auc(P_geom, Y)
        print(f"  w={w:.1f}  arith={auc_a:.4f}  geom={auc_g:.4f}  Δgeom={auc_g-auc_a:+.4f}")

# ===== Test 5: Surgical (exp019 keep majority, route only some classes) =====
print("\n=== Test 5: Surgical per-class routing with entropy weight ===")
# For each class, find best 1-way blend weight via cross-validation on the OOF
# (no leakage since we tune on the SAME OOF — would overfit but informative)
def best_per_class_blend(R_anchor, R_other, Y):
    """Find per-class optimal blend weight on this OOF."""
    weights = np.zeros(C)
    P_out = R_anchor.copy()
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        best_auc = roc_auc_score(Y[:, c], R_anchor[:, c]) if R_anchor[:, c].max() > R_anchor[:, c].min() else 0
        best_w = 0.0
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            P_c = (1-w) * R_anchor[:, c] + w * R_other[:, c]
            if P_c.max() == P_c.min(): continue
            a = roc_auc_score(Y[:, c], P_c)
            if a > best_auc:
                best_auc = a; best_w = w
        weights[c] = best_w
        P_out[:, c] = (1-best_w) * R_anchor[:, c] + best_w * R_other[:, c]
    return P_out, weights

for name in ["birdmae", "convnext_rag", "v73_rag", "mlp_5seed", "balanced_lr"]:
    if name not in ranks: continue
    R_other = ranks[name]
    P_oracle, w = best_per_class_blend(R_exp, R_other, Y)
    auc, _ = macro_auc(P_oracle, Y)
    n_active = int((w > 0).sum())
    print(f"  ORACLE: exp019 + {name:15s}: {auc:.4f}  Δ={auc-0.9618:+.4f}  active={n_active}/234 classes")

# Save best result
out_pred_path = f"{ETT}/exp2_outputs.npz"
np.savez(out_pred_path, P_exp019=P_exp019, Y=Y, R_exp=R_exp)
print(f"\nDone. Saved: {out_pred_path}")
