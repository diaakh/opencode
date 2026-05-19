"""Experiment 7: Build sub_v8 proxy OOF from its components.

sub_v8 final OOF was reported as 0.9748. Components we have:
- ConvNeXt-RAG (0.939)
- BirdAVES-RAG (0.916)
- mlp_5seed (0.857)
- lgb7
- balanced_lr

If we approximate sub_v8 final = average of these, we can validate slot12.
Then test what fixed blend ratios beat exp019 alone on [exp019, sub_v8_proxy, v73_raw].
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

cn = np.load(f"{CR}/convnext_rag.npz")["P_rag"]
ba = np.load(f"{CR}/birdaves_rag.npz")["P_rag_avg"]
mlp = np.load(f"{CR}/mlp_5seed_oof.npz")["P_mlp"]

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

# Build sub_v8 proxy = average of strongest components in rank space
R_cn = rank_norm(cn)
R_ba = rank_norm(ba)
R_mlp = rank_norm(mlp)
R_subv8_proxy = (R_cn + R_ba + R_mlp) / 3

# Stronger proxy: weighted by individual AUC
auc_cn = macro_auc(cn, Y)[0]
auc_ba = macro_auc(ba, Y)[0]
auc_mlp = macro_auc(mlp, Y)[0]
w_cn = auc_cn - 0.5
w_ba = auc_ba - 0.5
w_mlp = auc_mlp - 0.5
total = w_cn + w_ba + w_mlp
R_subv8_wproxy = (w_cn * R_cn + w_ba * R_ba + w_mlp * R_mlp) / total

auc_proxy, _ = macro_auc(R_subv8_proxy, Y)
auc_wproxy, _ = macro_auc(R_subv8_wproxy, Y)
print(f"sub_v8 proxy (mean of 3): {auc_proxy:.4f}")
print(f"sub_v8 proxy (auc-weighted): {auc_wproxy:.4f}")
print(f"exp019: {macro_auc(P_exp019, Y)[0]:.4f}")
print(f"V73 raw: {macro_auc(P_v73, Y)[0]:.4f}")
print()

R_exp = rank_norm(P_exp019)
R_v73 = rank_norm(P_v73)

# Grid search [exp019, sub_v8_proxy, v73] fixed weights
print("="*70)
print("Fixed 3-way blend grid: [w_exp, w_sub, w_v73]")
print("="*70)
results = []
for w_e in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    rest = 1 - w_e
    for sv_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w_sub = rest * sv_ratio
        w_v73 = rest * (1 - sv_ratio)
        P_b = w_e * R_exp + w_sub * R_subv8_wproxy + w_v73 * R_v73
        a, _ = macro_auc(P_b, Y)
        results.append((w_e, w_sub, w_v73, a))
        if a > 0.9618:
            print(f"  exp={w_e:.2f} sub={w_sub:.2f} v73={w_v73:.2f}: {a:.4f}  Δ={a-0.9618:+.4f}")

results.sort(key=lambda x: -x[3])
print("\nTop 5:")
for w_e, w_s, w_v, a in results[:5]:
    print(f"  exp={w_e:.2f} sub={w_s:.2f} v73={w_v:.2f}: {a:.4f}")

# Now entropy gate version
print("\n" + "="*70)
print("Entropy gate version (slot12 actual mechanism)")
print("="*70)
def entropy_gate(R_exp, R_sub, R_v73, raw_anchor, T=2.0, floor=0.5, ceil=0.95):
    p = np.clip(raw_anchor, 1e-7, 1-1e-7)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
    conf = (1 - H) ** T
    w_a = np.clip(conf, floor, ceil)
    w_h = (1 - w_a) / 2
    return w_a * R_exp + w_h * R_sub + w_h * R_v73

for floor in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    P_g = entropy_gate(R_exp, R_subv8_wproxy, R_v73, P_exp019, T=2.0, floor=floor)
    a, _ = macro_auc(P_g, Y)
    # Equivalent fixed blend (mean weights)
    p = np.clip(P_exp019, 1e-7, 1-1e-7)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
    conf = (1 - H) ** 2.0
    w_a_mean = np.clip(conf, floor, 0.95).mean()
    print(f"  floor={floor}: {a:.4f}  Δ={a-0.9618:+.4f}  (avg anchor weight: {w_a_mean:.3f})")

# What if we DON'T include V73 (since V73 raw OOF is bad)?
print("\n" + "="*70)
print("2-way: exp019 + sub_v8_proxy only (NO V73)")
print("="*70)
def entropy_gate_2way(R_exp, R_help, raw_anchor, T=2.0, floor=0.5, ceil=0.95):
    p = np.clip(raw_anchor, 1e-7, 1-1e-7)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
    conf = (1 - H) ** T
    w_a = np.clip(conf, floor, ceil)
    return w_a * R_exp + (1 - w_a) * R_help

for floor in [0.3, 0.4, 0.5, 0.6, 0.7]:
    P_g = entropy_gate_2way(R_exp, R_subv8_wproxy, P_exp019, floor=floor)
    a, _ = macro_auc(P_g, Y)
    print(f"  exp019 ⊕ subv8 only floor={floor}: {a:.4f}  Δ={a-0.9618:+.4f}")
