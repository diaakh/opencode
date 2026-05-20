"""Experiment 8: Final decision matrix.

Given exp019's flat probability distribution (avg=0.49, high entropy), the
"per-cell entropy gate" effectively reduces to a fixed weighted blend for
~95% of cells. This makes slot12 ≈ fixed weight blend.

Question: what's the OOF-optimal SIMPLE fixed blend that we trust to transfer
to LB given known leakage patterns?

Constraints:
- exp019 is the LB anchor (must dominate)
- BirdMAE leaks (slot6 -0.003 LB) — exclude
- V73 OOF bad but LB-strong — uncertain inclusion
- sub_v8 final has multiple components (Bruce leaks, but final 0.9748 OOF)
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

v73 = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]
cn = np.load(f"{CR}/convnext_rag.npz")["P_rag"]
ba = np.load(f"{CR}/birdaves_rag.npz")["P_rag_avg"]
mlp = np.load(f"{CR}/mlp_5seed_oof.npz")["P_mlp"]

# sub_v8 proxy = average of its strongest components
def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 1) / (N - 1)
    return R

R_exp = rank_norm(P_exp019)
R_cn = rank_norm(cn)
R_ba = rank_norm(ba)
R_mlp = rank_norm(mlp)
R_v73 = rank_norm(v73)
R_subv8 = (R_cn + R_ba + R_mlp) / 3  # rough proxy

def macro_auc(P, Y):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs), len(aucs)

base = macro_auc(P_exp019, Y)[0]
print(f"BASELINE exp019: {base:.4f}\n")

# Test all candidate slot12 variants
variants = {
    "slot12_v1 (current: 50/25/25 entropy gate)": (0.50, 0.25, 0.25),
    "slot12_v2 (60/20/20 fixed)": (0.60, 0.20, 0.20),
    "slot12_v3 (50/50/0 - no V73)": (0.50, 0.50, 0.00),
    "slot12_v4 (60/40/0 - no V73)": (0.60, 0.40, 0.00),
    "slot12_v5 (40/30/30 lower floor)": (0.40, 0.30, 0.30),
    "slot12_v6 (50/40/10)": (0.50, 0.40, 0.10),
    "slot12_v7 (60/30/10)": (0.60, 0.30, 0.10),
    "slot12_v8 (70/20/10)": (0.70, 0.20, 0.10),
    "slot12_v9 (40/40/20)": (0.40, 0.40, 0.20),
    "slot12_v10 (30/40/30 helpers-heavy)": (0.30, 0.40, 0.30),
}

print(f"{'variant':<55s} | OOF      | Δ")
print("-"*72)
for name, (we, ws, wv) in variants.items():
    P_b = we * R_exp + ws * R_subv8 + wv * R_v73
    a = macro_auc(P_b, Y)[0]
    print(f"{name:<55s} | {a:.4f}  | {a-base:+.4f}")

print()
print("="*72)
print("Multi-helper: anchor + helpers with diverse aggregation")
print("="*72)

# Use ACTUAL sub_v8 components (multi-arch) instead of proxy
def multi_aggregation(R_exp, helpers, weights):
    """weights = [w_exp, w_help1, w_help2, ...]"""
    out = weights[0] * R_exp
    for i, R_h in enumerate(helpers):
        out = out + weights[i+1] * R_h
    return out

# Test multi-helper compositions
print(f"\n4-way: [exp019, convnext_rag, birdaves_rag, v73]")
for w_e in [0.5, 0.6, 0.7]:
    rest = 1 - w_e
    for split in ["equal", "weight_v73_less"]:
        if split == "equal":
            ws = [rest/3]*3
        else:
            # Give V73 half-weight since its raw OOF is bad
            ws = [rest*0.4, rest*0.4, rest*0.2]
        P_b = multi_aggregation(R_exp, [R_cn, R_ba, R_v73], [w_e] + ws)
        a = macro_auc(P_b, Y)[0]
        print(f"  exp={w_e:.1f} split={split} weights={[w_e]+[round(w,3) for w in ws]}: {a:.4f}  Δ={a-base:+.4f}")

print(f"\n5-way: [exp019, convnext_rag, birdaves_rag, mlp, v73]")
for w_e in [0.5, 0.6, 0.7]:
    rest = 1 - w_e
    ws = [rest/4]*4
    P_b = multi_aggregation(R_exp, [R_cn, R_ba, R_mlp, R_v73], [w_e] + ws)
    a = macro_auc(P_b, Y)[0]
    print(f"  exp={w_e:.1f} equal-split: {a:.4f}  Δ={a-base:+.4f}")
    
    # Drop V73 (5 → 4)
    ws4 = [rest/3]*3
    P_b4 = multi_aggregation(R_exp, [R_cn, R_ba, R_mlp], [w_e] + ws4)
    a4 = macro_auc(P_b4, Y)[0]
    print(f"  exp={w_e:.1f} NO V73:       {a4:.4f}  Δ={a4-base:+.4f}")
