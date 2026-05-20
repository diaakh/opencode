"""Experiment 11: Perch v2 raw as truly out-of-distribution signal.

Key hypothesis (from user observation):
ALL our trained models (exp019, V73, BirdMAE, Bruce, ConvNeXt-RAG, BirdAVES-RAG,
MLP, LGB stacker, ...) are trained on BC2026 train_audio. They share the same
site/distribution biases. This is why labeled OOF metrics inflate and LB transfer
is hostile.

Perch v2 is Google's foundation model — pretrained on multi-taxa bioacoustic
data with ZERO exposure to BC2026 train_audio. Its 14,795-species raw output
mapped to BC2026's 234 species is the ONE signal that's genuinely external.

Test:
1. Perch v2 raw standalone OOF
2. exp019 + Perch v2 raw entropy-gated blend
3. Compare to slot12 v4 (exp019 + sub_v8) — which uses ONLY in-distribution helpers
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"

ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp = ex["P_exp019"]
Y = ex["Y"]
N, C = Y.shape

# Perch v2 raw OOF
pe = np.load("/tmp/perch20_blend.npz", allow_pickle=True)
P_perch_raw = pe["P_perch20"]
mapped = pe["mapped_cols"]
print(f"Perch v2 mapped classes: {int(mapped.sum())}/{C}")

# Compare to sub_v8 proxy
v73 = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]

def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 1) / (N - 1)
    return R

def macro_auc(P):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs), len(aucs)

def macro_auc_mapped_only(P):
    """AUC only on mapped classes."""
    aucs = []
    for c in np.where(mapped)[0]:
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs), len(aucs)

R_exp = rank_norm(P_exp)
R_perch = rank_norm(P_perch_raw)
R_v73 = rank_norm(v73)

print(f"\nexp019 baseline OOF: {macro_auc(P_exp)[0]:.4f}")
print(f"Perch v2 raw standalone OOF: {macro_auc(P_perch_raw)[0]:.4f}  ({macro_auc(P_perch_raw)[1]} active classes)")
print(f"Perch v2 raw OOF on mapped only: {macro_auc_mapped_only(P_perch_raw)[0]:.4f}")
print(f"V73 raw OOF: {macro_auc(v73)[0]:.4f}")
print(f"exp019 on mapped only: {macro_auc_mapped_only(P_exp)[0]:.4f}")

# Tests
print("\n" + "="*70)
print("2-way blend tests: exp019 + Perch v2 raw")
print("="*70)

# Blanket rank blend
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    P_b = (1-w) * R_exp + w * R_perch
    a = macro_auc(P_b)[0]
    a_mapped = macro_auc_mapped_only(P_b)[0]
    print(f"  w_perch={w:.2f}: full auc={a:.4f}  Δ={a-0.9618:+.4f}   mapped auc={a_mapped:.4f}")

# Per-cell entropy gate (calibrated)
print("\nCalibrated entropy gate (T=2) with Perch v2:")
eps = 1e-6
p = np.clip(P_exp, eps, 1-eps)
logit_e = np.log(p / (1-p))
p_calib = 1 / (1 + np.exp(-2.0 * logit_e))
p_calib = np.clip(p_calib, eps, 1-eps)
H = -(p_calib * np.log(p_calib) + (1-p_calib) * np.log(1-p_calib)) / np.log(2)
conf = (1 - H) ** 2.0

for floor in [0.30, 0.40, 0.50, 0.60, 0.70]:
    W_a = np.clip(conf, floor, 0.90)
    P_b = W_a * R_exp + (1 - W_a) * R_perch
    a = macro_auc(P_b)[0]
    print(f"  floor={floor}: full auc={a:.4f}  Δ={a-0.9618:+.4f}  (avg W_a={W_a.mean():.3f})")

print("\n" + "="*70)
print("3-way: exp019 + sub_v8_proxy + Perch v2")
print("="*70)
# Build sub_v8 proxy from creative dir (if we have it)
try:
    import glob
    # Try to load via stash
    for path in ['/tmp/perch20_blend.npz']:
        pass
    # Fallback: just use exp019 + Perch v2 + V73
    from pathlib import Path
    cn_path = Path('/tmp/cn_rag.npz')
    if not cn_path.exists():
        # Try git show
        import subprocess
        for fn in ['convnext_rag.npz', 'birdaves_rag.npz', 'mlp_5seed_oof.npz']:
            with open(f'/tmp/{fn}', 'wb') as f:
                subprocess.run(['git', 'show', f'7c984ba:birdclef-2026/analysis/creative/{fn}'], stdout=f, check=False)
    cn = np.load('/tmp/convnext_rag.npz')['P_rag']
    ba = np.load('/tmp/birdaves_rag.npz')['P_rag_avg']
    mlp = np.load('/tmp/mlp_5seed_oof.npz')['P_mlp']
    R_sub = (rank_norm(cn) + rank_norm(ba) + rank_norm(mlp)) / 3
    print(f"sub_v8 proxy OOF: {macro_auc(R_sub)[0]:.4f}")
    
    # 3-way blends
    for w_p in [0.05, 0.10, 0.15, 0.20, 0.30]:
        for w_s in [0.20, 0.30, 0.40, 0.50]:
            w_e = 1 - w_p - w_s
            if w_e < 0.3: continue
            P_b = w_e * R_exp + w_s * R_sub + w_p * R_perch
            a = macro_auc(P_b)[0]
            if a > 0.965:
                print(f"  exp={w_e:.2f} sub={w_s:.2f} perch={w_p:.2f}: {a:.4f}  Δ={a-0.9618:+.4f}")
except Exception as e:
    print(f"Couldn't compute sub_v8 proxy: {e}")

print("\n" + "="*70)
print("Logit-space transfer test: per-class AUC distribution")
print("="*70)

# Look at where Perch v2 specifically EXCELS vs exp019
perch_aucs = []
exp_aucs = []
for c in range(C):
    if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
    if not mapped[c]: continue
    try:
        a_exp = roc_auc_score(Y[:, c], P_exp[:, c])
        a_per = roc_auc_score(Y[:, c], P_perch_raw[:, c])
        perch_aucs.append(a_per)
        exp_aucs.append(a_exp)
    except: pass

perch_aucs = np.array(perch_aucs)
exp_aucs = np.array(exp_aucs)
perch_wins = (perch_aucs > exp_aucs).sum()
print(f"On mapped classes (n={len(perch_aucs)}):")
print(f"  Perch v2 mean AUC: {perch_aucs.mean():.4f}")
print(f"  exp019  mean AUC: {exp_aucs.mean():.4f}")
print(f"  Perch wins on {perch_wins}/{len(perch_aucs)} classes")
print(f"  Mean win margin (when Perch wins): {(perch_aucs - exp_aucs)[perch_aucs > exp_aucs].mean():+.4f}")
print(f"  Mean loss margin (when exp019 wins): {(perch_aucs - exp_aucs)[perch_aucs < exp_aucs].mean():+.4f}")
