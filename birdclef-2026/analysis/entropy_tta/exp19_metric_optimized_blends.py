"""Experiment 19: Use the validated metric to find OPTIMAL blends.

The metric (RMSE 0.002 on 5 anchors):
  LB ≈ 0.277 + 0.714 * site_mean - 0.894 * (overall_auc - site_mean)

Goal: find blends with HIGHEST predicted LB.

Constraints from public kernel evidence:
- Pure Perch v2 + trained head: ~0.91
- Adding ProtoSSM: ~0.93  
- Full ensemble (Nina EoS / exp019): ~0.948-0.949
- exp019 is at the public ceiling.

Strategy: only combine LOW-gap models (gap < +0.03) to keep predicted LB high.
Avoid sub_v8 components (high gap = bad LB transfer).
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import re

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp = ex["P_exp019"]
Y = ex["Y"]
row_fn = ex["row_filename"]
N, C = Y.shape

def site_of(fn):
    parts = str(fn).split('_')
    for p in parts:
        if p.startswith('S') and len(p) <= 3 and p[1:].isdigit():
            return p
    return "S??"
sites = np.array([site_of(fn) for fn in row_fn])

v73 = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]
bmae = np.load("/tmp/birdmae_oof_kernel/birdmae_labeled_oof.npz", allow_pickle=True)["P_birdmae"]
perch = np.load("/tmp/perch20_blend.npz", allow_pickle=True)["P_perch20"]
cn = np.load('/tmp/convnext_rag.npz')['P_rag']  # for completeness
ba = np.load('/tmp/birdaves_rag.npz')['P_rag_avg']
mlp = np.load('/tmp/mlp_5seed_oof.npz')['P_mlp']

def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 1) / (N - 1)
    return R

R_exp = rank_norm(P_exp)
R_v73 = rank_norm(v73)
R_bmae = rank_norm(bmae)
R_perch = rank_norm(perch)
R_cn = rank_norm(cn)
R_ba = rank_norm(ba)
R_mlp = rank_norm(mlp)

def macro_auc(P):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs) if aucs else 0

def site_metrics(P):
    site_data = []
    for site in sorted(set(sites)):
        mask = sites == site
        if mask.sum() < 15: continue
        Y_s, P_s = Y[mask], P[mask]
        aucs = []
        for c in range(C):
            if Y_s[:, c].sum() < 2 or Y_s[:, c].sum() == mask.sum(): continue
            if P_s[:, c].max() == P_s[:, c].min(): continue
            try: aucs.append(roc_auc_score(Y_s[:, c], P_s[:, c]))
            except: pass
        if aucs:
            site_data.append((mask.sum(), np.mean(aucs)))
    counts = np.array([c for c, _ in site_data])
    aucs = np.array([a for _, a in site_data])
    w = counts / counts.sum()
    mean = (w * aucs).sum()
    var = (w * (aucs - mean)**2).sum()
    return np.sqrt(var), mean

def predict_lb(P):
    """LB ≈ 0.277 + 0.714 * site_mean - 0.894 * (overall - site_mean)"""
    oa = macro_auc(P)
    _, mean = site_metrics(P)
    gap = oa - mean
    return 0.277 + 0.714 * mean - 0.894 * gap, oa, mean, gap

# Test the metric on the proposed slot13 candidates
print("="*90)
print("EVALUATING CANDIDATE BLENDS WITH NEW METRIC")
print("="*90)
print(f"{'Blend':<55s} | overall | mean   | gap    | PREDICTED LB")
print("-"*100)

blends = {
    "exp019 alone (anchor)": R_exp,
    "exp019 90% + V73 10%": 0.90*R_exp + 0.10*R_v73,
    "exp019 85% + V73 15%": 0.85*R_exp + 0.15*R_v73,
    "exp019 80% + V73 20%": 0.80*R_exp + 0.20*R_v73,
    "exp019 70% + V73 30%": 0.70*R_exp + 0.30*R_v73,
    "exp019 50% + V73 50%": 0.50*R_exp + 0.50*R_v73,
    "exp019 85% + BirdMAE 15%": 0.85*R_exp + 0.15*R_bmae,
    "exp019 80% + V73 10% + BMAE 10%": 0.80*R_exp + 0.10*R_v73 + 0.10*R_bmae,
    "exp019 75% + V73 15% + BMAE 10%": 0.75*R_exp + 0.15*R_v73 + 0.10*R_bmae,
    "exp019 70% + V73 20% + BMAE 10%": 0.70*R_exp + 0.20*R_v73 + 0.10*R_bmae,
    "exp019 80% + V73 10% + Perch 10%": 0.80*R_exp + 0.10*R_v73 + 0.10*R_perch,
    "exp019 70% + V73 20% + BMAE 5% + Perch 5%": 0.70*R_exp + 0.20*R_v73 + 0.05*R_bmae + 0.05*R_perch,
    "BAD: exp019 70% + sub_v8 30%": 0.70*R_exp + 0.10*R_cn + 0.10*R_ba + 0.10*R_mlp,
    "BAD: 50% exp + 50% sub_v8 proxy": 0.50*R_exp + 0.50*((R_cn+R_ba+R_mlp)/3),
    "slot12 v4 (current)": None,  # compute below
}

# slot12 v4
eps = 1e-6
p = np.clip(P_exp, eps, 1-eps)
logit_e = np.log(p / (1-p))
p_calib = 1 / (1 + np.exp(-2.0 * logit_e))
p_calib = np.clip(p_calib, eps, 1-eps)
H = -(p_calib * np.log(p_calib) + (1-p_calib) * np.log(1-p_calib)) / np.log(2)
conf = (1 - H) ** 2.0
W_a = np.clip(conf, 0.40, 0.90)
W_r = 1 - W_a
R_sub_proxy = (R_cn + R_ba + R_mlp) / 3
blends["slot12 v4 (current)"] = W_a * R_exp + (W_r * 0.95) * R_sub_proxy + (W_r * 0.05) * R_v73

results = []
for name, P in blends.items():
    if P is None: continue
    pred, oa, mean, gap = predict_lb(P)
    results.append((name, oa, mean, gap, pred))
    print(f"{name:<55s} | {oa:.4f}  | {mean:.4f} | {gap:+.4f} | {pred:.4f}")

results.sort(key=lambda x: -x[4])
print("\n" + "="*90)
print(f"TOP 5 BY PREDICTED LB:")
print("="*90)
for n, oa, m, g, p in results[:5]:
    print(f"  {p:.4f}  {n}")

print(f"\nBOTTOM 3 (Bruce-like risk):")
for n, oa, m, g, p in results[-3:]:
    print(f"  {p:.4f}  {n}")

# Now scan wider for the optimal exp019 + V73 + BMAE blend
print("\n" + "="*90)
print("FINE-GRAINED SCAN: exp019 + V73 + BirdMAE (no sub_v8 components)")
print("="*90)
best_results = []
for w_e in np.arange(0.50, 0.96, 0.05):
    for w_v in np.arange(0.0, 0.51, 0.05):
        w_b = 1 - w_e - w_v
        if w_b < 0 or w_b > 0.5: continue
        P_b = w_e * R_exp + w_v * R_v73 + w_b * R_bmae
        pred, oa, mean, gap = predict_lb(P_b)
        best_results.append((w_e, w_v, w_b, oa, mean, gap, pred))

best_results.sort(key=lambda x: -x[6])
print(f"{'w_exp':<6s} {'w_v73':<7s} {'w_bmae':<7s} | overall | site_mean | gap     | PREDICTED LB")
for we, wv, wb, oa, m, g, p in best_results[:10]:
    print(f"  {we:.2f}  {wv:.2f}    {wb:.2f}    | {oa:.4f}  | {m:.4f}    | {g:+.4f} | {p:.4f}")
