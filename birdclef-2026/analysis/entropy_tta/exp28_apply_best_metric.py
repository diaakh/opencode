"""Experiment 28: Apply the BEST metric to all our blends and zoo.

Best metric (LOO RMSE 0.006, ρ +0.99):
  LB ≈ 0.2786 - 0.8970 * overall_auc + 1.6087 * site_mean
"""
import numpy as np
import pandas as pd
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
bruce = np.load('/tmp/bruce_oof/labeled_oof_perch_bruce.npz', allow_pickle=True)['P_bruce']
cn = np.load('/tmp/convnext_rag.npz')['P_rag']
ba = np.load('/tmp/birdaves_rag.npz')['P_rag_avg']
mlp = np.load('/tmp/mlp_5seed_oof.npz')['P_mlp']
knn = np.load('/tmp/mega_knn.npz')['P']
bal_lr = np.load('/tmp/balanced_lr_oof.npz')['P_lr']
lgb = np.load('/tmp/lgb7_oof.npz')['P_lgb7']

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
    return np.mean(aucs) if aucs else 0

def site_mean(P):
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
    return (counts / counts.sum() * aucs).sum()

INT = 0.2786
A_oa = -0.8970
A_sm = 1.6087
def predict_lb(P):
    oa = macro_auc(P)
    sm = site_mean(P)
    return INT + A_oa * oa + A_sm * sm, oa, sm

# Model zoo
R_exp = rank_norm(P_exp)
R_bmae = rank_norm(bmae)
R_v73 = rank_norm(v73)
R_cn = rank_norm(cn)
R_ba = rank_norm(ba)
R_mlp = rank_norm(mlp)
R_perch = rank_norm(perch)
R_sub_proxy = (R_cn + R_ba + R_mlp) / 3

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
slot12_v4 = W_a * R_exp + (W_r * 0.95) * R_sub_proxy + (W_r * 0.05) * R_v73

# Various blends to evaluate
print("="*100)
print("BEST METRIC: LB ≈ 0.279 - 0.897 * overall_auc + 1.609 * site_mean")
print("="*100)
print(f"{'Blend':<60s} | overall  | site_mean | predicted LB")
print("-"*100)

blends = {
    "exp019 alone (KNOWN LB 0.949)": R_exp,
    "V73 alone (KNOWN LB 0.941)": R_v73,
    "Bruce alone (KNOWN LB 0.755)": rank_norm(bruce),
    "BirdMAE alone": R_bmae,
    "Perch_v2 alone": R_perch,
    "ConvNeXt-RAG": R_cn,
    "BirdAVES-RAG": R_ba,
    "MLP-5seed": R_mlp,
    "exp019 + 5% BirdMAE": 0.95*R_exp + 0.05*R_bmae,
    "exp019 + 10% BirdMAE": 0.90*R_exp + 0.10*R_bmae,
    "exp019 + 15% BirdMAE": 0.85*R_exp + 0.15*R_bmae,
    "exp019 + 20% BirdMAE": 0.80*R_exp + 0.20*R_bmae,
    "exp019 + 30% BirdMAE (slot6 ≈)": 0.70*R_exp + 0.30*R_bmae,
    "exp019 + 50% BirdMAE": 0.50*R_exp + 0.50*R_bmae,
    "exp019 + 10% V73": 0.90*R_exp + 0.10*R_v73,
    "exp019 + 20% V73": 0.80*R_exp + 0.20*R_v73,
    "exp019 + 30% V73": 0.70*R_exp + 0.30*R_v73,
    "exp019 + 10% Perch_v2": 0.90*R_exp + 0.10*R_perch,
    "exp019 + 10% BirdMAE + 10% V73": 0.80*R_exp + 0.10*R_bmae + 0.10*R_v73,
    "exp019 + 5% BirdMAE + 5% V73": 0.90*R_exp + 0.05*R_bmae + 0.05*R_v73,
    "slot12 v4 (current Kaggle kernel)": slot12_v4,
    "exp+BirdMAE GEOM mean 0.7/0.3": np.exp(0.7*np.log(R_exp+1e-6) + 0.3*np.log(R_bmae+1e-6)),
}

results = []
for name, P in blends.items():
    pred, oa, sm = predict_lb(P)
    results.append((name, oa, sm, pred))
    print(f"{name:<60s} | {oa:.4f}   | {sm:.4f}    | {pred:.4f}")

results.sort(key=lambda x: -x[3])
print("\n" + "="*100)
print("TOP 5 BY PREDICTED LB:")
print("="*100)
for n, oa, sm, p in results[:5]:
    print(f"  {p:.4f}  {n}  (oa={oa:.4f}, sm={sm:.4f})")

print("\nBOTTOM 5 (DO NOT SUBMIT):")
for n, oa, sm, p in results[-5:]:
    print(f"  {p:.4f}  {n}  (oa={oa:.4f}, sm={sm:.4f})")
