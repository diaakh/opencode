"""Experiment 10: Calibrate exp019 to sharpen confidence signal.

exp019's flat distribution makes entropy gate ~useless. What if we calibrate
exp019 first (e.g., via temperature scaling in logit space) to sharpen
distribution, THEN apply entropy gate?
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
CR = "/home/user/opencode/birdclef-2026/analysis/creative"

ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp = ex["P_exp019"]
Y = ex["Y"]
N, C = Y.shape

cn = np.load(f"{CR}/convnext_rag.npz")["P_rag"]
ba = np.load(f"{CR}/birdaves_rag.npz")["P_rag_avg"]
mlp = np.load(f"{CR}/mlp_5seed_oof.npz")["P_mlp"]
v73 = np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]

def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 1) / (N - 1)
    return R

def macro_auc(P, Y):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs)

R_exp = rank_norm(P_exp)
R_sub = (rank_norm(cn) + rank_norm(ba) + rank_norm(mlp)) / 3
R_v73 = rank_norm(v73)
base = macro_auc(P_exp, Y)
print(f"baseline: {base:.4f}\n")

# Calibration via temperature scaling in logit space
def temp_calibrate(P, T):
    """Apply T to logit: sigmoid(T * logit(P)). T>1 sharpens, T<1 flattens."""
    eps = 1e-6
    p = np.clip(P, eps, 1-eps)
    logit = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-T * logit))

# Calibration via rank quantile mapping → uniform [0, 1]
def rank_quantile(P):
    """Map each value to its rank quantile per class."""
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 0.5) / N
    return R

# Calibration via Beta-like transformation
def beta_calibrate(P, alpha, beta):
    """Transform: shifts mass away from 0.5."""
    return P**alpha / (P**alpha + (1-P)**beta)

print("=== Temp-scaled exp019 + sub_v8 50/50 ===")
for T in [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
    P_t = temp_calibrate(P_exp, T)
    # entropy distribution after calibration
    eps = 1e-6
    p = np.clip(P_t, eps, 1-eps)
    H = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
    print(f"  T={T}: avg entropy={H.mean():.3f}, <0.3: {(H<0.3).mean()*100:.1f}%, >0.7: {(H>0.7).mean()*100:.1f}%")
    
    # Apply gate using calibrated entropy
    R_calib = rank_norm(P_t)
    for floor in [0.3, 0.5]:
        conf = (1 - H) ** 2.0
        W_a = np.clip(conf, floor, 0.9)
        R_b = W_a * R_exp + (1 - W_a) * R_sub
        a = macro_auc(R_b, Y)
        print(f"    floor={floor}: gate auc={a:.4f}  Δ={a-base:+.4f}")

print()
print("=== Sharpen via rank quantile then gate ===")
R_q = rank_quantile(P_exp)
# entropy of quantile-calibrated
eps = 1e-6
p_q = np.clip(R_q, eps, 1-eps)
H_q = -(p_q * np.log(p_q) + (1-p_q) * np.log(1-p_q)) / np.log(2)
print(f"  quantile entropy: avg={H_q.mean():.3f}, <0.3: {(H_q<0.3).mean()*100:.1f}%, >0.7: {(H_q>0.7).mean()*100:.1f}%")
for floor in [0.3, 0.5]:
    conf = (1 - H_q) ** 2.0
    W_a = np.clip(conf, floor, 0.9)
    R_b = W_a * R_exp + (1 - W_a) * R_sub
    a = macro_auc(R_b, Y)
    print(f"    floor={floor}: gate auc={a:.4f}  Δ={a-base:+.4f}")

print()
print("=== Use sub_v8's confidence as gate signal instead ===")
# If sub_v8 has sharper distribution, use it
P_sub_avg = (cn + ba + mlp) / 3  # raw probability average
eps = 1e-6
p = np.clip(P_sub_avg, eps, 1-eps)
H_sub = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
print(f"  sub_v8 entropy: avg={H_sub.mean():.3f}, <0.3: {(H_sub<0.3).mean()*100:.1f}%, >0.7: {(H_sub>0.7).mean()*100:.1f}%")

# Gate using sub_v8's confidence - when sub_v8 is confident, give it MORE weight
conf_sub = (1 - H_sub) ** 2.0
for floor in [0.05, 0.1, 0.2, 0.3]:
    # When sub_v8 confident, its weight goes UP
    W_sub_min = floor  # min weight for sub_v8
    W_sub_max = 0.6   # max
    W_sub = np.clip(conf_sub, W_sub_min, W_sub_max)
    R_b = (1 - W_sub) * R_exp + W_sub * R_sub
    a = macro_auc(R_b, Y)
    print(f"    sub_v8 conf-gate floor={floor}: gate auc={a:.4f}  Δ={a-base:+.4f}  (W_sub avg={W_sub.mean():.3f})")

# Combine: anchor weight = (1 - H_exp_calib) * (H_sub_calib)
# i.e., anchor wins when EXP confident AND sub_v8 uncertain
print()
print("=== Disagreement gate: anchor wins on EXP_conf AND sub_uncertain ===")
H_exp_orig = -(np.clip(P_exp, eps, 1-eps) * np.log(np.clip(P_exp, eps, 1-eps)) + np.clip(1-P_exp, eps, 1-eps) * np.log(np.clip(1-P_exp, eps, 1-eps))) / np.log(2)
for floor in [0.3, 0.4, 0.5]:
    # Higher score → trust anchor more
    score = (1 - H_exp_orig) + H_sub  # both want this high
    score_norm = score / 2  # [0, 1]
    W_a = np.clip(score_norm, floor, 0.9)
    R_b = W_a * R_exp + (1 - W_a) * R_sub
    a = macro_auc(R_b, Y)
    print(f"  floor={floor}: gate auc={a:.4f}  Δ={a-base:+.4f}  (W_a avg={W_a.mean():.3f})")
