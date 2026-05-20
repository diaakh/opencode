"""Experiment 17: Decompose LB-prediction into two regimes.

Observation: V73 and Bruce both have ~similar site_mean (0.78/0.78) but
vastly different LB (0.94/0.76). They differ in one critical way:

V73:    overall=0.667 (LOW on OOF)  → LB=0.941 (HIGH)  ← labeled OOF UNDERESTIMATES
Bruce:  overall=0.859 (HIGH on OOF) → LB=0.755 (LOW)   ← labeled OOF OVERESTIMATES

The OOF→LB gap is the key signal. Models split into:
A. "Underestimated": overall_auc < site_mean (V73 type) → expect LB > overall
B. "Overestimated": overall_auc > site_mean (Bruce type) → expect LB < overall
C. "Calibrated":   overall_auc ≈ site_mean (exp019 type) → expect LB ≈ overall

Test: define a calibration-gap metric and use it.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, pearsonr, spearmanr

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

# Anchors with known LB
ranks = {
    "exp019":   (P_exp, rank_norm(P_exp), 0.949),
    "V73":      (v73, rank_norm(v73), 0.941),
    "Bruce":    (bruce, rank_norm(bruce), 0.755),
}

# Build slot blends
R_exp = rank_norm(P_exp)
R_bmae = rank_norm(bmae)
R_v73 = rank_norm(v73)
R_sub = (rank_norm(cn) + rank_norm(ba) + rank_norm(mlp)) / 3

slot6 = 0.7 * R_exp + 0.3 * R_bmae
slot11 = 0.97 * R_exp + 0.03 * R_sub

# slot12 v4 algorithm
eps = 1e-6
p = np.clip(P_exp, eps, 1-eps)
logit_e = np.log(p / (1-p))
p_calib = 1 / (1 + np.exp(-2.0 * logit_e))
p_calib = np.clip(p_calib, eps, 1-eps)
H = -(p_calib * np.log(p_calib) + (1-p_calib) * np.log(1-p_calib)) / np.log(2)
conf = (1 - H) ** 2.0
W_a = np.clip(conf, 0.40, 0.90)
W_r = 1 - W_a
slot12_v4 = W_a * R_exp + (W_r * 0.95) * R_sub + (W_r * 0.05) * R_v73

# All anchors as rank-normalized (for fair comparison)
all_blends = {
    "exp019":      (R_exp, 0.949),
    "V73":         (R_v73, 0.941),
    "Bruce":       (rank_norm(bruce), 0.755),
    "slot6":       (slot6, 0.946),
    "slot11":      (slot11, 0.949),
    "slot12 v4":   (slot12_v4, None),
}

print(f"{'Blend':<15s} | overall | std    | mean   | overall-mean (gap_metric) | LB")
print("-"*100)
data = []
for name, (P, lb) in all_blends.items():
    oa = macro_auc(P)
    std, mean = site_metrics(P)
    gap = oa - mean  # POSITIVE if labeled OOF overstates (Bruce-like), NEGATIVE if understates (V73-like)
    print(f"{name:<15s} | {oa:.4f}  | {std:.4f} | {mean:.4f} | {gap:+.4f}                    | {lb if lb else '?'}")
    if lb is not None:
        data.append((name, oa, std, mean, gap, lb))

# Correlation
arr = np.array(data, dtype=object)
print("\n" + "="*80)
print("Correlation analysis:")
print("="*80)

oa_v = np.array([d[1] for d in data])
std_v = np.array([d[2] for d in data])
mean_v = np.array([d[3] for d in data])
gap_v = np.array([d[4] for d in data])
lb_v = np.array([d[5] for d in data])

for label, v in [("overall_auc", oa_v), ("site_std", std_v), ("site_mean", mean_v), ("gap (oa-mean)", gap_v)]:
    if len(v) > 2:
        r, _ = pearsonr(v, lb_v)
        rho, _ = spearmanr(v, lb_v)
        print(f"  LB ↔ {label:<20s}: Pearson r={r:+.3f}  Spearman ρ={rho:+.3f}")

# Combined fit: LB = a + b*mean + c*gap
from numpy.linalg import lstsq
X = np.column_stack([np.ones(len(lb_v)), mean_v, gap_v])
coefs, _, _, _ = lstsq(X, lb_v, rcond=None)
preds = X @ coefs
rmse = np.sqrt(((preds - lb_v)**2).mean())
print(f"\nFit: LB = {coefs[0]:.3f} + {coefs[1]:.3f}*mean + {coefs[2]:.3f}*gap   RMSE={rmse:.4f}")
print(f"\nPredictions:")
for i, d in enumerate(data):
    print(f"  {d[0]:<12s}: predicted {preds[i]:.4f}, actual {d[5]:.4f}, error {preds[i]-d[5]:+.4f}")

# Now predict slot12 v4
oa_v4 = macro_auc(slot12_v4)
std_v4, mean_v4 = site_metrics(slot12_v4)
gap_v4 = oa_v4 - mean_v4
pred_v4 = coefs[0] + coefs[1] * mean_v4 + coefs[2] * gap_v4
print(f"\nslot12 v4: overall={oa_v4:.4f}, mean={mean_v4:.4f}, gap={gap_v4:+.4f}")
print(f"  Predicted LB: {pred_v4:.4f}")

# Score all our other models for "LB-trustworthiness"
print("\n" + "="*80)
print("FULL MODEL ZOO RANKING")
print("="*80)
print(f"{'Model':<14s} | overall | mean   | gap     | predicted_LB | trust")
print("-"*70)
zoo = {
    "exp019": P_exp, "V73": v73, "BirdMAE": bmae, "Bruce": bruce,
    "Perch_v2": perch, "ConvNeXt-RAG": cn, "BirdAVES-RAG": ba,
    "MLP-5seed": mlp, "mega_KNN": knn, "balanced_LR": bal_lr, "LGB-7": lgb,
}

zoo_results = []
for name, P in zoo.items():
    oa = macro_auc(P)
    std, mean = site_metrics(P)
    gap = oa - mean
    pred = coefs[0] + coefs[1] * mean + coefs[2] * gap
    # trust based on |gap|
    if abs(gap) < 0.02:
        trust = "HIGH (calibrated)"
    elif gap < -0.05:
        trust = "GOOD (OOF-deflated, V73-like)"
    elif gap > 0.05:
        trust = "LOW (OOF-inflated, Bruce-like)"
    else:
        trust = "MEDIUM"
    print(f"{name:<14s} | {oa:.4f}  | {mean:.4f} | {gap:+.4f} | {pred:.4f}       | {trust}")
    zoo_results.append((name, oa, mean, gap, pred, trust))

# Sort by predicted LB
zoo_results.sort(key=lambda x: -x[4])
print(f"\nRanked by predicted LB:")
for n, oa, m, g, p, t in zoo_results[:10]:
    print(f"  {n}: predicted {p:.4f} ({t})")
