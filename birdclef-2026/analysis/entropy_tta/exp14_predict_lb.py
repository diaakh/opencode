"""Experiment 14: Use w_site_std to predict slot12 v4's LB.

Also validate w_site_std with auxiliary LB data points:
- slot6 (exp019+BirdMAE 70/30) = LB 0.946
  Implies: slot6 site_std should be ~midway between exp019 and BirdMAE site_std
- slot11 (exp019+sub_v8+V73 surgical) = LB 0.949
  Implies: slot11 site_std should be ~exp019 site_std (mostly exp019)
- exp019 ALONE = LB 0.949 (anchor)

Also predict the LB of:
- slot12 v4 candidate blend
- exp019 + Perch v2 blends at various weights
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
R_sub_proxy = (R_cn + R_ba + R_mlp) / 3

def w_site_std(P):
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
    return np.sqrt(var)

def macro_auc(P):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs) if aucs else 0

# Reconstruct slot6, slot11, slot12 v4 blends on labeled OOF
print("="*80)
print("Validate w_site_std with auxiliary LB data points")
print("="*80)

# slot6 = exp019 (70%) + BirdMAE (30%) rank blend, blanket
slot6 = 0.70 * R_exp + 0.30 * R_bmae

# slot12 v4 algorithm: calibrated entropy gate, exp019 + sub_v8_proxy + V73
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

# slot11 surgical (~3% of cells modified) — approximate as 97% exp019 + 3% sub_v8
slot11_approx = 0.97 * R_exp + 0.03 * R_sub_proxy

# Also: pure exp019 + BirdMAE alternatives
ex_b_50 = 0.5 * R_exp + 0.5 * R_bmae
ex_b_30 = 0.7 * R_exp + 0.3 * R_bmae
ex_b_15 = 0.85 * R_exp + 0.15 * R_bmae

# Per-cell gate with calibration: exp019 + BirdMAE
slot12_with_bmae = W_a * R_exp + W_r * R_bmae

# exp019 + Perch v2
ex_p_10 = 0.9 * R_exp + 0.1 * R_perch
ex_p_20 = 0.8 * R_exp + 0.2 * R_perch
ex_p_gate = W_a * R_exp + W_r * R_perch

# Compute metrics for each
blends = {
    "exp019 (anchor)": (R_exp, 0.949),
    "BirdMAE alone": (R_bmae, None),  # unknown, slot6 math suggests ~0.94
    "V73 alone": (R_v73, 0.941),
    "Bruce alone": (rank_norm(bruce), 0.755),
    "Perch v2 alone": (R_perch, None),
    "slot6 (exp+bmae 70/30)": (slot6, 0.946),
    "slot11 surgical approx": (slot11_approx, 0.949),
    "slot12 v4 (sub_v8+v73 gate)": (slot12_v4, None),
    "slot12 with BirdMAE gate": (slot12_with_bmae, None),
    "exp+bmae 50/50": (ex_b_50, None),
    "exp+bmae 30/70": (ex_b_30, None),
    "exp+perch 10%": (ex_p_10, None),
    "exp+perch 20%": (ex_p_20, None),
    "exp+perch entropy gate": (ex_p_gate, None),
}

print(f"{'Blend':<35s} | overall_auc | w_site_std | site_min | known_LB")
print("-"*100)

table = {}
for name, (P, lb) in blends.items():
    overall = macro_auc(P)
    std = w_site_std(P)
    # site_min
    site_min_aucs = []
    for site in sorted(set(sites)):
        mask = sites == site
        if mask.sum() < 15: continue
        Ys, Ps = Y[mask], P[mask]
        aucs = []
        for c in range(C):
            if Ys[:, c].sum() < 2 or Ys[:, c].sum() == mask.sum(): continue
            if Ps[:, c].max() == Ps[:, c].min(): continue
            try: aucs.append(roc_auc_score(Ys[:, c], Ps[:, c]))
            except: pass
        if aucs:
            site_min_aucs.append(np.mean(aucs))
    site_min = min(site_min_aucs) if site_min_aucs else 0
    table[name] = {"overall": overall, "std": std, "site_min": site_min, "lb": lb}
    lb_str = f"{lb:.3f}" if lb else "?"
    print(f"{name:<35s} | {overall:.4f}      | {std:.4f}     | {site_min:.4f}   | {lb_str}")

# Build better predictor using more anchors
print("\n" + "="*80)
print("Robust LB predictor with 5 anchors (exp019, V73, Bruce, slot6, slot11)")
print("="*80)
known = [n for n in table if table[n]["lb"] is not None]
stds = np.array([table[n]["std"] for n in known])
lbs = np.array([table[n]["lb"] for n in known])
overalls = np.array([table[n]["overall"] for n in known])
mins = np.array([table[n]["site_min"] for n in known])

print(f"Anchors: {[(n, table[n]['lb']) for n in known]}")
print(f"Stds:    {stds}")
print(f"LBs:     {lbs}")

# Try multiple regression: LB ~ overall + std
from numpy.linalg import lstsq
X = np.column_stack([np.ones(len(stds)), stds])
coefs, _, _, _ = lstsq(X, lbs, rcond=None)
print(f"\nUnivariate fit: LB = {coefs[0]:.4f} + {coefs[1]:.4f} * w_site_std")
preds_known = X @ coefs
residuals = lbs - preds_known
print(f"Residuals: {dict(zip(known, [f'{r:+.4f}' for r in residuals]))}")
print(f"RMSE: {np.sqrt((residuals**2).mean()):.4f}")

# Also: LB ~ overall + std + site_min (multiple regression)
X2 = np.column_stack([np.ones(len(stds)), stds, overalls])
coefs2, _, _, _ = lstsq(X2, lbs, rcond=None)
preds2 = X2 @ coefs2
res2 = lbs - preds2
print(f"\nMultivariate: LB = {coefs2[0]:.4f} + {coefs2[1]:.4f}*std + {coefs2[2]:.4f}*overall")
print(f"RMSE: {np.sqrt((res2**2).mean()):.4f}")

# Predict for unknown blends
print(f"\nPredictions for unknown blends (using univariate model):")
for n in table:
    if table[n]["lb"] is not None: continue
    pred = coefs[0] + coefs[1] * table[n]["std"]
    print(f"  {n:<35s}: std={table[n]['std']:.4f}, predicted LB ≈ {pred:.4f}")
