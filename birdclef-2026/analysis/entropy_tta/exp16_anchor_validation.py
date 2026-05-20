"""Experiment 16: Validate site_std with all available LB anchors.

Internal anchor catalog:
  exp019:        LB 0.949 — full ensemble
  V73:           LB 0.941 — 5-fold mel-CNN standalone
  Bruce:         LB 0.755 — Ridge on Perch features
  slot6:         LB 0.946 — exp019 70% + BirdMAE 30%
  slot11:        LB 0.949 — exp019 + sub_v8 + V73 surgical (~3% modified)
  sub1:          LB 0.920 — exp019 + hour_prior (w=3.0) — exp019 + 25% hour
  sub3:          LB 0.914 — exp019 + hour_prior + alias + site_blind
  
Compute site_std for all reproducible blends. Plot correlation with LB.
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
row_hour = ex["row_start_sec"]  # actually it's start_sec; need real hour

# Get hours from filenames or aligned data
import re
HOUR_RE = re.compile(r"_(\d{8})_(\d{2})\d{4}")
hours = np.array([int(HOUR_RE.search(str(f)).group(2)) if HOUR_RE.search(str(f)) else -1 for f in row_fn])
print(f"hour distribution: {dict(zip(*np.unique(hours, return_counts=True)))}")

# Load models
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
R_bruce = rank_norm(bruce)
R_cn = rank_norm(cn)
R_ba = rank_norm(ba)
R_mlp = rank_norm(mlp)
R_sub_proxy = (R_cn + R_ba + R_mlp) / 3

# Build hour prior matrix (24 hours)
hour_counts = np.zeros((24, C), dtype=np.float32)
for i, h in enumerate(hours):
    if 0 <= h < 24:
        hour_counts[h] += Y[i]
hour_prior = hour_counts / (hour_counts.sum(axis=0, keepdims=True) + 1e-6)
# Build per-row hour prior
hour_prior_per_row = np.zeros((N, C), dtype=np.float32)
for i, h in enumerate(hours):
    if 0 <= h < 24:
        hour_prior_per_row[i] = hour_prior[h]
R_hour = rank_norm(hour_prior_per_row + 1e-7 * np.random.RandomState(42).rand(N, C))

# Reconstruct LB-anchored blends
# sub1: exp019 + hour_prior w=3.0 → exp019 + w/(w+1) * hour ~ exp019 + 0.75 * hour
# Actually "w=3.0" might mean different scaling. Let's try: rank-blend with w=0.25 hour
sub1_approx = 0.75 * R_exp + 0.25 * R_hour
# Alternative: w=0.20
sub1_alt = 0.80 * R_exp + 0.20 * R_hour

# sub2: Bruce + Perch (Perch v1 ≠ Perch v2 but proxy ok) + hour_prior w=3.0
# Standalone Bruce+Perch at 50/50 + hour at 25%
sub2_approx = 0.40 * R_bruce + 0.40 * R_perch + 0.20 * R_hour

# slot6: exp019 + BirdMAE 70/30 rank-blend (BLANKET)
slot6 = 0.7 * R_exp + 0.3 * R_bmae

# slot11 surgical approximation
slot11 = 0.97 * R_exp + 0.03 * R_sub_proxy

# slot12 v4 — re-compute
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
    return np.sqrt(var), mean

def macro_auc(P):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    return np.mean(aucs) if aucs else 0

# Anchor table
anchors = {
    "exp019":             (R_exp, 0.949),
    "V73":                (R_v73, 0.941),
    "Bruce":              (R_bruce, 0.755),
    "slot6":              (slot6, 0.946),
    "slot11":             (slot11, 0.949),
    "sub1 (w=3.0)":       (sub1_approx, 0.920),
    "sub2 (Bruce+Perch)": (sub2_approx, 0.755),
    "slot12 v4":          (slot12_v4, None),  # unknown
}

print(f"\n{'Blend':<22s} | overall | w_site_std | site_mean | known_LB | gap")
print("-"*80)
data = []
for name, (P, lb) in anchors.items():
    overall = macro_auc(P)
    std, mean = w_site_std(P)
    gap = (lb - overall) if lb else None
    gap_str = f"{gap:+.4f}" if gap is not None else "?"
    lb_str = f"{lb:.3f}" if lb else "?"
    print(f"{name:<22s} | {overall:.4f}  | {std:.4f}     | {mean:.4f}    | {lb_str}   | {gap_str}")
    if lb is not None:
        data.append((name, overall, std, mean, lb))

# Correlation
print("\n" + "="*80)
print("Correlation with LB:")
print("="*80)
arr = np.array([(s, m, lb) for n, oa, s, m, lb in data], dtype=np.float64)
names_ord = [d[0] for d in data]
stds_v = arr[:, 0]
means_v = arr[:, 1]
lbs_v = arr[:, 2]

r_std, _ = pearsonr(stds_v, lbs_v)
rho_std, _ = spearmanr(stds_v, lbs_v)
r_mean, _ = pearsonr(means_v, lbs_v)
rho_mean, _ = spearmanr(means_v, lbs_v)
r_overall, _ = pearsonr([d[1] for d in data], lbs_v)
rho_overall, _ = spearmanr([d[1] for d in data], lbs_v)

print(f"\nLB ↔ overall_auc:  Pearson r={r_overall:+.3f}  Spearman ρ={rho_overall:+.3f}")
print(f"LB ↔ w_site_std:   Pearson r={r_std:+.3f}  Spearman ρ={rho_std:+.3f}")
print(f"LB ↔ site_mean:    Pearson r={r_mean:+.3f}  Spearman ρ={rho_mean:+.3f}")

# Try combining metrics
print(f"\nDual fit: LB = a + b*site_std + c*site_mean")
from numpy.linalg import lstsq
X = np.column_stack([np.ones(len(stds_v)), stds_v, means_v])
coefs, _, _, _ = lstsq(X, lbs_v, rcond=None)
preds = X @ coefs
rmse = np.sqrt(((preds - lbs_v)**2).mean())
print(f"  LB ≈ {coefs[0]:.3f} + {coefs[1]:.3f}*std + {coefs[2]:.3f}*mean   RMSE={rmse:.4f}")
print(f"\nKnown predictions:")
for i, (name, oa, s, m, lb) in enumerate(data):
    print(f"  {name:<22s}: predicted {preds[i]:.4f}, actual {lb:.4f}, error {preds[i]-lb:+.4f}")

# Predict slot12 v4
std_v4, mean_v4 = w_site_std(slot12_v4)
oa_v4 = macro_auc(slot12_v4)
pred_v4 = coefs[0] + coefs[1] * std_v4 + coefs[2] * mean_v4
print(f"\nslot12 v4 prediction: site_std={std_v4:.4f}, site_mean={mean_v4:.4f}")
print(f"  Predicted LB: {pred_v4:.4f}")
print(f"  (overall_auc={oa_v4:.4f})")
