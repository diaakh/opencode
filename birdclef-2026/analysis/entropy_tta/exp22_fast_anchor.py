"""Experiment 22: Fast anchor validation using existing OOF predictions.

Key shortcut: instead of training new heads, validate metric using
KNOWN-LB models that we can approximate from existing data.

Specifically:
- Reconstruct hour_prior model (sub1 LB = 0.920)
- Reconstruct Bruce+Perch sub2 (LB = 0.755)
- Use these as additional anchors

These were submitted to LB and we can rebuild them locally.
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

# Get hours
HOUR_RE = re.compile(r"_(\d{8})_(\d{2})\d{4}")
hours = np.array([int(HOUR_RE.search(str(f)).group(2)) if HOUR_RE.search(str(f)) else -1 for f in row_fn])

# Build hour prior from Y (note: this leaks slightly, but matches public sub1)
hour_counts = np.zeros((24, C), dtype=np.float32)
for i, h in enumerate(hours):
    if 0 <= h < 24:
        hour_counts[h] += Y[i]
hour_freq = hour_counts.sum(axis=0)  # per-class prevalence
hour_prior = hour_counts / (hour_freq[None, :] + 1)  # normalize per class

# Per-row hour prior
P_hour = np.zeros((N, C), dtype=np.float32)
for i, h in enumerate(hours):
    if 0 <= h < 24:
        P_hour[i] = hour_prior[h]
# Add small noise for tie-breaking
P_hour = P_hour + 1e-7 * np.random.RandomState(42).rand(N, C)

# Load Bruce + Perch
bruce = np.load('/tmp/bruce_oof/labeled_oof_perch_bruce.npz', allow_pickle=True)['P_bruce']
perch_v1 = np.load('/tmp/bruce_oof/labeled_oof_perch_bruce.npz', allow_pickle=True)['P_perch_logits']
perch_v2 = np.load("/tmp/perch20_blend.npz", allow_pickle=True)["P_perch20"]

def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(C):
        R[:, c] = (rankdata(P[:, c]) - 1) / (N - 1)
    return R

# Build sub1: exp019 + hour_prior at "w=3.0" rank blend
# Interpretation: w=3.0 likely means w/(w+1) = 0.75 hour weight
R_exp = rank_norm(P_exp)
R_hour = rank_norm(P_hour)
R_bruce = rank_norm(bruce)
R_perch_v1 = rank_norm(perch_v1)

sub1_v1 = 0.25 * R_exp + 0.75 * R_hour  # w=3.0 interpretation
sub1_v2 = 0.5 * R_exp + 0.5 * R_hour    # alt interpretation
sub1_v3 = 0.75 * R_exp + 0.25 * R_hour  # conservative

# sub2: Bruce+Perch standalone at "batched + hour_prior w=3.0"
# Bruce+Perch blend ~50/50, then add hour
sub2_v1 = 0.5 * R_bruce + 0.5 * R_perch_v1  # just Bruce+Perch
sub2_v2 = 0.4 * R_bruce + 0.4 * R_perch_v1 + 0.2 * R_hour  # +hour
sub2_v3 = 0.3 * R_bruce + 0.3 * R_perch_v1 + 0.4 * R_hour  # heavy hour

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
    oa = macro_auc(P)
    _, mean = site_metrics(P)
    gap = oa - mean
    return 0.277 + 0.714 * mean - 0.894 * gap, oa, mean, gap

print("="*80)
print("RECONSTRUCTING LB-KNOWN SUBMISSIONS")
print("="*80)
print(f"{'submission':<30s} | overall | site_mean | gap     | PREDICTED LB | known LB")
print("-"*100)

reconstructions = {
    "exp019 only (anchor)": (R_exp, 0.949),
    "sub1 v1 (25%exp+75%hour)": (sub1_v1, 0.920),
    "sub1 v2 (50%exp+50%hour)": (sub1_v2, 0.920),
    "sub1 v3 (75%exp+25%hour)": (sub1_v3, 0.920),
    "sub2 v1 (Bruce+Perch 50/50)": (sub2_v1, 0.755),
    "sub2 v2 (B+P+20%hour)": (sub2_v2, 0.755),
    "sub2 v3 (B+P+40%hour)": (sub2_v3, 0.755),
}

for name, (P, lb) in reconstructions.items():
    pred, oa, mean, gap = predict_lb(P)
    error = pred - lb
    print(f"{name:<30s} | {oa:.4f}  | {mean:.4f}    | {gap:+.4f} | {pred:.4f}       | {lb:.3f}  (err {error:+.4f})")

# Now: try a richer fit with these added anchors
print("\n" + "="*80)
print("REFINED METRIC FIT WITH MORE ANCHORS")
print("="*80)
# Use best matches for sub1, sub2
all_anchors = [
    ("exp019", R_exp, 0.949),
    ("V73", rank_norm(np.load("/tmp/v73_oof/v73_labeled_oof.npz", allow_pickle=True)["P_v73"]), 0.941),
    ("Bruce", R_bruce, 0.755),
    ("slot6", 0.7*R_exp + 0.3*rank_norm(np.load("/tmp/birdmae_oof_kernel/birdmae_labeled_oof.npz", allow_pickle=True)["P_birdmae"]), 0.946),
    ("slot11", 0.97*R_exp + 0.01*R_bruce + 0.02*rank_norm(np.load("/tmp/perch20_blend.npz")["P_perch20"]), 0.949),
    # Add reconstructions with best-match LB
    ("sub1_v1", sub1_v1, 0.920),
    ("sub2_v1", sub2_v1, 0.755),
]

import numpy as np
from numpy.linalg import lstsq

X_data = []
y_data = []
names = []
for name, P, lb in all_anchors:
    oa = macro_auc(P)
    _, mean = site_metrics(P)
    gap = oa - mean
    X_data.append([1, mean, gap])
    y_data.append(lb)
    names.append((name, oa, mean, gap, lb))

X = np.array(X_data)
y = np.array(y_data)
coefs, _, _, _ = lstsq(X, y, rcond=None)
preds = X @ coefs
errors = preds - y
rmse = np.sqrt((errors**2).mean())

print(f"\nFit: LB = {coefs[0]:.4f} + {coefs[1]:.4f}*site_mean + {coefs[2]:.4f}*gap")
print(f"RMSE: {rmse:.4f}\n")
print(f"{'Anchor':<15s} | site_mean | gap     | predicted | actual | error")
print("-"*60)
for (n, oa, m, g, lb), p in zip(names, preds):
    print(f"{n:<15s} | {m:.4f}    | {g:+.4f} | {p:.4f}    | {lb:.4f} | {p-lb:+.4f}")
