"""Experiment 15: Use site_std + LB anchors to design OOF-LB-safe blends.

Approach:
1. We have 5 (validation, LB) pairs from real submissions
2. Linear fit gives RMSE ~0.05, but relationship not strictly linear
3. Use a more careful approach: rank models by site_std, prefer low-std helpers
4. Test if site_std minimization can produce blends predicted to ≥ 0.949
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

# Models
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

ranks = {
    "exp019": rank_norm(P_exp),
    "BirdMAE": rank_norm(bmae),
    "V73": rank_norm(v73),
    "Bruce": rank_norm(bruce),
    "Perch_v2": rank_norm(perch),
    "ConvNeXt-RAG": rank_norm(cn),
    "BirdAVES-RAG": rank_norm(ba),
    "MLP-5seed": rank_norm(mlp),
}

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

# 1. Rank individual models by site_std
print("="*80)
print("INDIVIDUAL MODEL RANKING BY SITE_STD (lower = better LB transfer)")
print("="*80)
individual_metrics = []
for n, R in ranks.items():
    std, mean = w_site_std(R)
    individual_metrics.append((n, std, mean, macro_auc(R)))
individual_metrics.sort(key=lambda x: x[1])
print(f"{'Rank':<5s} {'Model':<15s} | site_std | site_mean | overall_auc")
print("-"*60)
for i, (n, s, m, oa) in enumerate(individual_metrics):
    print(f"  {i+1:<3d} {n:<15s} | {s:.4f}   | {m:.4f}    | {oa:.4f}")

# 2. Build blends and rank
print("\n" + "="*80)
print("BLEND EXPLORATION: which blends MINIMIZE site_std while keeping high AUC?")
print("="*80)

# Search through 2-way blends
print("\n2-way: exp019 + helper (rank blend at various weights)")
print(f"{'Helper':<15s} | w=0.1 std/auc | w=0.2 std/auc | w=0.3 std/auc | w=0.5 std/auc")
for helper in ["BirdMAE", "V73", "Bruce", "Perch_v2", "ConvNeXt-RAG", "BirdAVES-RAG", "MLP-5seed"]:
    R_h = ranks[helper]
    row = f"{helper:<15s} |"
    for w in [0.1, 0.2, 0.3, 0.5]:
        P_b = (1-w) * ranks["exp019"] + w * R_h
        std, _ = w_site_std(P_b)
        a = macro_auc(P_b)
        row += f" {std:.4f}/{a:.4f}  |"
    print(row)

# 3. Find optimal blends — minimize site_std subject to AUC > exp019 baseline
print("\n" + "="*80)
print("OPTIMIZATION: find 2-way blends with site_std < exp019 (0.0997)")
print("="*80)
results = []
for helper in ["BirdMAE", "V73", "Perch_v2", "ConvNeXt-RAG", "MLP-5seed"]:
    R_h = ranks[helper]
    for w in np.arange(0.05, 0.61, 0.05):
        P_b = (1-w) * ranks["exp019"] + w * R_h
        std, _ = w_site_std(P_b)
        a = macro_auc(P_b)
        if std < 0.0997 and a >= 0.95:  # better site_std AND not catastrophic OOF
            results.append((helper, w, std, a, "exp019+"+helper))

results.sort(key=lambda x: x[2])  # sort by site_std ascending
print(f"{'Blend':<25s} | weight | site_std | overall_auc")
for h, w, s, a, _ in results[:15]:
    print(f"  exp019 + {h:<15s} | w={w:.2f}  | {s:.4f}   | {a:.4f}")

# 4. 3-way and 4-way blends
print("\n" + "="*80)
print("3-WAY BLENDS (anchor exp019 + 2 helpers)")
print("="*80)
helpers_3w = ["BirdMAE", "V73", "Perch_v2", "ConvNeXt-RAG", "MLP-5seed"]
results = []
for h1 in helpers_3w:
    for h2 in helpers_3w:
        if h1 >= h2: continue
        for we in [0.5, 0.6, 0.7]:
            for w1 in [0.1, 0.2, 0.3]:
                w2 = 1 - we - w1
                if w2 < 0.05: continue
                P_b = we * ranks["exp019"] + w1 * ranks[h1] + w2 * ranks[h2]
                std, _ = w_site_std(P_b)
                a = macro_auc(P_b)
                if std < 0.095 and a >= 0.96:
                    results.append((h1, h2, we, w1, w2, std, a))
results.sort(key=lambda x: x[5])
print(f"{'Helpers':<35s} | exp w | h1 w | h2 w | site_std | overall_auc")
for h1, h2, we, w1, w2, s, a in results[:15]:
    print(f"  {h1+', '+h2:<33s} | {we:.2f}  | {w1:.2f} | {w2:.2f} | {s:.4f}   | {a:.4f}")

# 5. Predict LB with our 5-anchor regression
known = {
    "exp019": (0.0997, 0.949),
    "V73": (0.1049, 0.941),
    "Bruce": (0.1129, 0.755),
    "slot6": (0.0912, 0.946),  # approx; need to recompute
    "slot11": (0.0995, 0.949),
}
stds = np.array([v[0] for v in known.values()])
lbs = np.array([v[1] for v in known.values()])
# linear fit
from numpy.linalg import lstsq
X = np.column_stack([np.ones(len(stds)), stds])
coefs, _, _, _ = lstsq(X, lbs, rcond=None)
print(f"\nLinear: LB ≈ {coefs[0]:.4f} + {coefs[1]:.4f} * site_std (R²={pearsonr(stds, lbs)[0]**2:.3f})")

# Show top blends with predicted LB
print("\nTop 10 by site_std with predicted LB:")
all_blends = []
for h1 in helpers_3w:
    for h2 in helpers_3w:
        if h1 >= h2: continue
        for we in np.arange(0.4, 0.91, 0.05):
            for w1 in np.arange(0.05, 0.61, 0.05):
                w2 = 1 - we - w1
                if w2 < 0.03 or w2 > 0.55: continue
                P_b = we * ranks["exp019"] + w1 * ranks[h1] + w2 * ranks[h2]
                std, _ = w_site_std(P_b)
                a = macro_auc(P_b)
                pred_lb = coefs[0] + coefs[1] * std
                all_blends.append((h1, h2, we, w1, w2, std, a, pred_lb))

all_blends.sort(key=lambda x: -x[7])  # by predicted LB desc
print(f"{'Helpers':<25s} | weights         | site_std | OOF auc | predicted LB")
for h1, h2, we, w1, w2, s, a, lb_p in all_blends[:10]:
    print(f"  {h1+'+'+h2:<23s} | {we:.2f}/{w1:.2f}/{w2:.2f} | {s:.4f}   | {a:.4f}  | {lb_p:.4f}")
