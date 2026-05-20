"""Experiment 20: Maximize signal from EXISTING predictions + LB inferences.

Strategy: instead of running 10 more Kaggle kernels (60min each), squeeze
all signal out of what we already have:

We have OOF predictions for:
  exp019, V73, BirdMAE, Bruce, Perch_v2_raw, ConvNeXt-RAG, BirdAVES-RAG,
  MLP-5seed, mega_KNN, balanced_LR, LGB-7

We have direct LB for: exp019 (0.949), V73 (0.941), Bruce (0.755)

Inferred LB from blend math:
  slot6 = 0.7*exp019 + 0.3*BirdMAE → LB 0.946
    → If exp019 LB=0.949 and slot6 LB=0.946, BirdMAE blanket-30%-impact = -0.003
    → BirdMAE standalone LB ≈ uncertain (could be 0.92-0.94 by various blend models)
  
  slot11 ~ 97% exp019 + 3% sub_v8 → LB 0.949
    → sub_v8 contribution at 3% gives 0 movement (consistent with sub_v8 standalone
       being similar to exp019, ~0.92-0.95)

Public LB tier validations from 100 kernel catalog:
  Pure Perch v2 (no other models): 0.905-0.912
  + ProtoSSM: 0.925
  + SED: 0.932-0.946
  + post-processing: 0.946-0.948

The metric LB ≈ 0.277 + 0.714*site_mean - 0.894*gap predicts:
  Perch_v2_raw: 0.858 (but public Perch v2 + trained head: 0.91)
    DISCREPANCY: 0.05 — likely because our Perch_v2 OOF is RAW (no head),
    public Perch v2 includes a classifier head trained on train_audio.

Refined metric usage:
  - For models with HIGH overall_auc (>0.95): metric extrapolates well
  - For models with HIGH gap (>+0.05): predicted LB realistically (Bruce-like)
  - For RAW external models (Perch v2 raw): metric UNDER-predicts because
    LB-similar architectures have additional trained components.

Action: trust metric for IN-DISTRIBUTION models (those trained on BC2026
train_audio), interpret cautiously for raw external models.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import json

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

# Load all models
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
    return np.sqrt(var), mean, aucs.min()

# Per-class AUC variance, top-K precision
def per_class_features(P):
    aucs = []
    for c in range(C):
        if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
        if P[:, c].max() == P[:, c].min(): continue
        try: aucs.append(roc_auc_score(Y[:, c], P[:, c]))
        except: pass
    if not aucs: return 0, 0, 0
    aucs = np.array(aucs)
    return aucs.mean(), aucs.std(), aucs.min()

# Calibration ECE
def ece(P, n_bins=10):
    p_flat = P.flatten()
    y_flat = Y.flatten()
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(p_flat, bins[1:-1])
    ece_v = 0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 30: continue
        conf = p_flat[mask].mean()
        acc = y_flat[mask].mean()
        ece_v += (mask.sum() / len(p_flat)) * abs(conf - acc)
    return ece_v

models = {
    "exp019": (P_exp, 0.949),
    "V73": (v73, 0.941),
    "BirdMAE": (bmae, None),  # ~0.94 inferred
    "Bruce": (bruce, 0.755),
    "Perch_v2_raw": (perch, None),  # public Perch+head ~0.91; raw uncertain
    "ConvNeXt-RAG": (cn, None),
    "BirdAVES-RAG": (ba, None),
    "MLP-5seed": (mlp, None),
    "mega_KNN": (knn, None),
    "balanced_LR": (bal_lr, None),
    "LGB-7": (lgb, None),
}

print(f"{'Model':<15s} | overall | site_mean | site_std | site_min | cls_std | ECE   | LB known | LB predicted")
print("-"*120)
rows = []
for name, (P, lb) in models.items():
    oa = macro_auc(P)
    std, mean, smin = site_metrics(P)
    _, cls_std, cls_min = per_class_features(P)
    e = ece(P)
    gap = oa - mean
    pred_lb = 0.277 + 0.714 * mean - 0.894 * gap
    lb_str = f"{lb:.3f}" if lb else "?"
    print(f"{name:<15s} | {oa:.4f}  | {mean:.4f}    | {std:.4f}   | {smin:.4f}   | {cls_std:.3f}   | {e:.3f} | {lb_str:<8s} | {pred_lb:.4f}")
    rows.append({"model": name, "overall": oa, "site_mean": mean, "site_std": std, 
                 "site_min": smin, "cls_std": cls_std, "ece": e, "gap": gap,
                 "lb_known": lb, "lb_predicted": pred_lb})

# Save
with open(f"{ETT}/model_zoo_metrics.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nSaved: {ETT}/model_zoo_metrics.json")

# Build "trust" classification
print("\n" + "="*80)
print("TRUST CLASSIFICATION based on metric:")
print("="*80)
print(f"{'Model':<15s} | LB predicted | gap     | trust tier")
print("-"*60)
for r in sorted(rows, key=lambda x: -x["lb_predicted"]):
    gap = r["gap"]
    if r["lb_predicted"] > 0.94: tier = "SAFE (LB-ceiling tier)"
    elif r["lb_predicted"] > 0.88: tier = "MEDIUM (might add noise)"
    elif r["lb_predicted"] > 0.78: tier = "RISKY (Bruce-like drag)"
    else: tier = "CATASTROPHIC (mega_KNN-like)"
    print(f"{r['model']:<15s} | {r['lb_predicted']:.4f}      | {gap:+.4f} | {tier}")
