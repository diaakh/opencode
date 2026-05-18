"""Multiple creative experiments — find what actually moves the needle.

1. Within-file temporal smoothing (aliozanmemetoglu's trick)
2. Multi-K KNN ensemble (K=5, 10, 20, 50, 100)
3. Hour-restricted KNN (only neighbors at same hour)
4. KNN over Bruce's logit-space (instead of Perch emb space)
5. Cross-model agreement filter
6. Per-class isotonic calibration on Bruce + KNN blend
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from scipy.stats import rankdata

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/creative"
os.makedirs(OUT, exist_ok=True)

# Data
br = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
Y = br["Y"]
emb_lab = br["embeddings"].astype(np.float32)
file_lab = br["row_filename"]
hour_lab = br["row_hour"].astype(int)
site_lab = br["row_site"].astype(str)
end_sec = br["row_end_sec"].astype(int)
P_bruce = br["P_bruce"]
P_perch = 1.0 / (1.0 + np.exp(-br["P_perch_logits"]))
N, C = Y.shape

tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
classes = list(br["classes"])
class_to_name = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
class_taxa = np.array([class_to_name.get(c, "Unknown") for c in classes])

is_texture = np.array([class_to_name.get(c) in ("Insecta", "Amphibia") for c in classes])
print(f"Texture classes (Insecta/Amphibia): {is_texture.sum()}; event: {(~is_texture).sum()}")


def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(P.shape[1]):
        col = P[:, c]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            R[valid, c] = rankdata(col[valid]) / valid.sum()
            R[~valid, c] = 0.5
    return R


def macro_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0 or y[:, c].sum() == len(y):
            continue
        col = p[:, c]
        mask = ~np.isnan(col)
        if mask.sum() < 5 or y[mask, c].sum() == 0 or y[mask, c].sum() == mask.sum():
            continue
        try: aucs.append(roc_auc_score(y[mask, c], col[mask]))
        except: pass
    return float(np.mean(aucs)) if aucs else float("nan")


# Build (file, win_idx) → row_idx lookup for within-file smoothing
file_to_rows = {}
for i, (f, e) in enumerate(zip(file_lab, end_sec)):
    win_idx = (e - 1) // 5
    file_to_rows.setdefault(f, {})[win_idx] = i


# ---------- E1: Within-file temporal smoothing ----------
def smooth_in_file(P, smooth_event=(0.20, 0.60, 0.20), smooth_texture=(0.35, 0.30, 0.35)):
    """For each file, smooth predictions across the 12 windows."""
    P_out = P.copy()
    for f, win_to_row in file_to_rows.items():
        windows = sorted(win_to_row.keys())
        if len(windows) < 3:
            continue
        for c in range(C):
            ker = smooth_texture if is_texture[c] else smooth_event
            cw = [win_to_row[w] for w in windows]
            cls_vals = P[cw, c]
            # Convolve [prev, curr, next] with kernel
            pad = np.pad(cls_vals, (1, 1), mode="edge")
            smoothed = pad[:-2] * ker[0] + pad[1:-1] * ker[1] + pad[2:] * ker[2]
            for k, w in enumerate(windows):
                P_out[win_to_row[w], c] = smoothed[k]
    return P_out


P_bruce_smoothed = smooth_in_file(P_bruce)
print(f"\nE1 Within-file smoothing on Bruce: {macro_auc(Y, P_bruce_smoothed):.4f} "
      f"(was {macro_auc(Y, P_bruce):.4f})")

# Try blending with KNN
probe_v2 = np.load(f"{REPO}/analysis/gbm_stacker/oof_predictions_v2.npz", allow_pickle=True)
P_knn = probe_v2["P_knn"]
R_bruce_smoothed = rank_norm(P_bruce_smoothed)
R_knn = rank_norm(P_knn)
print(f"  Bruce_smoothed + KNN blend: {macro_auc(Y, 0.5*R_bruce_smoothed + 0.5*R_knn):.4f}")


# ---------- E2: Multi-K KNN ensemble ----------
print("\n=== E2: Multi-K KNN ensemble (file-grouped) ===")
emb_lab_n = normalize(emb_lab, axis=1)
sim = emb_lab_n @ emb_lab_n.T
same_file = np.array([[file_lab[i] == file_lab[j] for j in range(N)] for i in range(N)])
sim_masked = np.where(same_file, -np.inf, sim)

knn_preds = {}
for K in [5, 10, 20, 50, 100]:
    knn_p = np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        idx = np.argsort(sim_masked[i])[::-1][:K]
        w = np.maximum(sim_masked[i, idx], 0)
        w = w / (w.sum() if w.sum() > 0 else 1.0)
        knn_p[i] = w @ Y[idx]
    auc = macro_auc(Y, knn_p)
    knn_preds[K] = knn_p
    print(f"  KNN K={K:3d}: {auc:.4f}")

# Average of multi-K KNN
multi_k = np.mean([rank_norm(knn_preds[K]) for K in [5, 10, 20, 50]], axis=0)
multi_k_auc = macro_auc(Y, multi_k)
print(f"  Multi-K (5,10,20,50) avg: {multi_k_auc:.4f}")

# Bruce + multi-K
R_bruce = rank_norm(P_bruce)
print(f"  Bruce + multi-K avg: {macro_auc(Y, 0.5*R_bruce + 0.5*multi_k):.4f}")


# ---------- E3: Hour-restricted KNN ----------
print("\n=== E3: Hour-restricted KNN (only same-hour neighbors) ===")
same_hour = hour_lab[:, None] == hour_lab[None, :]
sim_hour = np.where(same_hour & (~same_file), sim, -np.inf)
hour_knn = np.zeros((N, C), dtype=np.float32)
K = 20
for i in range(N):
    idx = np.argsort(sim_hour[i])[::-1][:K]
    n_valid = np.sum(sim_hour[i, idx] > -np.inf)
    if n_valid < 2:
        # Fall back to global KNN
        idx = np.argsort(sim_masked[i])[::-1][:K]
    w = np.maximum(sim_masked[i, idx], 0)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    hour_knn[i] = w @ Y[idx]
hour_knn_auc = macro_auc(Y, hour_knn)
print(f"  Hour-restricted KNN K=20: {hour_knn_auc:.4f}")
R_hour_knn = rank_norm(hour_knn)
print(f"  Bruce + hour-restricted-KNN: {macro_auc(Y, 0.5*R_bruce + 0.5*R_hour_knn):.4f}")


# ---------- E4: Cross-model agreement boost ----------
print("\n=== E4: Cross-model agreement boost ===")
# When Bruce + KNN + Probe all agree, boost confidence
R_probe = rank_norm(probe_v2["P_probe"])
agree = (R_bruce > 0.85) & (R_knn > 0.85) & (R_probe > 0.7)
# Multiplier
boost_factor = 1.5
R_blend = (R_bruce + R_knn) / 2
R_boosted = R_blend.copy()
R_boosted[agree] = np.minimum(R_boosted[agree] * boost_factor, 1.0)
print(f"  agreement count: {agree.sum()}")
print(f"  Bruce+KNN+agree-boost: {macro_auc(Y, R_boosted):.4f}")


# ---------- E5: Per-class isotonic calibration of Bruce+KNN ----------
print("\n=== E5: Per-class isotonic calibration (file-grouped) ===")
file_idx_arr = pd.Categorical(file_lab).codes
gkf = GroupKFold(n_splits=5)
R_blend_base = 0.5 * R_bruce + 0.5 * R_knn

# Per-fold per-class isotonic
isotonic_oof = R_blend_base.copy()
for fold, (tr, va) in enumerate(gkf.split(np.zeros(N), Y, groups=file_idx_arr)):
    for c in range(C):
        if Y[tr, c].sum() < 3 or Y[tr, c].sum() == len(tr):
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(R_blend_base[tr, c], Y[tr, c])
            isotonic_oof[va, c] = ir.transform(R_blend_base[va, c])
        except Exception:
            pass
iso_auc = macro_auc(Y, isotonic_oof)
print(f"  Bruce+KNN + per-class isotonic: {iso_auc:.4f}")


# ---------- E6: Weighted blend grid search ----------
print("\n=== E6: Weighted blend grid (Bruce, KNN, multi-K, smoothed) ===")
R_smoothed = rank_norm(P_bruce_smoothed)
results = []
for wb in [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:
    for wk in [0, 0.2, 0.4, 0.5, 0.6]:
        for ws in [0, 0.2, 0.4]:
            wp = 1 - wb - wk - ws
            if wp < 0: continue
            blend = wb*R_bruce + wk*R_knn + ws*R_smoothed + wp*rank_norm(P_perch)
            if blend.sum() == 0: continue
            auc = macro_auc(Y, blend)
            results.append((wb, wk, ws, wp, auc))
top = sorted(results, key=lambda x: -x[-1])[:10]
print(f"  Top 10 blends:")
print(f"  {'wB':>5s} {'wK':>5s} {'wS':>5s} {'wP':>5s}  AUC")
for r in top:
    print(f"  {r[0]:>5.2f} {r[1]:>5.2f} {r[2]:>5.2f} {r[3]:>5.2f}  {r[4]:.4f}")


# ---------- Summary ----------
print("\n=== EXPERIMENTAL SUMMARY ===")
print(f"  Bruce alone:                                0.8670")
print(f"  Bruce + KNN simple blend (prior best):      0.8874")
print(f"  Bruce_smoothed (within-file):               {macro_auc(Y, P_bruce_smoothed):.4f}")
print(f"  Bruce_smoothed + KNN blend:                 {macro_auc(Y, 0.5*R_bruce_smoothed + 0.5*R_knn):.4f}")
print(f"  Multi-K KNN avg:                            {multi_k_auc:.4f}")
print(f"  Bruce + multi-K:                            {macro_auc(Y, 0.5*R_bruce + 0.5*multi_k):.4f}")
print(f"  Hour-restricted KNN K=20:                   {hour_knn_auc:.4f}")
print(f"  Bruce + hour-restricted KNN:                {macro_auc(Y, 0.5*R_bruce + 0.5*R_hour_knn):.4f}")
print(f"  Bruce+KNN+agree-boost:                      {macro_auc(Y, R_boosted):.4f}")
print(f"  Bruce+KNN+isotonic:                         {iso_auc:.4f}")
print(f"  Best grid search:                           {top[0][-1]:.4f}")
