"""Experiment 31: Use UNLABELED data for cross-model agreement metric.

KEY INSIGHT FROM USER:
- Public Perch v2 cache files have 708 rows of predictions on train_soundscapes
- We have labels for only ~649 of those rows (749 - 90 unmatched)
- The 59-90 UNMATCHED rows are model predictions on UNLABELED windows
- We also have our model7 predictions on 792 windows (53 unlabeled)

CROSS-MODEL AGREEMENT FEATURE (no labels needed):
For each model M and each row in UNLABELED set:
  - Look at the TOP-K predicted classes
  - Check what fraction of OTHER models also have those classes in their top-K
  - Sum across rows: model's "consensus alignment"

Hypothesis: models that consistently align with ensemble consensus on unlabeled
data have less site-specific overfitting → better LB transfer.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr, pearsonr
from itertools import combinations
import re

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
Y_full = ex["Y"]
row_fn = ex["row_filename"]
row_start = ex["row_start_sec"]
N_full, C = Y_full.shape

# Load model7 from exp019-labeled-oof (792 rows = ALL train_soundscapes windows incl. unlabeled)
m7 = np.load("/tmp/exp019_oof/labeled_oof_model7_pre_align.npz")
m7_pred = m7["P"]   # (792, 234)
m7_ids = m7["row_ids"]  # 792 row IDs

# Identify which m7 rows are UNLABELED (not in our 739 labels)
def our_key(fn, start):
    return f"{str(fn).replace('.ogg', '')}_{int(start)}"
labeled_keys = set(our_key(row_fn[i], row_start[i]) for i in range(N_full))
unlabeled_idx_m7 = []
labeled_idx_m7 = []
m7_to_label_idx = {}
for i, rid in enumerate(m7_ids):
    # m7 row format: BC2026_Train_NNNN_S08_20250606_030007_5
    # remove "BC2026_Train_NNNN_" prefix
    parts = rid.split('_')
    key = '_'.join(parts[3:-1]) + '_' + parts[-1]  # S08_20250606_030007_5 normalized
    # build our_key would be S08_20250606_030007_5 too
    
    # Build matching key the way we built labeled_keys
    fn_part = '_'.join(parts[3:-1])
    start = int(parts[-1])
    # But our labels use FILENAME like "BC2026_Train_0001_S08_20250606_030007.ogg"
    # Need to find the BC2026_Train_NNNN_ prefix back
    # Look up by site/date/time → BC2026 form
    # Actually, easier: just match by site+date+time
    for our_i in range(N_full):
        our_fn = str(row_fn[our_i])
        if fn_part in our_fn and int(row_start[our_i]) == start - 5:  # we use start (0-based seconds)
            labeled_idx_m7.append(i)
            m7_to_label_idx[i] = our_i
            break
    else:
        unlabeled_idx_m7.append(i)

print(f"model7 (792 rows): {len(labeled_idx_m7)} labeled, {len(unlabeled_idx_m7)} unlabeled")

# Load public predictions on 708 rows
public_npz = [
    ("/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_arrays.npz", "scores"),
    ("/tmp/pub_outputs/mtoshidesu_birdclef-2026-0-947-lb-public-pipeline-reproduced/cache/perch_arrays.npz", "scores"),
    ("/tmp/pub_outputs/needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s/perch_cache/full_oof_meta_features.npz", "oof_base"),
    ("/tmp/pub_outputs/koushikrudra_0-928-winner-position/perch_cache/full_oof_meta_features.npz", "oof_base"),
    ("/tmp/pub_outputs/baidalinadilzhan_perch-improved-lb-0-904/perch_cache/full_oof_meta_features.npz", "oof_base"),
    ("/tmp/pub_outputs/saurabhrajvarma_birdclef-2026-audio-classification-0-922/perch_cache/full_oof_meta_features.npz", "oof_base"),
    ("/tmp/pub_outputs/itshyao_birdclef-2026-s106-eos5-0949-safealign2/perch_cache/full_oof_meta_features.npz", "oof_base"),
]

safar_meta = pd.read_parquet("/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_arrays.npz".replace("perch_arrays.npz", "perch_meta.parquet"))
their_keys = safar_meta["row_id"].astype(str).values

# Build map: their idx → our_idx (or None for unlabeled)
their_to_our = {}
for i, k in enumerate(their_keys):
    if k in labeled_keys:
        # match to our index
        for our_i in range(N_full):
            if our_key(row_fn[our_i], row_start[our_i]) == k:
                their_to_our[i] = our_i
                break
        else:
            their_to_our[i] = None  # unlabeled
    else:
        their_to_our[i] = None  # unlabeled (key not in labels)

n_unlabeled_public = sum(1 for v in their_to_our.values() if v is None)
print(f"Public 708 rows: {708 - n_unlabeled_public} labeled, {n_unlabeled_public} unlabeled")

# Get all model predictions on the SAME 708 row index
# Convert to ranks (cross-model comparable)
def rank_norm(P):
    R = np.zeros_like(P, dtype=np.float32)
    n = P.shape[0]
    for c in range(P.shape[1]):
        R[:, c] = (rankdata(P[:, c]) - 1) / max(n - 1, 1)
    return R

model_preds_pub = {}
for path, key in public_npz:
    name = path.split('/')[-3] if 'cache' in path else path.split('/')[-2]
    try:
        d = np.load(path)
        p = d[key]
        if p.min() < -1:
            p = 1 / (1 + np.exp(-p))
        model_preds_pub[name] = rank_norm(p)
    except: pass

print(f"\nPublic models loaded for agreement: {list(model_preds_pub.keys())}")

# Compute CROSS-MODEL agreement on unlabeled rows
unlabeled_their_idx = [i for i, v in their_to_our.items() if v is None]
print(f"Computing agreement on {len(unlabeled_their_idx)} unlabeled rows")

# Stack model predictions on unlabeled rows
model_list = list(model_preds_pub.keys())
M = len(model_list)
if M >= 2:
    # (M, n_unlabeled, C)
    stack_unlabeled = np.stack([model_preds_pub[m][unlabeled_their_idx] for m in model_list], axis=0)
    print(f"Stack shape: {stack_unlabeled.shape}")
    
    # CONSENSUS: per (row, class), mean of all models' rank
    consensus = stack_unlabeled.mean(axis=0)  # (n_unl, C)
    
    # AGREEMENT score per model: how close is this model's rank to consensus per cell?
    # Use Spearman rank correlation per row
    print(f"\n{'Model':<55s} | mean rank corr with consensus (unlabeled)")
    print("-"*90)
    for i, m in enumerate(model_list):
        m_preds = stack_unlabeled[i]  # (n_unl, C)
        consensus_other = (stack_unlabeled.sum(axis=0) - m_preds) / (M - 1)  # leave-one-out consensus
        # Per-row Spearman rho
        rhos = []
        for r in range(m_preds.shape[0]):
            try:
                rho, _ = spearmanr(m_preds[r], consensus_other[r])
                if not np.isnan(rho): rhos.append(rho)
            except: pass
        print(f"{m:<55s} | {np.mean(rhos):+.3f}")

# Also compute agreement on LABELED rows where models PREDICT the labels correctly
labeled_their_idx = [i for i, v in their_to_our.items() if v is not None]
print(f"\nLabeled rows in public: {len(labeled_their_idx)}")
