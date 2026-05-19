"""Experiment 23: Use public kernel OOF outputs to validate metric.

Found in public kernel outputs:
- Perch v2 logits + embeddings on labeled OOF (708 rows): safar1, mtoshidesu, 
  youssefmo (LB 0.943-0.948)
- needless090 oof_base + oof_prior (708 rows): LB 0.934-0.935
- koushikrudra oof_base: LB 0.928

These let us validate the metric on actual high-LB models without re-running.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import pandas as pd

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
ex = np.load(f"{ETT}/exp019_aligned.npz", allow_pickle=True)
P_exp = ex["P_exp019"]
Y_full = ex["Y"]
row_fn = ex["row_filename"]
N_full, C = Y_full.shape

# Load all public OOF
print("=== Inspecting public OOF files ===\n")
# safar1 cache (LB 0.948)
safar_d = np.load("/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_arrays.npz")
safar_scores = safar_d["scores"]  # (708, 234) logits
safar_embs = safar_d["embs"]      # (708, 1536) Perch v2 embs
safar_labels = safar_d["primary_labels"]
print(f"safar1 scores: {safar_scores.shape}, labels: {len(safar_labels)}")

# meta parquet (the row IDs)
safar_meta = pd.read_parquet("/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_meta.parquet")
print(f"safar1 meta cols: {safar_meta.columns.tolist()}")
print(safar_meta.head(3))
print(f"shape: {safar_meta.shape}")

# needless090 (LB 0.934)
need_d = np.load("/tmp/pub_outputs/needless090_birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s/perch_cache/full_oof_meta_features.npz")
print(f"\nneedless090 oof_base: {need_d['oof_base'].shape}")
print(f"needless090 fold_id: range {need_d['fold_id'].min()}-{need_d['fold_id'].max()}, distribution {np.bincount(need_d['fold_id'])}")

# Look at row IDs in meta
print(f"\nsafar1 meta head:")
print(safar_meta.iloc[:3, :3].to_string())
if 'row_id' in safar_meta.columns or 'window_id' in safar_meta.columns:
    print(f"Has IDs!")
