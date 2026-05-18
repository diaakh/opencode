"""Test 3 untried high-value insights:

1. Labeled hourly_species_priors (ground-truth, no Perch bias) — vs pseudo
2. Bruce raw probability blend with exp019-like (ensemble diversity)
3. Per-class temperature derived from Bruce OOF (proper calibration vector)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

import postproc_v3 as pp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPUS = ROOT / "meta_corpus" / "datasets"
META = ROOT / "meta_analysis"
PRIORS_DIR = ROOT / "inference_notebooks" / "priors_bundle"


def macro_auc(y, score):
    return float(np.mean([roc_auc_score(y[:, c], score[:, c])
                          for c in range(y.shape[1]) if y[:, c].sum() > 0]))


def to_rank_power(prob, rp=0.6, lo=0.477, hi=0.555):
    n = prob.shape[0]
    out = np.zeros_like(prob, dtype=np.float32)
    for c in range(prob.shape[1]):
        r = rankdata(prob[:, c]) / n
        rp_v = r ** rp
        out[:, c] = lo + (rp_v - rp_v.min()) / max(rp_v.max() - rp_v.min(), 1e-9) * (hi - lo)
    return np.clip(out, 0.001, 0.999)


samp = pd.read_csv(DATA / "sample_submission.csv")
class_cols = [c for c in samp.columns if c != "row_id"]
oof = np.load(CORPUS / "teacher_oof_predictions.npz")
rows = pd.read_parquet(CORPUS / "teacher_eval_rows.parquet")
bruce_prob = 1.0 / (1.0 + np.exp(-oof["oof"].astype(np.float32)))
y_true = oof["y_true"].astype(np.int32)
row_ids = rows["row_id"].tolist()
hours = rows["hour_utc"].values

exp_like = to_rank_power(bruce_prob)
priors = pp.load_priors_filled(PRIORS_DIR)

# Synthetic dead-hour
np.random.seed(42)
mask_dead = np.random.rand(len(hours)) < 0.20
fake_hours = hours.copy()
fake_hours[mask_dead] = 11
new_row_ids = []
for i, rid in enumerate(row_ids):
    if fake_hours[i] != hours[i]:
        new_rid = pp._ROW_RE.sub(lambda m: m.group(0).replace(m.group(3), "110000"), rid)
        new_row_ids.append(new_rid)
    else:
        new_row_ids.append(rid)

base_b = macro_auc(y_true, exp_like)
base_a = macro_auc(y_true, exp_like)
best_known_b = 0.9585  # hour 0.05 + site 0.2
print(f"baseline B: {base_b:.4f}  |  best known stack B: {best_known_b:.4f}")

# ============================================================
# TEST 1: Labeled hourly prior vs pseudo
# ============================================================
print("\n" + "=" * 80)
print(" TEST 1: LABELED hourly priors (ground truth) vs pseudo-cache")
print("=" * 80)
lab_prior = pd.read_csv(META / "hourly_species_priors.csv").set_index("hour")
print(f"Labeled prior: shape {lab_prior.shape}, hours {sorted(lab_prior.index.tolist())}")
# Fill missing hours with global
for h in range(24):
    if h not in lab_prior.index:
        lab_prior.loc[h] = lab_prior.mean(axis=0)
lab_prior = lab_prior.sort_index()
lab_arr = pp.align(lab_prior, class_cols)
# Some classes will be all-zero in labeled (those not in any of the 9 labeled sites)
zero_classes = (lab_arr.sum(axis=0) == 0).sum()
print(f"Classes with zero labeled-prior coverage: {zero_classes}/234")
# Fill zero-class columns with pseudo (avoid crush)
pseudo_arr = pp.align(priors["pseudo_hour"], class_cols)
mask_zero = lab_arr.sum(axis=0) == 0
combined = lab_arr.copy()
combined[:, mask_zero] = pseudo_arr[:, mask_zero]
print(f"Built hybrid prior: labeled-covered + pseudo-fill ({mask_zero.sum()} pseudo-filled classes)")

print(f"\nUsing LABELED-only (with global fill, no pseudo blend):")
for w in [0.025, 0.05, 0.1, 0.2]:
    pred = pp.p_hour_prior(exp_like.copy().astype(np.float64), fake_hours, lab_arr, w)
    auc = macro_auc(y_true, pred)
    print(f"  w={w}: {auc:.4f}")

print(f"\nUsing HYBRID (labeled + pseudo-fill for missing classes):")
for w in [0.025, 0.05, 0.1, 0.2]:
    pred = pp.p_hour_prior(exp_like.copy().astype(np.float64), fake_hours, combined, w)
    auc = macro_auc(y_true, pred)
    print(f"  w={w}: {auc:.4f}")

print(f"\nBlend: alpha*labeled + (1-alpha)*pseudo at w=0.05:")
for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    blended = alpha * lab_arr + (1 - alpha) * pseudo_arr
    pred = pp.p_hour_prior(exp_like.copy().astype(np.float64), fake_hours, blended, 0.05)
    auc = macro_auc(y_true, pred)
    print(f"  alpha={alpha}: {auc:.4f}")

# ============================================================
# TEST 2: Ensemble blend with Bruce raw probabilities
# ============================================================
print("\n" + "=" * 80)
print(" TEST 2: Ensemble blend — exp019-like + Bruce raw probabilities")
print("=" * 80)
# Rank-space blend (more robust to scale differences)
n = exp_like.shape[0]
rank_exp = np.zeros_like(exp_like)
rank_bruce = np.zeros_like(exp_like)
for c in range(exp_like.shape[1]):
    rank_exp[:, c] = rankdata(exp_like[:, c]) / n
    rank_bruce[:, c] = rankdata(bruce_prob[:, c]) / n

print("Pure rank-blend (no priors applied):")
for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]:
    blend = alpha * rank_exp + (1 - alpha) * rank_bruce
    auc = macro_auc(y_true, blend)
    print(f"  alpha_exp={alpha}: {auc:.4f}")

# What if we patch BOTH first then blend?
print("\nPatched ensemble: patch(exp_like) blended with patch(bruce_prob), then test:")
patched_exp = pp.apply_all(exp_like, new_row_ids, class_cols, priors, {"w_hour": 0.05, "w_site": 0.2})
patched_bruce = pp.apply_all(bruce_prob, new_row_ids, class_cols, priors, {"w_hour": 0.5, "w_site": 0.2})
# Re-rank
rank_pe = np.zeros_like(patched_exp)
rank_pb = np.zeros_like(patched_bruce)
for c in range(patched_exp.shape[1]):
    rank_pe[:, c] = rankdata(patched_exp[:, c]) / n
    rank_pb[:, c] = rankdata(patched_bruce[:, c]) / n
print(f"patched_exp alone (B): {macro_auc(y_true, patched_exp):.4f}")
print(f"patched_bruce alone (B): {macro_auc(y_true, patched_bruce):.4f}")
for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
    blend = alpha * rank_pe + (1 - alpha) * rank_pb
    auc = macro_auc(y_true, blend)
    print(f"  alpha_patched_exp={alpha}: {auc:.4f}")

# ============================================================
# TEST 3: Per-class temperature from labeled OOF
# ============================================================
print("\n" + "=" * 80)
print(" TEST 3: Per-class temperature from Bruce OOF")
print("=" * 80)
# For each class with positives, find T that maximizes AUC
# (Note: temperature is monotonic per-class, so it doesn't change SINGLE-CLASS AUC.
# But applied uniformly with prior, it might shift the relative scale.)
# A more meaningful approach: use Platt scaling per class to get calibrated probs,
# then add prior. Let's test this.

from sklearn.linear_model import LogisticRegression

cal_logit = np.zeros_like(exp_like)
for c in range(exp_like.shape[1]):
    if y_true[:, c].sum() < 5:
        cal_logit[:, c] = exp_like[:, c]
        continue
    # Logistic regression on the exp_like value to fit class probability
    lr = LogisticRegression(C=1.0, max_iter=100)
    try:
        lr.fit(exp_like[:, c:c+1], y_true[:, c])
        cal_logit[:, c] = lr.predict_proba(exp_like[:, c:c+1])[:, 1]
    except Exception:
        cal_logit[:, c] = exp_like[:, c]

print(f"Platt-calibrated exp_like baseline: {macro_auc(y_true, cal_logit):.4f}")
print(f"Original exp_like baseline:         {macro_auc(y_true, exp_like):.4f}")
# Now apply stack to Platt-calibrated:
patched_cal = pp.apply_all(cal_logit, new_row_ids, class_cols, priors, {"w_hour": 0.05, "w_site": 0.2})
print(f"Platt + stack (B): {macro_auc(y_true, patched_cal):.4f}")

print(f"\n** Note: Platt is fit ON the labeled OOF — strict leak. Real LB would be lower.")

# ============================================================
# TEST 4: combined: LABELED prior + Bruce ensemble + stack
# ============================================================
print("\n" + "=" * 80)
print(" TEST 4: Full mega-stack with labeled prior + bruce ensemble")
print("=" * 80)
# Use hybrid prior in apply_all
hybrid_priors = dict(priors)
import copy as _copy
hour_combined = pd.DataFrame(combined, columns=class_cols, index=range(24))
hour_combined.index.name = "hour"
hybrid_priors["pseudo_hour"] = hour_combined  # override

patched_hybrid = pp.apply_all(exp_like, new_row_ids, class_cols, hybrid_priors, {"w_hour": 0.05, "w_site": 0.2})
print(f"exp_like + stack with HYBRID prior:      {macro_auc(y_true, patched_hybrid):.4f}")
patched_pseudo = pp.apply_all(exp_like, new_row_ids, class_cols, priors, {"w_hour": 0.05, "w_site": 0.2})
print(f"exp_like + stack with PURE PSEUDO prior: {macro_auc(y_true, patched_pseudo):.4f}")

# Now blend patched_hybrid with patched_bruce (using pure pseudo)
patched_bruce_pseudo = pp.apply_all(bruce_prob, new_row_ids, class_cols, priors, {"w_hour": 0.5, "w_site": 0.2})
rank_ph = np.zeros_like(patched_hybrid)
rank_pbp = np.zeros_like(patched_bruce_pseudo)
for c in range(patched_hybrid.shape[1]):
    rank_ph[:, c] = rankdata(patched_hybrid[:, c]) / n
    rank_pbp[:, c] = rankdata(patched_bruce_pseudo[:, c]) / n
print(f"\nFinal mega-stack: alpha*patched_hybrid + (1-alpha)*patched_bruce")
for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    blend = alpha * rank_ph + (1 - alpha) * rank_pbp
    auc = macro_auc(y_true, blend)
    print(f"  alpha={alpha}: {auc:.4f}")
