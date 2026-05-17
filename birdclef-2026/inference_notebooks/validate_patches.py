"""
Validate post-processing patches on Bruce Wu's pre-computed OOF predictions
(739 labeled windows on the BC2026 train_soundscapes).

This is our local proxy for "Kaggle Perch outputs" since Bruce's clip_student_bundle
includes Ridge predictions on the labeled OOF, and we have ground truth for these.

Reports macro-AUC per patch configuration. Honest macro-AUC numbers here strongly
predict similar deltas on the actual Kaggle private LB (typical 0.01-0.02 OOF->LB gap).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import postproc_patches as pp

ROOT = Path(__file__).resolve().parents[1]  # birdclef-2026/
META = ROOT / "meta_analysis"
DATA = ROOT / "data"
CORPUS = ROOT / "meta_corpus" / "datasets"


def macro_auc(y_true, y_score):
    """Macro-AUC skipping classes with 0 positives (mirrors competition metric)."""
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    # Load OOF + ground truth
    oof = np.load(CORPUS / "teacher_oof_predictions.npz")
    rows = pd.read_parquet(CORPUS / "teacher_eval_rows.parquet")
    samp = pd.read_csv(DATA / "sample_submission.csv")
    class_cols = [c for c in samp.columns if c != "row_id"]

    # Bruce's final OOF prediction (post-pipeline) — saved as LOGITS, convert to prob
    bruce_logit = oof["oof"].astype(np.float32)
    bruce = 1.0 / (1.0 + np.exp(-bruce_logit))
    y_true = oof["y_true"].astype(np.int32)
    raw_logit = oof["raw_scores"].astype(np.float32)
    raw = 1.0 / (1.0 + np.exp(-raw_logit))

    row_ids = rows["row_id"].tolist()
    assert bruce.shape == (739, 234), bruce.shape
    assert y_true.shape == bruce.shape

    print(f"Loaded {bruce.shape[0]} OOF rows, {y_true.sum()} positive labels, {y_true.any(axis=0).sum()} non-empty classes")

    priors = pp.load_priors(META)
    print(f"Loaded priors: {list(priors.keys())}")

    # Sanity-baselines
    base_bruce = macro_auc(y_true, bruce)
    base_raw = macro_auc(y_true, raw)
    print(f"\n=== Baselines ===")
    print(f"Raw Perch (no Bruce):      {base_raw:.4f}")
    print(f"Bruce's pipeline (anchor): {base_bruce:.4f}")

    # Each patch combo
    configs = [
        ("hour_prior_w1.0", ["hour_prior"], dict(hour_prior_weight=1.0)),
        ("hour_prior_w2.0", ["hour_prior"], dict(hour_prior_weight=2.0)),
        ("hour_prior_w3.0 ⭐", ["hour_prior"], dict(hour_prior_weight=3.0)),
        ("hour_prior_w4.0", ["hour_prior"], dict(hour_prior_weight=4.0)),
        ("site_hour_prior_w2.5", ["site_hour_prior"], dict(site_hour_weight=2.5)),
        ("perch_calib only", ["perch_calib"], {}),
        ("hour_prior+perch_calib", ["hour_prior", "perch_calib"], dict(hour_prior_weight=3.0)),
        ("hour_prior+sonotype_alias", ["hour_prior", "sonotype_aliases"], dict(hour_prior_weight=3.0)),
        ("hour_prior+missing_class", ["hour_prior", "missing_class"], dict(hour_prior_weight=3.0)),
        ("hour_prior+chorus_boost", ["hour_prior", "pantanal_chorus"], dict(hour_prior_weight=3.0)),
        ("hour_prior+site_blind", ["hour_prior", "site_blind"], dict(hour_prior_weight=3.0)),
        ("MAX_KITCHEN_SINK", ["hour_prior", "perch_calib", "sonotype_aliases",
                              "missing_class", "pantanal_chorus", "site_blind"],
         dict(hour_prior_weight=3.0)),
    ]

    print(f"\n=== Patch evaluations on Bruce OOF ===")
    print(f"{'Config':<35}  macro-AUC   Δ-from-bruce")
    print(f"{'-' * 35}  ---------   --------------")
    for name, patches, kwargs in configs:
        new_prob = pp.apply_postproc(bruce, row_ids, class_cols, priors, patches, **kwargs)
        auc = macro_auc(y_true, new_prob)
        delta = auc - base_bruce
        marker = "  ⬆" if delta > 0.001 else ("  ≈" if abs(delta) <= 0.001 else "  ⬇")
        print(f"{name:<35}  {auc:.4f}     {delta:+.4f}{marker}")


if __name__ == "__main__":
    main()
