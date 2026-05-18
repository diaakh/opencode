"""Benchmark every postproc_v3 technique individually + greedy stack.

Two test surfaces:
  A) Bruce OOF (739 rows, all night/dawn hours) — pure ranking
  B) Bruce OOF + 20% synthetic dead-hour exposure — realistic LB proxy

We use exp019-like RANK-POWER inputs derived from Bruce's raw logits, since
exp019 is what we patch on Kaggle (rank-power values in [0.477, 0.555]).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

import postproc_v3 as pp

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "meta_analysis"
DATA = ROOT / "data"
CORPUS = ROOT / "meta_corpus" / "datasets"
PRIORS_DIR = ROOT / "inference_notebooks" / "priors_bundle"
RANK_POWER = 0.6


def macro_auc(y, score):
    aucs = []
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y[:, c], score[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def to_rank_power_like_exp019(prob, target_lo=0.477, target_hi=0.555):
    """Convert per-class to rank/N, raise to RANK_POWER, rescale to [lo, hi]."""
    n = prob.shape[0]
    out = np.zeros_like(prob, dtype=np.float32)
    for c in range(prob.shape[1]):
        r = rankdata(prob[:, c]) / n
        rp = r ** RANK_POWER
        rpmin = rp.min()
        rpmax = rp.max()
        if rpmax - rpmin < 1e-9:
            out[:, c] = (target_lo + target_hi) / 2
        else:
            out[:, c] = target_lo + (rp - rpmin) / (rpmax - rpmin) * (target_hi - target_lo)
    return np.clip(out, 0.001, 0.999)


def derive_texture_classes(y_true, threshold=0.20):
    """Texture classes = those that have HIGH frequency across rows (persistent).
    Event classes = sparse, peaky."""
    prevalence = y_true.mean(axis=0)
    return [i for i in range(len(prevalence)) if prevalence[i] >= threshold]


def main():
    samp = pd.read_csv(DATA / "sample_submission.csv")
    class_cols = [c for c in samp.columns if c != "row_id"]
    oof = np.load(CORPUS / "teacher_oof_predictions.npz")
    rows = pd.read_parquet(CORPUS / "teacher_eval_rows.parquet")

    bruce_logit = oof["oof"].astype(np.float32)
    bruce_prob = 1.0 / (1.0 + np.exp(-bruce_logit))
    y_true = oof["y_true"].astype(np.int32)
    row_ids = rows["row_id"].tolist()
    hours = rows["hour_utc"].values

    # Build exp019-like rank-power inputs
    exp_like = to_rank_power_like_exp019(bruce_prob)
    print(f"exp019-like input: min={exp_like.min():.4f}, max={exp_like.max():.4f}")
    base_auc = macro_auc(y_true, exp_like)
    print(f"Baseline macro-AUC (no patches): {base_auc:.4f}")

    # Build texture classes from labels themselves (rough heuristic)
    prevalence = y_true.mean(axis=0)
    tex_idx = np.where(prevalence >= 0.10)[0]
    texture_classes = [class_cols[i] for i in tex_idx]
    print(f"Identified {len(texture_classes)} texture classes (prevalence >= 10%)")

    priors = pp.load_priors_filled(PRIORS_DIR)
    print(f"Loaded priors: hour shape={priors['pseudo_hour'].shape}, "
          f"site_hour shape={priors['pseudo_site_hour'].shape}")

    # Build synthetic test scenario: 20% rows pushed to dead hour 11
    np.random.seed(42)
    mask_dead = np.random.rand(len(hours)) < 0.20
    fake_hours = hours.copy()
    fake_hours[mask_dead] = 11

    # Make synthetic row_ids that have hour=11 for those rows
    new_row_ids = []
    for i, rid in enumerate(row_ids):
        if fake_hours[i] != hours[i]:
            # Rewrite the HHMMSS group in the original row_id with 11xxxx
            new_rid = pp._ROW_RE.sub(lambda m: m.group(0).replace(m.group(3), "110000"), rid)
            new_row_ids.append(new_rid)
        else:
            new_row_ids.append(rid)

    print(f"Synthetic dead-hour rows: {mask_dead.sum()} ({100*mask_dead.mean():.1f}%)")

    # ================================================================
    # INDIVIDUAL benchmarks
    # ================================================================
    print("\n" + "=" * 80)
    print("INDIVIDUAL BENCHMARKS  —  on exp019-like rank-power inputs")
    print("(Scenario A: Bruce OOF original hours — no dead-hour exposure)")
    print("(Scenario B: 20% synthetic dead-hour exposure — LB-like)")
    print("=" * 80)

    configs = [
        ("baseline (no patches)",          {}),
        ("hour_prior w=0.05",              {"w_hour": 0.05}),
        ("hour_prior w=0.1",               {"w_hour": 0.10}),
        ("hour_prior w=0.2",               {"w_hour": 0.20}),
        ("site_hour_prior w=0.05",         {"w_sh": 0.05}),
        ("site_hour_prior w=0.1",          {"w_sh": 0.10}),
        ("site_shrinkage w=0.1",           {"w_site": 0.10}),
        ("site_shrinkage w=0.3",           {"w_site": 0.30}),
        ("calibration only",               {"calib": 1}),
        ("temperature 0.8",                {"temps": np.full(234, 0.8)}),
        ("temperature 1.2",                {"temps": np.full(234, 1.2)}),
        ("two-pass SSM α=0.2",             {"ssm_alpha": 0.2}),
        ("two-pass SSM α=0.4",             {"ssm_alpha": 0.4}),
        ("adaptive_delta",                 {"adaptive_alpha_event": 0.6,
                                            "adaptive_alpha_texture": 0.3,
                                            "texture_classes": texture_classes}),
        ("file_confidence",                {"file_conf": True}),
        ("rank_power 0.5",                 {"rank_power": 0.5}),
        ("sonotype alias",                 {"alias": True}),
        ("howler coupling 0.9",            {"howler": 0.9}),
        ("weeping->chiasmo",               {"weeping": True}),
        ("chorus boost 0.10",              {"chorus": 0.10}),
        ("missing broadcast 0.8",          {"broadcast": 0.8}),
        ("missing floor 0.5",              {"floor": 0.5}),
    ]

    print(f"\n{'Config':<35}  {'A: clean':>10}  {'B: dead20':>10}  Δ_A      Δ_B")
    print("-" * 85)
    base_a = macro_auc(y_true, exp_like)

    # Compute scenario B baseline
    base_b_pred = exp_like.copy()  # baseline isn't affected by row_id changes
    base_b = macro_auc(y_true, base_b_pred)
    print(f"{'baseline':<35}  {base_a:>10.4f}  {base_b:>10.4f}  {0:+.4f}   {0:+.4f}")

    results = []
    for name, params in configs[1:]:
        try:
            pred_a = pp.apply_all(exp_like, row_ids, class_cols, priors, params)
            auc_a = macro_auc(y_true, pred_a)
            pred_b = pp.apply_all(exp_like, new_row_ids, class_cols, priors, params)
            auc_b = macro_auc(y_true, pred_b)
            d_a = auc_a - base_a
            d_b = auc_b - base_b
            results.append((name, params, auc_a, auc_b, d_a, d_b))
            star_a = " ⭐" if d_a > 0.001 else ("  ≈" if abs(d_a) < 0.001 else "")
            star_b = " ⭐" if d_b > 0.001 else ("  ≈" if abs(d_b) < 0.001 else "")
            print(f"{name:<35}  {auc_a:>10.4f}  {auc_b:>10.4f}  {d_a:+.4f}{star_a:<3}  {d_b:+.4f}{star_b}")
        except Exception as e:
            print(f"{name:<35}  ERROR: {e}")

    # ================================================================
    # GREEDY STACKING
    # ================================================================
    print("\n" + "=" * 80)
    print("GREEDY STACKING on scenario B (LB-like dead-hour exposure)")
    print("=" * 80)

    # Sort techniques by their scenario B delta and stack while delta > 0
    results.sort(key=lambda r: r[5], reverse=True)
    print("Ranked by scenario B delta:")
    for r in results[:15]:
        print(f"  {r[0]:<35} dB={r[5]:+.4f}")

    # Greedy: add techniques one by one, keep only if cumulative score improves
    stacked_params = {}
    current_pred = exp_like.copy()
    current_auc = base_b
    print(f"\nStarting greedy stack from baseline (B={base_b:.4f}):")
    for name, params, _, _, d_a, d_b in results:
        if d_b <= 0:
            continue
        merged = {**stacked_params, **params}
        new_pred = pp.apply_all(exp_like, new_row_ids, class_cols, priors, merged)
        new_auc = macro_auc(y_true, new_pred)
        delta = new_auc - current_auc
        verdict = "ADD" if delta > 0 else "skip"
        print(f"  {verdict:<5} {name:<35}  new_auc={new_auc:.4f}  delta={delta:+.4f}")
        if delta > 0:
            stacked_params = merged
            current_auc = new_auc

    print(f"\n=== STACKED RECIPE ===")
    print(f"Final scenario-B macro-AUC: {current_auc:.4f}  (delta {current_auc - base_b:+.4f})")
    print(f"Active params: {stacked_params}")

    # Final sanity: also score on scenario A
    final_pred_a = pp.apply_all(exp_like, row_ids, class_cols, priors, stacked_params)
    final_auc_a = macro_auc(y_true, final_pred_a)
    print(f"Scenario A (no dead hour) with same recipe: {final_auc_a:.4f}  (delta {final_auc_a - base_a:+.4f})")
    return stacked_params, current_auc, final_auc_a


if __name__ == "__main__":
    main()
