"""2D sweep: (hour_prior weight, site_shrinkage weight) on scenario B + best blend search."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

import postproc_v3 as pp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPUS = ROOT / "meta_corpus" / "datasets"
PRIORS_DIR = ROOT / "inference_notebooks" / "priors_bundle"


def macro_auc(y, score):
    aucs = [roc_auc_score(y[:, c], score[:, c]) for c in range(y.shape[1]) if y[:, c].sum() > 0]
    return float(np.mean(aucs))


def to_rank_power(prob, rp=0.6, lo=0.477, hi=0.555):
    n = prob.shape[0]
    out = np.zeros_like(prob, dtype=np.float32)
    for c in range(prob.shape[1]):
        r = rankdata(prob[:, c]) / n
        rp_v = r ** rp
        out[:, c] = lo + (rp_v - rp_v.min()) / max(rp_v.max() - rp_v.min(), 1e-9) * (hi - lo)
    return np.clip(out, 0.001, 0.999)


def main():
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

    # Build dead-hour synthetic
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

    # ============================================================
    # 2D sweep
    # ============================================================
    print("=" * 90)
    print(" 2D sweep on Scenario B (LB-like)")
    print("=" * 90)
    hour_ws = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
    site_ws = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    best = (None, None, 0.0)
    grid = np.zeros((len(hour_ws), len(site_ws)))
    for i, hw in enumerate(hour_ws):
        for j, sw in enumerate(site_ws):
            params = {}
            if hw > 0: params["w_hour"] = hw
            if sw > 0: params["w_site"] = sw
            pred = pp.apply_all(exp_like, new_row_ids, class_cols, priors, params)
            auc = macro_auc(y_true, pred)
            grid[i, j] = auc
            if auc > best[2]:
                best = (hw, sw, auc)

    print("                site_shrinkage:")
    print(f"           {'':>4}  " + "  ".join(f"{sw:>7.2f}" for sw in site_ws))
    print(f"           {'baseline':<10}  {base_b:>7.4f}")
    for i, hw in enumerate(hour_ws):
        row = "  ".join(f"{grid[i, j]:>7.4f}" for j in range(len(site_ws)))
        marker = " ⬅ best row" if i == np.argmax(grid.max(axis=1)) else ""
        print(f"hour_w={hw:<5.3f}  {row}{marker}")
    print(f"\nBest 2D: hour_w={best[0]}, site_w={best[1]}, AUC={best[2]:.4f}  (delta {best[2]-base_b:+.4f})")

    # ============================================================
    # Add additional patches on top of best (hw, sw)
    # ============================================================
    print("\n" + "=" * 90)
    print(" Adding patches on top of best 2D point")
    print("=" * 90)
    base_params = {"w_hour": best[0], "w_site": best[1]}
    p0 = pp.apply_all(exp_like, new_row_ids, class_cols, priors, base_params)
    auc0 = macro_auc(y_true, p0)
    print(f"Starting from hour_w={best[0]}, site_w={best[1]}: AUC={auc0:.4f}")

    extras = [
        ("+ssm α=0.1",          {"ssm_alpha": 0.1}),
        ("+ssm α=0.2",          {"ssm_alpha": 0.2}),
        ("+ssm α=0.3",          {"ssm_alpha": 0.3}),
        ("+alias",              {"alias": True}),
        ("+howler 0.9",         {"howler": 0.9}),
        ("+chorus 0.05",        {"chorus": 0.05}),
        ("+chorus 0.10",        {"chorus": 0.10}),
        ("+broadcast 0.5",      {"broadcast": 0.5}),
        ("+broadcast 0.8",      {"broadcast": 0.8}),
        ("+floor 0.3",          {"floor": 0.3}),
        ("+floor 0.5",          {"floor": 0.5}),
        ("+calib only",         {"calib": 1, "calib_lo": 0.7, "calib_hi": 1.43}),
    ]
    keep = dict(base_params)
    for name, p in extras:
        merged = {**keep, **p}
        pred = pp.apply_all(exp_like, new_row_ids, class_cols, priors, merged)
        auc = macro_auc(y_true, pred)
        delta = auc - auc0
        verdict = "KEEP" if delta > 0.0001 else "skip"
        if delta > 0.0001:
            keep.update(p)
            auc0 = auc
        print(f"  {verdict:<5} {name:<25}  AUC={auc:.4f}  cumulative={auc0:.4f}  delta_step={delta:+.5f}")

    print(f"\n=== FINAL STACKED RECIPE ===")
    print(f"Scenario B AUC: {auc0:.4f}  (delta {auc0 - base_b:+.4f} over baseline)")
    # Scenario A
    final_a = pp.apply_all(exp_like, row_ids, class_cols, priors, keep)
    auc_a = macro_auc(y_true, final_a)
    base_a = macro_auc(y_true, exp_like)
    print(f"Scenario A AUC: {auc_a:.4f}  (delta {auc_a - base_a:+.4f})")
    print(f"Active params: {keep}")

    # ============================================================
    # Linear blend with control
    # ============================================================
    print("\n" + "=" * 90)
    print(" Linear blend with control (no-patch) at scenario B")
    print("=" * 90)
    patched_b = pp.apply_all(exp_like, new_row_ids, class_cols, priors, keep)
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        blend = alpha * patched_b + (1 - alpha) * exp_like
        auc = macro_auc(y_true, blend)
        print(f"  α={alpha:.1f}: AUC={auc:.4f}  (alpha=1.0 is full patch, 0.0 is control)")

    return keep, auc0, auc_a


if __name__ == "__main__":
    main()
