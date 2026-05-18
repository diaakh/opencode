"""Test if SHARPENING the rank-power signal first (lower temperature on logits)
gives our priors more discriminative power over exp019's signal.
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


def sharpen(prob, temperature):
    """Apply per-class temperature: scale logits by 1/T."""
    logit = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
    return 1.0 / (1.0 + np.exp(-logit / temperature))


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
    print(f"baseline: {base_b:.4f}")

    # ============================================================
    # Sweep: sharpen factor vs hour_w
    # ============================================================
    print("\n" + "=" * 88)
    print(" Sharpen + hour_prior sweep on scenario B")
    print("=" * 88)
    print(f"{'sharpen T':<12} | " + "  ".join(f"hw={hw:.3f}" for hw in [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]))
    print("-" * 110)
    for T in [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05]:
        sharpened = sharpen(exp_like, T)
        row = []
        for hw in [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]:
            params = {"w_hour": hw} if hw > 0 else {}
            params["w_site"] = 0.20
            pred = pp.apply_all(sharpened, new_row_ids, class_cols, priors, params)
            row.append(macro_auc(y_true, pred))
        best = max(row)
        flag = " ⭐" if best > 0.959 else ""
        print(f"T={T:<8.2f} | " + "  ".join(f"{v:.4f}" for v in row) + f"  (best {best:.4f}{flag})")

    # Different rank_power input variants — what if exp019 had different dynamic range?
    print("\n" + "=" * 88)
    print(" Wider-dynamic-range inputs (simulating less-aggressive rank power)")
    print("=" * 88)
    print(f"baseline (lo=0.477, hi=0.555): {base_b:.4f}")
    for lo, hi in [(0.477, 0.555), (0.4, 0.6), (0.3, 0.7), (0.1, 0.9), (0.05, 0.95)]:
        exp_var = to_rank_power(bruce_prob, lo=lo, hi=hi)
        base_var = macro_auc(y_true, exp_var)
        params = {"w_hour": 0.05, "w_site": 0.20}
        pred = pp.apply_all(exp_var, new_row_ids, class_cols, priors, params)
        auc = macro_auc(y_true, pred)
        # tune hour weight for this range
        best = (0, 0)
        for hw in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            p = pp.apply_all(exp_var, new_row_ids, class_cols, priors, {"w_hour": hw, "w_site": 0.2})
            a = macro_auc(y_true, p)
            if a > best[1]:
                best = (hw, a)
        print(f"[{lo:.2f}, {hi:.2f}]: base={base_var:.4f}, w=0.05 patch={auc:.4f}, best hw={best[0]:.3f} -> {best[1]:.4f}")


if __name__ == "__main__":
    main()
