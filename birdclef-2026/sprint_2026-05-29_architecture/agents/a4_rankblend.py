"""A4 — Rank-blend + LOCAL macro-ROC-AUC evaluation for BirdCLEF+ 2026.

THE METRIC: macro-averaged ROC-AUC, skipping classes with no positives.
It is a *per-class ranking* metric. Only the within-class ordering of each
score column across all row_ids matters. Monotone per-class transforms
(temperature, sigmoid, per-class affine, calibration) DO NOT change it.
Cross-row ranking is everything; per-class calibration is irrelevant.

This module is self-contained and runs on the repo's existing labeled OOF
(analysis/entropy_tta/exp019_aligned.npz + analysis/creative/*.npz), so blend
search is NOT blind. Run:  python3 a4_rankblend.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]  # birdclef-2026/
SITE_RE = re.compile(r"_(S\d{2})_")


# ---------------------------------------------------------------------------
# 1. Per-class rank transform (the AUC-invariant normalization)
# ---------------------------------------------------------------------------
def rank_transform(P: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Per-COLUMN rank in [0,1]. `power` warps the rank distribution.

    Within a single column this is a strictly monotone map, so it leaves that
    column's OWN macro-AUC unchanged. Its ONLY effect is to put every model on
    a common 0..1 scale BEFORE averaging columns across models -> this is what
    makes ensemble averaging well-behaved (a raw 0.99-vs-0.01 model can't drown
    out a 0.6-vs-0.4 model). `power<1` compresses the top, `power>1` sharpens.
    """
    R = np.empty_like(P, dtype=np.float64)
    n = P.shape[0]
    for c in range(P.shape[1]):
        r = rankdata(P[:, c], method="average") / n
        R[:, c] = r ** power if power != 1.0 else r
    return R


# ---------------------------------------------------------------------------
# 2. LOCAL macro-ROC-AUC eval (overall + site-balanced, the LB proxy)
# ---------------------------------------------------------------------------
def macro_auc(Y: np.ndarray, P: np.ndarray) -> float:
    """Macro ROC-AUC over classes with >=1 positive and <n positives."""
    aucs = []
    for c in range(Y.shape[1]):
        y = Y[:, c]
        s = y.sum()
        if s == 0 or s == len(y):
            continue
        col = P[:, c]
        if col.min() == col.max():
            aucs.append(0.5)
            continue
        aucs.append(roc_auc_score(y, col))
    return float(np.mean(aucs)) if aucs else float("nan")


def infer_sites(row_ids) -> np.ndarray:
    out = []
    for r in row_ids:
        m = SITE_RE.search(str(r))
        out.append(m.group(1) if m else "S??")
    return np.array(out, dtype=object)


def site_balanced_auc(Y, P, sites, min_rows=2):
    """Mean of per-SITE macro-AUC. The repo found this (and the overall-minus-
    site GAP) is the part of local eval that best tracks LB transfer: it
    penalizes models that win only on the one giant site (S22) and downweights
    train_audio site contamination. NOTE: calibrated to RANK ORDER, not absolute
    LB (see METRIC_REALITY_CHECK.md — refit RMSE 0.05 across pipelines)."""
    overall = macro_auc(Y, P)
    per = []
    for s in sorted(set(sites)):
        m = sites == s
        if m.sum() < min_rows:
            continue
        a = macro_auc(Y[m], P[m])
        if np.isfinite(a):
            per.append(a)
    site_mean = float(np.mean(per)) if per else float("nan")
    gap = overall - site_mean if np.isfinite(site_mean) else float("nan")
    return overall, site_mean, gap


def local_lb_proxy(Y, P, sites):
    """Single scalar LB proxy: site-balanced macro-AUC penalized by the gap.
    Use the RANK of this scalar across candidates, not its absolute value."""
    overall, site_mean, gap = site_balanced_auc(Y, P, sites)
    if not np.isfinite(site_mean):
        return overall
    return site_mean - 0.5 * max(gap, 0.0)


# ---------------------------------------------------------------------------
# 3. Rank-blend + weight search against the local proxy (no leaderboard)
# ---------------------------------------------------------------------------
def rank_blend(mats, weights, power=1.0):
    """Weighted average of per-class rank transforms. The OUTPUT is itself a
    valid submission (AUC only cares about order, so rank-space is fine)."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    acc = np.zeros_like(mats[0], dtype=np.float64)
    for wi, M in zip(w, mats):
        acc += wi * rank_transform(M, power=power)
    return acc


def prob_blend(mats, weights):
    w = np.asarray(weights, float)
    w = w / w.sum()
    return sum(wi * M for wi, M in zip(w, mats))


def search_blend_weights(mats, Y, sites, anchor_idx=0, grid=None, powers=(0.5, 1.0)):
    """Coordinate search of helper weights vs the local proxy.

    anchor (the trusted high-LB model, e.g. exp019) gets weight 1.0; each helper
    gets weight in `grid`. Greedily add the helper+weight+power that most
    improves the SITE-BALANCED proxy. Conservative: only accept a helper if it
    beats the anchor-alone proxy (mirrors the repo's 'only blend that exceeds
    exp019' finding)."""
    if grid is None:
        grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    n = len(mats)
    helpers = [i for i in range(n) if i != anchor_idx]
    best_w = {i: (1.0 if i == anchor_idx else 0.0) for i in range(n)}
    best_power = 1.0

    def proxy_for(weights, power):
        active = [i for i in range(n) if weights[i] > 0]
        P = rank_blend([mats[i] for i in active], [weights[i] for i in active], power=power)
        return local_lb_proxy(Y, P, sites)

    base = max(proxy_for(best_w, p) for p in powers)
    best = base
    for power in powers:
        for h in helpers:
            for g in grid:
                trial = dict(best_w)
                trial[h] = g
                v = proxy_for(trial, power)
                if v > best + 1e-6:
                    best, best_w[h], best_power = v, g, power
    return best_w, best_power, base, best


# ---------------------------------------------------------------------------
# Demo / self-test on real repo OOF
# ---------------------------------------------------------------------------
def _demo():
    base = np.load(REPO / "analysis/entropy_tta/exp019_aligned.npz", allow_pickle=True)
    Y = base["Y"].astype(np.float32)
    P_exp = base["P_exp019"].astype(np.float32)
    row_ids = [f"{f}_{int(s)}" for f, s in zip(base["row_filename"], base["row_start_sec"])]
    sites = infer_sites(row_ids)

    helpers = {}
    for path, key in [
        ("analysis/creative/best_oof_predictions.npz", "P_blend"),
        ("analysis/creative/mlp_5seed_final_oof.npz", "P_mlp"),
        ("analysis/creative/birdaves_oof.npz", "P_aves_ridge"),
    ]:
        d = np.load(REPO / path, allow_pickle=True)
        if key in d.files and d[key].shape == P_exp.shape:
            helpers[Path(path).stem + ":" + key] = d[key].astype(np.float32)

    names = ["exp019"] + list(helpers)
    mats = [P_exp] + list(helpers.values())

    print("=== invariance check (calibration is IRRELEVANT to macro-AUC) ===")
    a_raw = macro_auc(Y, P_exp)
    a_rank = macro_auc(Y, rank_transform(P_exp))
    a_temp = macro_auc(Y, 1 / (1 + np.exp(-(np.log(P_exp + 1e-9) - np.log(1 - P_exp + 1e-9)) / 2.0)))
    print(f"  raw={a_raw:.5f}  rank-transformed={a_rank:.5f}  temperature(T=2)={a_temp:.5f}")
    print("  (identical -> any per-class monotone map leaves macro-AUC fixed)")

    print("\n=== per-model local macro-AUC + site-balanced proxy ===")
    for nm, M in zip(names, mats):
        o, sm, gap = site_balanced_auc(Y, M, sites)
        print(f"  {nm:34s} overall={o:.4f} site_mean={sm:.4f} gap={gap:+.4f} proxy={local_lb_proxy(Y,M,sites):.4f}")

    print("\n=== rank-blend vs prob-blend (exp019 + best helper, 0.8/0.2) ===")
    if len(mats) >= 2:
        rb = macro_auc(Y, rank_blend([mats[0], mats[1]], [0.8, 0.2]))
        pb = macro_auc(Y, prob_blend([mats[0], mats[1]], [0.8, 0.2]))
        print(f"  rank-blend overall macro-AUC={rb:.5f}   prob-blend={pb:.5f}")

    print("\n=== greedy weight search vs local site-balanced proxy ===")
    w, power, base_proxy, best_proxy = search_blend_weights(mats, Y, sites, anchor_idx=0)
    print(f"  anchor-alone proxy={base_proxy:.5f} -> searched proxy={best_proxy:.5f} (power={power})")
    print("  weights:", {names[i]: round(v, 3) for i, v in w.items() if v > 0})
    active = [i for i in range(len(mats)) if w[i] > 0]
    final = rank_blend([mats[i] for i in active], [w[i] for i in active], power=power)
    print(f"  final blend overall macro-AUC={macro_auc(Y, final):.5f} (exp019 alone={a_raw:.5f})")


if __name__ == "__main__":
    _demo()
