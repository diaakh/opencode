"""T1 experiment — proxy-driven rank-blend search over the FULL on-disk OOF pool.

Reuses A4's tested machinery (a4_rankblend.py). Loads the exp019 anchor + labels,
then harvests every 739x234 prediction matrix found in analysis/ + meta_analysis/
as a candidate ensemble member, and runs the site-balanced-proxy greedy search to
see if ANY existing asset gives an orthogonal lift over exp019-alone.

Run:  python3 t1_blend_search.py
"""
from __future__ import annotations
import glob, os
import numpy as np
from pathlib import Path

from a4_rankblend import (
    REPO, infer_sites, macro_auc, site_balanced_auc, local_lb_proxy,
    rank_blend, search_blend_weights,
)

base = np.load(REPO / "analysis/entropy_tta/exp019_aligned.npz", allow_pickle=True)
Y = base["Y"].astype(np.float32)
P_exp = base["P_exp019"].astype(np.float32)
row_ids = [f"{f}_{int(s)}" for f, s in zip(base["row_filename"], base["row_start_sec"])]
sites = infer_sites(row_ids)
N, C = P_exp.shape
print(f"anchor exp019: {P_exp.shape}  pos-classes={int((Y.sum(0)>0).sum())}  sites={len(set(sites))}")

# Harvest every 739x234 prediction matrix as a candidate helper.
helpers = {}
for f in sorted(glob.glob(str(REPO / "analysis/**/*.npz"), recursive=True)) + \
         sorted(glob.glob(str(REPO / "meta_analysis/*.npz"))):
    try:
        d = np.load(f, allow_pickle=True)
    except Exception:
        continue
    for k in d.files:
        arr = d[k]
        if getattr(arr, "shape", None) == (N, C) and np.issubdtype(arr.dtype, np.number):
            if k == "Y":
                continue
            a = arr.astype(np.float32)
            if not np.isfinite(a).all():
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            name = f"{Path(f).stem}:{k}"
            helpers[name] = a

print(f"harvested {len(helpers)} candidate (739x234) helper matrices")

# Rank each helper by its OWN proxy and by marginal lift when 0.85/0.15 rank-blended w/ exp019.
rows = []
base_proxy = local_lb_proxy(Y, P_exp, sites)
base_auc = macro_auc(Y, P_exp)
for nm, M in helpers.items():
    solo = local_lb_proxy(Y, M, sites)
    bl3 = rank_blend([P_exp, M], [0.85, 0.15])
    marg = local_lb_proxy(Y, bl3, sites) - base_proxy
    rows.append((nm, solo, marg, macro_auc(Y, bl3) - base_auc))
rows.sort(key=lambda r: r[2], reverse=True)
print(f"\nexp019 alone: proxy={base_proxy:.5f}  true-macroAUC={base_auc:.5f}")
print("\nTOP 15 helpers by MARGINAL proxy lift in 0.85/0.15 rank-blend:")
print(f"  {'helper':38s} {'solo_proxy':>10s} {'Δproxy':>9s} {'Δtrue':>9s}")
for nm, solo, marg, dtrue in rows[:15]:
    print(f"  {nm:38s} {solo:10.4f} {marg:+9.4f} {dtrue:+9.4f}")

# Full greedy multi-helper search (anchor + best candidates by marginal lift, capped for speed).
TOPK = 25
pool_names = ["exp019"] + [r[0] for r in rows[:TOPK]]
pool_mats = [P_exp] + [helpers[r[0]] for r in rows[:TOPK]]
w, power, bp, best = search_blend_weights(pool_mats, Y, sites, anchor_idx=0)
sel = {pool_names[i]: round(v, 3) for i, v in w.items() if v > 0}
active = [i for i in range(len(pool_mats)) if w[i] > 0]
final = rank_blend([pool_mats[i] for i in active], [w[i] for i in active], power=power)
print(f"\n=== greedy proxy search over anchor + top-{TOPK} candidates ===")
print(f"  proxy: {bp:.5f} -> {best:.5f}   power={power}")
print(f"  selected weights: {sel}")
print(f"  FINAL true macro-AUC = {macro_auc(Y, final):.5f}  (exp019 alone = {base_auc:.5f}, "
      f"Δ = {macro_auc(Y, final)-base_auc:+.5f})")
