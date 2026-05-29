"""T1 — leave-one-site-out (LOSO) validation of orthogonal-helper rank-blends.

The in-sample search (t1_blend_search.py) inflated +0.024 via leaky RAG/kNN helpers.
This script does the HONEST test: for each candidate helper, choose the blend weight on
N-1 sites and score the held-out site, concatenating held-out predictions for one overall
out-of-site macro-AUC. Independent models (BirdMAE, Perch20) should hold up; leaky
retrieval helpers (P_ctx_knn) should COLLAPSE out-of-site. Run: python3 t1_loso_validate.py
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from a4_rankblend import REPO, infer_sites, macro_auc, rank_transform

base = np.load(REPO / "analysis/entropy_tta/exp019_aligned.npz", allow_pickle=True)
Y = base["Y"].astype(np.float32)
P_exp = base["P_exp019"].astype(np.float32)
row_ids = [f"{f}_{int(s)}" for f, s in zip(base["row_filename"], base["row_start_sec"])]
sites = infer_sites(row_ids)
uniq = sorted(set(sites))
N, C = P_exp.shape

def load(path, key):
    d = np.load(REPO / path, allow_pickle=True)
    a = d[key].astype(np.float32)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

# candidates: (label, matrix, expected) — include a leaky control
cands = {
    "birdmae (independent)": load("analysis/creative/birdmae_blend.npz", "P_birdmae"),
    "perch20 (independent)": load("analysis/creative/perch20_blend.npz", "P_perch20"),
    "P_ctx_knn (LEAKY ctrl)": load("analysis/creative/rag_embeddings.npz", "P_ctx_knn"),
    "db_ridge (LEAKY ctrl)": load("analysis/creative/db_ridge_lr_oof.npz", "P_db_ridge"),
}
WGRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

def loso_auc(helper):
    """Concatenate held-out-site predictions under per-fold-selected weight; one overall AUC."""
    oof_pred = np.zeros_like(P_exp)
    Re = rank_transform(P_exp); Rh = rank_transform(helper)
    for s in uniq:
        te = sites == s; tr = ~te
        # pick w on training sites by their macro-AUC
        best_w, best = 0.0, -1
        for w in WGRID:
            bl  = (1 - w) * Re[tr] + w * Rh[tr]
            a = macro_auc(Y[tr], bl) if w > 0 else macro_auc(Y[tr], Re[tr])
            if a > best: best, best_w = a, w
        oof_pred[te] = (1 - best_w) * Re[te] + best_w * Rh[te]
    return macro_auc(Y, oof_pred)

base_loso = macro_auc(Y, rank_transform(P_exp))
print(f"sites={uniq}")
print(f"exp019 alone (rank) overall macro-AUC = {base_loso:.5f}\n")
print(f"  {'helper':28s} {'LOSO-blend AUC':>15s} {'Δ vs exp019':>12s}")
for label, M in cands.items():
    a = loso_auc(M)
    print(f"  {label:28s} {a:15.5f} {a-base_loso:+12.5f}")
print("\n(Independent models that survive LOSO are real; leaky controls should go ~0 or negative.)")
