# ============================================================================
# SUB_V2_5_W02_ALIAS
#
# exp019 + hour_prior w=0.2 + sonotype_aliases. More aggressive prior to see if the curve is still rising or starts hurting.
# ============================================================================

# =============================================================================
# BirdCLEF+ 2026 — post-processing v2 (BUG-FIXED for May-18 LB findings)
# =============================================================================
#
# Fixes applied vs v1:
#  1. Dead-hour fallback: hours 11-16 have no pseudo-cache data. v1 clipped
#     to EPS=1e-7 -> applied -48 logit shift -> randomized predictions for
#     ~20% of test files. v2 fills dead hours with the GLOBAL mean prior.
#  2. Default weight reduced: exp019 outputs rank-power values in [0.477, 0.555]
#     with ~0.31 logit dynamic range. v1 used w=3.0 (-> 125x signal domination).
#     v2 default is w=0.05 (matches the per-class signal scale for exp019).
#
# REQUIRED Kaggle dataset attached as input:
#   - adkasd/birdclef-2026-priors-research
#
# Configure PATCHES + HOUR_PRIOR_WEIGHT at the top of the cell to switch
# between variants.
# =============================================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd

PATCHES = ['hour_prior', 'sonotype_aliases']
HOUR_PRIOR_WEIGHT = 0.2
SITE_HOUR_WEIGHT = 0.2
SITE_BLIND_WEIGHT = 0.2

INPUT_SUBMISSION = Path("submission.csv")
OUTPUT_SUBMISSION = Path("submission.csv")

def _find_priors_dir():
    candidates = list(Path("/kaggle/input").rglob("pseudo_hour_priors.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find pseudo_hour_priors.csv anywhere under /kaggle/input — "
            "attach the birdclef-2026-priors-research dataset to the notebook."
        )
    return candidates[0].parent


PRIORS_DIR = _find_priors_dir()
print(f"[v2] Priors found at {PRIORS_DIR}")
EPS = 1e-7

# ---- row_id parser ----------------------------------------------------------
_ROW_RE = re.compile(r"^BC2026_(?:Test|Train)_\d+_(S\d+)_(\d{8})_(\d{6})_(\d+)$")


def parse_row_ids(row_ids):
    sites = np.empty(len(row_ids), dtype=object)
    hours = np.empty(len(row_ids), dtype=np.int32)
    for i, rid in enumerate(row_ids):
        m = _ROW_RE.match(str(rid))
        if not m:
            sites[i], hours[i] = "UNK", -1
            continue
        sites[i] = m.group(1)
        hours[i] = int(m.group(3)[:2])
    return sites, hours


# ---- prior loaders ----------------------------------------------------------
def load_priors_filled(d):
    """Load priors and FILL dead hours with the GLOBAL mean prior."""
    out = {}
    hour_df = pd.read_csv(d / "pseudo_hour_priors.csv").set_index("hour")
    # Identify dead hours (sum across classes == 0)
    hourly_sum = hour_df.sum(axis=1)
    covered = hourly_sum[hourly_sum > 0].index.tolist()
    dead = [h for h in range(24) if h not in covered]
    if dead:
        global_prior = hour_df.loc[covered].mean(axis=0)
        for h in dead:
            if h not in hour_df.index:
                hour_df.loc[h] = global_prior
            else:
                hour_df.loc[h] = global_prior
        hour_df = hour_df.sort_index()
        print(f"[v2] Filled dead hours {dead} with global prior (mean={global_prior.mean():.4f})")
    out["pseudo_hour"] = hour_df

    sh = pd.read_csv(d / "pseudo_site_hour_priors.csv")
    sh.columns = ["site", "hour"] + sh.columns.tolist()[2:]
    sh["site_hour"] = sh["site"].astype(str) + "_h" + sh["hour"].astype(int).astype(str).str.zfill(2)
    sh_indexed = sh.set_index("site_hour").drop(columns=["site", "hour"])
    # Build set of dead site-hour keys; if missing, will fall back to hour prior later
    out["pseudo_site_hour"] = sh_indexed

    sp = pd.read_csv(d / "pseudo_site_priors.csv")
    sp = sp.rename(columns={sp.columns[0]: "site"})
    out["pseudo_site"] = sp.set_index("site")
    out["perch_calib"] = pd.read_csv(d / "perch_calibration.csv").set_index("class")
    return out


def align_class_order(prior_df, class_cols):
    arr = prior_df.reindex(columns=class_cols).to_numpy(dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0)


# ---- patches ----------------------------------------------------------------
def patch_hour_prior(prob, hours, hp_arr, weight):
    """Apply per-hour prior in logit space. Dead hours already filled with global."""
    logit_p = np.log(prob + EPS) - np.log(1.0 - prob + EPS)
    log_prior = np.log(np.clip(hp_arr, EPS, 1.0))
    shift = np.zeros_like(prob, dtype=np.float64)
    valid = (hours >= 0) & (hours < hp_arr.shape[0])
    shift[valid] = weight * log_prior[hours[valid]]
    return (1.0 / (1.0 + np.exp(-(logit_p + shift)))).astype(np.float32)


def patch_site_hour_prior(prob, sites, hours, priors, weight):
    """Apply (site, hour) prior; fall back to filled hour prior when missing."""
    sh_df = priors["pseudo_site_hour"]
    hour_df = priors["pseudo_hour"]
    hour_arr = hour_df.to_numpy(dtype=np.float64)
    logit_p = np.log(prob + EPS) - np.log(1.0 - prob + EPS)
    shift = np.zeros_like(prob, dtype=np.float64)
    sh_keys = set(sh_df.index)
    for i, (s, h) in enumerate(zip(sites, hours)):
        if h < 0 or h >= 24:
            continue
        key = f"{s}_h{h:02d}"
        row = sh_df.loc[key].to_numpy(dtype=np.float64) if key in sh_keys else hour_arr[h]
        shift[i] = weight * np.log(np.clip(row, EPS, 1.0))
    return (1.0 / (1.0 + np.exp(-(logit_p + shift)))).astype(np.float32)


def patch_perch_calib(prob, ratios, lo=0.5, hi=2.0):
    """Clipped per-class calibration. Smaller bounds for rank-power inputs."""
    safe = np.clip(ratios, lo, hi)
    log_r = np.log(safe)
    logit_p = np.log(prob + EPS) - np.log(1.0 - prob + EPS)
    return (1.0 / (1.0 + np.exp(-(logit_p + log_r[None, :])))).astype(np.float32)


def patch_sonotype_aliases(prob, class_cols):
    idx = {c: i for i, c in enumerate(class_cols)}
    out = prob.copy().astype(np.float32)
    for a, b in [("47158son15", "47158son16"), ("47158son22", "47158son23")]:
        if a in idx and b in idx:
            avg = (out[:, idx[a]] + out[:, idx[b]]) / 2.0
            out[:, idx[a]] = avg
            out[:, idx[b]] = avg
    if "43435" in idx and "47158son14" in idx:
        out[:, idx["47158son14"]] = np.maximum(out[:, idx["47158son14"]], out[:, idx["43435"]] * 0.9)
    return out


UNLABELED_SITES = {"S01", "S02", "S04", "S05", "S06", "S07", "S10", "S11", "S12",
                   "S14", "S16", "S17", "S20", "S21"}


def patch_site_blind(prob, sites, fallback_prior, weight):
    out = prob.copy().astype(np.float32)
    mask = np.array([s in UNLABELED_SITES for s in sites])
    if mask.any():
        logit_p = np.log(out[mask] + EPS) - np.log(1.0 - out[mask] + EPS)
        log_prior = np.log(np.clip(fallback_prior[mask], EPS, 1.0))
        out[mask] = (1.0 / (1.0 + np.exp(-(logit_p + weight * log_prior)))).astype(np.float32)
    return out


# ---- main -------------------------------------------------------------------
print(f"[v2] Reading {INPUT_SUBMISSION}...")
sub = pd.read_csv(INPUT_SUBMISSION)
assert "row_id" in sub.columns
class_cols = [c for c in sub.columns if c != "row_id"]
assert len(class_cols) == 234, f"Expected 234 class cols, got {len(class_cols)}"

row_ids = sub["row_id"].tolist()
prob = sub[class_cols].to_numpy(dtype=np.float64)
prob_clipped = np.clip(prob, EPS, 1.0 - EPS)

print(f"[v2] Loaded {len(row_ids)} rows × {prob.shape[1]} classes; "
      f"min={prob.min():.4f}, max={prob.max():.4f}, mean={prob.mean():.4f}")

priors = load_priors_filled(PRIORS_DIR)
sites, hours = parse_row_ids(row_ids)
n_in_dead = ((hours >= 11) & (hours <= 16)).sum()
print(f"[v2] Sites: {len(set(sites))} unique, Hours: {sorted(set(hours.tolist()))}, "
      f"dead-hour rows: {n_in_dead} ({100*n_in_dead/len(hours):.1f}%)")

new_prob = prob_clipped.copy()

if "hour_prior" in PATCHES:
    hp = align_class_order(priors["pseudo_hour"], class_cols)
    new_prob = patch_hour_prior(new_prob, hours, hp, HOUR_PRIOR_WEIGHT)
    print(f"[v2] Applied hour_prior (w={HOUR_PRIOR_WEIGHT}, dead-hour=global)")

if "site_hour_prior" in PATCHES:
    new_prob = patch_site_hour_prior(new_prob, sites, hours, priors, SITE_HOUR_WEIGHT)
    print(f"[v2] Applied site_hour_prior (w={SITE_HOUR_WEIGHT})")

if "perch_calib" in PATCHES:
    ratios = np.array([priors["perch_calib"].loc[c, "ratio"] if c in priors["perch_calib"].index else 1.0
                       for c in class_cols], dtype=np.float64)
    ratios = np.where(np.isfinite(ratios) & (ratios > 0), ratios, 1.0)
    new_prob = patch_perch_calib(new_prob, ratios)
    print(f"[v2] Applied perch_calib (clipped [0.5, 2.0])")

if "sonotype_aliases" in PATCHES:
    new_prob = patch_sonotype_aliases(new_prob, class_cols)
    print(f"[v2] Applied sonotype_aliases")

if "site_blind" in PATCHES:
    site_df = priors["pseudo_site"]
    site_arr = align_class_order(site_df, class_cols)
    site_idx_map = {s: i for i, s in enumerate(site_df.index)}
    fb = np.tile(site_arr.mean(axis=0), (len(row_ids), 1))
    for i, s in enumerate(sites):
        if s in site_idx_map:
            fb[i] = site_arr[site_idx_map[s]]
    new_prob = patch_site_blind(new_prob, sites, fb, SITE_BLIND_WEIGHT)
    print(f"[v2] Applied site_blind (w={SITE_BLIND_WEIGHT})")

new_prob = np.clip(new_prob, 0.0, 1.0).astype(np.float32)
assert np.isfinite(new_prob).all(), "NaN/inf after post-processing!"
assert new_prob.min() >= 0.0 and new_prob.max() <= 1.0

out_df = pd.DataFrame(new_prob, columns=class_cols)
out_df.insert(0, "row_id", row_ids)
out_df.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"[v2] Wrote {OUTPUT_SUBMISSION}: rows={len(out_df)}, cols={out_df.shape[1]}, "
      f"min={new_prob.min():.4f}, max={new_prob.max():.4f}")
