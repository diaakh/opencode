# ============================================================================
# SUBMISSION 5 / 5 — exp019 + hour_prior (w=3.0) + perch_calib + sonotype_alias
# 
# Adds per-class Perch calibration ratios from ROUND 23 (Perch under-predicts
# 517063/24321 frogs by 100x+, over-predicts compot1/compau nightjars by 80x+).
# Calibration is neutral on Bruce OOF (chicken-and-egg) but should help on actual
# test data where the per-class biases still apply.
#
# Paste as a new code cell at the END of your exp019 Kaggle notebook fork,
# after submission.csv has been written. Attach the
# birdclef-2026-priors-research dataset.
# ============================================================================

# =============================================================================
# BirdCLEF+ 2026 — post-processing cell (paste at END of exp019 notebook)
# =============================================================================
#
# Reads the submission.csv produced by exp019 (Model_7 Karnakbayev LB 0.948
# baseline), applies a configurable sequence of post-processing patches based
# on local research findings, and overwrites submission.csv.
#
# REQUIRED Kaggle dataset attached as input:
#   - <your-username>/birdclef-2026-priors-research
#     (contains pseudo_hour_priors.csv, perch_calibration.csv,
#      missing_class_strategy.csv, etc.)
#
# Configure PATCHES + HOUR_PRIOR_WEIGHT at the top of the cell to switch
# between variants sub1..sub5.
#
# Local OOF validation (Bruce 739-row labeled set):
#   sub1 hour_prior_w3.0                  0.9586  (+0.028 over Bruce)
#   sub3 hour_prior_w2.0 + alias + blind  0.9583
#   sub4 hour_prior_w3.0 + alias          0.9586
#   sub5 hour_prior_w3.0 + calib + alias  0.9586
# =============================================================================

import re
from pathlib import Path

import numpy as np
import pandas as pd

PATCHES = ['hour_prior', 'perch_calib', 'sonotype_aliases']
HOUR_PRIOR_WEIGHT = 3.0
SITE_HOUR_WEIGHT = 2.5
SITE_BLIND_WEIGHT = 0.2

# ---- CONFIG (override before pasting into cell) -----------------------------

# Submission paths
INPUT_SUBMISSION = Path("submission.csv")  # exp019 wrote it here in /kaggle/working
OUTPUT_SUBMISSION = Path("submission.csv")  # overwrite

# Prior CSV root — auto-discover from /kaggle/input
def _find_priors_dir():
    candidates = list(Path("/kaggle/input").rglob("pseudo_hour_priors.csv"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find pseudo_hour_priors.csv anywhere under /kaggle/input — "
            "attach the birdclef-2026-priors-research dataset to the notebook."
        )
    return candidates[0].parent


PRIORS_DIR = _find_priors_dir()
print(f"Priors found at {PRIORS_DIR}")
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
def load_priors(d):
    out = {}
    out["pseudo_hour"] = pd.read_csv(d / "pseudo_hour_priors.csv").set_index("hour")
    sh = pd.read_csv(d / "pseudo_site_hour_priors.csv")
    sh.columns = ["site", "hour"] + sh.columns.tolist()[2:]
    sh["site_hour"] = sh["site"].astype(str) + "_h" + sh["hour"].astype(int).astype(str).str.zfill(2)
    out["pseudo_site_hour"] = sh.set_index("site_hour").drop(columns=["site", "hour"])
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
    logit_p = np.log(prob + EPS) - np.log(1.0 - prob + EPS)
    log_prior = np.log(np.clip(hp_arr, EPS, 1.0))
    shift = np.zeros_like(prob, dtype=np.float64)
    valid = (hours >= 0) & (hours < hp_arr.shape[0])
    shift[valid] = weight * log_prior[hours[valid]]
    return (1.0 / (1.0 + np.exp(-(logit_p + shift)))).astype(np.float32)


def patch_site_hour_prior(prob, sites, hours, priors, weight):
    sh_df = priors["pseudo_site_hour"]
    hour_df = priors["pseudo_hour"]
    class_cols = list(hour_df.columns)
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


def patch_perch_calib(prob, ratios, lo=0.1, hi=10.0):
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
print(f"Reading {INPUT_SUBMISSION}...")
sub = pd.read_csv(INPUT_SUBMISSION)
assert "row_id" in sub.columns
class_cols = [c for c in sub.columns if c != "row_id"]
assert len(class_cols) == 234, f"Expected 234 class cols, got {len(class_cols)}"

row_ids = sub["row_id"].tolist()
prob = sub[class_cols].to_numpy(dtype=np.float64)
prob_clipped = np.clip(prob, EPS, 1.0 - EPS)

print(f"Loaded {len(row_ids)} rows × {prob.shape[1]} classes; min={prob.min():.4f}, max={prob.max():.4f}")

priors = load_priors(PRIORS_DIR)
sites, hours = parse_row_ids(row_ids)
print(f"Parsed sites (unique): {sorted(set(sites))}")
print(f"Parsed hours (unique): {sorted(set(hours.tolist()))}")

new_prob = prob_clipped.copy()

if "hour_prior" in PATCHES:
    hp = align_class_order(priors["pseudo_hour"], class_cols)
    new_prob = patch_hour_prior(new_prob, hours, hp, HOUR_PRIOR_WEIGHT)
    print(f"Applied hour_prior (w={HOUR_PRIOR_WEIGHT})")

if "site_hour_prior" in PATCHES:
    new_prob = patch_site_hour_prior(new_prob, sites, hours, priors, SITE_HOUR_WEIGHT)
    print(f"Applied site_hour_prior (w={SITE_HOUR_WEIGHT})")

if "perch_calib" in PATCHES:
    ratios = np.array([priors["perch_calib"].loc[c, "ratio"] if c in priors["perch_calib"].index else 1.0
                       for c in class_cols], dtype=np.float64)
    ratios = np.where(np.isfinite(ratios) & (ratios > 0), ratios, 1.0)
    new_prob = patch_perch_calib(new_prob, ratios)
    print(f"Applied perch_calib")

if "sonotype_aliases" in PATCHES:
    new_prob = patch_sonotype_aliases(new_prob, class_cols)
    print(f"Applied sonotype_aliases")

if "site_blind" in PATCHES:
    site_df = priors["pseudo_site"]
    site_arr = align_class_order(site_df, class_cols)
    site_idx_map = {s: i for i, s in enumerate(site_df.index)}
    fb = np.tile(site_arr.mean(axis=0), (len(row_ids), 1))
    for i, s in enumerate(sites):
        if s in site_idx_map:
            fb[i] = site_arr[site_idx_map[s]]
    new_prob = patch_site_blind(new_prob, sites, fb, SITE_BLIND_WEIGHT)
    print(f"Applied site_blind (w={SITE_BLIND_WEIGHT})")

new_prob = np.clip(new_prob, 0.0, 1.0).astype(np.float32)

# Sanity checks
assert np.isfinite(new_prob).all(), "NaN/inf after post-processing!"
assert new_prob.min() >= 0.0 and new_prob.max() <= 1.0

# Write
out_df = pd.DataFrame(new_prob, columns=class_cols)
out_df.insert(0, "row_id", row_ids)
out_df.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Wrote {OUTPUT_SUBMISSION}: rows={len(out_df)}, cols={out_df.shape[1]}, "
      f"min={new_prob.min():.4f}, max={new_prob.max():.4f}")
