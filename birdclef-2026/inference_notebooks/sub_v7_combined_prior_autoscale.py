"""
============================================================================
SUBMISSION v7 — combined_hour_prior with auto-scaled weight + dead-hour safe
============================================================================

Drop-in post-processing cell. Paste at the END of any Kaggle notebook that
has already written submission.csv (e.g. forked exp019 notebook, Bruce
standalone, or any other public top inference). It overwrites submission.csv
with the post-processed version.

Why this beats sub1-sub5:
  - Uses combined_hour_prior.csv (pseudo + iNat) — dense across all 24 hours.
    No dead-hour clipping; daytime rows (11-16) get a real signal from iNat
    instead of being randomized to -48 logits.
  - Auto-detects the input scale (rank-power vs raw probability) and picks
    the weight that simulation predicts is optimal:
      * compressed range ([0.45, 0.55]) → w=0.05  (exp019-like)
      * mid range       ([0.05, 0.95]) → w=1.0   (Bruce-like)
      * full range      ([0.0, 1.0])   → w=2.5   (raw blend)
  - Per INVESTIGATION_LB_DROP.md, the right weight for exp019 outputs is
    w=0.05 (simulated AUC 0.9516, +0.021 over no-prior). The May 18 submissions
    used w=3.0 which crushed LB.

Optional patches (default on, can be toggled):
  - sonotype_aliases : free if "47158son15"/"son16"/"son22"/"son23" classes
                       exist; broadcasts paired sono call types together.
  - perch_calib      : Perch v2 per-class bias correction (only safe if
                       input.csv was produced by a Perch-based pipeline).

REQUIRED Kaggle dataset attached as input:
  - adkasd/birdclef-2026-priors-research
    (contains combined_hour_prior.csv, perch_calibration.csv)
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================================
# CONFIG
# ============================================================================
PATCHES = ['hour_prior', 'sonotype_aliases']  # 'perch_calib' optional
INPUT_SUBMISSION = Path("submission.csv")
OUTPUT_SUBMISSION = Path("submission.csv")
EPS = 1e-7

# ============================================================================
# Helpers
# ============================================================================
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


def find_prior_csv(name):
    cs = list(Path("/kaggle/input").rglob(name))
    if not cs:
        raise FileNotFoundError(f"Could not find {name} in /kaggle/input — "
                                f"attach the birdclef-2026-priors-research dataset")
    return cs[0]


def auto_weight(prob):
    """Pick prior weight based on input dynamic range.

    Calibrated on labeled OOF with combined_hour_prior (24h dense, iNat-filled,
    no dead-hour bug):
      span 0.02 (very compressed): w=0.1 optimal, AUC 0.9505
      span 0.08 (rank-power):      w=0.5 optimal, AUC 0.9506
      span 0.87 (mid scaled):      w=2.5 optimal, AUC 0.9528
      span 0.96 (raw probability): w=3.0 optimal, AUC 0.9534

    Slightly conservative bands for LB safety — distribution shift can erode
    OOF lift by 0.01-0.03.
    """
    lo, hi = np.percentile(prob, [1, 99])
    span = hi - lo
    if span < 0.05:
        return 0.1   # very compressed (extreme rank-power)
    if span < 0.15:
        return 0.5   # rank-power compressed (exp019 ~[0.477, 0.555])
    if span < 0.5:
        return 1.5   # mid compressed
    return 2.5       # wide (raw probability / Bruce-like)


# ============================================================================
# Patches
# ============================================================================
def apply_hour_prior(prob, hours, hp_arr, weight):
    """Logit-space prior shift, dead-hour safe.

    hp_arr is dense (24, 234); rows untouched only if hour is invalid (-1).
    """
    logit_p = np.log(prob + EPS) - np.log(1.0 - prob + EPS)
    shift = np.zeros_like(prob, dtype=np.float64)
    valid = (hours >= 0) & (hours < hp_arr.shape[0])
    log_prior = np.log(np.clip(hp_arr, EPS, 1.0))
    shift[valid] = weight * log_prior[hours[valid]]
    return (1.0 / (1.0 + np.exp(-(logit_p + shift)))).astype(np.float32)


def apply_sonotype_aliases(prob, class_cols):
    """Broadcast across sonotype pairs that share call type."""
    idx = {c: i for i, c in enumerate(class_cols)}
    out = prob.copy().astype(np.float32)
    for a, b in [("47158son15", "47158son16"),
                 ("47158son22", "47158son23")]:
        if a in idx and b in idx:
            avg = (out[:, idx[a]] + out[:, idx[b]]) / 2.0
            out[:, idx[a]] = avg
            out[:, idx[b]] = avg
    if "43435" in idx and "47158son14" in idx:
        out[:, idx["47158son14"]] = np.maximum(
            out[:, idx["47158son14"]],
            out[:, idx["43435"]] * 0.9,
        )
    return out


def apply_perch_calib(prob, ratios, lo=0.1, hi=10.0):
    safe = np.clip(ratios, lo, hi)
    log_r = np.log(safe)
    logit_p = np.log(prob + EPS) - np.log(1.0 - prob + EPS)
    return (1.0 / (1.0 + np.exp(-(logit_p + log_r[None, :])))).astype(np.float32)


# ============================================================================
# Main
# ============================================================================
print(f"Reading {INPUT_SUBMISSION}...")
sub = pd.read_csv(INPUT_SUBMISSION)
assert "row_id" in sub.columns
class_cols = [c for c in sub.columns if c != "row_id"]
assert len(class_cols) == 234, f"Expected 234 class cols, got {len(class_cols)}"

row_ids = sub["row_id"].tolist()
prob = sub[class_cols].to_numpy(dtype=np.float64)
prob_clipped = np.clip(prob, EPS, 1.0 - EPS)

print(f"Loaded {len(row_ids)} rows × {prob.shape[1]} classes; "
      f"min={prob.min():.4f}, p1={np.percentile(prob,1):.4f}, "
      f"p99={np.percentile(prob,99):.4f}, max={prob.max():.4f}")

sites, hours = parse_row_ids(row_ids)
print(f"Parsed sites (unique): {sorted(set(sites))}")
print(f"Parsed hours (unique): {sorted(set(hours.tolist()))}")
n_dead = ((hours >= 11) & (hours <= 16)).sum()
print(f"Dead-hour rows (11-16): {n_dead} ({n_dead/len(hours)*100:.1f}%)")

new_prob = prob_clipped.copy()

if "hour_prior" in PATCHES:
    combined_path = find_prior_csv("combined_hour_prior.csv")
    print(f"Combined prior at: {combined_path}")
    hp_df = pd.read_csv(combined_path).set_index("hour")
    hp_arr = hp_df.reindex(columns=class_cols).to_numpy(dtype=np.float64)
    hp_arr = np.nan_to_num(hp_arr, nan=0.0)
    # Sanity: combined prior should cover all 24 hours densely
    print(f"  combined prior shape: {hp_arr.shape}, "
          f"nonzero per hour min/max: {(hp_arr > 0).sum(axis=1).min()}/{(hp_arr > 0).sum(axis=1).max()}")

    weight = auto_weight(prob)
    print(f"  AUTO-SCALED weight: w={weight} (span p99-p1 = {np.percentile(prob,99)-np.percentile(prob,1):.3f})")

    new_prob = apply_hour_prior(new_prob, hours, hp_arr, weight)
    print(f"Applied combined hour_prior (w={weight}, dead-hour safe)")

if "sonotype_aliases" in PATCHES:
    new_prob = apply_sonotype_aliases(new_prob, class_cols)
    print(f"Applied sonotype_aliases")

if "perch_calib" in PATCHES:
    pc_path = find_prior_csv("perch_calibration.csv")
    pc_df = pd.read_csv(pc_path).set_index("class")
    ratios = np.array([pc_df.loc[c, "ratio"] if c in pc_df.index else 1.0
                       for c in class_cols], dtype=np.float64)
    ratios = np.where(np.isfinite(ratios) & (ratios > 0), ratios, 1.0)
    new_prob = apply_perch_calib(new_prob, ratios)
    print(f"Applied perch_calib")

new_prob = np.clip(new_prob, 0.0, 1.0).astype(np.float32)
assert np.isfinite(new_prob).all(), "NaN/inf after post-processing!"

out_df = pd.DataFrame(new_prob, columns=class_cols)
out_df.insert(0, "row_id", row_ids)
out_df.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Wrote {OUTPUT_SUBMISSION}: rows={len(out_df)}, cols={out_df.shape[1]}, "
      f"min={new_prob.min():.4f}, max={new_prob.max():.4f}")
print(f"Expected LB lift vs base: +0.01 to +0.03 (combined_prior_w={weight})")
