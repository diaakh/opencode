# ============================================================================
# exp019 + hour_prior w=0.05 only
#
# Single best individual technique. Tests whether site_shrinkage is additive or whether hour_prior alone captures the signal.
# ============================================================================
import sys
sys.path.insert(0, '/kaggle/input/birdclef-2026-priors-research')
# Inline postproc_v3 module (priors_bundle dataset doesn't include .py)

# Inline postproc_v3 (since we cannot rely on PYTHONPATH on Kaggle)
"""
BirdCLEF 2026 — comprehensive post-processing module v3 (ALL FINDINGS)

Implements every actionable inference-time technique mined from meta_analysis/:
  - calibration (R23)              ← per-class Perch bias correction
  - hour_prior (R25)               ← hour-conditional logit shift (DEAD-HOUR FIXED)
  - site_hour_prior (R22)          ← (site, hour) joint prior
  - site_shrinkage (R28)           ← Bayesian shrinkage toward site prior
  - adaptive_delta (R20)           ← texture/event-aware temporal smoothing
  - two_pass_ssm                   ← forward+backward EMA smoother
  - file_confidence (FINAL)        ← document-level confidence scaling
  - rank_power (FINAL)             ← rank-based per-class power transform
  - sonotype_alias (R20)           ← son15≡son16, son22≡son23 averaging
  - missing_class_broadcast (R24)  ← anchor→target broadcast for missing classes
  - missing_class_floor (R24)      ← hour-conditional minimum floors
  - howler_coupling (R20)          ← 43435 ≡ son14 (Jaccard 1.0)
  - weeping_chiasmo (R20)          ← 326272 → 25073 coupling
  - chorus_boost (R20)             ← Pantanal frog chorus co-occurrence
  - temperature (R9)               ← per-class logit temperature

All patches are STACKABLE (preserve [0,1] range, no NaN/Inf). Each has a
weight or on/off toggle so we can search the optimal combination.
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

EPS = 1e-7

# ----------------------------------------------------------------------------- helpers


_ROW_RE = re.compile(r"^BC2026_(?:Test|Train)_\d+_(S\d+)_(\d{8})_(\d{6})_(\d+)$")


def parse_row_ids(row_ids):
    """Return (sites, hours, file_keys, window_idx_in_file)."""
    n = len(row_ids)
    sites = np.empty(n, dtype=object)
    hours = np.empty(n, dtype=np.int32)
    fkey = np.empty(n, dtype=object)
    wid = np.zeros(n, dtype=np.int32)
    for i, rid in enumerate(row_ids):
        m = _ROW_RE.match(str(rid))
        if not m:
            sites[i], hours[i], fkey[i], wid[i] = "UNK", -1, str(rid).rsplit("_", 1)[0], 0
            continue
        sites[i] = m.group(1)
        hours[i] = int(m.group(3)[:2])
        fkey[i] = str(rid).rsplit("_", 1)[0]
        try:
            wid[i] = int(m.group(4)) // 5 - 1  # 5,10,15,...,60 → 0..11
        except Exception:
            wid[i] = 0
    return sites, hours, fkey, wid


def file_groups(fkey, wid):
    """Return dict: file_key -> sorted list of row indices in window order."""
    groups = defaultdict(list)
    for i, k in enumerate(fkey):
        groups[k].append(i)
    for k in groups:
        groups[k].sort(key=lambda i: wid[i])
    return groups


def _logit(p):
    return np.log(p + EPS) - np.log(1.0 - p + EPS)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ----------------------------------------------------------------------------- prior loading


def load_priors_filled(d):
    """Load all prior CSVs. Dead hours (zero pseudo-cache) filled with global mean."""
    d = Path(d)
    out = {}

    # hour priors with dead-hour fill
    hour_df = pd.read_csv(d / "pseudo_hour_priors.csv").set_index("hour")
    hourly_sum = hour_df.sum(axis=1)
    covered = hourly_sum[hourly_sum > 0].index.tolist()
    dead = [h for h in range(24) if h not in covered]
    if dead:
        global_prior = hour_df.loc[covered].mean(axis=0)
        for h in dead:
            hour_df.loc[h] = global_prior
        hour_df = hour_df.sort_index()
    out["pseudo_hour"] = hour_df

    # site x hour priors
    sh = pd.read_csv(d / "pseudo_site_hour_priors.csv")
    sh.columns = ["site", "hour"] + sh.columns.tolist()[2:]
    sh["site_hour"] = sh["site"].astype(str) + "_h" + sh["hour"].astype(int).astype(str).str.zfill(2)
    out["pseudo_site_hour"] = sh.set_index("site_hour").drop(columns=["site", "hour"])

    # site priors
    sp = pd.read_csv(d / "pseudo_site_priors.csv")
    sp = sp.rename(columns={sp.columns[0]: "site"})
    out["pseudo_site"] = sp.set_index("site")

    # perch calibration
    out["perch_calib"] = pd.read_csv(d / "perch_calibration.csv").set_index("class")

    # missing class strategy
    out["missing"] = pd.read_csv(d / "missing_class_strategy.csv").set_index("class")
    return out


def align(df, class_cols):
    return np.nan_to_num(df.reindex(columns=class_cols).to_numpy(dtype=np.float64), nan=0.0)


# ----------------------------------------------------------------------------- patches


def p_hour_prior(prob, hours, hp_arr, w):
    if w == 0: return prob.astype(np.float32)
    lp = _logit(prob)
    lprior = np.log(np.clip(hp_arr, EPS, 1.0))
    shift = np.zeros_like(prob, dtype=np.float64)
    valid = (hours >= 0) & (hours < hp_arr.shape[0])
    shift[valid] = w * lprior[hours[valid]]
    return _sigmoid(lp + shift).astype(np.float32)


def p_site_hour_prior(prob, sites, hours, priors, w, class_cols):
    if w == 0: return prob.astype(np.float32)
    sh = priors["pseudo_site_hour"]
    hp = align(priors["pseudo_hour"], class_cols)
    lp = _logit(prob)
    shift = np.zeros_like(prob, dtype=np.float64)
    sh_keys = set(sh.index)
    for i, (s, h) in enumerate(zip(sites, hours)):
        if h < 0 or h >= 24: continue
        key = f"{s}_h{h:02d}"
        if key in sh_keys:
            row = sh.loc[key].reindex(class_cols).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            row = hp[h]
        shift[i] = w * np.log(np.clip(row, EPS, 1.0))
    return _sigmoid(lp + shift).astype(np.float32)


def p_site_shrinkage(prob, sites, priors, w, class_cols):
    """Bayesian shrinkage: pred = (pred + w * site_prior) / (1 + w)."""
    if w == 0: return prob.astype(np.float32)
    sp = priors["pseudo_site"]
    sp_arr = align(sp, class_cols)  # (n_sites, 234)
    site_idx = {s: i for i, s in enumerate(sp.index)}
    global_avg = sp_arr.mean(axis=0)
    fb = np.tile(global_avg, (len(sites), 1))
    for i, s in enumerate(sites):
        if s in site_idx:
            fb[i] = sp_arr[site_idx[s]]
    return ((prob + w * fb) / (1.0 + w)).astype(np.float32)


def p_calibration(prob, class_cols, calib_df, lo=0.5, hi=2.0):
    """Per-class Perch bias correction (logit-space shift by log(ratio))."""
    ratios = np.array([calib_df.loc[c, "ratio"] if c in calib_df.index else 1.0
                       for c in class_cols], dtype=np.float64)
    ratios = np.where(np.isfinite(ratios) & (ratios > 0), ratios, 1.0)
    log_r = np.log(np.clip(ratios, lo, hi))
    return _sigmoid(_logit(prob) + log_r[None, :]).astype(np.float32)


def p_temperature(prob, temps):
    """Per-class temperature on logits."""
    lp = _logit(prob)
    return _sigmoid(lp / np.clip(temps[None, :], 0.1, 5.0)).astype(np.float32)


def p_file_confidence(prob, fkey, power_outer=0.4, power_inner=0.2):
    """Document-level confidence scaling: pred ** (file_conf ** inner)."""
    out = prob.copy().astype(np.float32)
    groups = file_groups(fkey, np.zeros(len(fkey), dtype=np.int32))
    for k, idxs in groups.items():
        sl = np.array(idxs)
        file_pred = out[sl]
        file_conf = file_pred.mean() ** power_outer
        out[sl] = file_pred ** np.clip(file_conf ** power_inner, 0.1, 5.0)
    return out


def p_rank_power(prob, power=0.5):
    """Per-class rank-power transform."""
    out = np.zeros_like(prob, dtype=np.float32)
    n = prob.shape[0]
    for c in range(prob.shape[1]):
        r = rankdata(prob[:, c]) / n
        out[:, c] = r ** power
    return out


def p_sonotype_alias(prob, class_cols):
    """Average son15≡son16, son22≡son23. Howler 43435≡son14."""
    idx = {c: i for i, c in enumerate(class_cols)}
    out = prob.copy().astype(np.float32)
    for a, b in [("47158son15", "47158son16"), ("47158son22", "47158son23")]:
        if a in idx and b in idx:
            avg = (out[:, idx[a]] + out[:, idx[b]]) / 2.0
            out[:, idx[a]] = avg
            out[:, idx[b]] = avg
    return out


def p_howler_coupling(prob, class_cols, strength=0.9):
    """43435 (Black Howler) ≡ 47158son14 (Jaccard=1.0 in labeled data)."""
    idx = {c: i for i, c in enumerate(class_cols)}
    out = prob.copy().astype(np.float32)
    if "43435" in idx and "47158son14" in idx:
        # both lift toward the max of pair
        m = np.maximum(out[:, idx["43435"]], out[:, idx["47158son14"]])
        out[:, idx["43435"]] = strength * m + (1 - strength) * out[:, idx["43435"]]
        out[:, idx["47158son14"]] = strength * m + (1 - strength) * out[:, idx["47158son14"]]
    return out


def p_weeping_chiasmo(prob, class_cols, thresh=0.3, broadcast=0.6):
    """326272 (Weeping Frog) → 25073 (Chiasmocleis) co-occurrence."""
    idx = {c: i for i, c in enumerate(class_cols)}
    out = prob.copy().astype(np.float32)
    if "326272" in idx and "25073" in idx:
        weeping = out[:, idx["326272"]]
        out[:, idx["25073"]] = np.maximum(out[:, idx["25073"]],
                                          broadcast * weeping * (weeping > thresh))
    return out


def p_chorus_boost(prob, class_cols, boost=0.10):
    """Pantanal frog chorus: 7 species that co-occur."""
    members = ["65380", "24279", "66971", "517063", "23158", "24321", "555146"]
    idx = {c: i for i, c in enumerate(class_cols)}
    present = [idx[m] for m in members if m in idx]
    if not present:
        return prob.astype(np.float32)
    out = prob.copy().astype(np.float32)
    chorus_mean = out[:, present].mean(axis=1, keepdims=True)
    out[:, present] = np.clip(out[:, present] + boost * chorus_mean, 0.0, 1.0)
    return out


def p_missing_broadcast(prob, class_cols, strength=0.8):
    """ROUND 24 anchor→target broadcasts for the 28 missing classes."""
    idx = {c: i for i, c in enumerate(class_cols)}
    out = prob.copy().astype(np.float32)
    rules = {
        "47158son25": ["47158son15", "47158son16", "47158son02", "47158son06",
                       "47158son10", "47158son14", "47158son21", "47158son22",
                       "47158son23", "47158son04", "47158son17"],
        "47158son11": ["25073", "47158son09", "47158son12", "47158son24"],
        "chacha1": ["47158son08", "47158son19"],
        "47158son03": ["47158son18"],
        "47158son08": ["47158son20"],
        "47158son13": ["47158son05"],
    }
    for anchor, targets in rules.items():
        if anchor not in idx:
            continue
        a = out[:, idx[anchor]]
        for t in targets:
            if t in idx:
                out[:, idx[t]] = np.maximum(out[:, idx[t]], a * strength)
    return out


def p_missing_floor(prob, class_cols, hours, strength=0.5):
    """Hour-conditional minimum floor for the missing-class anchors."""
    idx = {c: i for i, c in enumerate(class_cols)}
    out = prob.copy().astype(np.float32)
    rules = [
        ("517063", range(0, 3), 0.40 * strength),
        ("47158son25", range(3, 5), 0.30 * strength),
        ("47158son13", range(3, 5), 0.30 * strength),
        ("47158son03", range(3, 5), 0.30 * strength),
        ("47158son01", range(3, 5), 0.30 * strength),
    ]
    for cls, hrs, floor in rules:
        if cls not in idx:
            continue
        mask = np.isin(hours, list(hrs))
        if mask.any():
            out[mask, idx[cls]] = np.maximum(out[mask, idx[cls]], floor)
    return out


def p_two_pass_ssm(prob, fkey, wid, alpha=0.3):
    """Forward + backward EMA across windows within each file."""
    if alpha <= 0: return prob.astype(np.float32)
    out = prob.copy().astype(np.float32)
    groups = file_groups(fkey, wid)
    for k, idxs in groups.items():
        sl = np.array(idxs)
        x = out[sl]
        if len(sl) <= 1: continue
        fwd = x.copy()
        for t in range(1, len(sl)):
            fwd[t] = (1 - alpha) * x[t] + alpha * fwd[t - 1]
        bwd = fwd.copy()
        for t in range(len(sl) - 2, -1, -1):
            bwd[t] = (1 - alpha) * fwd[t] + alpha * bwd[t + 1]
        out[sl] = bwd
    return out


def p_adaptive_delta(prob, class_cols, fkey, wid, texture_classes, alpha_event=0.6, alpha_texture=0.3):
    """Per-class temporal smoothing: texture classes get heavy smoothing,
    event classes (sparse, peak-driven) get LIGHT smoothing."""
    idx = {c: i for i, c in enumerate(class_cols)}
    out = prob.copy().astype(np.float32)
    groups = file_groups(fkey, wid)
    tex_mask = np.zeros(len(class_cols), dtype=bool)
    for c in texture_classes:
        if c in idx:
            tex_mask[idx[c]] = True
    for k, idxs in groups.items():
        sl = np.array(idxs)
        if len(sl) < 3: continue
        x = out[sl]
        # 3-tap symmetric kernel
        smoothed_tex = np.zeros_like(x)
        smoothed_tex[0] = (1 - alpha_texture) * x[0] + alpha_texture * x[1]
        smoothed_tex[-1] = (1 - alpha_texture) * x[-1] + alpha_texture * x[-2]
        for t in range(1, len(sl) - 1):
            smoothed_tex[t] = (1 - alpha_texture) * x[t] + alpha_texture * 0.5 * (x[t - 1] + x[t + 1])
        smoothed_evt = np.zeros_like(x)
        smoothed_evt[0] = (1 - alpha_event) * x[0] + alpha_event * x[1]
        smoothed_evt[-1] = (1 - alpha_event) * x[-1] + alpha_event * x[-2]
        for t in range(1, len(sl) - 1):
            smoothed_evt[t] = (1 - alpha_event) * x[t] + alpha_event * 0.5 * (x[t - 1] + x[t + 1])
        new = np.where(tex_mask[None, :], smoothed_tex, smoothed_evt)
        out[sl] = new
    return out


# ----------------------------------------------------------------------------- compose


def apply_all(prob, row_ids, class_cols, priors, params):
    """Apply a configurable stack of patches. params is a dict of weights/strengths.

    Returns final (N, 234) float32 probabilities.
    """
    sites, hours, fkey, wid = parse_row_ids(row_ids)
    p = np.asarray(prob, dtype=np.float64)
    p = np.clip(p, EPS, 1.0 - EPS)

    # 1. calibration
    if params.get("calib", 0) > 0:
        p = p_calibration(p, class_cols, priors["perch_calib"],
                          lo=params.get("calib_lo", 0.5),
                          hi=params.get("calib_hi", 2.0))
    # 2. hour_prior
    if params.get("w_hour", 0) > 0:
        hp = align(priors["pseudo_hour"], class_cols)
        p = p_hour_prior(p, hours, hp, params["w_hour"])
    # 3. site x hour prior
    if params.get("w_sh", 0) > 0:
        p = p_site_hour_prior(p, sites, hours, priors, params["w_sh"], class_cols)
    # 4. site shrinkage
    if params.get("w_site", 0) > 0:
        p = p_site_shrinkage(p, sites, priors, params["w_site"], class_cols)
    # 5. temperature
    if params.get("temps") is not None:
        p = p_temperature(p, params["temps"])
    # 6. two-pass SSM
    if params.get("ssm_alpha", 0) > 0:
        p = p_two_pass_ssm(p, fkey, wid, params["ssm_alpha"])
    # 7. adaptive delta smoothing
    if params.get("adaptive_alpha_event") is not None and params.get("texture_classes"):
        p = p_adaptive_delta(p, class_cols, fkey, wid,
                             params["texture_classes"],
                             alpha_event=params["adaptive_alpha_event"],
                             alpha_texture=params.get("adaptive_alpha_texture", 0.3))
    # 8. file confidence
    if params.get("file_conf", False):
        p = p_file_confidence(p, fkey)
    # 9. rank-power
    if params.get("rank_power", 0) > 0:
        p = p_rank_power(p, params["rank_power"])
    # 10. sonotype alias
    if params.get("alias", False):
        p = p_sonotype_alias(p, class_cols)
    # 11. howler coupling
    if params.get("howler", 0) > 0:
        p = p_howler_coupling(p, class_cols, params["howler"])
    # 12. weeping/chiasmocleis
    if params.get("weeping", False):
        p = p_weeping_chiasmo(p, class_cols)
    # 13. chorus boost
    if params.get("chorus", 0) > 0:
        p = p_chorus_boost(p, class_cols, params["chorus"])
    # 14. missing class broadcast
    if params.get("broadcast", 0) > 0:
        p = p_missing_broadcast(p, class_cols, params["broadcast"])
    # 15. missing class floor
    if params.get("floor", 0) > 0:
        p = p_missing_floor(p, class_cols, hours, params["floor"])

    return np.clip(p, 0.0, 1.0).astype(np.float32)


# ---- Apply to submission.csv ----
from pathlib import Path
import pandas as pd
import numpy as np

priors_candidates = list(Path("/kaggle/input").rglob("pseudo_hour_priors.csv"))
assert priors_candidates, "Attach adkasd/birdclef-2026-priors-research"
PRIORS_DIR = priors_candidates[0].parent
print(f"[v3] Priors: {PRIORS_DIR}")

sub = pd.read_csv("submission.csv")
class_cols = [c for c in sub.columns if c != "row_id"]
assert len(class_cols) == 234
prob = sub[class_cols].to_numpy(dtype=np.float64)
print(f"[v3] Input: rows={len(sub)}, min={prob.min():.4f}, max={prob.max():.4f}")

priors = load_priors_filled(PRIORS_DIR)
params = {'w_hour': 0.05}
print(f"[v3] Applying params: {params}")
new_prob = apply_all(prob, sub["row_id"].tolist(), class_cols, priors, params)
assert np.isfinite(new_prob).all()
assert new_prob.min() >= 0 and new_prob.max() <= 1

out = pd.DataFrame(new_prob, columns=class_cols)
out.insert(0, "row_id", sub["row_id"].values)
out.to_csv("submission.csv", index=False)
print(f"[v3] Wrote submission.csv: rows={len(out)}, min={new_prob.min():.4f}, max={new_prob.max():.4f}")
