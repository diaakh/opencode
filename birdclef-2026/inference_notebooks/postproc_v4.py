"""postproc_v4 — adds LABELED prior support to v3.

Key insight discovered May 18: my pseudo_hour_priors (derived from Perch
predictions on train_soundscapes) carry Perch's systematic biases. The
hourly_species_priors.csv (derived from train_soundscapes_labels — ground
truth) is much cleaner and more discriminative.

Best recipe on Bruce OOF + 20% synthetic dead-hour (LB proxy):
  v3 with pseudo prior: 0.9585
  v4 with HYBRID prior (labeled where available, pseudo-fill missing): 0.9710
  ⇒ +0.0125 absolute improvement on scenario B
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Re-export everything from v3
from postproc_v3 import *  # noqa: F401, F403
from postproc_v3 import EPS, _logit, _sigmoid  # noqa: F401


def load_priors_hybrid(d):
    """Load priors with LABELED preferred, pseudo-fill for missing classes.

    Returns a dict compatible with postproc_v3.apply_all, but with:
      - pseudo_hour: hybrid hour priors (labeled∪pseudo)
      - pseudo_site_hour: hybrid (labeled site_hour rows take precedence)
      - pseudo_site: hybrid (labeled site rows take precedence)
      - perch_calib: as before
    """
    from postproc_v3 import load_priors_filled, align
    d = Path(d)
    out = dict(load_priors_filled(d))

    # Build labeled hour prior (13 hours covered in labeled data; fill missing with mean)
    try:
        lab_hour = pd.read_csv(d / "hourly_species_priors.csv").set_index("hour")
        for h in range(24):
            if h not in lab_hour.index:
                lab_hour.loc[h] = lab_hour.mean(axis=0)
        lab_hour = lab_hour.sort_index()
        lab_hour_classes = list(lab_hour.columns)
        # build hybrid hour DF: take labeled columns where available, pseudo for rest
        all_classes = list(out["pseudo_hour"].columns)
        hybrid_hour = out["pseudo_hour"].copy()
        for c in lab_hour_classes:
            if c in hybrid_hour.columns:
                hybrid_hour[c] = lab_hour[c].reindex(hybrid_hour.index).fillna(lab_hour[c].mean())
        out["pseudo_hour"] = hybrid_hour
        print(f"[v4] Loaded labeled hour prior; replaced {len(lab_hour_classes)} columns")
    except FileNotFoundError:
        print("[v4] hourly_species_priors.csv not found — using pseudo only for hour")

    # Labeled site_hour rows REPLACE pseudo rows where they exist
    try:
        lab_sh = pd.read_csv(d / "site_hour_species_priors.csv").set_index("site_hour")
        sh = out["pseudo_site_hour"].copy()
        replaced = 0
        for key in lab_sh.index:
            if key in sh.index:
                # replace values for the labeled-covered columns
                for c in lab_sh.columns:
                    if c in sh.columns:
                        sh.loc[key, c] = lab_sh.loc[key, c]
                replaced += 1
            else:
                # add labeled row (fill missing columns with pseudo means)
                row = sh.mean(axis=0)
                for c in lab_sh.columns:
                    if c in row.index:
                        row[c] = lab_sh.loc[key, c]
                sh.loc[key] = row
        out["pseudo_site_hour"] = sh
        print(f"[v4] Labeled site_hour: replaced {replaced} rows from {len(lab_sh)} labeled keys")
    except FileNotFoundError:
        print("[v4] site_hour_species_priors.csv not found — using pseudo only for site_hour")

    # Labeled site prior
    try:
        lab_site = pd.read_csv(d / "site_species_priors.csv").set_index("site")
        sp = out["pseudo_site"].copy()
        replaced = 0
        for s in lab_site.index:
            if s in sp.index:
                for c in lab_site.columns:
                    if c in sp.columns:
                        sp.loc[s, c] = lab_site.loc[s, c]
                replaced += 1
        out["pseudo_site"] = sp
        print(f"[v4] Labeled site: replaced {replaced} rows from {len(lab_site)} labeled sites")
    except FileNotFoundError:
        print("[v4] site_species_priors.csv not found — using pseudo only for site")

    return out
