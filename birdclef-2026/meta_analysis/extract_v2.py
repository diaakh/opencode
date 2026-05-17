"""Improved feature extractor with score quality grading.

Parse scores with confidence levels:
  HIGH: 'LB': '0.XXX' in config dict (most reliable — these are claimed LBs)
       or "LB X.XXX" in TITLE first line (next most reliable)
  MED:  "LB X.XXX" anywhere in markdown
  LOW:  Bare "0.9XX" in markdown (could be CV, etc.)

Reports:
  title_score      — score in the FIRST markdown line (kernel's own title claim)
  config_lb_scores — list of 'LB': '0.XXX' from config dicts (these are sub-model LB scores)
  all_lb_mentions  — every "LB 0.XXX" pattern
  bare_scores      — every "0.9XX" pattern (weak signal)
"""
import json, re, os, sys
from pathlib import Path
import pandas as pd

CORPUS = Path(__file__).parent
NB_DIR = CORPUS / "notebooks"

# Score-extraction patterns by confidence
PAT_CONFIG_LB = re.compile(r"['\"]LB['\"]\s*:\s*['\"](0\.\d{3,4})['\"]")
PAT_LB_PREFIX = re.compile(r"\bLB[\s\-_=:.]*(0\.\d{3,4})\b", re.I)
PAT_SCORE_PREFIX = re.compile(r"\b(?:public[_ ]?score|score)[\s\-_=:]*(0\.\d{3,4})\b", re.I)
PAT_BARE_SCORE = re.compile(r"\b(0\.9\d{2,3}|0\.8\d{2,3})\b")
PAT_TITLE_SCORE = re.compile(r"(0\.\d{3,4})")

FEATURE_KW = {
    "uses_perch": [r"\bPerch\b", r"\bperch\b"],
    "uses_perch_v2": [r"perch[\s_]*v?2", r"perch.*2\.0"],
    "uses_birdnet": [r"\bBirdNET\b", r"\bbirdnet\b"],
    "uses_sed": [r"\bSED\b"],
    "uses_protossm": [r"ProtoSSM", r"LightProtoSSM"],
    "uses_residual_ssm": [r"ResidualSSM"],
    "uses_efficientnet": [r"EfficientNet", r"efficientnet_b\d", r"efficientnetv2"],
    "uses_hgnet": [r"HGNet", r"hgnetv2"],
    "uses_convnext": [r"ConvNeXt"],
    "uses_resnet": [r"\bResNet\b", r"\bresnet\d"],
    "uses_passt": [r"PaSST"],
    "uses_beats": [r"BEATs\b"],
    "uses_mae": [r"BirdMAE", r"masked.*auto[\s_]?encoder"],
    "uses_tucker": [r"\bTucker\b"],
    "uses_mlp_probes": [r"MLPClassifier", r"MLPProbes", r"mlp.*prob"],
    "uses_train_audio": [r"train_audio"],
    "aug_mixup": [r"mixup"],
    "aug_cutmix": [r"cutmix"],
    "aug_specaugment": [r"SpecAugment", r"spec.*augment"],
    "aug_time_shift": [r"time.?shift", r"shift_tta", r"temporal_shift", r"shifts\s*="],
    "aug_flip": [r"temporal[_\s]flip", r"time_flip"],
    "aug_clipping": [r"clipping[_\s]aug", r"random[_\s]clip", r"saturat[ei].*aug"],
    "aug_gain": [r"random[_\s]gain", r"gain.*aug"],
    "loss_focal": [r"focal[_\s]bce", r"focal[_\s]loss", r"FocalLoss"],
    "loss_bce": [r"BCEWithLogits"],
    "loss_softauc": [r"SoftAUC"],
    "ssl_pseudo": [r"pseudo.?label", r"noisy.?student", r"self[_\s]distill"],
    "ssl_iterative": [r"multi[_\s]iterative", r"iterative.*pseudo"],
    "tweak_A_per_class": [r"Tweak[_\s]A", r"per.class.*ensemble.*weight"],
    "tweak_C_residual": [r"Tweak[_\s]C", r"correction_weight.*grid"],
    "tweak_D_hour_smooth": [r"Tweak[_\s]D", r"circular.*gaussian.*hour"],
    "tweak_E_wide_mlp": [r"Tweak[_\s]E", r"wider[_\s]MLP"],
    "tweak_F_flip_tta": [r"Tweak[_\s]F", r"temporal[_\s]flip.*TTA"],
    "tweak_G_birdnet_weight": [r"Tweak[_\s]G", r"unmapped.*BirdNET"],
    "sonotype_mirror": [r"sonotype[_\s]mirror"],
    "site_hour_prior": [r"build_prior_tables", r"apply_prior", r"lambda_prior"],
    "rank_aware": [r"rank_aware_scaling"],
    "file_confidence": [r"file_confidence_scale"],
    "adaptive_delta": [r"adaptive_delta_smooth"],
    "uses_onnx": [r"onnxruntime", r"\.onnx"],
    "uses_torch": [r"import torch"],
    "uses_tf": [r"import tensorflow", r"tensorflow as tf"],
    "val_groupkfold": [r"GroupKFold"],
    "val_stratified": [r"StratifiedKFold", r"StratifiedGroup"],
    "ext_data": [r"external[_\s]data", r"additional[_\s]xc"],
}

NUMERIC_PATTERNS = {
    "lambda_prior": re.compile(r"lambda_prior\s*=\s*(0\.\d+|\d+\.\d+)"),
    "rank_power": re.compile(r"rank_aware_scaling\([^)]*power\s*=\s*(0\.\d+)"),
    "file_conf_power": re.compile(r"file_confidence_scale\([^)]*power\s*=\s*(0\.\d+)"),
    "alpha_blend": re.compile(r"alpha_blend\s*=\s*(0\.\d+)"),
    "ensemble_w_mapped": re.compile(r"np\.where\(MAPPED_MASK,\s*(0\.\d+),\s*(0\.\d+)"),
    "correction_weight": re.compile(r"correction_weight\s*=\s*(0\.\d+)"),
    "lambda_in_apply_prior": re.compile(r"apply_prior\([^)]*lambda_prior\s*=\s*(0\.\d+)"),
    "n_windows": re.compile(r"N_WINDOWS\s*=\s*(\d+)"),
    "window_sec": re.compile(r"WINDOW_SEC\s*=\s*(\d+)"),
    "n_classes": re.compile(r"N_CLASSES\s*=\s*(\d+)"),
}

def extract_scores(text, first_md=""):
    """Extract scores with confidence levels."""
    out = {}
    # Title score (first md cell — kernel's claimed LB)
    title_text = first_md[:500]
    # Try TITLE patterns first
    title_lb = PAT_LB_PREFIX.findall(title_text)
    title_lb = [float(s) for s in title_lb if 0.5 <= float(s) <= 0.999]
    if title_lb:
        out["title_score"] = max(title_lb)
    else:
        # Maybe just a bare score in first line
        first_line = title_text.split("\n")[0]
        bare = PAT_BARE_SCORE.findall(first_line)
        bare = [float(s) for s in bare if 0.5 <= float(s) <= 0.999]
        if bare:
            out["title_score"] = max(bare)

    # Config 'LB' scores
    config_lbs = [float(s) for s in PAT_CONFIG_LB.findall(text) if 0.5 <= float(s) <= 0.999]
    out["config_lb_max"] = max(config_lbs) if config_lbs else None
    out["config_lb_list"] = "|".join(f"{s:.4f}" for s in sorted(set(config_lbs), reverse=True))
    out["n_config_lb"] = len(set(config_lbs))

    # ALL LB-prefix scores
    all_lb = [float(s) for s in PAT_LB_PREFIX.findall(text) if 0.5 <= float(s) <= 0.999]
    out["lb_prefix_max"] = max(all_lb) if all_lb else None
    out["n_lb_prefix"] = len(set(all_lb))

    # Bare scores in markdown — least reliable
    bare = [float(s) for s in PAT_BARE_SCORE.findall(text) if 0.5 <= float(s) <= 0.999]
    out["bare_score_max"] = max(bare) if bare else None
    out["bare_score_min"] = min(bare) if bare else None
    out["n_bare_scores"] = len(set(bare))

    # BEST estimate: title > config_max > lb_prefix > bare
    best = (out.get("title_score") or
            out.get("config_lb_max") or
            out.get("lb_prefix_max") or
            out.get("bare_score_max"))
    out["best_score"] = best
    return out

def parse_notebook(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
    except Exception as e:
        return {"error": str(e)}
    cells = nb.get("cells", [])
    md_text, code_text = [], []
    for c in cells:
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "markdown":
            md_text.append(src)
        elif c.get("cell_type") == "code":
            code_text.append(src)
    all_text = "\n".join(md_text + code_text)
    first_md = md_text[0] if md_text else ""

    out = {"n_cells": len(cells),
           "n_code_cells": sum(1 for c in cells if c.get("cell_type") == "code"),
           "n_md_cells": len(md_text),
           "n_md_chars": sum(len(s) for s in md_text),
           "n_code_chars": sum(len(s) for s in code_text),
           "first_md": first_md[:400]}

    # Scores
    out.update(extract_scores(all_text, first_md))

    # Feature detection
    for fname, patterns in FEATURE_KW.items():
        out[f"feat__{fname}"] = any(re.search(p, all_text, re.I) for p in patterns)

    # Numerics
    for nname, pat in NUMERIC_PATTERNS.items():
        m = pat.search(all_text)
        if m:
            out[f"num__{nname}"] = m.group(1)

    # Forks
    kaggle_urls = re.findall(r"kaggle\.com/(?:code|competitions/[^/]+/code)/([^/\)\"\'\s\?\#]+/[^/\)\"\'\s\?\#]+)", all_text)
    forks = sorted(set(kaggle_urls))
    out["n_fork_refs"] = len(forks)
    out["fork_refs"] = "|".join(forks[:8])

    return out

def main():
    out_rows = []
    nb_dirs = sorted(NB_DIR.iterdir())
    print(f"Found {len(nb_dirs)} kernel dirs", flush=True)
    for i, d in enumerate(nb_dirs):
        if not d.is_dir(): continue
        ipynbs = list(d.glob("*.ipynb"))
        if not ipynbs:
            pys = list(d.glob("*.py"))
            if pys:
                # treat as text
                slug = d.name
                txt = pys[0].read_text(errors="replace")
                row = {"slug": slug, "ref": slug.replace("__", "/"),
                       "filename": pys[0].name, "n_cells": 0,
                       "n_code_chars": len(txt)}
                row.update(extract_scores(txt))
                out_rows.append(row)
            continue
        for nb in ipynbs:
            slug = d.name
            try:
                row = {"slug": slug, "ref": slug.replace("__", "/"), "filename": nb.name}
                row.update(parse_notebook(nb))
                out_rows.append(row)
            except Exception as e:
                pass
        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(nb_dirs)}", flush=True)

    df = pd.DataFrame(out_rows)
    df.to_csv(CORPUS / "features_v2.csv", index=False)
    print(f"\nWrote features_v2.csv: {len(df)} rows × {len(df.columns)} cols")
    print(f"\nScore extraction quality:")
    print(f"  title_score:  {df['title_score'].notna().sum()}")
    print(f"  config_lb:    {df['config_lb_max'].notna().sum()}")
    print(f"  lb_prefix:    {df['lb_prefix_max'].notna().sum()}")
    print(f"  bare_score:   {df['bare_score_max'].notna().sum()}")
    print(f"  best_score:   {df['best_score'].notna().sum()}")
    print(f"\nbest_score distribution:")
    print(df["best_score"].describe())
    print(f"\nTop 30 by best_score:")
    cols = ["ref", "title_score", "config_lb_max", "lb_prefix_max", "best_score"]
    print(df.dropna(subset=["best_score"]).nlargest(30, "best_score")[cols].to_string(index=False))

if __name__ == "__main__":
    main()
