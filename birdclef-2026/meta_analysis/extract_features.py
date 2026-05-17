"""Extract meta-features from each pulled notebook.

For each .ipynb:
  - Title (from filename or first markdown cell)
  - All markdown text (collected)
  - All code text (collected, compressed)
  - Claimed LB scores (parsed from text)
  - Model architectures mentioned
  - Hyperparameters (lambda_prior, power, gain, weights, etc.)
  - Tweaks from the Karnakbayev playbook (A, B, C, D, E, F, G, etc.)
  - References / forks (URLs to other kernels)

Output: features.csv (one row per notebook).
"""
import json, re, os, sys
from pathlib import Path
from collections import Counter
import pandas as pd

CORPUS = Path(__file__).parent
NB_DIR = CORPUS / "notebooks"

# Score patterns — exhaustive
SCORE_PATTERNS = [
    re.compile(r"\bLB[\s\-_=:]*(0\.\d{3,4})", re.I),
    re.compile(r"\bscore[\s\-_=:]*(0\.\d{3,4})", re.I),
    re.compile(r"'LB'[\s:]*'(0\.\d{3,4})'", re.I),
    re.compile(r'"LB"[\s:]*"(0\.\d{3,4})"', re.I),
    re.compile(r"publicScore[\s:=]*(0\.\d{3,4})", re.I),
    re.compile(r"public[_ ]LB[\s:=]*(0\.\d{3,4})", re.I),
    re.compile(r"(0\.9\d{2,3})[\s]*LB", re.I),
    re.compile(r"baseline.*?(0\.9\d{2,3})", re.I),
    # Standalone scores in markdown headers
    re.compile(r"\|\s*(0\.9\d{2,3})\s*\|"),
    re.compile(r"^#+ .*?(0\.9\d{2,3})", re.M),
]

# Features to detect
FEATURE_KW = {
    # Model architectures
    "uses_perch": ["Perch", "perch", "google.*perch"],
    "uses_perch_v2": ["perch[\\s_]*v?2", "perch.*2\\.0"],
    "uses_birdnet": ["BirdNET", "birdnet"],
    "uses_sed": ["SED", "sound[_\\s]event[_\\s]detection"],
    "uses_protossm": ["ProtoSSM", "LightProtoSSM", "Light.*Proto"],
    "uses_residual_ssm": ["ResidualSSM", "residual.*ssm"],
    "uses_selective_ssm": ["SelectiveSSM"],
    "uses_efficientnet": ["EfficientNet", "efficientnet_b\\d", "efficientnetv2"],
    "uses_hgnet": ["HGNet", "hgnetv2"],
    "uses_convnext": ["ConvNeXt", "convnext"],
    "uses_resnet": ["ResNet", "resnet"],
    "uses_passt": ["PaSST", "passt"],
    "uses_beats": ["BEATs", "beats"],
    "uses_mae": ["BirdMAE", "MAE", "masked.*auto"],
    "uses_tucker": ["Tucker", "tucker"],
    "uses_mlp_probes": ["MLPClassifier", "MLPProbes", "mlp.*prob"],
    # Train_audio usage (this is the BIG one)
    "uses_train_audio": ["train_audio"],
    # Augmentations
    "aug_mixup": ["mixup"],
    "aug_cutmix": ["cutmix"],
    "aug_specaugment": ["SpecAugment", "spec.*augment"],
    "aug_time_shift": ["time.?shift", "shift_tta", "temporal_shift"],
    "aug_flip": ["temporal[_\\s]flip", "time_flip"],
    "aug_clipping": ["clipping[_\\s]aug", "random[_\\s]clip"],
    # Loss functions
    "loss_focal": ["focal[_\\s]bce", "focal[_\\s]loss", "FocalLoss"],
    "loss_bce": ["BCEWithLogits", "binary[_\\s]cross"],
    "loss_softauc": ["SoftAUC", "soft.*auc"],
    # SSL / pseudo-labeling
    "ssl_pseudo": ["pseudo.?label", "noisy.?student", "self[_\\s]distill"],
    "ssl_iterative": ["multi[_\\s]iterative", "iterative.*pseudo"],
    # Post-processing tweaks
    "tweak_A_per_class": ["Tweak[_\\s]A", "per.class.*ensemble.*weight"],
    "tweak_B": ["Tweak[_\\s]B"],
    "tweak_C_residual": ["Tweak[_\\s]C", "correction_weight.*grid"],
    "tweak_D_hour_smooth": ["Tweak[_\\s]D", "circular.*gaussian.*hour"],
    "tweak_E_wide_mlp": ["Tweak[_\\s]E", "wider[_\\s]MLP"],
    "tweak_F_flip_tta": ["Tweak[_\\s]F", "temporal[_\\s]flip.*TTA"],
    "tweak_G_birdnet_weight": ["Tweak[_\\s]G", "BirdNET[_\\s]weight", "unmapped.*BirdNET"],
    "sonotype_mirror": ["sonotype[_\\s]mirror"],
    "site_hour_prior": ["build_prior_tables", "apply_prior", "lambda_prior"],
    "rank_aware": ["rank_aware_scaling"],
    "file_confidence": ["file_confidence_scale", "file_level_scaling"],
    "adaptive_delta": ["adaptive_delta_smooth"],
    # Inference framework
    "uses_onnx": ["onnxruntime", "import\\s*onnx", "\\.onnx"],
    "uses_torch": ["import torch", "torch\\.nn"],
    "uses_tf": ["import tensorflow", "tensorflow as tf"],
    "uses_tflite": ["tflite", "TFLITE"],
    # Validation
    "val_groupkfold": ["GroupKFold"],
    "val_stratified": ["StratifiedKFold", "StratifiedGroup"],
    # External data
    "ext_data": ["external[_\\s]data", "additional[_\\s]xc", "xeno.canto.*extra"],
}

NUMERIC_PATTERNS = {
    "lambda_prior": re.compile(r"lambda_prior\s*=\s*(0\.\d+|\d+\.\d+)"),
    "rank_power": re.compile(r"rank_aware_scaling\([^)]*power\s*=\s*(0\.\d+)"),
    "file_conf_power": re.compile(r"file_confidence_scale\([^)]*power\s*=\s*(0\.\d+)"),
    "n_windows": re.compile(r"N_WINDOWS\s*=\s*(\d+)"),
    "window_sec": re.compile(r"WINDOW_SEC\s*=\s*(\d+)"),
    "sr": re.compile(r"SR\s*=\s*(\d+[_\d]*)"),
    "alpha_blend": re.compile(r"alpha_blend\s*=\s*(0\.\d+)"),
    "lr": re.compile(r"\bLR\s*=\s*(\d+e?-?\d*|\d\.\d+e-\d+)"),
    "batch_size": re.compile(r"batch.?size\s*[:=]\s*(\d+)"),
    "epochs": re.compile(r"\bn_epochs?\s*[:=]\s*(\d+)|EPOCHS\s*=\s*(\d+)"),
    "ensemble_w_mapped": re.compile(r"ENSEMBLE_W_PER_CLASS\s*=\s*np\.where\([^,]+,\s*(0\.\d+)"),
}

def parse_notebook(path):
    """Parse a single .ipynb file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
    except Exception as e:
        return {"error": str(e)}
    cells = nb.get("cells", [])
    md_text = []
    code_text = []
    for c in cells:
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "markdown":
            md_text.append(src)
        elif c.get("cell_type") == "code":
            code_text.append(src)
    all_text = "\n".join(md_text + code_text)
    md_combined = "\n".join(md_text)

    # Extract scores
    scores = []
    for pat in SCORE_PATTERNS:
        for m in pat.findall(all_text):
            try:
                s = float(m)
                if 0.5 <= s <= 1.0:
                    scores.append(s)
            except: pass
    scores = sorted(set(scores), reverse=True)
    max_score = max(scores) if scores else None
    n_scores = len(scores)

    # Extract features
    feats = {}
    text_lower = all_text.lower()
    for fname, patterns in FEATURE_KW.items():
        feats[fname] = any(re.search(p, all_text, re.I) for p in patterns)

    # Extract numerics
    numerics = {}
    for nname, pat in NUMERIC_PATTERNS.items():
        m = pat.search(all_text)
        if m:
            v = m.group(1) if m.lastindex == 1 else (m.group(1) or m.group(2))
            numerics[nname] = v

    # Lineage: kaggle.com URLs and "fork" mentions
    kaggle_urls = re.findall(r"https?://(?:www\.)?kaggle\.com/(?:code|c/[^/]+/notebooks|competitions/[^/]+/code)/[^\s\)\"\']+", all_text)
    fork_refs = []
    for url in kaggle_urls:
        m = re.search(r"kaggle\.com/(?:code|competitions/[^/]+/code)/([^/]+/[^/?\#\s]+)", url)
        if m:
            fork_refs.append(m.group(1))

    # First markdown line (often title or description)
    first_md = md_text[0][:300] if md_text else ""

    return {
        "title_md": first_md,
        "n_cells": len(cells),
        "n_code_cells": sum(1 for c in cells if c.get("cell_type") == "code"),
        "n_md_cells": sum(1 for c in cells if c.get("cell_type") == "markdown"),
        "n_md_chars": sum(len(s) for s in md_text),
        "n_code_chars": sum(len(s) for s in code_text),
        "max_claimed_score": max_score,
        "n_scores_mentioned": n_scores,
        "all_scores": "|".join(f"{s:.4f}" for s in scores[:10]),
        "n_fork_refs": len(set(fork_refs)),
        "fork_refs": "|".join(set(fork_refs))[:300],
        **{f"feat__{k}": v for k, v in feats.items()},
        **{f"num__{k}": v for k, v in numerics.items()},
    }

def main():
    out_rows = []
    nb_dirs = sorted(NB_DIR.iterdir())
    print(f"Found {len(nb_dirs)} kernel dirs", flush=True)
    for i, d in enumerate(nb_dirs):
        if not d.is_dir(): continue
        ipynbs = list(d.glob("*.ipynb"))
        if not ipynbs:
            # Maybe a .py?
            pys = list(d.glob("*.py")) + list(d.glob("*.R"))
            if pys:
                # Treat as text
                ipynbs = pys
            else:
                continue
        for nb in ipynbs:
            slug = d.name
            try:
                row = {"slug": slug, "ref": slug.replace("__", "/"), "filename": nb.name}
                if nb.suffix == ".ipynb":
                    row.update(parse_notebook(nb))
                else:
                    txt = nb.read_text(errors="replace")
                    # Bare-bones parse for .py
                    scores = []
                    for pat in SCORE_PATTERNS:
                        scores.extend([float(m) for m in pat.findall(txt) if 0.5 <= float(m) <= 1.0])
                    row.update({
                        "n_code_chars": len(txt),
                        "max_claimed_score": max(scores) if scores else None,
                        "n_scores_mentioned": len(set(scores)),
                    })
                out_rows.append(row)
            except Exception as e:
                print(f"  ERR {slug}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(nb_dirs)}", flush=True)

    df = pd.DataFrame(out_rows)
    df.to_csv(CORPUS / "features.csv", index=False)
    print(f"\nWrote features.csv: {len(df)} rows × {len(df.columns)} cols")
    # Quick stats
    n_with_score = df["max_claimed_score"].notna().sum()
    print(f"Notebooks with a parsable score: {n_with_score}/{len(df)}")
    print(f"Score distribution:")
    print(df["max_claimed_score"].describe())

if __name__ == "__main__":
    main()
