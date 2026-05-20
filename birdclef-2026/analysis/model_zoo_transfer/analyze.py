from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from features import compute_feature_row
from normalize_predictions import load_label_backbone, load_prediction
from registry import default_registry


def _markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
    if frame.empty:
        return ""
    table = frame.reset_index() if index else frame.copy()
    columns = [str(col) for col in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = ["" if pd.isna(value) else str(value) for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scored = df[df["known_lb"].notna()].copy()
    if len(scored) < 3:
        return pd.DataFrame(columns=["feature", "n", "pearson", "spearman"])

    for col in scored.columns:
        if col in {"model_id", "source", "category", "coverage", "risk_tier", "known_lb"}:
            continue
        values = pd.to_numeric(scored[col], errors="coerce")
        mask = values.notna()
        if mask.sum() < 3:
            continue
        feature_values = values[mask]
        lb_values = scored.loc[mask, "known_lb"]
        if feature_values.nunique() < 2 or lb_values.nunique() < 2:
            continue
        pear = pearsonr(feature_values, lb_values).statistic
        spear = spearmanr(feature_values, lb_values).correlation
        if not np.isfinite(pear) or not np.isfinite(spear):
            continue
        rows.append({"feature": col, "n": int(mask.sum()), "pearson": pear, "spearman": spear})

    if not rows:
        return pd.DataFrame(columns=["feature", "n", "pearson", "spearman"])
    return pd.DataFrame(rows).sort_values("spearman", ascending=False)


def _write_report(df: pd.DataFrame, corr: pd.DataFrame, path: Path) -> None:
    known_lb = df[df["known_lb"].notna()]
    known_lb_table = known_lb[
        ["model_id", "category", "known_lb", "labeled_macro_auc", "site_gap", "risk_tier"]
    ].sort_values("known_lb", ascending=False)

    lines = [
        "# BirdCLEF Model Zoo Transfer Report",
        "",
        f"Models ingested: {len(df)}",
        f"Models with known LB: {int(df['known_lb'].notna().sum())}",
        "",
        "## Risk Tiers",
        "",
        _markdown_table(df["risk_tier"].value_counts().rename("count").to_frame(), index=True),
        "",
        "## Top LB-Correlated Features",
        "",
        _markdown_table(corr.head(15)) if not corr.empty else "Not enough known-LB models for correlations.",
        "",
        "## Known-LB Models",
        "",
        _markdown_table(known_lb_table),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="birdclef-2026")
    parser.add_argument("--out-dir", default="birdclef-2026/analysis/model_zoo_transfer")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = load_label_backbone(root / "analysis" / "entropy_tta" / "exp019_aligned.npz")
    registry = default_registry(root)
    loaded = [load_prediction(item, backbone) for item in registry]
    predictions = [item for item in loaded if item is not None]

    anchors = {}
    for pred in predictions:
        if pred.model_id == "exp019":
            anchors["exp019"] = pred.predictions

    rows = [compute_feature_row(pred, backbone, anchors) for pred in predictions]
    df = pd.DataFrame(rows)
    corr = _correlations(df)

    df.to_csv(out_dir / "model_zoo_features.csv", index=False)
    corr.to_csv(out_dir / "model_zoo_feature_correlations.csv", index=False)
    _write_report(df, corr, out_dir / "model_zoo_report.md")


if __name__ == "__main__":
    main()
