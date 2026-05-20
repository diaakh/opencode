from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from features import compute_feature_row
from normalize_predictions import load_label_backbone, load_prediction
from registry import default_registry


LEAVE_ONE_OUT_COLUMNS = [
    "model_id",
    "actual_lb",
    "predicted_lb",
    "absolute_error",
    "nearest_model",
    "usable_features",
]
METADATA_COLUMNS = {"model_id", "source", "category", "coverage", "risk_tier", "known_lb"}


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


def _leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["known_lb"].notna()].copy()
    if len(known) < 2:
        return pd.DataFrame(columns=LEAVE_ONE_OUT_COLUMNS)

    feature_frame = pd.DataFrame(index=known.index)
    for col in known.columns:
        if col in METADATA_COLUMNS:
            continue
        values = pd.to_numeric(known[col], errors="coerce")
        if values.notna().any():
            feature_frame[col] = values

    rows = []
    for heldout_idx, heldout in known.iterrows():
        train = known.drop(index=heldout_idx)
        train_features = feature_frame.loc[train.index]
        heldout_features = feature_frame.loc[heldout_idx]

        finite_train = np.isfinite(train_features).all(axis=0)
        finite_heldout = np.isfinite(heldout_features)
        std = train_features.loc[:, finite_train].std(axis=0, ddof=0)
        usable = finite_train & finite_heldout & (std > 0)
        usable_cols = usable[usable].index.tolist()
        if not usable_cols:
            continue

        train_values = train_features[usable_cols]
        heldout_values = heldout_features[usable_cols]
        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0, ddof=0)
        train_scaled = (train_values - mean) / std
        heldout_scaled = (heldout_values - mean) / std
        distances = np.sqrt(((train_scaled - heldout_scaled) ** 2).sum(axis=1))
        nearest_idx = distances.idxmin()
        predicted_lb = float(known.loc[nearest_idx, "known_lb"])
        actual_lb = float(heldout["known_lb"])
        rows.append(
            {
                "model_id": heldout["model_id"],
                "actual_lb": actual_lb,
                "predicted_lb": predicted_lb,
                "absolute_error": abs(actual_lb - predicted_lb),
                "nearest_model": known.loc[nearest_idx, "model_id"],
                "usable_features": len(usable_cols),
            }
        )

    if not rows:
        return pd.DataFrame(columns=LEAVE_ONE_OUT_COLUMNS)
    return pd.DataFrame(rows, columns=LEAVE_ONE_OUT_COLUMNS)


def _write_report(df: pd.DataFrame, corr: pd.DataFrame, loo: pd.DataFrame, path: Path) -> None:
    known_lb = df[df["known_lb"].notna()]
    known_lb_table = known_lb[
        ["model_id", "category", "known_lb", "labeled_macro_auc", "site_gap", "risk_tier"]
    ].sort_values("known_lb", ascending=False)
    loo_table = loo.sort_values("absolute_error", ascending=False) if not loo.empty else loo

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
        "## Leave-One-Out Validation",
        "",
        _markdown_table(loo_table)
        if not loo_table.empty
        else "Not enough known-LB models or usable numeric features for leave-one-out validation.",
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
    loo = _leave_one_out(df)

    df.to_csv(out_dir / "model_zoo_features.csv", index=False)
    corr.to_csv(out_dir / "model_zoo_feature_correlations.csv", index=False)
    loo.to_csv(out_dir / "model_zoo_leave_one_out.csv", index=False)
    _write_report(df, corr, loo, out_dir / "model_zoo_report.md")


if __name__ == "__main__":
    main()
