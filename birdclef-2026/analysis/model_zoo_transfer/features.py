from __future__ import annotations

import re

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


SITE_RE = re.compile(r"_(S\d{2})_")
HOUR_RE = re.compile(r"_\d{8}_(\d{2})\d{4}_")


def infer_sites(row_ids: np.ndarray) -> np.ndarray:
    sites = []
    for row_id in row_ids:
        match = SITE_RE.search(str(row_id))
        sites.append(match.group(1) if match else "S??")
    return np.array(sites, dtype=object)


def infer_hours(row_ids: np.ndarray) -> np.ndarray:
    hours = []
    for row_id in row_ids:
        match = HOUR_RE.search(str(row_id))
        hours.append(int(match.group(1)) if match else -1)
    return np.array(hours, dtype=int)


def macro_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    aucs = []
    for col in range(labels.shape[1]):
        y = labels[:, col]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        if predictions[:, col].min() == predictions[:, col].max():
            aucs.append(0.5)
            continue
        aucs.append(roc_auc_score(y, predictions[:, col]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _rank_matrix(values: np.ndarray) -> np.ndarray:
    ranked = np.zeros_like(values, dtype=np.float32)
    for col in range(values.shape[1]):
        ranked[:, col] = rankdata(values[:, col], method="average") / max(len(values), 1)
    return ranked


def _mean_rank_agreement(predictions: np.ndarray, anchor: np.ndarray) -> float:
    pred_rank = _rank_matrix(predictions)
    anchor_rank = _rank_matrix(anchor)
    scores = []
    for col in range(predictions.shape[1]):
        corr = spearmanr(pred_rank[:, col], anchor_rank[:, col]).correlation
        if np.isfinite(corr):
            scores.append(corr)
    return float(np.mean(scores)) if scores else float("nan")


def risk_tier(labeled_macro_auc: float, known_lb: float | None, site_gap: float) -> str:
    if known_lb is not None:
        if known_lb < 0.80:
            return "catastrophic"
        if known_lb < 0.93:
            return "risky"
        if known_lb < 0.948:
            return "safe"
        return "ceiling"
    if not np.isfinite(labeled_macro_auc):
        return "risky"
    if site_gap > 0.08:
        return "risky"
    if labeled_macro_auc >= 0.96:
        return "safe"
    return "risky"


def _aligned_labels(pred, backbone) -> np.ndarray:
    if pred.predictions.ndim != 2:
        raise ValueError(f"{pred.model_id} predictions must be a 2D matrix")
    if pred.row_ids.shape[0] != pred.predictions.shape[0]:
        raise ValueError(
            f"{pred.model_id} has {pred.row_ids.shape[0]} row_ids but "
            f"{pred.predictions.shape[0]} prediction rows"
        )
    if len(set(pred.row_ids.tolist())) != pred.row_ids.shape[0]:
        raise ValueError(f"{pred.model_id} has duplicate row_ids")
    if pred.classes.shape[0] != pred.predictions.shape[1]:
        raise ValueError(
            f"{pred.model_id} has {pred.classes.shape[0]} classes but "
            f"{pred.predictions.shape[1]} prediction columns"
        )
    if not np.array_equal(pred.classes, backbone.classes):
        raise ValueError(f"{pred.model_id} classes do not align with backbone classes")

    backbone_index = {row_id: idx for idx, row_id in enumerate(backbone.row_ids.tolist())}
    if len(backbone_index) != backbone.row_ids.shape[0]:
        raise ValueError("backbone has duplicate row_ids")

    indices = []
    for row_id in pred.row_ids.tolist():
        if row_id not in backbone_index:
            raise ValueError(f"{pred.model_id} row_id {row_id} is not in backbone row_ids")
        indices.append(backbone_index[row_id])
    labels = backbone.labels[np.array(indices, dtype=int)]
    if labels.shape != pred.predictions.shape:
        raise ValueError(
            f"{pred.model_id} prediction shape {pred.predictions.shape} does not match "
            f"aligned label shape {labels.shape}"
        )
    return labels


def compute_feature_row(pred, backbone, anchors: dict[str, np.ndarray]) -> dict[str, float | str | int | None]:
    labels = _aligned_labels(pred, backbone)
    predictions = np.clip(pred.predictions.astype(np.float32), 1e-6, 1 - 1e-6)
    sites = infer_sites(pred.row_ids)
    hours = infer_hours(pred.row_ids)

    overall = macro_auc(labels, predictions)
    site_aucs = []
    for site in sorted(set(sites)):
        mask = sites == site
        if mask.sum() < 2:
            continue
        score = macro_auc(labels[mask], predictions[mask])
        if np.isfinite(score):
            site_aucs.append(score)
    site_mean = float(np.mean(site_aucs)) if site_aucs else float("nan")
    site_gap = float(overall - site_mean) if np.isfinite(overall) and np.isfinite(site_mean) else float("nan")

    entropy = -(predictions * np.log2(predictions) + (1 - predictions) * np.log2(1 - predictions))
    row = {
        "model_id": pred.model_id,
        "source": pred.source,
        "category": pred.category,
        "known_lb": pred.known_lb,
        "coverage": pred.coverage,
        "coverage_labeled_rows": pred.n_rows if pred.coverage.startswith("labeled") or pred.coverage == "both" else 0,
        "coverage_unlabeled_rows": pred.n_rows if pred.coverage == "unlabeled" else 0,
        "coverage_classes": pred.n_classes,
        "n_sites": int(len(set(sites))),
        "n_hours": int(len(set(hours.tolist()) - {-1})),
        "labeled_macro_auc": overall,
        "site_mean_auc": site_mean,
        "site_gap": site_gap,
        "entropy_mean": float(np.mean(entropy)),
        "entropy_p90": float(np.quantile(entropy, 0.90)),
        "confidence_rate_gt_0_9": float((predictions > 0.9).mean()),
        "probability_median": float(np.median(predictions)),
        "probability_p99": float(np.quantile(predictions, 0.99)),
    }
    for anchor_name, anchor_predictions in anchors.items():
        if anchor_predictions.shape == predictions.shape:
            row[f"agreement_{anchor_name}"] = _mean_rank_agreement(predictions, anchor_predictions)
    row["risk_tier"] = risk_tier(overall, pred.known_lb, site_gap if np.isfinite(site_gap) else 0.0)
    return row
