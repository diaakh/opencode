from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelBackbone:
    row_ids: np.ndarray
    classes: np.ndarray
    labels: np.ndarray
    filenames: np.ndarray
    start_seconds: np.ndarray


@dataclass(frozen=True)
class NormalizedPrediction:
    model_id: str
    row_ids: np.ndarray
    classes: np.ndarray
    predictions: np.ndarray
    source: str
    category: str
    known_lb: float | None
    coverage: str
    artifact_path: str

    @property
    def n_rows(self) -> int:
        return int(self.predictions.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.predictions.shape[1])

    @property
    def path(self) -> Path:
        return Path(self.artifact_path)


def make_row_ids(filenames: np.ndarray, start_seconds: np.ndarray) -> np.ndarray:
    values = []
    for filename, start in zip(filenames, start_seconds):
        stem = str(filename).replace(".ogg", "")
        values.append(f"{stem}_{int(start) + 5}")
    return np.array(values, dtype=object)


def load_label_backbone(path: Path) -> LabelBackbone:
    data = np.load(path, allow_pickle=True)
    filenames = data["row_filename"]
    start_seconds = data["row_start_sec"].astype(int)
    return LabelBackbone(
        row_ids=make_row_ids(filenames, start_seconds),
        classes=data["classes"].astype(str),
        labels=data["Y"].astype(np.float32),
        filenames=filenames,
        start_seconds=start_seconds,
    )


def load_internal_npz(item, backbone: LabelBackbone) -> NormalizedPrediction:
    if item.prediction_key is None:
        raise ValueError(f"{item.model_id} has no prediction_key")
    data = np.load(item.artifact_path, allow_pickle=True)
    if item.prediction_key not in data.files:
        raise KeyError(f"{item.prediction_key} not found in {item.artifact_path}")
    predictions = data[item.prediction_key].astype(np.float32)
    if predictions.shape != backbone.labels.shape:
        raise ValueError(
            f"{item.model_id} shape {predictions.shape} does not match backbone {backbone.labels.shape}"
        )
    return NormalizedPrediction(
        model_id=item.model_id,
        row_ids=backbone.row_ids.copy(),
        classes=backbone.classes.copy(),
        predictions=predictions,
        source=item.source,
        category=item.category,
        known_lb=item.known_lb,
        coverage=item.coverage,
        artifact_path=str(item.artifact_path),
    )


def load_public_cache_csv(item, backbone: LabelBackbone) -> NormalizedPrediction:
    df = pd.read_csv(item.artifact_path)
    if "row_id" not in df.columns:
        raise ValueError(f"{item.artifact_path} has no row_id column")
    missing = [cls for cls in backbone.classes if cls not in df.columns]
    if missing:
        raise ValueError(f"{item.model_id} missing class columns: {missing[:5]}")

    aligned = df.set_index("row_id").reindex(backbone.row_ids)
    present = aligned[backbone.classes].notna().all(axis=1).to_numpy()
    if not present.any():
        raise ValueError(f"{item.model_id} has no rows matching label backbone")

    matched_row_ids = backbone.row_ids[present]
    predictions = aligned.loc[present, backbone.classes].to_numpy(dtype=float)
    return NormalizedPrediction(
        model_id=item.model_id,
        row_ids=matched_row_ids.copy(),
        classes=backbone.classes.copy(),
        predictions=predictions,
        source=item.source,
        category=item.category,
        known_lb=item.known_lb,
        coverage=item.coverage,
        artifact_path=str(item.artifact_path),
    )


def _sigmoid_if_logits(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        clipped = np.clip(values, -50.0, 50.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)
    return values.astype(np.float32)


def _paired_meta_path(score_path: Path) -> Path | None:
    candidates = []
    name = score_path.name
    if name == "perch_arrays.npz":
        candidates.append(score_path.with_name("perch_meta.parquet"))
    elif name == "full_perch_arrays.npz":
        candidates.append(score_path.with_name("full_perch_meta.parquet"))
    elif name == "full_oof_meta_features.npz":
        candidates.append(score_path.with_name("full_perch_meta.parquet"))
    elif name.endswith("_arrays.npz"):
        candidates.append(score_path.with_name(re.sub(r"_arrays\.npz$", "_meta.parquet", name)))
    elif name.endswith("_arrays.local.npz"):
        candidates.append(score_path.with_name(re.sub(r"_arrays\.local\.npz$", "_meta.local.parquet", name)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _public_score_array(data: np.lib.npyio.NpzFile, prediction_key: str | None) -> np.ndarray:
    if prediction_key is not None:
        if prediction_key not in data.files:
            raise ValueError(f"cache does not contain {prediction_key}")
        return data[prediction_key]
    for key in ["scores", "scores_full_raw", "oof_base"]:
        if key in data.files:
            return data[key]
    raise ValueError("cache does not contain a supported score array")


def load_public_perch_npz(item, backbone: LabelBackbone) -> NormalizedPrediction:
    meta_path = _paired_meta_path(item.artifact_path)
    if meta_path is None:
        raise ValueError(f"{item.artifact_path} has no paired metadata parquet")

    data = np.load(item.artifact_path, allow_pickle=True)
    scores = _public_score_array(data, item.prediction_key)
    if "primary_labels" in data.files:
        labels = data["primary_labels"].astype(str)
        class_to_idx = {label: idx for idx, label in enumerate(labels.tolist())}
        missing = [cls for cls in backbone.classes if cls not in class_to_idx]
        if missing:
            raise ValueError(f"{item.model_id} missing class labels: {missing[:5]}")
        class_idx = [class_to_idx[cls] for cls in backbone.classes]
    elif scores.shape[1] == len(backbone.classes):
        class_idx = list(range(len(backbone.classes)))
    else:
        raise ValueError(f"{item.model_id} has no labels and {scores.shape[1]} columns")

    meta = pd.read_parquet(meta_path)
    if "row_id" not in meta.columns:
        raise ValueError(f"{meta_path} has no row_id column")
    if len(meta) != scores.shape[0]:
        raise ValueError(f"{item.model_id} metadata rows do not match score rows")

    df = pd.DataFrame(_sigmoid_if_logits(scores[:, class_idx]), index=meta["row_id"].astype(str))
    aligned = df.reindex(backbone.row_ids)
    present = aligned.notna().all(axis=1).to_numpy()
    if not present.any():
        raise ValueError(f"{item.model_id} has no rows matching label backbone")

    return NormalizedPrediction(
        model_id=item.model_id,
        row_ids=backbone.row_ids[present].copy(),
        classes=backbone.classes.copy(),
        predictions=aligned.loc[present].to_numpy(dtype=np.float32),
        source=item.source,
        category=item.category,
        known_lb=item.known_lb,
        coverage=item.coverage,
        artifact_path=str(item.artifact_path),
    )


def load_prediction(item, backbone: LabelBackbone) -> NormalizedPrediction | None:
    if item.source == "ours":
        return load_internal_npz(item, backbone)
    if item.artifact_path.suffix == ".csv":
        try:
            return load_public_cache_csv(item, backbone)
        except ValueError:
            return None
    if item.artifact_path.suffix == ".npz":
        try:
            return load_public_perch_npz(item, backbone)
        except ValueError:
            return None
    return None
