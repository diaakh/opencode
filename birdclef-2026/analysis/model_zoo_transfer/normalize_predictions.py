from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

    predictions = aligned.loc[present, backbone.classes].to_numpy(dtype=float)
    return NormalizedPrediction(
        model_id=item.model_id,
        row_ids=backbone.row_ids[present],
        classes=backbone.classes.copy(),
        predictions=predictions,
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
    return None
