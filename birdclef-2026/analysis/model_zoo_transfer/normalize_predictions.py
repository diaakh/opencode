from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
