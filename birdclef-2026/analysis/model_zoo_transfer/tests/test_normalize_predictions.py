from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from normalize_predictions import NormalizedPrediction


def test_normalized_prediction_records_coverage_counts():
    pred = NormalizedPrediction(
        model_id="toy",
        row_ids=np.array(["file_5", "file_10"]),
        classes=np.array(["a", "b", "c"]),
        predictions=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        source="ours",
        category="single",
        known_lb=0.9,
        coverage="labeled",
        artifact_path="toy.npz",
    )

    assert pred.n_rows == 2
    assert pred.n_classes == 3
    assert pred.predictions.dtype == np.float32
