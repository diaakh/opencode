from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import compute_feature_row, infer_sites, infer_hours, macro_auc, risk_tier
from normalize_predictions import LabelBackbone, NormalizedPrediction


def test_infer_sites_and_hours_from_row_ids():
    row_ids = np.array([
        "BC2026_Train_0001_S22_20211231_201500_5",
        "BC2026_Train_0002_S08_20250606_030007_10",
    ])

    assert infer_sites(row_ids).tolist() == ["S22", "S08"]
    assert infer_hours(row_ids).tolist() == [20, 3]


def test_compute_feature_row_has_expected_metrics():
    backbone = LabelBackbone(
        row_ids=np.array([
            "BC2026_Train_0001_S22_20211231_201500_5",
            "BC2026_Train_0001_S22_20211231_201500_10",
            "BC2026_Train_0002_S08_20250606_030007_5",
            "BC2026_Train_0002_S08_20250606_030007_10",
        ]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f1.ogg", "f2.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 5, 0, 5]),
    )
    pred = NormalizedPrediction(
        model_id="good",
        row_ids=backbone.row_ids,
        classes=backbone.classes,
        predictions=np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.8], [0.2, 0.9]], dtype=np.float32),
        source="ours",
        category="single",
        known_lb=0.95,
        coverage="labeled",
        artifact_path="toy.npz",
    )

    row = compute_feature_row(pred, backbone, anchors={})

    assert row["model_id"] == "good"
    assert row["coverage_labeled_rows"] == 4
    assert row["coverage_classes"] == 2
    assert row["labeled_macro_auc"] == 1.0
    assert 0.0 <= row["entropy_mean"] <= 1.0
    assert row["risk_tier"] == "ceiling"


def test_compute_feature_row_aligns_partial_prediction_rows_to_backbone():
    backbone = LabelBackbone(
        row_ids=np.array([
            "BC2026_Train_0001_S22_20211231_201500_5",
            "BC2026_Train_0002_S08_20250606_030007_5",
        ]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 0]),
    )
    pred = NormalizedPrediction(
        model_id="partial",
        row_ids=np.array(["BC2026_Train_0002_S08_20250606_030007_5"]),
        classes=backbone.classes,
        predictions=np.array([[0.2, 0.9]], dtype=np.float32),
        source="public",
        category="public_cache",
        known_lb=0.91,
        coverage="labeled",
        artifact_path="submission.csv",
    )

    row = compute_feature_row(pred, backbone, anchors={})

    assert row["coverage_labeled_rows"] == 1
    assert row["coverage_classes"] == 2
    assert row["n_sites"] == 1
    assert np.isfinite(row["entropy_mean"])


def test_macro_auc_counts_constant_active_class_as_chance():
    labels = np.array([
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
    ], dtype=np.float32)
    predictions = np.array([
        [0.1, 0.5],
        [0.8, 0.5],
        [0.2, 0.5],
        [0.9, 0.5],
    ], dtype=np.float32)

    assert macro_auc(labels, predictions) == 0.75
