from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import _leave_one_out
from features import compute_feature_row
from normalize_predictions import LabelBackbone, NormalizedPrediction


def test_leave_one_out_uses_nearest_known_lb_model():
    df = pd.DataFrame(
        [
            {"model_id": "a", "known_lb": 0.80, "feature": 0.0, "constant": 1.0, "source": "ours"},
            {"model_id": "b", "known_lb": 0.90, "feature": 1.0, "constant": 1.0, "source": "ours"},
            {"model_id": "c", "known_lb": 0.92, "feature": 1.2, "constant": 1.0, "source": "public"},
        ]
    )

    loo = _leave_one_out(df)

    row = loo[loo["model_id"] == "c"].iloc[0]
    assert row["actual_lb"] == 0.92
    assert row["predicted_lb"] == 0.90
    assert abs(row["absolute_error"] - 0.02) < 1e-12
    assert row["nearest_model"] == "b"
    assert row["usable_features"] == 1


def test_leave_one_out_returns_empty_schema_without_enough_known_lb():
    df = pd.DataFrame([{"model_id": "a", "known_lb": 0.80, "feature": 0.0}])

    loo = _leave_one_out(df)

    assert loo.empty
    assert loo.columns.tolist() == [
        "model_id",
        "actual_lb",
        "predicted_lb",
        "absolute_error",
        "nearest_model",
        "usable_features",
    ]


def test_assumed_order_labeled_coverage_counts_as_labeled_rows():
    backbone = LabelBackbone(
        row_ids=pd.Series(["f1_5", "f2_10"]).to_numpy(),
        classes=pd.Series(["a", "b"]).to_numpy(),
        labels=pd.DataFrame([[1, 0], [0, 1]]).to_numpy(dtype="float32"),
        filenames=pd.Series(["f1.ogg", "f2.ogg"]).to_numpy(),
        start_seconds=pd.Series([0, 5]).to_numpy(),
    )
    pred = NormalizedPrediction(
        model_id="assumed",
        row_ids=backbone.row_ids,
        classes=backbone.classes,
        predictions=pd.DataFrame([[0.9, 0.1], [0.2, 0.8]]).to_numpy(dtype="float32"),
        source="public",
        category="public_cache",
        known_lb=0.9,
        coverage="labeled_assumed_order",
        artifact_path="x.npz",
    )

    row = compute_feature_row(pred, backbone, {})

    assert row["coverage_labeled_rows"] == 2
