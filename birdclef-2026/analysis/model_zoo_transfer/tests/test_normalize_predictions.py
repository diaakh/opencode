from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registry import ModelArtifact
from normalize_predictions import LabelBackbone, NormalizedPrediction, load_internal_npz, load_prediction, load_public_cache_csv


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


def test_load_internal_npz_aligns_to_backbone(tmp_path):
    artifact = tmp_path / "toy.npz"
    np.savez(
        artifact,
        P_model=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    )
    backbone = LabelBackbone(
        row_ids=np.array(["BC2026_Train_0001_S01_20260101_010000_5", "BC2026_Train_0001_S01_20260101_010000_10"]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["BC2026_Train_0001_S01_20260101_010000.ogg", "BC2026_Train_0001_S01_20260101_010000.ogg"]),
        start_seconds=np.array([0, 5]),
    )
    item = ModelArtifact("toy", "ours", "single", artifact, "P_model", None, "labeled")

    pred = load_internal_npz(item, backbone)

    assert pred.model_id == "toy"
    assert pred.row_ids.tolist() == backbone.row_ids.tolist()
    assert pred.classes.tolist() == ["a", "b"]
    assert pred.predictions.shape == (2, 2)


def test_load_public_cache_csv_reindexes_to_backbone(tmp_path):
    cache = tmp_path / "submission.csv"
    cache.write_text("row_id,a,b\nf2_10,0.8,0.2\nf1_5,0.1,0.9\n")
    backbone = LabelBackbone(
        row_ids=np.array(["f1_5", "f2_10"]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 5]),
    )
    item = ModelArtifact("public_toy", "public", "public_cache", cache, None, 0.947, "labeled")

    pred = load_public_cache_csv(item, backbone)

    assert pred.predictions.tolist() == [[0.1, 0.9], [0.8, 0.2]]


def test_load_prediction_skips_public_cache_csv_with_no_matching_rows(tmp_path):
    cache = tmp_path / "submission.csv"
    cache.write_text("row_id,a,b\nother_5,0.8,0.2\n")
    backbone = LabelBackbone(
        row_ids=np.array(["f1_5", "f2_10"]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 5]),
    )
    item = ModelArtifact("public_toy", "public", "public_cache", cache, None, 0.947, "labeled")

    assert load_prediction(item, backbone) is None


def test_load_prediction_skips_public_cache_csv_with_partial_matching_rows(tmp_path):
    cache = tmp_path / "submission.csv"
    cache.write_text("row_id,a,b\nf1_5,0.8,0.2\n")
    backbone = LabelBackbone(
        row_ids=np.array(["f1_5", "f2_10"]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 5]),
    )
    item = ModelArtifact("public_toy", "public", "public_cache", cache, None, 0.947, "labeled")

    assert load_prediction(item, backbone) is None
