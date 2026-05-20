from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from registry import ModelArtifact
from normalize_predictions import (
    LabelBackbone,
    NormalizedPrediction,
    load_internal_npz,
    load_prediction,
    load_public_cache_csv,
    load_public_perch_npz,
)


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


def test_load_prediction_keeps_public_cache_csv_with_partial_matching_rows(tmp_path):
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

    pred = load_prediction(item, backbone)

    assert pred is not None
    assert pred.row_ids.tolist() == ["f1_5"]
    assert pred.predictions.tolist() == [[0.8, 0.2]]


def test_load_public_perch_npz_aligns_rows_classes_and_converts_logits(tmp_path):
    cache = tmp_path / "perch_arrays.npz"
    np.savez(
        cache,
        scores=np.array([[0.0, 2.0], [-2.0, 4.0]], dtype=np.float32),
        primary_labels=np.array(["b", "a"]),
    )
    meta = tmp_path / "perch_meta.parquet"
    meta.write_bytes(b"")
    import pandas as pd

    pd.DataFrame({"row_id": ["f2_10", "f1_5"]}).to_parquet(meta)
    backbone = LabelBackbone(
        row_ids=np.array(["f1_5", "f2_10"]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 5]),
    )
    item = ModelArtifact("public_perch", "public", "public_cache", cache, None, 0.947, "labeled")

    pred = load_public_perch_npz(item, backbone)

    assert pred.row_ids.tolist() == ["f1_5", "f2_10"]
    assert np.allclose(pred.predictions[0], [1 / (1 + np.exp(-4.0)), 1 / (1 + np.exp(2.0))])
    assert np.allclose(pred.predictions[1], [1 / (1 + np.exp(-2.0)), 0.5])


def test_load_public_full_perch_npz_uses_backbone_order_without_labels(tmp_path):
    cache = tmp_path / "full_perch_arrays.npz"
    np.savez(
        cache,
        scores_full_raw=np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32),
    )
    import pandas as pd

    pd.DataFrame({"row_id": ["f1_5", "f2_10"]}).to_parquet(tmp_path / "full_perch_meta.parquet")
    backbone = LabelBackbone(
        row_ids=np.array(["f1_5", "f2_10"]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 5]),
    )
    item = ModelArtifact("public_full_perch", "public", "public_cache", cache, None, 0.949, "labeled")

    pred = load_public_perch_npz(item, backbone)

    assert np.allclose(pred.predictions, [[0.2, 0.8], [0.4, 0.6]])


def test_load_public_oof_npz_uses_requested_prediction_key(tmp_path):
    cache = tmp_path / "full_oof_meta_features.npz"
    np.savez(
        cache,
        oof_base=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        oof_prior=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
    )
    import pandas as pd

    pd.DataFrame({"row_id": ["f1_5", "f2_10"]}).to_parquet(tmp_path / "full_perch_meta.parquet")
    backbone = LabelBackbone(
        row_ids=np.array(["f1_5", "f2_10"]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 5]),
    )
    item = ModelArtifact("public_oof_prior", "public", "public_cache", cache, "oof_prior", 0.949, "labeled")

    pred = load_public_perch_npz(item, backbone)

    assert np.allclose(pred.predictions, [[0.5, 0.6], [0.7, 0.8]])


def test_load_public_oof_npz_can_use_explicit_meta_template(tmp_path):
    cache = tmp_path / "full_oof_meta_features.npz"
    np.savez(
        cache,
        oof_base=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    )
    import pandas as pd

    template = tmp_path / "template.parquet"
    pd.DataFrame({"row_id": ["f2_10", "f1_5"]}).to_parquet(template)
    backbone = LabelBackbone(
        row_ids=np.array(["f1_5", "f2_10"]),
        classes=np.array(["a", "b"]),
        labels=np.array([[1, 0], [0, 1]], dtype=np.float32),
        filenames=np.array(["f1.ogg", "f2.ogg"]),
        start_seconds=np.array([0, 5]),
    )
    item = ModelArtifact("public_oof_base", "public", "public_cache", cache, "oof_base", 0.922, "labeled")

    pred = load_public_perch_npz(item, backbone, public_meta_template=template)

    assert pred.coverage == "labeled_assumed_order"
    assert pred.row_ids.tolist() == ["f1_5", "f2_10"]
    assert np.allclose(pred.predictions, [[0.3, 0.4], [0.1, 0.2]])
