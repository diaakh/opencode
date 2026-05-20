from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g124_assets import (
    RankBlendConfig,
    apply_rank_blend,
    find_asset_dir,
    list_fixed_windows,
    validate_submission_frame,
)
from infer import build_parser
from package_assets import build_dataset_metadata
from train_g124 import build_parser as build_train_parser
from train_g124 import parse_soundscape_row_id


def test_find_asset_dir_requires_infer_and_checkpoint(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "g124_fold1_fp16.pt").write_bytes(b"checkpoint")
    good = tmp_path / "good"
    good.mkdir()
    (good / "infer.py").write_text("print('ok')\n")
    (good / "g124_fold1_fp16.pt").write_bytes(b"checkpoint")

    found = find_asset_dir([bad, good], ["g124_fold1_fp16.pt"])

    assert found == good


def test_list_fixed_windows_creates_competition_row_ids(tmp_path):
    input_dir = tmp_path / "train_soundscapes"
    input_dir.mkdir()
    (input_dir / "BC2026_Train_0001_S08_20250606_030007.ogg").write_bytes(b"")
    (input_dir / "BC2026_Train_0002_S09_20250607_040000.wav").write_bytes(b"")

    rows = list_fixed_windows(input_dir, window_seconds=5.0, assume_duration=60.0, limit_files=1)

    assert len(rows) == 12
    assert rows[0].row_id == "BC2026_Train_0001_S08_20250606_030007_5"
    assert rows[-1].row_id == "BC2026_Train_0001_S08_20250606_030007_60"
    assert rows[3].start_seconds == 15.0
    assert rows[3].end_seconds == 20.0


def test_validate_submission_frame_reorders_and_rejects_bad_values():
    df = pd.DataFrame(
        {
            "row_id": ["r2", "r1"],
            "b": [0.8, 0.1],
            "a": [0.2, 0.9],
        }
    )

    out = validate_submission_frame(df, ["a", "b"], name="toy")

    assert out.columns.tolist() == ["row_id", "a", "b"]
    assert out["row_id"].tolist() == ["r2", "r1"]

    bad = df.copy()
    bad.loc[0, "a"] = 1.5
    try:
        validate_submission_frame(bad, ["a", "b"], name="bad")
    except ValueError as exc:
        assert "outside [0, 1]" in str(exc)
    else:
        raise AssertionError("expected invalid probabilities to fail")


def test_apply_rank_blend_changes_only_allowed_top_rank_cells():
    anchor = pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3"],
            "a": [0.9, 0.2, 0.1],
            "b": [0.1, 0.8, 0.2],
            "c": [0.2, 0.1, 0.7],
        }
    )
    side = pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3"],
            "a": [0.1, 0.2, 0.9],
            "b": [0.2, 0.9, 0.1],
            "c": [0.8, 0.1, 0.2],
        }
    )
    config = RankBlendConfig(
        weight=0.25,
        anchor_topk=1,
        side_topk=1,
        min_anchor_rank_for_side=0.0,
        min_top3_overlap=0.0,
        min_top10_overlap=0.0,
    )

    blended = apply_rank_blend(anchor, side, ["a", "b", "c"], config)

    values = blended[["a", "b", "c"]].to_numpy()
    assert values.shape == (3, 3)
    assert np.isfinite(values).all()
    assert values.min() > 0.0
    assert values.max() < 1.0
    assert not np.allclose(values, anchor[["a", "b", "c"]].to_numpy())


def test_infer_parser_accepts_s124_sidecar_arguments(tmp_path):
    args = build_parser().parse_args(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--input-dir",
            str(tmp_path / "test_soundscapes"),
            "--output",
            str(tmp_path / "submission.csv"),
            "--device",
            "cpu",
            "--batch-size",
            "64",
            "--num-workers",
            "0",
            "--window-seconds",
            "5.0",
            "--tta-shifts",
            "0",
            "--prior-weight",
            "0.0",
            "--smooth-weight",
            "0.12",
            "--disable-context-postprocess",
            "--fast-fixed-60s",
            "--assume-sr",
            "32000",
            "--assume-duration",
            "60",
            "--checkpoint",
            str(tmp_path / "g124_fold1_fp16.pt"),
        ]
    )

    assert args.batch_size == 64
    assert args.fast_fixed_60s is True
    assert args.checkpoint == [str(tmp_path / "g124_fold1_fp16.pt")]


def test_train_parser_supports_2025_pretrain_and_2026_pseudo_modes(tmp_path):
    args = build_train_parser().parse_args(
        [
            "--competition-dir",
            str(tmp_path / "birdclef-2026"),
            "--output-dir",
            str(tmp_path / "out"),
            "--fold",
            "1",
            "--stage",
            "finetune2026",
            "--pretrained-checkpoint",
            str(tmp_path / "g124_2025pre.pt"),
            "--pseudo-csv",
            str(tmp_path / "pseudo.csv"),
            "--epochs",
            "2",
        ]
    )

    assert args.stage == "finetune2026"
    assert args.fold == 1
    assert args.pseudo_csv.endswith("pseudo.csv")


def test_build_dataset_metadata_uses_kaggle_dataset_id():
    meta = build_dataset_metadata(
        dataset_id="diaakh/birdclef2026-g124-effv2s-2025pre-pseudo-assets",
        title="BirdCLEF 2026 G124 assets",
    )

    assert meta["id"] == "diaakh/birdclef2026-g124-effv2s-2025pre-pseudo-assets"
    assert meta["title"] == "BirdCLEF 2026 G124 assets"
    assert meta["licenses"] == [{"name": "CC0-1.0"}]


def test_parse_soundscape_row_id_recovers_filename_and_start():
    filename, start = parse_soundscape_row_id("BC2026_Train_0001_S08_20250606_030007_35")

    assert filename == "BC2026_Train_0001_S08_20250606_030007.ogg"
    assert start == 30.0
