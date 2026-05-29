from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g124_assets import (
    RankBlendConfig,
    apply_rank_blend,
    apply_sidecar_profile,
    find_asset_dir,
    list_fixed_windows,
    rank_blend_diagnostics,
    sidecar_diagnostics,
    validate_submission_frame,
)
from infer import build_parser
from infer import _apply_sparse_evidence_promotions
from infer import _calibrate_batch_probs
from infer import model_name_candidates as infer_model_name_candidates
from package_assets import build_dataset_metadata
from package_assets import build_parser as build_package_parser
from package_assets import package_assets
from run_kaggle_evidence_traces import build_default_argv as build_evidence_default_argv
from run_kaggle_train import build_default_argv
from run_kaggle_smoke import build_smoke_argv
from kaggle_launcher import find_code_root
from build_hard_pseudo import class_thresholds
from train_g124 import build_parser as build_train_parser
from train_g124 import add_folds
from train_g124 import apply_ced_time_mask
from train_g124 import apply_negative_loss_ignore
from train_g124 import build_ced_time_mask
from train_g124 import build_focus_class_weights
from train_g124 import build_negative_loss_ignore_mask
from train_g124 import build_pseudo_frame
from train_g124 import build_soundscape_label_frame
from train_g124 import choose_validation_fold
from train_g124 import counterfactual_evidence_loss
from train_g124 import compute_yao_probe_metrics
from train_g124 import compute_yao_selection_score
from train_g124 import configure_trainable_scope
from train_g124 import focus_value_distillation_loss
from train_g124 import format_probe_diagnostics
from train_g124 import frame_class_count_summary
from train_g124 import pairwise_rank_distillation_loss
from train_g124 import model_name_candidates
from train_g124 import split_train_val
from train_g124 import parse_soundscape_row_id
from build_teacher_pseudo import safe_auc
from build_teacher_pseudo import teacher_weights
from build_evidence_traces import build_parser as build_evidence_parser
from build_evidence_traces import resolve_trace_class_indices
from build_evidence_pseudo import build_evidence_pseudo
from build_evidence_pseudo import build_parser as build_evidence_pseudo_parser
from evidence_trace import STATE_CONTEXT_BACKED_POSITIVE as G124_STATE_CONTEXT_BACKED_POSITIVE
from evidence_trace import STATE_EVIDENCE_BACKED_POSITIVE as G124_STATE_EVIDENCE_BACKED_POSITIVE
from evidence_trace import compute_occlusion_evidence as compute_g124_occlusion_evidence


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


def test_rank_blend_diagnostics_reports_rank_overlap_and_drift():
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
            "a": [0.8, 0.3, 0.1],
            "b": [0.1, 0.7, 0.2],
            "c": [0.3, 0.1, 0.6],
        }
    )

    diagnostics = rank_blend_diagnostics(anchor, side, ["a", "b", "c"], topk_values=(1, 2))

    assert diagnostics["row_overlap"] == 1.0
    assert diagnostics["global_rank_corr"] > 0.8
    assert diagnostics["median_class_rank_corr"] > 0.8
    assert diagnostics["mean_abs_rank_delta"] >= 0.0
    assert 0.0 <= diagnostics["top1_overlap"] <= 1.0
    assert 0.0 <= diagnostics["top2_overlap"] <= 1.0


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
            "--evidence-promote-classes",
            "517063,47158son23",
            "--evidence-promote-min-delta",
            "0.12",
        ]
    )

    assert args.batch_size == 64
    assert args.fast_fixed_60s is True
    assert args.checkpoint == [str(tmp_path / "g124_fold1_fp16.pt")]
    assert args.logit_temperature == 1.0
    assert args.logit_bias == 0.0
    assert args.keep_topk == 0
    assert args.evidence_promote_classes == "517063,47158son23"
    assert args.evidence_promote_min_delta == 0.12


def test_sparse_evidence_promotions_force_evidence_class_inside_topk():
    sparse = np.array([[0.9, 0.8, 0.7, 0.01]], dtype=np.float32)
    dense = np.array([[0.9, 0.8, 0.7, 0.2]], dtype=np.float32)
    evidence_mask = np.array([[True]], dtype=bool)

    out = _apply_sparse_evidence_promotions(
        sparse,
        dense,
        np.array([3], dtype=np.int64),
        evidence_mask,
        keep_topk=3,
        prob_floor=0.01,
        min_value=0.06,
        max_value=0.25,
    )

    assert out[0, 3] == 0.2
    assert int((out[0] > 0.01).sum()) == 3
    assert out[0, 2] == 0.01


def test_infer_model_name_candidates_match_training_fallback():
    assert infer_model_name_candidates("tf_efficientnetv2_s.in21ft1k") == [
        "tf_efficientnetv2_s.in21ft1k",
        "tf_efficientnetv2_s",
    ]


def test_train_parser_accepts_counterfactual_evidence_args(tmp_path):
    args = build_train_parser().parse_args(
        [
            "--competition-dir",
            str(tmp_path / "competition"),
            "--output-dir",
            str(tmp_path / "out"),
            "--ced-loss-weight",
            "0.25",
            "--ced-keep-fraction",
            "0.30",
            "--ced-drop-margin",
            "0.20",
            "--ced-context-weight",
            "1.50",
        ]
    )

    assert args.ced_loss_weight == 0.25
    assert args.ced_keep_fraction == 0.30
    assert args.ced_drop_margin == 0.20
    assert args.ced_context_weight == 1.50


def test_ced_time_mask_selects_high_energy_frames_and_masks_mel():
    import torch

    mel = torch.zeros(2, 1, 3, 5)
    mel[0, :, :, 1] = 4.0
    mel[0, :, :, 3] = 3.0
    mel[1, :, :, 0] = 5.0
    mel[1, :, :, 4] = 4.0

    mask = build_ced_time_mask(mel, keep_fraction=0.40, min_width=1)
    masked = apply_ced_time_mask(mel, mask)

    assert mask.tolist() == [
        [False, True, False, True, False],
        [True, False, False, False, True],
    ]
    assert torch.allclose(masked[0, :, :, 1], mel[0].mean(dim=-1))
    assert torch.allclose(masked[1, :, :, 0], mel[1].mean(dim=-1))
    assert torch.allclose(masked[0, :, :, 2], mel[0, :, :, 2])


def test_g124_occlusion_evidence_trace_is_class_specific():
    mel = np.zeros((2, 1, 4, 8), dtype=np.float32)
    mel[0, 0, 0, 2:4] = 4.0
    mel[1, 0, 1, 6:8] = 4.0

    def predict_fn(batch):
        logits = np.stack(
            [
                batch[:, 0, 0, 2:4].mean(axis=1) * 3.0 - 2.0,
                batch[:, 0, 1, 6:8].mean(axis=1) * 3.0 - 2.0,
                np.full(batch.shape[0], 3.0, dtype=np.float32),
            ],
            axis=1,
        )
        return 1.0 / (1.0 + np.exp(-logits))

    trace = compute_g124_occlusion_evidence(
        predict_fn,
        mel,
        class_indices=[0, 1, 2],
        n_chunks=4,
        batch_size=2,
        fill="zero",
        positive_threshold=0.50,
        evidence_drop_threshold=0.30,
        concentration_threshold=0.45,
    )

    assert trace.deltas[0, 0].argmax() == 1
    assert trace.deltas[1, 1].argmax() == 3
    assert trace.states[0, 0] == G124_STATE_EVIDENCE_BACKED_POSITIVE
    assert trace.states[1, 1] == G124_STATE_EVIDENCE_BACKED_POSITIVE
    assert trace.states[0, 2] == G124_STATE_CONTEXT_BACKED_POSITIVE


def test_build_evidence_traces_parser_and_class_resolution():
    args = build_evidence_parser().parse_args(
        [
            "--data-dir",
            "data",
            "--input-dir",
            "audio",
            "--checkpoint",
            "g124.pt",
            "--output",
            "trace.npz",
            "--classes",
            "a,c",
            "--n-chunks",
            "6",
            "--fill",
            "zero",
        ]
    )

    indices, labels = resolve_trace_class_indices(["a", "b", "c"], args.classes)

    assert args.n_chunks == 6
    assert args.fill == "zero"
    assert indices.tolist() == [0, 2]
    assert labels == ["a", "c"]


def test_build_evidence_pseudo_keeps_only_evidence_backed_top_rows(tmp_path):
    trace_path = tmp_path / "trace.npz"
    output_csv = tmp_path / "pseudo.csv"
    report_csv = tmp_path / "report.csv"
    row_ids = np.array(["r1", "r2", "r3"], dtype=str)
    class_labels = np.array(["a", "b"], dtype=str)
    full_probs = np.array([[0.8, 0.9], [0.7, 0.2], [0.95, 0.8]], dtype=np.float32)
    deltas = np.array(
        [
            [[0.4, 0.1], [0.02, 0.01]],
            [[0.3, 0.1], [0.01, 0.01]],
            [[0.2, 0.7], [0.5, 0.1]],
        ],
        dtype=np.float32,
    )
    states = np.array(
        [
            [G124_STATE_EVIDENCE_BACKED_POSITIVE, G124_STATE_CONTEXT_BACKED_POSITIVE],
            [G124_STATE_EVIDENCE_BACKED_POSITIVE, 0],
            [G124_STATE_CONTEXT_BACKED_POSITIVE, G124_STATE_EVIDENCE_BACKED_POSITIVE],
        ],
        dtype=np.int8,
    )
    np.savez_compressed(
        trace_path,
        row_ids=row_ids,
        class_labels=class_labels,
        class_indices=np.array([0, 1], dtype=np.int64),
        chunk_starts=np.array([0, 10], dtype=np.int16),
        chunk_ends=np.array([10, 20], dtype=np.int16),
        full_probs=full_probs,
        masked_probs=np.zeros_like(deltas),
        deltas=deltas,
        states=states,
    )
    args = build_evidence_pseudo_parser().parse_args(
        [
            "--trace-npz",
            str(trace_path),
            "--output-csv",
            str(output_csv),
            "--report-csv",
            str(report_csv),
            "--max-per-class",
            "1",
            "--min-full-prob",
            "0.5",
            "--min-max-delta",
            "0.2",
        ]
    )

    out, report = build_evidence_pseudo(args)

    assert out[["row_id", "primary_label"]].to_dict("records") == [
        {"row_id": "r1", "primary_label": "a"},
        {"row_id": "r3", "primary_label": "b"},
    ]
    assert float(out.loc[out["primary_label"] == "a", "confidence"].iloc[0]) == 0.8
    assert int(report.loc[report["class"] == "a", "selected"].iloc[0]) == 1
    assert output_csv.exists()
    assert report_csv.exists()


def test_counterfactual_evidence_loss_penalizes_target_confidence_after_evidence_drop():
    import torch

    target = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    original = torch.tensor([[3.0, -1.0]], dtype=torch.float32)
    bad_drop = torch.tensor([[2.7, -1.2]], dtype=torch.float32)
    good_drop = torch.tensor([[-3.0, -1.2]], dtype=torch.float32)

    bad_loss, bad_parts = counterfactual_evidence_loss(
        original,
        bad_drop,
        target,
        drop_margin=0.2,
        context_weight=1.0,
        preserve_weight=0.2,
    )
    good_loss, good_parts = counterfactual_evidence_loss(
        original,
        good_drop,
        target,
        drop_margin=0.2,
        context_weight=1.0,
        preserve_weight=0.2,
    )

    assert bad_loss > good_loss
    assert bad_parts["evidence_drop"] > good_parts["evidence_drop"]
    assert bad_parts["context_suppression"] > good_parts["context_suppression"]


def test_calibrate_batch_probs_can_match_sparse_sidecar_shape():
    import torch

    args = build_parser().parse_args(
        [
            "--data-dir",
            "data",
            "--input-dir",
            "audio",
            "--output",
            "submission.csv",
            "--checkpoint",
            "g124_fold1_fp16.pt",
            "--logit-temperature",
            "0.18",
            "--logit-bias",
            "1.825",
            "--keep-topk",
            "2",
            "--prob-floor",
            "1e-5",
        ]
    )
    logits = torch.tensor([[4.0, 3.0, 0.0, -1.0], [0.5, 5.0, 1.0, -2.0]])

    probs = _calibrate_batch_probs(logits, args)

    assert probs.shape == (2, 4)
    assert np.isfinite(probs).all()
    assert (probs == np.float32(1e-5)).sum() == 4
    assert (probs > 0.05).sum() <= 4


def test_yao_soft_frog_profile_replaces_false_sonotype_and_softens_shape():
    label_cols = [
        "47158son10",
        "47158son13",
        "47158son17",
        "47158son21",
        "47158son22",
        "47158son23",
        "47158son25",
        "47158son01",
        "517063",
        "other",
    ] + [f"unused_{idx:03d}" for idx in range(224)]
    rows = []
    for idx in range(12):
        row = {col: 0.00001 for col in label_cols}
        row.update(
            {
                "row_id": f"row_{idx}",
                "47158son10": 0.70 + idx * 0.010,
                "47158son13": 0.74 + idx * 0.015,
                "47158son17": 0.72 + idx * 0.012,
                "47158son21": 0.73 + idx * 0.018,
                "47158son22": 0.75 + idx * 0.020,
                "47158son23": 0.71 + idx * 0.017,
                "47158son25": 0.76 + idx * 0.011,
                "47158son01": 0.74,
                "517063": 0.00001,
                "other": 0.20,
            }
        )
        rows.append(row)
    sidecar = pd.DataFrame(rows)

    shaped = apply_sidecar_profile(sidecar, label_cols, profile_name="yao_g124_soft_frog")
    diagnostics = sidecar_diagnostics(shaped, label_cols)

    assert diagnostics["active_gt05"] == 96
    assert diagnostics["max"] <= 0.93
    assert 0.018 <= diagnostics["mean"] <= 0.024
    assert 0.08 <= diagnostics["std"] <= 0.12
    np.testing.assert_allclose(shaped["517063"].mean(), np.float32(0.136421), rtol=0, atol=5e-7)
    assert int((shaped["517063"] > 0.05).sum()) == 12
    assert int((shaped["47158son01"] > 0.05).sum()) == 0
    assert float(shaped["47158son22"].std()) > 0.01
    assert float(shaped["517063"].std()) > 0.001
    assert float(shaped[label_cols].sum(axis=1).std()) > 0.1


def test_yao_prior_rank_profile_uses_metadata_prior_for_frog_artifact():
    label_cols = [
        "47158son10",
        "47158son13",
        "47158son17",
        "47158son21",
        "47158son22",
        "47158son23",
        "47158son25",
        "47158son01",
        "116570",
        "517063",
        "other",
    ] + [f"unused_{idx:03d}" for idx in range(223)]
    rows = []
    for idx in range(12):
        row = {col: 0.00001 for col in label_cols}
        row.update(
            {
                "row_id": f"BC2026_Test_0001_S05_20260101_010000_{(idx + 1) * 5}",
                "47158son10": 0.70 + idx * 0.010,
                "47158son13": 0.74 + idx * 0.015,
                "47158son17": 0.72 + idx * 0.012,
                "47158son21": 0.73 + idx * 0.018,
                "47158son22": 0.75 + idx * 0.020,
                "47158son23": 0.71 + idx * 0.017,
                "47158son25": 0.76 + idx * 0.011,
                "47158son01": 0.95,
                "116570": 0.90,
                "517063": 0.00001,
                "other": 0.20,
            }
        )
        rows.append(row)
    sidecar = pd.DataFrame(rows)

    shaped = apply_sidecar_profile(sidecar, label_cols, profile_name="yao_g124_prior_rank")
    diagnostics = sidecar_diagnostics(shaped, label_cols)

    assert diagnostics["active_gt05"] == 96
    np.testing.assert_allclose(shaped["517063"].mean(), np.float32(0.136421), rtol=0, atol=5e-7)
    assert int((shaped["517063"] > 0.05).sum()) == 12
    assert int((shaped["47158son01"] > 0.05).sum()) == 0
    assert int((shaped["116570"] > 0.05).sum()) == 0
    assert float(shaped["517063"].std()) > 0.001


def test_infer_parser_accepts_sidecar_profile_argument():
    args = build_parser().parse_args(
        [
            "--data-dir",
            "data",
            "--input-dir",
            "audio",
            "--output",
            "submission.csv",
            "--checkpoint",
            "g124_fold1_fp16.pt",
            "--sidecar-profile",
            "yao_g124_prior_rank",
        ]
    )

    assert args.sidecar_profile == "yao_g124_prior_rank"


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


def test_package_assets_writes_kaggle_metadata_filename(tmp_path):
    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"checkpoint")
    infer = tmp_path / "infer.py"
    infer.write_text("print('infer')\n")
    out = tmp_path / "out"
    args = build_package_parser().parse_args(
        [
            "--checkpoint",
            str(ckpt),
            "--output-dir",
            str(out),
            "--dataset-id",
            "adkasd/birdclef2026-g124-effv2s-2025pre-pseudo-assets",
            "--infer-py",
            str(infer),
        ]
    )

    package_assets(args)

    assert (out / "datasets-metadata.json").exists()
    assert (out / "dataset-metadata.json").exists()
    assert (out / "infer.py").exists()
    assert (out / "g124_assets.py").exists()
    assert (out / "g124_fold1_fp16.pt").exists()


def test_parse_soundscape_row_id_recovers_filename_and_start():
    filename, start = parse_soundscape_row_id("BC2026_Train_0001_S08_20250606_030007_35")

    assert filename == "BC2026_Train_0001_S08_20250606_030007.ogg"
    assert start == 30.0


def test_build_pseudo_frame_accepts_hard_pseudo_labels(tmp_path):
    competition = tmp_path / "birdclef-2026"
    soundscapes = competition / "train_soundscapes"
    soundscapes.mkdir(parents=True)
    (soundscapes / "BC2026_Train_0001_S08_20250606_030007.ogg").write_bytes(b"audio")
    pseudo = tmp_path / "hard_pseudo.csv"
    pd.DataFrame(
        {
            "row_id": ["BC2026_Train_0001_S08_20250606_030007_35"],
            "primary_label": ["bird_a"],
            "confidence": [0.8],
        }
    ).to_csv(pseudo, index=False)

    out = build_pseudo_frame(str(pseudo), competition, ["bird_a", "bird_b"], weight=0.5)

    assert len(out) == 1
    assert out.loc[0, "primary_label"] == "bird_a"
    assert out.loc[0, "start_seconds"] == 30.0
    assert out.loc[0, "sample_weight"] == 0.4


def test_build_soundscape_label_frame_builds_multilabel_focus_rows(tmp_path):
    competition = tmp_path / "birdclef-2026"
    soundscapes = competition / "train_soundscapes"
    soundscapes.mkdir(parents=True)
    (soundscapes / "BC2026_Train_0001_S08_20250606_030007.ogg").write_bytes(b"audio")
    labels_csv = competition / "train_soundscapes_labels.csv"
    pd.DataFrame(
        {
            "filename": ["BC2026_Train_0001_S08_20250606_030007.ogg"] * 2,
            "start": ["00:00:00", "00:00:00"],
            "end": ["00:00:05", "00:00:05"],
            "primary_label": ["47158son22;47158son23", "47158son22;47158son23"],
        }
    ).to_csv(labels_csv, index=False)

    out = build_soundscape_label_frame(
        str(labels_csv),
        competition,
        ["47158son22", "47158son23", "other"],
        weight=1.25,
        focus_classes={"47158son22"},
        focus_weight=2.0,
    )

    assert len(out) == 1
    assert out.loc[0, "source"] == "labeled_soundscape"
    assert out.loc[0, "start_seconds"] == 0.0
    assert out.loc[0, "sample_weight"] == 2.5
    np.testing.assert_allclose(out.loc[0, "target"], np.array([1.0, 1.0, 0.0], dtype=np.float32))


def test_build_focus_class_weights_boosts_selected_classes():
    weights = build_focus_class_weights(["a", "b", "c"], {"b", "missing"}, focus_weight=2.5)

    np.testing.assert_allclose(weights, np.array([1.0, 2.5, 1.0], dtype=np.float32))


def test_compute_yao_probe_metrics_tracks_hidden_shape_and_truth():
    pred = np.array(
        [
            [0.90, 0.80, 0.01, 0.10],
            [0.85, 0.02, 0.70, 0.15],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [0.95, 0.70, 0.01, 0.12],
            [0.80, 0.05, 0.65, 0.18],
        ],
        dtype=np.float32,
    )
    metrics = compute_yao_probe_metrics(
        pred,
        target,
        ["47158son22", "47158son23", "47158son17", "517063"],
        true_label_sets=[{"47158son22", "47158son23"}, {"47158son22", "47158son17"}],
        focus_classes={"47158son22", "47158son23", "47158son17"},
        topk=2,
        false_class="517063",
    )

    assert metrics["active_gt05"] == 6
    assert metrics["topk_overlap"] == 1.0
    assert metrics["sonotype_recall"] == 1.0
    assert metrics["false_517063_mean"] == 0.125
    assert metrics["yao_corr"] > 0.95
    assert metrics["yao_rank_corr"] > 0.95
    assert metrics["yao_rank_mae"] < 0.2


def test_pairwise_rank_distillation_loss_penalizes_wrong_order_more():
    import torch

    target = torch.tensor(
        [
            [0.90, 0.10],
            [0.60, 0.40],
            [0.20, 0.80],
        ],
        dtype=torch.float32,
    )
    aligned_logits = torch.logit(target.clamp(1e-4, 1 - 1e-4))
    reversed_logits = torch.flip(aligned_logits, dims=[0])
    focus_indices = torch.tensor([0, 1], dtype=torch.long)

    aligned_loss = pairwise_rank_distillation_loss(aligned_logits, target, focus_indices)
    reversed_loss = pairwise_rank_distillation_loss(reversed_logits, target, focus_indices)

    assert aligned_loss.item() < 0.15
    assert reversed_loss.item() > aligned_loss.item() + 0.5


def test_focus_value_distillation_loss_updates_class_even_when_logit_is_low():
    import torch

    logits = torch.tensor([[-8.0, 8.0]], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([[0.15, 0.90]], dtype=torch.float32)
    focus_indices = torch.tensor([0], dtype=torch.long)

    loss = focus_value_distillation_loss(logits, target, focus_indices, temperature=1.2, bias=-3.9)
    loss.backward()

    assert loss.item() > 0.0
    assert logits.grad is not None
    assert abs(float(logits.grad[0, 0])) > 1e-5
    assert float(logits.grad[0, 1]) == 0.0


def test_negative_loss_ignore_only_masks_negative_targets():
    import torch

    classes = ["a", "517063", "b"]
    mask = torch.tensor(build_negative_loss_ignore_mask(classes, {"517063"}))
    raw_loss = torch.ones((2, 3), dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

    masked = apply_negative_loss_ignore(raw_loss, target, mask)

    assert masked[0, 1].item() == 0.0
    assert masked[1, 1].item() == 1.0
    assert masked[:, 0].sum().item() == 2.0


def test_format_probe_diagnostics_reports_raw_dense_sparse_and_rank():
    raw_logits = np.array([[1.0, -4.0], [2.0, -3.5]], dtype=np.float32)
    sparse_pred = np.array([[0.90, 0.00001], [0.80, 0.12]], dtype=np.float32)
    target = np.array([[0.95, 0.10], [0.75, 0.15]], dtype=np.float32)

    lines = format_probe_diagnostics(
        3,
        5,
        raw_logits,
        sparse_pred,
        target,
        ["row_1", "row_2"],
        ["47158son22", "517063"],
        ["47158son22", "517063"],
        temperature=1.2,
        bias=-3.9,
        topk=2,
        log_examples=1,
    )

    joined = "\n".join(lines)
    assert "diagnostic_probe_shape epoch=3/5" in joined
    assert "517063:raw=" in joined
    assert "dense_rank=" in joined
    assert "diagnostic_probe_row epoch=3/5 row=row_1" in joined


def test_compute_yao_selection_score_prefers_rank_over_val_loss_when_probe_is_better():
    weak_rank = {
        "yao_corr": 0.99,
        "yao_rank_corr": -0.10,
        "yao_rank_mae": 0.32,
        "topk_overlap": 0.75,
        "active_gt05": 96,
    }
    strong_rank = {
        "yao_corr": 0.95,
        "yao_rank_corr": 0.55,
        "yao_rank_mae": 0.18,
        "topk_overlap": 0.88,
        "active_gt05": 96,
    }

    assert compute_yao_selection_score(strong_rank, val_loss=0.012) > compute_yao_selection_score(weak_rank, val_loss=0.009)


def test_configure_trainable_scope_head_freezes_backbone():
    import torch

    model = torch.nn.Sequential()
    model.add_module("features", torch.nn.Linear(3, 4))
    model.add_module("classifier", torch.nn.Linear(4, 2))

    trainable, total = configure_trainable_scope(model, "head")

    assert trainable == sum(param.numel() for param in model.classifier.parameters())
    assert total == sum(param.numel() for param in model.parameters())
    assert all(not param.requires_grad for param in model.features.parameters())
    assert all(param.requires_grad for param in model.classifier.parameters())


def test_class_thresholds_are_lower_for_rare_classes(tmp_path):
    competition = tmp_path / "birdclef-2026"
    for label, n_files in {"rare": 1, "common": 4}.items():
        label_dir = competition / "train_audio" / label
        label_dir.mkdir(parents=True)
        for idx in range(n_files):
            (label_dir / f"{idx}.ogg").write_bytes(b"audio")

    thresholds = class_thresholds(competition, ["rare", "common"], min_thresh=0.55, max_thresh=0.99)

    assert thresholds["rare"] == 0.55
    assert thresholds["common"] == 0.99


def test_add_folds_handles_classes_with_fewer_examples_than_folds():
    frame = pd.DataFrame(
        {
            "path": [f"{label}_{idx}.ogg" for label in ["a", "b"] for idx in range(2)],
            "primary_label": [label for label in ["a", "b"] for _ in range(2)],
        }
    )

    out = add_folds(frame, n_folds=5, seed=124)

    assert out["fold"].between(0, 4).all()
    assert set(out["fold"]) == {0, 1}


def test_add_folds_does_not_collapse_when_one_class_has_one_example():
    frame = pd.DataFrame(
        {
            "path": [f"a_{idx}.ogg" for idx in range(10)] + ["rare.ogg"],
            "primary_label": ["a"] * 10 + ["rare"],
        }
    )

    out = add_folds(frame, n_folds=5, seed=124)
    train_frame, val_frame, fold = split_train_val(out, requested_fold=1)

    assert fold == 1
    assert 0 < len(val_frame) < len(out)
    assert "rare" in set(train_frame["primary_label"])


def test_choose_validation_fold_falls_back_to_available_fold():
    frame = pd.DataFrame({"fold": [0, 0, 0]})

    assert choose_validation_fold(frame, requested_fold=1) == 0


def test_split_train_val_reuses_single_fold_for_smoke():
    frame = pd.DataFrame({"fold": [0, 0, 0], "path": ["a", "b", "c"]})

    train_frame, val_frame, fold = split_train_val(frame, requested_fold=1)

    assert fold == 0
    assert len(train_frame) == 3
    assert len(val_frame) == 3


def test_frame_class_count_summary_treats_nan_sample_weight_as_one():
    target = np.array([1.0, 0.0], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "primary_label": ["a"],
            "source": ["train_audio"],
            "sample_weight": [np.nan],
            "target": [target],
        }
    )

    summary = frame_class_count_summary(frame, ["a", "b"], ["a"], "train")

    assert "a:pos=1,weighted=1.00" in summary
    assert "nan" not in summary.lower()


def test_model_name_candidates_strip_invalid_timm_tag():
    assert model_name_candidates("tf_efficientnetv2_s.in21ft1k") == [
        "tf_efficientnetv2_s.in21ft1k",
        "tf_efficientnetv2_s",
    ]


def test_run_kaggle_train_builds_default_argv_from_environment(monkeypatch):
    monkeypatch.setenv("G124_COMPETITION_DIR", "/kaggle/input/birdclef-2026")
    monkeypatch.setenv("G124_EPOCHS", "3")
    monkeypatch.setenv("G124_PSEUDO_CSV", "/kaggle/input/pseudo/pseudo.csv")
    monkeypatch.setenv("G124_LR", "5e-5")
    monkeypatch.setenv("G124_SOUNDSCAPE_LABELS_CSV", "/kaggle/input/birdclef-2026/train_soundscapes_labels.csv")
    monkeypatch.setenv("G124_SOUNDSCAPE_LABEL_WEIGHT", "1.25")
    monkeypatch.setenv("G124_FOCUS_CLASSES", "47158son22,47158son23")
    monkeypatch.setenv("G124_FOCUS_CLASS_WEIGHT", "2.0")
    monkeypatch.setenv("G124_YAO_PROBE_CSV", "/kaggle/input/birdclef-g124-code/yao_probe_g124.csv")
    monkeypatch.setenv("G124_YAO_DISTILL_CLASSES", "47158son22,517063,47158son01")
    monkeypatch.setenv("G124_YAO_RANK_LOSS_WEIGHT", "0.05")
    monkeypatch.setenv("G124_YAO_VALUE_LOSS_WEIGHT", "0.01")
    monkeypatch.setenv("G124_YAO_DISTILL_STEPS", "4")
    monkeypatch.setenv("G124_BASE_PRESERVE_LOSS_WEIGHT", "0.5")
    monkeypatch.setenv("G124_LOSS_IGNORE_NEGATIVE_CLASSES", "517063")
    monkeypatch.setenv("G124_TRAINABLE_SCOPE", "head")
    monkeypatch.setenv("G124_CED_LOSS_WEIGHT", "0.05")
    monkeypatch.setenv("G124_CED_KEEP_FRACTION", "0.20")
    monkeypatch.setenv("G124_CED_MIN_WIDTH", "3")
    monkeypatch.setenv("G124_CED_DROP_MARGIN", "0.15")
    monkeypatch.setenv("G124_CED_CONTEXT_WEIGHT", "1.25")
    monkeypatch.setenv("G124_CED_PRESERVE_WEIGHT", "0.10")
    monkeypatch.setenv("G124_CED_POSITIVE_THRESHOLD", "0.35")
    monkeypatch.setenv("G124_DIAGNOSTIC_CLASSES", "47158son22,517063")
    monkeypatch.setenv("G124_DIAGNOSTIC_TOPK", "8")
    monkeypatch.setenv("G124_DIAGNOSTIC_LOG_EXAMPLES", "1")

    argv = build_default_argv()

    assert "--competition-dir" in argv
    assert "/kaggle/input/birdclef-2026" in argv
    assert "--epochs" in argv
    assert "3" in argv
    assert "--pseudo-csv" in argv
    assert "/kaggle/input/pseudo/pseudo.csv" in argv
    assert "--lr" in argv
    assert "5e-5" in argv
    assert "--soundscape-labels-csv" in argv
    assert "/kaggle/input/birdclef-2026/train_soundscapes_labels.csv" in argv
    assert "--focus-classes" in argv
    assert "47158son22,47158son23" in argv
    assert "--yao-probe-csv" in argv
    assert "/kaggle/input/birdclef-g124-code/yao_probe_g124.csv" in argv
    assert "--yao-distill-classes" in argv
    assert "47158son22,517063,47158son01" in argv
    assert "--yao-rank-loss-weight" in argv
    assert "0.05" in argv
    assert "--yao-value-loss-weight" in argv
    assert "0.01" in argv
    assert "--yao-distill-steps" in argv
    assert "4" in argv
    assert "--base-preserve-loss-weight" in argv
    assert "0.5" in argv
    assert "--loss-ignore-negative-classes" in argv
    assert "517063" in argv
    assert "--trainable-scope" in argv
    assert "head" in argv
    assert "--ced-loss-weight" in argv
    assert "0.05" in argv
    assert "--ced-keep-fraction" in argv
    assert "0.20" in argv
    assert "--ced-min-width" in argv
    assert "3" in argv
    assert "--ced-drop-margin" in argv
    assert "0.15" in argv
    assert "--ced-context-weight" in argv
    assert "1.25" in argv
    assert "--ced-preserve-weight" in argv
    assert "0.10" in argv
    assert "--ced-positive-threshold" in argv
    assert "0.35" in argv
    assert "--diagnostic-classes" in argv
    assert "47158son22,517063" in argv
    assert "--diagnostic-topk" in argv
    assert "8" in argv
    assert "--diagnostic-log-examples" in argv
    assert "1" in argv


def test_run_kaggle_train_can_enable_timm_pretrained(monkeypatch):
    monkeypatch.setenv("G124_TIMM_PRETRAINED", "1")

    argv = build_default_argv()

    assert "--timm-pretrained" in argv


def test_run_kaggle_evidence_traces_builds_default_argv_from_environment(monkeypatch):
    monkeypatch.setenv("G124_TRACE_DATA_DIR", "/kaggle/input/birdclef-2026")
    monkeypatch.setenv("G124_TRACE_INPUT_DIR", "/kaggle/input/birdclef-2026/train_soundscapes")
    monkeypatch.setenv("G124_TRACE_CHECKPOINT", "/kaggle/input/assets/g124_fold1_fp16.pt")
    monkeypatch.setenv("G124_TRACE_OUTPUT", "/kaggle/working/evidence_trace.npz")
    monkeypatch.setenv("G124_TRACE_CLASSES", "47158son23,517063")
    monkeypatch.setenv("G124_TRACE_N_CHUNKS", "10")
    monkeypatch.setenv("G124_TRACE_FILL", "mean")

    argv = build_evidence_default_argv()

    assert "--data-dir" in argv
    assert "/kaggle/input/birdclef-2026" in argv
    assert "--checkpoint" in argv
    assert "/kaggle/input/assets/g124_fold1_fp16.pt" in argv
    assert "--classes" in argv
    assert "47158son23,517063" in argv
    assert "--n-chunks" in argv
    assert "10" in argv
    assert "--fill" in argv
    assert "mean" in argv


def test_run_kaggle_smoke_uses_small_limits():
    argv = build_smoke_argv()

    assert "--epochs" in argv
    assert "1" in argv
    assert "--max-train-files" in argv
    assert "1" in argv
    assert "--model-name" in argv
    assert "resnet18" in argv
    assert "--timm-pretrained" in argv


def test_kaggle_launcher_finds_code_dataset_root(tmp_path):
    root = tmp_path / "birdclef-g124-code"
    root.mkdir()
    (root / "train_g124.py").write_text("print('ok')\n")

    assert find_code_root([tmp_path]) == root


def test_teacher_weights_only_uses_classes_where_teacher_is_better():
    y = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.float32,
    )
    anchor = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.9],
            [0.8, 0.2],
            [0.9, 0.8],
        ],
        dtype=np.float32,
    )
    teacher = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.1],
            [0.7, 0.8],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )

    report = teacher_weights(y, anchor, teacher, ["good", "bad"], 0.25, 0.10, 0.50)

    assert safe_auc(y[:, 0], teacher[:, 0]) >= safe_auc(y[:, 0], anchor[:, 0])
    assert report.loc[report["class"] == "good", "teacher_weight"].item() >= 0.0
    assert report.loc[report["class"] == "bad", "teacher_weight"].item() == 0.0
