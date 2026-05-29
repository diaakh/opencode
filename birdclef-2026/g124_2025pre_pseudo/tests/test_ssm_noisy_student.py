"""Tests for the SSM head (T2-B) and noisy-student loop (T2-A).

Mirrors the existing tests/ style (sys.path insert + direct module imports, plain
asserts). Self-contained: imports only the new modules and torch/numpy/pandas, so it
runs even though the repo's legacy test module references a few never-committed
``evidence_*`` helpers. CPU-only.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ssm_head import (
    SSMHeadConfig,
    build_ssm_head,
    build_sonotype_group_index,
    flops_per_file,
    param_count,
    sonotype_mirror_logits,
    summarize_head,
)
from noisy_student import (
    NoiseSchedule,
    NoisyStudentConfig,
    RoundResult,
    cap_pseudo_ratio,
    ensemble_teacher_probs,
    make_soft_targets,
    mixup_embeddings,
    power_transform,
    run_noisy_student,
    select_best_round,
    subsample_pseudo_frame,
)
from train_g124 import (
    build_noisy_student_config,
    build_ssm_head_config,
    build_parser as build_train_parser,
    load_pseudo_parquet,
)


# ----------------------------------------------------------------- SSM head (T2-B)


def test_ssm_head_param_count_matches_module():
    cfg = SSMHeadConfig(num_classes=234)
    head = build_ssm_head(cfg)
    actual = sum(p.numel() for p in head.parameters())
    assert actual == param_count(cfg)["total"]


def test_ssm_head_forward_shape_and_backward():
    cfg = SSMHeadConfig(embed_dim=1280, num_classes=234, proj_dim=64, state_dim=16)
    head = build_ssm_head(cfg)
    embeddings = torch.randn(3, 12, 1280)
    logits = head(embeddings)
    assert tuple(logits.shape) == (3, 12, 234)
    logits.sigmoid().mean().backward()
    assert head.proto.grad is not None
    assert head.alpha.grad is not None


def test_ssm_head_bidirectional_differs_from_unidirectional():
    torch.manual_seed(0)
    embeddings = torch.randn(2, 12, 1280)
    bi = build_ssm_head(SSMHeadConfig(num_classes=20, proj_dim=32, state_dim=8))
    uni = build_ssm_head(
        SSMHeadConfig(num_classes=20, proj_dim=32, state_dim=8, bidirectional=False)
    )
    # different parameter counts: two scan directions vs one
    assert sum(p.numel() for p in bi.parameters()) > sum(p.numel() for p in uni.parameters())


def test_ssm_head_pure_pytorch_scan_runs_without_kernel():
    cfg = SSMHeadConfig(num_classes=10, proj_dim=16, state_dim=8, use_mamba_ssm=False)
    head = build_ssm_head(cfg)
    assert head._use_kernel is False
    out = head(torch.randn(1, 12, 1280))
    assert torch.isfinite(out).all()


def test_sonotype_mirror_maxpools_within_group():
    group_id = build_sonotype_group_index(4, {0: "a", 1: "a", 2: "b", 3: "b"})
    assert group_id is not None and group_id.shape[0] == 4
    logits = torch.tensor([[[1.0, 5.0, 2.0, 0.0]]])  # (1,1,4)
    mirrored = sonotype_mirror_logits(logits, group_id)
    # classes 0,1 share a group -> both become max(1,5)=5; classes 2,3 -> max(2,0)=2
    assert float(mirrored[0, 0, 0]) == 5.0
    assert float(mirrored[0, 0, 1]) == 5.0
    assert float(mirrored[0, 0, 2]) == 2.0
    assert float(mirrored[0, 0, 3]) == 2.0


def test_ssm_seed_prototypes_only_touches_nonzero_rows():
    cfg = SSMHeadConfig(num_classes=6, proj_dim=8, state_dim=4)
    head = build_ssm_head(cfg)
    before = head.proto.data.clone()
    class_embeddings = torch.randn(6, 1280)
    class_embeddings[3:] = 0.0  # simulate zero-train_audio classes
    head.seed_prototypes_from_embeddings(class_embeddings)
    after = head.proto.data
    assert not torch.allclose(before[0], after[0])  # seeded class changed
    assert torch.allclose(before[5], after[5])       # zero-embedding class untouched


def test_flops_per_file_is_small_and_dominated_by_proj():
    cfg = SSMHeadConfig()
    flops = flops_per_file(cfg, n_windows=12)
    assert flops["total"] < 50e6  # well under 50 MFLOPs/file (A1: ~5-10M)
    assert "ssm_head" in summarize_head(cfg)


# ----------------------------------------------------------- noisy-student (T2-A)


def test_power_transform_zeros_weak_and_sharpens_strong():
    p = np.array([0.1, 0.35, 0.9], dtype=np.float32)
    out = power_transform(p, threshold=0.3, power=2.0)
    np.testing.assert_allclose(out, [0.01, 0.35 + 0.35**2, 1.0], atol=1e-5)


def test_power_transform_torch_matches_numpy():
    p = np.array([0.05, 0.4, 0.7], dtype=np.float32)
    np_out = power_transform(p, 0.3, 2.0)
    torch_out = power_transform(torch.tensor(p), 0.3, 2.0).numpy()
    np.testing.assert_allclose(np_out, torch_out, atol=1e-6)


def test_make_soft_targets_mixes_pseudo_and_hard():
    cfg = NoisyStudentConfig(pseudo_alpha=0.7, pseudo_threshold=0.3, pseudo_power=2.0)
    teacher = torch.full((2, 4), 0.8)
    hard = torch.zeros(2, 4)
    hard[0, 0] = 1.0
    target = make_soft_targets(teacher, hard, cfg)
    # row 1 is unlabeled -> 0.7 * powertransform(0.8); row 0 col 0 adds 0.3 hard
    assert float(target[1, 0]) > 0.6
    assert float(target[0, 0]) > float(target[1, 0])


def test_cap_pseudo_ratio_enforces_max_fraction():
    keep = cap_pseudo_ratio(n_labeled=600, n_pseudo=10000, ratio_cap=0.4)
    assert keep == 400
    assert keep / (keep + 600) <= 0.4 + 1e-9
    assert cap_pseudo_ratio(0, 100, 0.4) == 0


def test_subsample_pseudo_frame_keeps_highest_confidence():
    frame = pd.DataFrame(
        {
            "filename": [f"f{i}" for i in range(100)],
            "file_confidence": np.linspace(0.0, 1.0, 100, dtype=np.float32),
        }
    )
    capped = subsample_pseudo_frame(frame, n_labeled=10, ratio_cap=0.4, seed=0)
    # cap: 0.4/0.6*10 = ~6.67 -> 7 rows, all from the high-confidence tail
    assert len(capped) == cap_pseudo_ratio(10, 100, 0.4)
    assert float(capped["file_confidence"].min()) > 0.5


def test_noise_schedule_ramps_with_round():
    cfg = NoisyStudentConfig(noise_base=0.1, noise_step=0.1, noise_max=0.5)
    strengths = [NoiseSchedule.for_round(cfg, r).strength for r in range(6)]
    np.testing.assert_allclose(strengths, [0.1, 0.2, 0.3, 0.4, 0.5, 0.5], atol=1e-6)
    assert strengths[-1] == strengths[-2]  # capped at noise_max


def test_ensemble_teacher_probs_averages_sigmoids():
    probs = ensemble_teacher_probs([torch.zeros(2, 3), torch.full((2, 3), 10.0)])
    np.testing.assert_allclose(probs.numpy(), 0.75, atol=1e-3)


def test_mixup_embeddings_blends_pairs():
    sched = NoiseSchedule(strength=0.4, mixup_alpha=0.4)
    x = torch.randn(4, 12, 8)
    y = torch.eye(4).unsqueeze(1).expand(4, 12, 4).clone()
    mx, my = mixup_embeddings(x, y, sched)
    assert mx.shape == x.shape and my.shape == y.shape


def test_select_best_round_picks_highest_score():
    results = [
        RoundResult(0, 0.2, {}, None),
        RoundResult(1, 0.5, {}, None),
        RoundResult(2, 0.3, {}, None),
    ]
    assert select_best_round(results).round_idx == 1


def test_run_noisy_student_loop_selects_best_round_with_ssm_head():
    torch.manual_seed(0)
    cfg = NoisyStudentConfig(n_rounds=3, final_tss=True)
    hcfg = SSMHeadConfig(embed_dim=64, num_classes=6, proj_dim=16, state_dim=8)
    labeled = torch.randn(8, 12, 64)
    labels = torch.zeros(8, 12, 6)
    for i in range(8):
        labels[i, :, i % 6] = 1.0

    def init_student(_round):
        return build_ssm_head(hcfg)

    def train_one_round(student, _teacher, noise, _round_idx, _use_tss):
        opt = torch.optim.AdamW(student.parameters(), lr=1e-2)
        for _ in range(3):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                student(labeled), labels
            )
            loss.backward()
            opt.step()
        return student

    scores = {0: 0.2, 1: 0.9, 2: 0.4}

    def evaluate(_student):
        return {}

    calls = {"i": -1}

    def selection_score_fn(_metrics):
        calls["i"] += 1
        return scores[calls["i"]]

    best = run_noisy_student(
        config=cfg,
        init_student=init_student,
        train_one_round=train_one_round,
        evaluate=evaluate,
        selection_score_fn=selection_score_fn,
        teacher_init=None,
        logger=lambda *_a, **_k: None,
    )
    assert best.round_idx == 1
    # checkpoints are detached fp16 CPU tensors
    assert all(v.dtype == torch.float16 for v in best.state_dict.values())


# ----------------------------------------------------------- train_g124 wiring


def test_train_parser_accepts_ssm_and_noisy_student_flags(tmp_path):
    parser = build_train_parser()
    args = parser.parse_args(
        [
            "--competition-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--ssm-head",
            "--noisy-student-rounds",
            "4",
            "--ns-final-tss",
            "--ns-pseudo-ratio-cap",
            "0.4",
            "--ssm-no-mamba-kernel",
        ]
    )
    assert args.ssm_head is True
    assert args.noisy_student_rounds == 4
    assert args.ns_final_tss is True
    ssm_cfg = build_ssm_head_config(args)
    assert ssm_cfg.num_classes == 234
    assert ssm_cfg.use_mamba_ssm is False
    ns_cfg = build_noisy_student_config(args)
    assert ns_cfg.n_rounds == 4
    assert ns_cfg.pseudo_alpha == 0.7
    assert ns_cfg.pseudo_ratio_cap == 0.4


def test_load_pseudo_parquet_csv_validates_schema(tmp_path):
    classes = [f"c{i}" for i in range(234)]
    data = {label: np.random.rand(12).astype(np.float32) for label in classes}
    data["filename"] = ["f1"] * 12
    data["start_sec"] = list(range(0, 60, 5))
    data["file_confidence"] = [0.9] * 12
    frame = pd.DataFrame(data)
    csv_path = tmp_path / "pseudo.csv"
    frame.to_csv(csv_path, index=False)

    meta, soft = load_pseudo_parquet(str(csv_path), classes)
    assert list(meta.columns) == ["filename", "start_sec", "file_confidence"]
    assert soft.shape == (12, 234)
    assert float(soft.min()) >= 0.0 and float(soft.max()) <= 1.0

    raised = False
    try:
        load_pseudo_parquet(str(csv_path), classes + ["MISSING_CLASS"])
    except ValueError:
        raised = True
    assert raised


def test_load_pseudo_parquet_none_returns_none():
    meta, soft = load_pseudo_parquet(None, [])
    assert meta is None and soft is None


def test_smoke_runner_completes_end_to_end_fast():
    from run_ssm_smoke import build_parser as build_smoke_parser, run_smoke

    args = build_smoke_parser().parse_args(
        ["--rounds", "2", "--num-classes", "8", "--n-labeled", "8", "--n-unlabeled", "6", "--n-probe", "4"]
    )
    result = run_smoke(args)
    assert 0 <= result["best_round"] <= 1
    assert result["elapsed"] < 60.0
    assert result["params"] > 0
