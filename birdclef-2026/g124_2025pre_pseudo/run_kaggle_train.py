from __future__ import annotations

import os
from pathlib import Path

from train_g124 import main as train_main


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def build_default_argv() -> list[str]:
    competition_dir = _env("G124_COMPETITION_DIR", "/kaggle/input/competitions/birdclef-2026")
    if not Path(competition_dir).exists() and Path("/kaggle/input/birdclef-2026").exists():
        competition_dir = "/kaggle/input/birdclef-2026"

    argv = [
        "--competition-dir",
        competition_dir,
        "--output-dir",
        _env("G124_OUTPUT_DIR", "/kaggle/working/g124_assets"),
        "--stage",
        _env("G124_STAGE", "finetune2026"),
        "--fold",
        _env("G124_FOLD", "1"),
        "--epochs",
        _env("G124_EPOCHS", "8"),
        "--batch-size",
        _env("G124_BATCH_SIZE", "48"),
        "--num-workers",
        _env("G124_NUM_WORKERS", "4"),
    ]
    optional = [
        ("G124_PRETRAINED_CHECKPOINT", "--pretrained-checkpoint"),
        ("G124_PSEUDO_CSV", "--pseudo-csv"),
        ("G124_MAX_TRAIN_FILES", "--max-train-files"),
        ("G124_START_EPOCH", "--start-epoch"),
        ("G124_TOTAL_EPOCHS", "--total-epochs"),
        ("G124_LR", "--lr"),
        ("G124_PSEUDO_WEIGHT", "--pseudo-weight"),
        ("G124_SOUNDSCAPE_LABELS_CSV", "--soundscape-labels-csv"),
        ("G124_SOUNDSCAPE_LABEL_WEIGHT", "--soundscape-label-weight"),
        ("G124_FOCUS_CLASSES", "--focus-classes"),
        ("G124_FOCUS_CLASS_WEIGHT", "--focus-class-weight"),
        ("G124_YAO_PROBE_CSV", "--yao-probe-csv"),
        ("G124_YAO_PROBE_TOPK", "--yao-probe-topk"),
        ("G124_YAO_PROBE_TEMPERATURE", "--yao-probe-temperature"),
        ("G124_YAO_PROBE_BIAS", "--yao-probe-bias"),
        ("G124_YAO_PROBE_PROB_FLOOR", "--yao-probe-prob-floor"),
        ("G124_YAO_PROBE_FALSE_CLASS", "--yao-probe-false-class"),
        ("G124_YAO_DISTILL_CLASSES", "--yao-distill-classes"),
        ("G124_YAO_RANK_LOSS_WEIGHT", "--yao-rank-loss-weight"),
        ("G124_YAO_VALUE_LOSS_WEIGHT", "--yao-value-loss-weight"),
        ("G124_YAO_DISTILL_STEPS", "--yao-distill-steps"),
        ("G124_BASE_PRESERVE_LOSS_WEIGHT", "--base-preserve-loss-weight"),
        ("G124_LOSS_IGNORE_NEGATIVE_CLASSES", "--loss-ignore-negative-classes"),
        ("G124_TRAINABLE_SCOPE", "--trainable-scope"),
        ("G124_CED_LOSS_WEIGHT", "--ced-loss-weight"),
        ("G124_CED_KEEP_FRACTION", "--ced-keep-fraction"),
        ("G124_CED_MIN_WIDTH", "--ced-min-width"),
        ("G124_CED_DROP_MARGIN", "--ced-drop-margin"),
        ("G124_CED_CONTEXT_WEIGHT", "--ced-context-weight"),
        ("G124_CED_PRESERVE_WEIGHT", "--ced-preserve-weight"),
        ("G124_CED_POSITIVE_THRESHOLD", "--ced-positive-threshold"),
        ("G124_DIAGNOSTIC_CLASSES", "--diagnostic-classes"),
        ("G124_DIAGNOSTIC_TOPK", "--diagnostic-topk"),
        ("G124_DIAGNOSTIC_LOG_EXAMPLES", "--diagnostic-log-examples"),
        ("G124_GRAD_CLIP", "--grad-clip"),
        ("G124_AUDIO_CACHE_MB", "--audio-cache-mb"),
    ]
    for env_name, flag in optional:
        value = os.environ.get(env_name)
        if value:
            argv.extend([flag, value])
    if os.environ.get("G124_TIMM_PRETRAINED", "").lower() in {"1", "true", "yes"}:
        argv.append("--timm-pretrained")
    if os.environ.get("G124_AMP", "").lower() in {"1", "true", "yes"}:
        argv.append("--amp")
    return argv


if __name__ == "__main__":
    raise SystemExit(train_main(build_default_argv()))
