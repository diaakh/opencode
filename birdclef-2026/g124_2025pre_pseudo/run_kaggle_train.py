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
    ]
    for env_name, flag in optional:
        value = os.environ.get(env_name)
        if value:
            argv.extend([flag, value])
    if os.environ.get("G124_TIMM_PRETRAINED", "").lower() in {"1", "true", "yes"}:
        argv.append("--timm-pretrained")
    return argv


if __name__ == "__main__":
    raise SystemExit(train_main(build_default_argv()))
