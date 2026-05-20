from __future__ import annotations

from run_kaggle_train import build_default_argv
from train_g124 import main as train_main


def _replace_flag(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv.extend([flag, value])


def build_smoke_argv() -> list[str]:
    argv = build_default_argv()
    _replace_flag(argv, "--output-dir", "/kaggle/working/g124_smoke")
    _replace_flag(argv, "--epochs", "1")
    _replace_flag(argv, "--batch-size", "8")
    _replace_flag(argv, "--num-workers", "2")
    _replace_flag(argv, "--max-train-files", "1")
    _replace_flag(argv, "--model-name", "resnet18")
    if "--timm-pretrained" not in argv:
        argv.append("--timm-pretrained")
    return argv


if __name__ == "__main__":
    raise SystemExit(train_main(build_smoke_argv()))
