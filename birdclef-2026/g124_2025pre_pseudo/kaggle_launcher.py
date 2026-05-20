from __future__ import annotations

import argparse
from pathlib import Path
import sys


def find_code_root(search_roots: list[Path] | None = None) -> Path:
    roots = search_roots or [Path("/kaggle/input")]
    candidates = [
        "birdclef-g124-code",
        "birdclef2026-g124-code",
        "g124-2025pre-pseudo-code",
    ]
    for root in roots:
        for name in candidates:
            candidate = root / name
            if (candidate / "train_g124.py").exists():
                return candidate
        if root.exists():
            for candidate in root.rglob("train_g124.py"):
                return candidate.parent
    raise FileNotFoundError("G124 code dataset with train_g124.py not found")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch G124 code from a Kaggle input dataset")
    parser.add_argument("--mode", choices=["smoke", "train"], default="train")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code_root = find_code_root()
    sys.path.insert(0, str(code_root))
    if args.mode == "smoke":
        from run_kaggle_smoke import build_smoke_argv
        from train_g124 import main as train_main

        return train_main(build_smoke_argv())

    from run_kaggle_train import build_default_argv
    from train_g124 import main as train_main

    return train_main(build_default_argv())


if __name__ == "__main__":
    raise SystemExit(main())
