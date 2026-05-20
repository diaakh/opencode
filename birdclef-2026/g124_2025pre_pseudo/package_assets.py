from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def build_dataset_metadata(dataset_id: str, title: str) -> dict:
    return {
        "id": dataset_id,
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package G124 sidecar assets as a Kaggle dataset folder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--title", default="BirdCLEF 2026 G124 EfficientNetV2-S 2025pre pseudo assets")
    parser.add_argument("--infer-py", default=str(Path(__file__).with_name("infer.py")))
    return parser


def package_assets(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    infer_py = Path(args.infer_py)
    if not infer_py.exists():
        raise FileNotFoundError(infer_py)

    shutil.copy2(infer_py, output_dir / "infer.py")
    shutil.copy2(checkpoint, output_dir / "g124_fold1_fp16.pt")
    metadata = build_dataset_metadata(args.dataset_id, args.title)
    (output_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = package_assets(args)
    print(f"Packaged G124 assets in {output_dir}")
    print(f"Next: kaggle datasets create -p {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
