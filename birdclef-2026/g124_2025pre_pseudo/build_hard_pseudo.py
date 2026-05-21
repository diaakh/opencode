from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_teacher_pseudo import row_ids, teacher_weights


def class_thresholds(competition_dir: Path, classes: list[str], min_thresh: float, max_thresh: float) -> dict[str, float]:
    counts = {}
    train_audio = competition_dir / "train_audio"
    for label in classes:
        label_dir = train_audio / label
        if label_dir.exists():
            counts[label] = len(list(label_dir.glob("*.ogg"))) + len(list(label_dir.glob("*.wav"))) + len(list(label_dir.glob("*.flac")))
        else:
            counts[label] = 0
    values = np.array(list(counts.values()), dtype=np.float32)
    min_count = float(values.min()) if len(values) else 0.0
    max_count = float(values.max()) if len(values) else 1.0
    out = {}
    for label, count in counts.items():
        commonness = (float(count) - min_count) / max(max_count - min_count, 1e-6)
        out[label] = float(min_thresh + (commonness**0.5) * (max_thresh - min_thresh))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build sparse hard pseudo labels from train_soundscape teacher caches")
    parser.add_argument("--competition-dir", required=True)
    parser.add_argument("--exp019-npz", required=True)
    parser.add_argument("--teacher-npz", required=True)
    parser.add_argument("--teacher-key", default="R_final")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--delta-full-weight", type=float, default=0.10)
    parser.add_argument("--min-teacher-auc", type=float, default=0.80)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-per-class", type=int, default=400)
    parser.add_argument("--min-threshold", type=float, default=0.55)
    parser.add_argument("--max-threshold", type=float, default=0.99)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    competition_dir = Path(args.competition_dir)
    exp = np.load(args.exp019_npz, allow_pickle=True)
    teacher_npz = np.load(args.teacher_npz, allow_pickle=True)
    classes = [str(label) for label in exp["classes"].tolist()]
    anchor = np.clip(np.nan_to_num(exp["P_exp019"].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    teacher = np.clip(np.nan_to_num(teacher_npz[args.teacher_key].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    truth = exp["Y"].astype(np.float32)
    if teacher.shape != anchor.shape:
        raise ValueError(f"teacher shape {teacher.shape} != exp019 shape {anchor.shape}")
    report = teacher_weights(
        truth,
        anchor,
        teacher,
        classes,
        max_weight=args.max_weight,
        delta_full_weight=args.delta_full_weight,
        min_teacher_auc=args.min_teacher_auc,
    )
    weights = report["teacher_weight"].to_numpy(dtype=np.float32)[None, :]
    pseudo = np.clip((1.0 - weights) * anchor + weights * teacher, 0.0, 1.0)
    thresholds = class_thresholds(competition_dir, classes, args.min_threshold, args.max_threshold)
    ids = row_ids(exp["row_filename"], exp["row_start_sec"])
    top_n = max(1, min(int(args.top_n), len(classes)))
    by_class: dict[str, list[tuple[float, str]]] = {label: [] for label in classes}
    for row_idx, row_id in enumerate(ids):
        top_idx = np.argsort(pseudo[row_idx])[-top_n:][::-1]
        for class_idx in top_idx:
            label = classes[int(class_idx)]
            confidence = float(pseudo[row_idx, class_idx])
            threshold = max(float(args.min_confidence), thresholds.get(label, args.max_threshold))
            if confidence >= threshold:
                by_class[label].append((confidence, row_id))

    rows = []
    for label, entries in by_class.items():
        for confidence, row_id in sorted(entries, reverse=True)[: max(1, int(args.max_per_class))]:
            rows.append({"row_id": row_id, "primary_label": label, "confidence": confidence, "threshold": thresholds[label]})
    out = pd.DataFrame(rows).sort_values(["primary_label", "confidence"], ascending=[True, False])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    report.to_csv(args.report_csv, index=False)
    print(f"wrote {args.output_csv}: rows={len(out)} classes={out['primary_label'].nunique() if len(out) else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
