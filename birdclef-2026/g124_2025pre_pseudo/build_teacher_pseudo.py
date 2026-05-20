from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def teacher_weights(
    y_true: np.ndarray,
    anchor: np.ndarray,
    teacher: np.ndarray,
    classes: list[str],
    max_weight: float,
    delta_full_weight: float,
    min_teacher_auc: float,
) -> pd.DataFrame:
    rows = []
    for idx, label in enumerate(classes):
        anchor_auc = safe_auc(y_true[:, idx], anchor[:, idx])
        teacher_auc = safe_auc(y_true[:, idx], teacher[:, idx])
        if np.isnan(anchor_auc) or np.isnan(teacher_auc):
            weight = 0.0
            delta = float("nan")
        else:
            delta = teacher_auc - anchor_auc
            if teacher_auc < min_teacher_auc or delta <= 0:
                weight = 0.0
            else:
                weight = min(max_weight, max_weight * delta / max(delta_full_weight, 1e-6))
        rows.append(
            {
                "class": label,
                "positives": int(y_true[:, idx].sum()),
                "anchor_auc": anchor_auc,
                "teacher_auc": teacher_auc,
                "delta_auc": delta,
                "teacher_weight": weight,
            }
        )
    return pd.DataFrame(rows)


def row_ids(row_filename: np.ndarray, row_start_sec: np.ndarray) -> list[str]:
    return [
        f"{Path(str(filename)).stem}_{int(start) + 5}"
        for filename, start in zip(row_filename.tolist(), row_start_sec.tolist())
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build class-gated G124 pseudo labels from exp019 and v8 teacher caches")
    parser.add_argument("--exp019-npz", required=True)
    parser.add_argument("--teacher-npz", required=True)
    parser.add_argument("--teacher-key", default="R_final")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report-csv", required=True)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--delta-full-weight", type=float, default=0.10)
    parser.add_argument("--min-teacher-auc", type=float, default=0.80)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exp = np.load(args.exp019_npz, allow_pickle=True)
    teacher_npz = np.load(args.teacher_npz, allow_pickle=True)
    classes = [str(label) for label in exp["classes"].tolist()]
    anchor = np.nan_to_num(exp["P_exp019"].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    teacher = np.nan_to_num(teacher_npz[args.teacher_key].astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    truth = exp["Y"].astype(np.float32)
    if teacher.shape != anchor.shape:
        raise ValueError(f"teacher shape {teacher.shape} != exp019 shape {anchor.shape}")
    anchor = np.clip(anchor, 0.0, 1.0)
    teacher = np.clip(teacher, 0.0, 1.0)
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
    out = pd.DataFrame(pseudo.astype(np.float32), columns=classes)
    out.insert(0, "row_id", row_ids(exp["row_filename"], exp["row_start_sec"]))
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    report.to_csv(args.report_csv, index=False)
    active = int((report["teacher_weight"] > 0).sum())
    print(
        f"wrote {args.output_csv}: rows={len(out)} classes={len(classes)} "
        f"active_teacher_classes={active} max_weight={report['teacher_weight'].max():.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
