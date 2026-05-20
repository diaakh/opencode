from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


AUDIO_SUFFIXES = {".ogg", ".wav", ".flac", ".mp3"}


@dataclass(frozen=True)
class SoundscapeWindow:
    row_id: str
    path: Path
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class RankBlendConfig:
    weight: float = 0.115
    anchor_topk: int = 48
    side_topk: int = 32
    min_anchor_rank_for_side: float = 0.46
    min_top3_overlap: float = 0.56
    min_top10_overlap: float = 0.68
    eps: float = 1e-5


def find_asset_dir(search_roots: list[Path], checkpoint_names: list[str]) -> Path:
    for root in search_roots:
        if not root.exists():
            continue
        if (root / "infer.py").exists() and all((root / name).exists() for name in checkpoint_names):
            return root
    raise FileNotFoundError(f"asset directory with infer.py and {checkpoint_names} not found")


def list_fixed_windows(
    input_dir: Path,
    window_seconds: float,
    assume_duration: float,
    limit_files: int | None = None,
) -> list[SoundscapeWindow]:
    files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in AUDIO_SUFFIXES)
    if limit_files is not None:
        files = files[:limit_files]

    rows: list[SoundscapeWindow] = []
    n_windows = int(round(assume_duration / window_seconds))
    for path in files:
        stem = path.stem
        for idx in range(n_windows):
            start = idx * window_seconds
            end = start + window_seconds
            rows.append(
                SoundscapeWindow(
                    row_id=f"{stem}_{int(round(end))}",
                    path=path,
                    start_seconds=start,
                    end_seconds=end,
                )
            )
    return rows


def validate_submission_frame(df: pd.DataFrame, label_cols: list[str], name: str) -> pd.DataFrame:
    if "row_id" not in df.columns:
        raise ValueError(f"{name}: row_id missing")
    if df["row_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate row_id")
    missing = [col for col in label_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing label columns {missing[:5]}")

    out = df[["row_id", *label_cols]].copy()
    out["row_id"] = out["row_id"].astype(str)
    values = out[label_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: non-finite values")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError(f"{name}: probabilities outside [0, 1]")
    return out


def _rank_cols(values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values)
    return frame.rank(axis=0, pct=True).to_numpy(dtype=np.float32)


def _top_mask(values: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), values.shape[1]))
    idx = np.argpartition(values, -k, axis=1)[:, -k:]
    mask = np.zeros(values.shape, dtype=bool)
    rows = np.arange(values.shape[0])[:, None]
    mask[rows, idx] = True
    return mask


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = max(1, min(int(k), a.shape[1], b.shape[1]))
    ia = np.argpartition(a, -k, axis=1)[:, -k:]
    ib = np.argpartition(b, -k, axis=1)[:, -k:]
    return float(np.mean([len(set(x.tolist()) & set(y.tolist())) / k for x, y in zip(ia, ib)]))


def apply_rank_blend(
    anchor_df: pd.DataFrame,
    side_df: pd.DataFrame,
    label_cols: list[str],
    config: RankBlendConfig = RankBlendConfig(),
) -> pd.DataFrame:
    anchor_df = validate_submission_frame(anchor_df, label_cols, "anchor")
    side_df = validate_submission_frame(side_df, label_cols, "sidecar")

    anchor_ids = anchor_df["row_id"].astype(str)
    if not set(anchor_ids).issubset(set(side_df["row_id"].astype(str))):
        return anchor_df

    side_df = side_df.set_index("row_id").loc[anchor_ids].reset_index()
    anchor = np.clip(anchor_df[label_cols].to_numpy(dtype=np.float32), config.eps, 1.0 - config.eps)
    side = np.clip(side_df[label_cols].to_numpy(dtype=np.float32), config.eps, 1.0 - config.eps)
    anchor_rank = _rank_cols(anchor)
    side_rank = _rank_cols(side)
    mask = _top_mask(anchor_rank, config.anchor_topk) | (
        _top_mask(side_rank, config.side_topk) & (anchor_rank >= config.min_anchor_rank_for_side)
    )
    blended = anchor_rank.copy()
    blended[mask] = (1.0 - config.weight) * anchor_rank[mask] + config.weight * side_rank[mask]
    blended = np.clip(blended, config.eps, 1.0 - config.eps).astype(np.float32)

    top3 = _topk_overlap(anchor_rank, blended, 3)
    top10 = _topk_overlap(anchor_rank, blended, 10)
    if top3 < config.min_top3_overlap or top10 < config.min_top10_overlap:
        return anchor_df

    out = anchor_df.copy()
    out[label_cols] = blended
    return out
