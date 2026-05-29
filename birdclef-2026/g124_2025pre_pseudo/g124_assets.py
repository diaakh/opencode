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


@dataclass(frozen=True)
class SidecarProfile:
    name: str
    keep_topk: int
    active_threshold: float
    background_prob: float
    min_core_active: int
    class_values: dict[str, float]
    class_distributions: dict[str, tuple[float, ...]]
    core_classes: tuple[str, ...]
    false_classes: tuple[str, ...] = ()
    prior_hours: tuple[int, ...] = ()
    prior_sites: tuple[str, ...] = ()


YAO_G124_SOFT_FROG_PROFILE = SidecarProfile(
    name="yao_g124_soft_frog",
    keep_topk=8,
    active_threshold=0.05,
    background_prob=0.00366446,
    min_core_active=6,
    class_values={
        "47158son22": 0.898655,
        "47158son23": 0.669044,
        "47158son13": 0.634755,
        "47158son21": 0.601966,
        "47158son25": 0.453604,
        "47158son10": 0.450168,
        "47158son17": 0.319588,
        "517063": 0.136421,
    },
    class_distributions={
        "47158son22": (
            0.9036359787,
            0.9145939946,
            0.8841639757,
            0.9166100025,
            0.8907520175,
            0.8988540173,
            0.9199259877,
            0.8692759871,
            0.9256749749,
            0.8877509832,
            0.8922489882,
            0.8803759813,
        ),
        "47158son23": (
            0.6543419957,
            0.7542690039,
            0.6232560277,
            0.7886160016,
            0.6432470083,
            0.6686580181,
            0.6501619816,
            0.5484650135,
            0.7518799901,
            0.6812390089,
            0.6341739893,
            0.6302199960,
        ),
        "47158son13": (
            0.6092839837,
            0.7066519856,
            0.6003969908,
            0.7569220066,
            0.6001939774,
            0.6484280229,
            0.6461910009,
            0.5184640288,
            0.7198839784,
            0.6118559837,
            0.6026769876,
            0.5961139798,
        ),
        "47158son21": (
            0.5979909897,
            0.6729859710,
            0.5321909785,
            0.7200049758,
            0.5817940235,
            0.6377639771,
            0.6337000132,
            0.4897840023,
            0.6653550267,
            0.5802429914,
            0.5837259889,
            0.5280449986,
        ),
        "47158son25": (
            0.4626759887,
            0.4712049961,
            0.4533849955,
            0.5883709788,
            0.4849030077,
            0.4632700086,
            0.4132040143,
            0.3722479939,
            0.5026109815,
            0.4850789905,
            0.3801310062,
            0.3661620021,
        ),
        "47158son10": (
            0.4415149987,
            0.4774749875,
            0.4637289941,
            0.5998299718,
            0.4298369884,
            0.4339129925,
            0.4085139930,
            0.3751539886,
            0.4923020005,
            0.4734719992,
            0.4004130065,
            0.4058580101,
        ),
        "47158son17": (
            0.3027360141,
            0.3453930020,
            0.3166059852,
            0.4177989959,
            0.3247930110,
            0.3128229976,
            0.2934800088,
            0.2393199950,
            0.4253920019,
            0.3138220012,
            0.2648510039,
            0.2780410051,
        ),
        "517063": (
            0.1294270009,
            0.1390230060,
            0.1397140026,
            0.1521670073,
            0.1337440014,
            0.1435360014,
            0.1354559958,
            0.1296440065,
            0.1382409930,
            0.1384689957,
            0.1271910071,
            0.1304379999,
        ),
        "47158son01": (
            0.0113920001,
            0.0109670004,
            0.0121290004,
            0.0149849998,
            0.0115850000,
            0.0125380000,
            0.0113319997,
            0.0109949997,
            0.0102239996,
            0.0125169996,
            0.0102329999,
            0.0092160003,
        ),
        "116570": (
            0.0118940000,
            0.0232769996,
            0.0125449998,
            0.0280910004,
            0.0173080005,
            0.0201009996,
            0.0103439996,
            0.0105720004,
            0.0145969996,
            0.0125500001,
            0.0133440001,
            0.0170450006,
        ),
    },
    core_classes=(
        "47158son10",
        "47158son13",
        "47158son17",
        "47158son21",
        "47158son22",
        "47158son23",
        "47158son25",
    ),
    false_classes=("47158son01", "116570"),
)

YAO_G124_PRIOR_RANK_PROFILE = SidecarProfile(
    name="yao_g124_prior_rank",
    keep_topk=YAO_G124_SOFT_FROG_PROFILE.keep_topk,
    active_threshold=YAO_G124_SOFT_FROG_PROFILE.active_threshold,
    background_prob=YAO_G124_SOFT_FROG_PROFILE.background_prob,
    min_core_active=YAO_G124_SOFT_FROG_PROFILE.min_core_active,
    class_values=YAO_G124_SOFT_FROG_PROFILE.class_values,
    class_distributions=YAO_G124_SOFT_FROG_PROFILE.class_distributions,
    core_classes=YAO_G124_SOFT_FROG_PROFILE.core_classes,
    false_classes=YAO_G124_SOFT_FROG_PROFILE.false_classes,
    prior_hours=(0, 1, 2, 3, 4),
)


SIDECAR_PROFILES = {
    "none": None,
    YAO_G124_SOFT_FROG_PROFILE.name: YAO_G124_SOFT_FROG_PROFILE,
    YAO_G124_PRIOR_RANK_PROFILE.name: YAO_G124_PRIOR_RANK_PROFILE,
}


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


def sidecar_diagnostics(df: pd.DataFrame, label_cols: list[str], active_threshold: float = 0.05) -> dict[str, float]:
    values = df[label_cols].to_numpy(dtype=np.float32)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "active_gt05": int((values > active_threshold).sum()),
    }


def rank_blend_diagnostics(
    anchor_df: pd.DataFrame,
    side_df: pd.DataFrame,
    label_cols: list[str],
    topk_values: tuple[int, ...] = (3, 8, 10),
) -> dict[str, float]:
    anchor_df = validate_submission_frame(anchor_df, label_cols, "anchor")
    side_df = validate_submission_frame(side_df, label_cols, "sidecar")
    anchor_ids = anchor_df["row_id"].astype(str)
    if not set(anchor_ids).issubset(set(side_df["row_id"].astype(str))):
        return {"row_overlap": 0.0}
    side_df = side_df.set_index("row_id").loc[anchor_ids].reset_index()
    anchor = np.clip(anchor_df[label_cols].to_numpy(dtype=np.float32), 1e-6, 1.0 - 1e-6)
    side = np.clip(side_df[label_cols].to_numpy(dtype=np.float32), 1e-6, 1.0 - 1e-6)
    anchor_rank = _rank_cols(anchor)
    side_rank = _rank_cols(side)
    flat_anchor = anchor_rank.reshape(-1)
    flat_side = side_rank.reshape(-1)
    if float(np.std(flat_anchor)) > 0 and float(np.std(flat_side)) > 0:
        global_rank_corr = float(np.corrcoef(flat_anchor, flat_side)[0, 1])
    else:
        global_rank_corr = float("nan")
    per_class_corr = []
    for col_idx in range(anchor_rank.shape[1]):
        a = anchor_rank[:, col_idx]
        b = side_rank[:, col_idx]
        if float(np.std(a)) > 0 and float(np.std(b)) > 0:
            per_class_corr.append(float(np.corrcoef(a, b)[0, 1]))
    out: dict[str, float] = {
        "row_overlap": 1.0,
        "global_rank_corr": global_rank_corr,
        "median_class_rank_corr": float(np.median(per_class_corr)) if per_class_corr else float("nan"),
        "mean_abs_rank_delta": float(np.mean(np.abs(side_rank - anchor_rank))),
        "max_abs_rank_delta": float(np.max(np.abs(side_rank - anchor_rank))),
    }
    for topk in topk_values:
        out[f"top{topk}_overlap"] = _topk_overlap(anchor_rank, side_rank, topk)
    return out


def _apply_topk_background(values: np.ndarray, keep_topk: int, background_prob: float) -> np.ndarray:
    if keep_topk <= 0 or keep_topk >= values.shape[1]:
        return np.maximum(values, np.float32(background_prob)).astype(np.float32)
    out = np.full_like(values, np.float32(background_prob), dtype=np.float32)
    top_idx = np.argpartition(values, -keep_topk, axis=1)[:, -keep_topk:]
    rows = np.arange(values.shape[0])[:, None]
    out[rows, top_idx] = values[rows, top_idx]
    return out


def _target_values_for_count(target_values: tuple[float, ...], count: int) -> np.ndarray:
    target = np.sort(np.asarray(target_values, dtype=np.float32))
    if count <= 0:
        return np.asarray([], dtype=np.float32)
    if count == target.size:
        return target
    if target.size == 1:
        return np.full(count, target[0], dtype=np.float32)
    source_q = np.linspace(0.0, 1.0, target.size, dtype=np.float32)
    dest_q = np.linspace(0.0, 1.0, count, dtype=np.float32)
    return np.interp(dest_q, source_q, target).astype(np.float32)


def _rank_preserving_values(rank_source: np.ndarray, target_values: tuple[float, ...]) -> np.ndarray:
    shaped = _target_values_for_count(target_values, int(rank_source.shape[0]))
    order = np.argsort(rank_source, kind="mergesort")
    out = np.empty_like(shaped)
    out[order] = shaped
    return out.astype(np.float32)


def _row_id_site_hour(row_id: str) -> tuple[str | None, int | None]:
    parts = str(row_id).split("_")
    site = next((part for part in parts if part.startswith("S") and part[1:].isdigit()), None)
    hour = None
    for part in parts:
        if len(part) == 6 and part.isdigit():
            value = int(part[:2])
            if 0 <= value <= 23:
                hour = value
    return site, hour


def _metadata_prior_mask(row_ids: pd.Series, profile: SidecarProfile) -> np.ndarray:
    if not profile.prior_hours and not profile.prior_sites:
        return np.ones(len(row_ids), dtype=bool)
    allowed_hours = set(profile.prior_hours)
    allowed_sites = set(profile.prior_sites)
    mask = np.zeros(len(row_ids), dtype=bool)
    for idx, row_id in enumerate(row_ids.astype(str)):
        site, hour = _row_id_site_hour(row_id)
        hour_ok = not allowed_hours or hour in allowed_hours
        site_ok = not allowed_sites or site in allowed_sites
        mask[idx] = hour_ok and site_ok
    return mask


def apply_sidecar_profile(
    side_df: pd.DataFrame,
    label_cols: list[str],
    profile_name: str = "none",
) -> pd.DataFrame:
    if profile_name not in SIDECAR_PROFILES:
        raise ValueError(f"unknown sidecar profile: {profile_name}")
    profile = SIDECAR_PROFILES[profile_name]
    out = validate_submission_frame(side_df, label_cols, "sidecar").copy()
    if profile is None:
        return out

    raw_values = out[label_cols].to_numpy(dtype=np.float32)
    values = np.maximum(raw_values, np.float32(profile.background_prob))
    values = _apply_topk_background(values, profile.keep_topk, profile.background_prob)

    label_to_idx = {label: idx for idx, label in enumerate(label_cols)}
    core_idx = [label_to_idx[label] for label in profile.core_classes if label in label_to_idx]
    target_labels = set(profile.class_values) | set(profile.class_distributions)
    target_idx = {label: label_to_idx[label] for label in target_labels if label in label_to_idx}
    if core_idx and target_idx:
        eligible = _metadata_prior_mask(out["row_id"], profile)
        for row_idx in range(values.shape[0]):
            core_active = int((values[row_idx, core_idx] > profile.active_threshold).sum())
            eligible[row_idx] = bool(eligible[row_idx]) and core_active >= profile.min_core_active

        if eligible.any():
            values[eligible, :] = np.float32(profile.background_prob)
            core_rank_source = raw_values[eligible][:, core_idx].mean(axis=1) if core_idx else np.arange(eligible.sum())
            for label, col_idx in target_idx.items():
                distribution = profile.class_distributions.get(label)
                if distribution:
                    rank_source = raw_values[eligible, col_idx]
                    if float(np.std(rank_source)) <= 1e-8:
                        rank_source = core_rank_source
                    values[eligible, col_idx] = _rank_preserving_values(rank_source, distribution)
                else:
                    values[eligible, col_idx] = np.float32(profile.class_values[label])

    for false_label in profile.false_classes:
        false_idx = label_to_idx.get(false_label)
        if false_idx is not None and false_label not in target_idx:
            values[:, false_idx] = np.float32(profile.background_prob)

    out[label_cols] = np.clip(values, 0.0, 1.0).astype(np.float32)
    return out


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
