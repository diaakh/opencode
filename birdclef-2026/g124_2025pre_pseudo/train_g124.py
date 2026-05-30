from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import random
import re
import time

import numpy as np
import pandas as pd
import soundfile as sf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train G124 EfficientNetV2-S BirdCLEF sidecar checkpoint")
    parser.add_argument("--competition-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=["pretrain2025", "finetune2026"], default="finetune2026")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--pseudo-csv", default=None)
    parser.add_argument("--model-name", default="tf_efficientnetv2_s.in21ft1k")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--total-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=124)
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=16000.0)
    parser.add_argument("--pseudo-weight", type=float, default=0.35)
    parser.add_argument("--soundscape-labels-csv", default=None)
    parser.add_argument("--soundscape-label-weight", type=float, default=0.0)
    parser.add_argument("--focus-classes", default="")
    parser.add_argument("--focus-class-weight", type=float, default=1.0)
    parser.add_argument("--yao-probe-csv", default=None)
    parser.add_argument("--yao-probe-topk", type=int, default=8)
    parser.add_argument("--yao-probe-temperature", type=float, default=1.20)
    parser.add_argument("--yao-probe-bias", type=float, default=-3.90)
    parser.add_argument("--yao-probe-prob-floor", type=float, default=1e-5)
    parser.add_argument("--yao-probe-false-class", default="517063")
    parser.add_argument("--yao-distill-classes", default="")
    parser.add_argument("--yao-rank-loss-weight", type=float, default=0.0)
    parser.add_argument("--yao-value-loss-weight", type=float, default=0.0)
    parser.add_argument("--yao-distill-steps", type=int, default=1)
    parser.add_argument("--base-preserve-loss-weight", type=float, default=0.0)
    parser.add_argument("--loss-ignore-negative-classes", default="")
    parser.add_argument("--trainable-scope", choices=["all", "head"], default="all")
    parser.add_argument("--ced-loss-weight", type=float, default=0.0)
    parser.add_argument("--ced-keep-fraction", type=float, default=0.20)
    parser.add_argument("--ced-min-width", type=int, default=1)
    parser.add_argument("--ced-drop-margin", type=float, default=0.15)
    parser.add_argument("--ced-context-weight", type=float, default=1.0)
    parser.add_argument("--ced-preserve-weight", type=float, default=0.25)
    parser.add_argument("--ced-positive-threshold", type=float, default=0.5)
    parser.add_argument("--diagnostic-classes", default="")
    parser.add_argument("--diagnostic-topk", type=int, default=8)
    parser.add_argument("--diagnostic-log-examples", type=int, default=2)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--audio-cache-mb", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--timm-pretrained", action="store_true", help="Allow timm pretrained weight loading when internet/cache is available")
    # ---- T2-A: noisy-student / self-distillation loop (see noisy_student.py) ----
    parser.add_argument("--noisy-student-rounds", type=int, default=0,
                        help="If >0, run the multi-iteration noisy-student loop for this many rounds (A6 sweet spot 3-4). 0 disables it (default supervised path).")
    parser.add_argument("--pseudo-parquet", default=None,
                        help="B1 pseudo-label parquet (filename, start_sec, <234 soft cols>, file_confidence). Soft soundscape targets for the student.")
    parser.add_argument("--ns-pseudo-alpha", type=float, default=0.7)
    parser.add_argument("--ns-pseudo-threshold", type=float, default=0.3)
    parser.add_argument("--ns-pseudo-power", type=float, default=2.0)
    parser.add_argument("--ns-pseudo-ratio-cap", type=float, default=0.4)
    parser.add_argument("--ns-final-tss", action="store_true", default=False,
                        help="Final round re-labels with the stricter TSS threshold (5th place pseudo_tss_th).")
    parser.add_argument("--ns-tss-threshold", type=float, default=0.7)
    parser.add_argument("--ns-noise-base", type=float, default=0.1)
    parser.add_argument("--ns-noise-step", type=float, default=0.1)
    parser.add_argument("--ns-noise-max", type=float, default=0.5)
    # ---- T2-B: bidirectional SSM / prototype head over Perch embeddings ----
    parser.add_argument("--ssm-head", action="store_true", default=False,
                        help="Use the bidirectional Mamba-2/SSD + prototype head over Perch embedding sequences instead of the EfficientNet mel pipeline.")
    parser.add_argument("--perch-embeddings", default=None,
                        help="Path to precomputed Perch embedding sequences (.npz with E:(F,T,D), filenames). Required when --ssm-head is set for the real run.")
    parser.add_argument("--ssm-embed-dim", type=int, default=1280)
    parser.add_argument("--ssm-proj-dim", type=int, default=256)
    parser.add_argument("--ssm-state-dim", type=int, default=16)
    parser.add_argument("--ssm-protos-per-class", type=int, default=1)
    parser.add_argument("--ssm-tau", type=float, default=16.0)
    parser.add_argument("--ssm-no-bidirectional", action="store_true", default=False)
    parser.add_argument("--ssm-no-mamba-kernel", action="store_true", default=False,
                        help="Force the pure-PyTorch scan even if mamba-ssm is importable (CPU smoke path).")
    parser.add_argument("--ssm-batch-size", type=int, default=256,
                        help="Minibatch size (files) for the SSM head; embeddings stream from pinned host memory to the GPU per batch.")
    return parser


def build_ssm_head_config(args):
    """Build an :class:`ssm_head.SSMHeadConfig` from parsed CLI args."""
    from ssm_head import SSMHeadConfig

    return SSMHeadConfig(
        embed_dim=int(args.ssm_embed_dim),
        proj_dim=int(args.ssm_proj_dim),
        state_dim=int(args.ssm_state_dim),
        num_classes=234,
        protos_per_class=int(args.ssm_protos_per_class),
        tau=float(args.ssm_tau),
        bidirectional=not bool(args.ssm_no_bidirectional),
        use_mamba_ssm=not bool(args.ssm_no_mamba_kernel),
    )


def build_noisy_student_config(args):
    """Build a :class:`noisy_student.NoisyStudentConfig` from parsed CLI args."""
    from noisy_student import NoisyStudentConfig

    return NoisyStudentConfig(
        n_rounds=int(args.noisy_student_rounds),
        pseudo_alpha=float(args.ns_pseudo_alpha),
        pseudo_threshold=float(args.ns_pseudo_threshold),
        pseudo_power=float(args.ns_pseudo_power),
        pseudo_ratio_cap=float(args.ns_pseudo_ratio_cap),
        final_tss=bool(args.ns_final_tss),
        tss_threshold=float(args.ns_tss_threshold),
        noise_base=float(args.ns_noise_base),
        noise_step=float(args.ns_noise_step),
        noise_max=float(args.ns_noise_max),
    )


# Pseudo-label parquet schema assumed for B1's precompute output (reconcile if it differs):
#   filename       : str   soundscape file stem or basename (e.g. BC2026_Train_0001_...)
#   start_sec      : float window start in seconds (0,5,...,55 for a 60 s file)
#   <234 columns>  : float soft scores in [0,1], one per BirdCLEF-2026 class code
#   file_confidence: float per-file confidence used to weight / subsample pseudo rows
PSEUDO_PARQUET_META_COLS = ("filename", "start_sec", "file_confidence")


def load_pseudo_parquet(path: str | None, classes: list[str]):
    """Load B1's pseudo-label parquet into (meta_df, soft_targets ndarray).

    Tolerates a missing ``file_confidence`` column (defaults to 1.0) and validates that
    all 234 class columns are present so a schema mismatch fails loudly rather than
    silently mislabeling. Returns ``(None, None)`` when ``path`` is falsy.
    """
    if not path:
        return None, None
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if parquet_path.suffix == ".parquet":
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(parquet_path)
    if "filename" not in df.columns:
        raise ValueError(f"{parquet_path}: missing 'filename' column (schema: {PSEUDO_PARQUET_META_COLS} + 234 class cols)")
    if "start_sec" not in df.columns:
        if "start_seconds" in df.columns:
            df = df.rename(columns={"start_seconds": "start_sec"})
        else:
            raise ValueError(f"{parquet_path}: missing 'start_sec' column")
    if "file_confidence" not in df.columns:
        # B1's per-window parquet carries per-window confidence as `win_confidence`;
        # use it so A6's pseudo-ratio gating sees real confidence, not a constant.
        df["file_confidence"] = df["win_confidence"] if "win_confidence" in df.columns else 1.0
    missing = [label for label in classes if label not in df.columns]
    if missing:
        raise ValueError(f"{parquet_path}: missing {len(missing)} class columns e.g. {missing[:5]}")
    soft = np.clip(np.nan_to_num(df[classes].to_numpy(dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    meta = df[list(PSEUDO_PARQUET_META_COLS)].reset_index(drop=True)
    return meta, soft


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_classes(competition_dir: Path) -> list[str]:
    sample = competition_dir / "sample_submission.csv"
    if sample.exists():
        return pd.read_csv(sample, nrows=1).columns[1:].astype(str).tolist()
    taxonomy = competition_dir / "taxonomy.csv"
    return pd.read_csv(taxonomy)["primary_label"].astype(str).tolist()


def build_train_audio_frame(competition_dir: Path, classes: list[str], max_files: int | None = None) -> pd.DataFrame:
    train_audio = competition_dir / "train_audio"
    rows = []
    for label in classes:
        label_dir = train_audio / label
        if not label_dir.exists():
            continue
        files = sorted(label_dir.glob("*.ogg")) + sorted(label_dir.glob("*.wav")) + sorted(label_dir.glob("*.flac"))
        if max_files is not None:
            files = files[:max_files]
        for path in files:
            rows.append({"path": str(path), "primary_label": label, "source": "train_audio"})
    if not rows:
        raise FileNotFoundError(f"no class audio files found under {train_audio}")
    return pd.DataFrame(rows)


ROW_ID_RE = re.compile(r"(.+)_(\d+)$")


def parse_soundscape_row_id(row_id: str) -> tuple[str, float]:
    match = ROW_ID_RE.match(str(row_id))
    if match is None:
        raise ValueError(f"invalid soundscape row_id: {row_id}")
    stem, end_seconds = match.groups()
    return f"{stem}.ogg", float(end_seconds) - 5.0


def parse_class_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def parse_time_seconds(value) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60.0 + float(seconds)
        return float(text)
    except ValueError as exc:
        raise ValueError(f"invalid time value: {value}") from exc


def split_labels(value) -> list[str]:
    if pd.isna(value):
        return []
    return [label.strip() for label in str(value).replace(",", ";").split(";") if label.strip()]


def build_pseudo_frame(pseudo_csv: str | None, competition_dir: Path, classes: list[str], weight: float) -> pd.DataFrame:
    if not pseudo_csv:
        return pd.DataFrame()
    pseudo_path = Path(pseudo_csv)
    if not pseudo_path.exists():
        raise FileNotFoundError(pseudo_path)
    df = pd.read_csv(pseudo_path)
    if "row_id" not in df.columns:
        raise ValueError(f"{pseudo_path} has no row_id column")
    if "primary_label" in df.columns:
        rows = []
        train_soundscapes = competition_dir / "train_soundscapes"
        class_set = set(classes)
        for _, row in df.iterrows():
            label = str(row["primary_label"])
            if label not in class_set:
                continue
            filename, start_seconds = parse_soundscape_row_id(str(row["row_id"]))
            path = train_soundscapes / filename
            if not path.exists():
                continue
            confidence = float(row.get("confidence", 1.0))
            if not np.isfinite(confidence):
                confidence = 1.0
            rows.append(
                {
                    "path": str(path),
                    "primary_label": label,
                    "source": "hard_pseudo_soundscape",
                    "start_seconds": start_seconds,
                    "sample_weight": float(weight) * float(np.clip(confidence, 0.0, 1.0)),
                }
            )
        return pd.DataFrame(rows)
    missing = [label for label in classes if label not in df.columns]
    if missing:
        raise ValueError(f"{pseudo_path} missing class columns: {missing[:5]}")

    rows = []
    train_soundscapes = competition_dir / "train_soundscapes"
    for _, row in df.iterrows():
        filename, start_seconds = parse_soundscape_row_id(str(row["row_id"]))
        path = train_soundscapes / filename
        if not path.exists():
            continue
        target = row[classes].to_numpy(dtype=np.float32)
        rows.append(
            {
                "path": str(path),
                "primary_label": "__pseudo__",
                "source": "pseudo_soundscape",
                "start_seconds": start_seconds,
                "target": target,
                "sample_weight": float(weight),
            }
        )
    return pd.DataFrame(rows)


def build_soundscape_label_frame(
    labels_csv: str | None,
    competition_dir: Path,
    classes: list[str],
    weight: float,
    focus_classes: set[str] | None = None,
    focus_weight: float = 1.0,
) -> pd.DataFrame:
    if not labels_csv or weight <= 0:
        return pd.DataFrame()
    labels_path = Path(labels_csv)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    df = pd.read_csv(labels_path)
    required = {"filename", "start", "end", "primary_label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{labels_path} missing columns: {missing}")
    df = df.drop_duplicates(subset=["filename", "start", "end", "primary_label"]).reset_index(drop=True)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    train_soundscapes = competition_dir / "train_soundscapes"
    focus_classes = focus_classes or set()
    rows = []
    for _, row in df.iterrows():
        path = train_soundscapes / str(row["filename"])
        if not path.exists():
            continue
        labels = [label for label in split_labels(row["primary_label"]) if label in class_to_idx]
        if not labels:
            continue
        target = np.zeros(len(classes), dtype=np.float32)
        for label in labels:
            target[class_to_idx[label]] = 1.0
        sample_weight = float(weight)
        if focus_classes.intersection(labels):
            sample_weight *= float(focus_weight)
        rows.append(
            {
                "path": str(path),
                "primary_label": ";".join(labels),
                "source": "labeled_soundscape",
                "start_seconds": parse_time_seconds(row["start"]),
                "target": target,
                "sample_weight": sample_weight,
            }
        )
    return pd.DataFrame(rows)


def build_focus_class_weights(classes: list[str], focus_classes: set[str] | None, focus_weight: float) -> np.ndarray:
    weights = np.ones(len(classes), dtype=np.float32)
    if not focus_classes or focus_weight <= 1.0:
        return weights
    for idx, label in enumerate(classes):
        if label in focus_classes:
            weights[idx] = float(focus_weight)
    return weights


def build_negative_loss_ignore_mask(classes: list[str], ignore_classes: set[str] | None) -> np.ndarray:
    mask = np.zeros(len(classes), dtype=bool)
    if not ignore_classes:
        return mask
    for idx, label in enumerate(classes):
        if label in ignore_classes:
            mask[idx] = True
    return mask


def apply_negative_loss_ignore(raw_loss, target, ignore_mask):
    import torch

    if ignore_mask is None or ignore_mask.numel() == 0 or not bool(ignore_mask.any()):
        return raw_loss
    class_mask = ignore_mask.to(device=raw_loss.device, dtype=torch.bool).unsqueeze(0)
    return raw_loss.masked_fill(class_mask & (target <= 0.0), 0.0)


def build_ced_time_mask(mel, keep_fraction: float = 0.20, min_width: int = 1):
    import torch

    if mel.ndim != 4:
        raise ValueError(f"CED mel tensor must be (B,C,F,T), got {tuple(mel.shape)}")
    score = mel.detach().float().abs().mean(dim=(1, 2))
    keep = int(round(float(keep_fraction) * int(score.shape[1])))
    keep = min(max(keep, 1), int(score.shape[1]))
    indices = torch.topk(score, k=keep, dim=1).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    width = max(int(min_width), 1)
    if width > 1:
        radius = width // 2
        widened = mask.clone()
        for shift in range(1, radius + 1):
            widened[:, shift:] |= mask[:, :-shift]
            widened[:, :-shift] |= mask[:, shift:]
        mask = widened
    return mask


def apply_ced_time_mask(mel, mask):
    import torch

    if mel.ndim != 4:
        raise ValueError(f"CED mel tensor must be (B,C,F,T), got {tuple(mel.shape)}")
    if mask.shape != (mel.shape[0], mel.shape[-1]):
        raise ValueError(f"CED mask shape {tuple(mask.shape)} does not match {(mel.shape[0], mel.shape[-1])}")
    fill = mel.mean(dim=-1, keepdim=True)
    expanded = mask.to(device=mel.device, dtype=torch.bool).unsqueeze(1).unsqueeze(2).expand_as(mel)
    return torch.where(expanded, fill.expand_as(mel), mel)


def counterfactual_evidence_loss(
    original_logits,
    dropped_logits,
    target,
    drop_margin: float = 0.15,
    context_weight: float = 1.0,
    preserve_weight: float = 0.25,
    positive_threshold: float = 0.5,
):
    import torch

    if original_logits.shape != dropped_logits.shape or original_logits.shape != target.shape:
        raise ValueError(
            "CED logits and target shapes must match; "
            f"got {tuple(original_logits.shape)}, {tuple(dropped_logits.shape)}, {tuple(target.shape)}"
        )
    target = target.float()
    positive = target > float(positive_threshold)
    negative = ~positive
    original_prob = torch.sigmoid(original_logits.float())
    dropped_prob = torch.sigmoid(dropped_logits.float())
    zero = original_logits.sum() * 0.0
    if bool(positive.any()):
        evidence_drop = torch.relu(dropped_prob[positive] - original_prob[positive] + float(drop_margin)).mean()
        context_suppression = torch.nn.functional.binary_cross_entropy_with_logits(
            dropped_logits[positive].float(),
            torch.zeros_like(dropped_logits[positive].float()),
        )
    else:
        evidence_drop = zero
        context_suppression = zero
    if bool(negative.any()):
        preserve = torch.nn.functional.mse_loss(dropped_prob[negative], original_prob.detach()[negative])
    else:
        preserve = zero
    total = evidence_drop + float(context_weight) * context_suppression + float(preserve_weight) * preserve
    return total, {
        "evidence_drop": evidence_drop,
        "context_suppression": context_suppression,
        "preserve": preserve,
    }


def build_yao_probe_frame(
    probe_csv: str | None,
    competition_dir: Path,
    classes: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not probe_csv:
        return pd.DataFrame(), np.empty((0, len(classes)), dtype=np.float32), []
    probe_path = Path(probe_csv)
    if not probe_path.exists():
        raise FileNotFoundError(probe_path)
    df = pd.read_csv(probe_path)
    if "row_id" not in df.columns:
        raise ValueError(f"{probe_path} has no row_id column")
    missing = [label for label in classes if label not in df.columns]
    if missing:
        raise ValueError(f"{probe_path} missing class columns: {missing[:5]}")
    rows = []
    keep_indices = []
    row_ids = []
    train_soundscapes = competition_dir / "train_soundscapes"
    for idx, row in df.iterrows():
        filename, start_seconds = parse_soundscape_row_id(str(row["row_id"]))
        path = train_soundscapes / filename
        if not path.exists():
            continue
        rows.append(
            {
                "path": str(path),
                "primary_label": "__probe__",
                "source": "yao_probe",
                "start_seconds": start_seconds,
                "target": np.zeros(len(classes), dtype=np.float32),
                "sample_weight": 1.0,
            }
        )
        keep_indices.append(idx)
        row_ids.append(str(row["row_id"]))
    target = df.loc[keep_indices, classes].to_numpy(dtype=np.float32) if keep_indices else np.empty((0, len(classes)), dtype=np.float32)
    target = np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    target = np.clip(target, 0.0, 1.0)
    return pd.DataFrame(rows), target, row_ids


def build_probe_true_label_sets(labels_csv: str | None, row_ids: list[str]) -> list[set[str]]:
    if not labels_csv or not row_ids or not Path(labels_csv).exists():
        return [set() for _ in row_ids]
    labels = pd.read_csv(labels_csv)
    lookup: dict[tuple[str, float], set[str]] = {}
    for _, row in labels.drop_duplicates(subset=["filename", "start", "end", "primary_label"]).iterrows():
        key = (str(row["filename"]), parse_time_seconds(row["end"]))
        lookup.setdefault(key, set()).update(split_labels(row["primary_label"]))
    true_sets = []
    for row_id in row_ids:
        filename, start_seconds = parse_soundscape_row_id(row_id)
        end_seconds = start_seconds + 5.0
        true_sets.append(set(lookup.get((filename, end_seconds), set())))
    return true_sets


def calibrate_probe_logits(logits, temperature: float, bias: float, keep_topk: int, prob_floor: float):
    import torch

    calibrated = torch.sigmoid((logits - float(bias)) / max(float(temperature), 1e-6))
    keep_topk = int(keep_topk)
    if keep_topk > 0 and keep_topk < calibrated.shape[1]:
        values, indices = torch.topk(calibrated, k=keep_topk, dim=1)
        sparse = torch.full_like(calibrated, float(prob_floor))
        sparse.scatter_(1, indices, values)
        calibrated = sparse
    if prob_floor > 0:
        calibrated = calibrated.clamp_min(float(prob_floor))
    return calibrated


def pairwise_rank_distillation_loss(logits, target, focus_indices=None, min_target_delta: float = 1e-4):
    import torch

    if logits.numel() == 0 or target.numel() == 0 or logits.shape[0] < 2:
        return logits.sum() * 0.0
    pred = torch.sigmoid(logits.float())
    target = target.float().to(pred.device)
    if focus_indices is not None:
        focus_indices = torch.as_tensor(focus_indices, dtype=torch.long, device=pred.device)
        if focus_indices.numel() > 0:
            pred = pred.index_select(1, focus_indices)
            target = target.index_select(1, focus_indices)
    row_i, row_j = torch.triu_indices(pred.shape[0], pred.shape[0], offset=1, device=pred.device)
    pred_delta = pred[row_i] - pred[row_j]
    target_delta = target[row_i] - target[row_j]
    valid = target_delta.abs() > float(min_target_delta)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    direction = target_delta.sign()
    weights = target_delta.abs().clamp_min(1e-3)
    losses = torch.nn.functional.softplus(-direction * pred_delta * 12.0)
    return (losses[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-6)


def focus_value_distillation_loss(logits, target, focus_indices=None, temperature: float = 1.0, bias: float = 0.0):
    import torch

    if logits.numel() == 0 or target.numel() == 0:
        return logits.sum() * 0.0
    adjusted_logits = (logits.float() - float(bias)) / max(float(temperature), 1e-6)
    pred = torch.sigmoid(adjusted_logits)
    target = target.float().to(pred.device)
    if focus_indices is not None:
        focus_indices = torch.as_tensor(focus_indices, dtype=torch.long, device=pred.device)
        if focus_indices.numel() > 0:
            pred = pred.index_select(1, focus_indices)
            target = target.index_select(1, focus_indices)
    mse = torch.nn.functional.mse_loss(pred, target)
    bce = torch.nn.functional.binary_cross_entropy(pred.clamp(1e-5, 1.0 - 1e-5), target.clamp(0.0, 1.0))
    return mse + 0.25 * bce


def compute_yao_selection_score(metrics: dict, val_loss: float | None = None) -> float:
    def finite(name: str, default: float = 0.0) -> float:
        value = float(metrics.get(name, default))
        return value if np.isfinite(value) else default

    rank_corr = finite("yao_rank_corr")
    value_corr = finite("yao_corr")
    topk_overlap = finite("topk_overlap")
    rank_mae = finite("yao_rank_mae", 1.0)
    active_gt05 = finite("active_gt05", 96.0)
    active_score = max(0.0, 1.0 - abs(active_gt05 - 96.0) / 96.0)
    val_penalty = 0.0
    if val_loss is not None and np.isfinite(float(val_loss)):
        val_penalty = min(max(float(val_loss), 0.0), 0.1)
    return (
        0.40 * rank_corr
        + 0.25 * value_corr
        + 0.15 * topk_overlap
        + 0.10 * max(0.0, 1.0 - rank_mae)
        + 0.10 * active_score
        - 0.05 * val_penalty
    )


def compute_yao_probe_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    classes: list[str],
    true_label_sets: list[set[str]] | None = None,
    focus_classes: set[str] | None = None,
    topk: int = 8,
    false_class: str = "517063",
) -> dict[str, float | int | str]:
    if pred.size == 0 or target.size == 0:
        return {}
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    flat_pred = pred.reshape(-1)
    flat_target = target.reshape(-1)
    if np.std(flat_pred) > 0 and np.std(flat_target) > 0:
        yao_corr = float(np.corrcoef(flat_pred, flat_target)[0, 1])
    else:
        yao_corr = float("nan")
    pred_rank = pd.DataFrame(pred).rank(axis=0, pct=True).to_numpy(dtype=np.float32)
    target_rank = pd.DataFrame(target).rank(axis=0, pct=True).to_numpy(dtype=np.float32)
    flat_pred_rank = pred_rank.reshape(-1)
    flat_target_rank = target_rank.reshape(-1)
    if np.std(flat_pred_rank) > 0 and np.std(flat_target_rank) > 0:
        yao_rank_corr = float(np.corrcoef(flat_pred_rank, flat_target_rank)[0, 1])
    else:
        yao_rank_corr = float("nan")
    topk = min(max(int(topk), 1), pred.shape[1])
    pred_top = np.argsort(-pred, axis=1)[:, :topk]
    target_top = np.argsort(-target, axis=1)[:, :topk]
    overlaps = [
        len(set(pred_top[idx]).intersection(set(target_top[idx]))) / float(topk)
        for idx in range(pred.shape[0])
    ]
    focus_classes = focus_classes or set()
    true_label_sets = true_label_sets or [set() for _ in range(pred.shape[0])]
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    recalls = []
    for idx, labels in enumerate(true_label_sets):
        focus_truth = [label for label in labels if label in focus_classes and label in class_to_idx]
        if not focus_truth:
            continue
        pred_labels = {classes[class_idx] for class_idx in pred_top[idx]}
        recalls.append(len(pred_labels.intersection(focus_truth)) / float(len(focus_truth)))
    false_mean = 0.0
    if false_class in class_to_idx:
        false_mean = float(pred[:, class_to_idx[false_class]].mean())
    first_row_top = [classes[idx] for idx in pred_top[0]]
    return {
        "yao_corr": yao_corr,
        "yao_rank_corr": yao_rank_corr,
        "yao_rank_mae": float(np.mean(np.abs(pred_rank - target_rank))),
        "yao_mae": float(np.mean(np.abs(pred - target))),
        "topk_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "sonotype_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "active_gt05": int((pred > 0.05).sum()),
        "false_517063_mean": false_mean,
        "topk_first": ",".join(first_row_top),
    }


def resolve_diagnostic_labels(
    classes: list[str],
    requested: set[str] | None,
    focus_classes: set[str] | None,
    false_class: str | None,
) -> list[str]:
    class_set = set(classes)
    labels = list(requested or [])
    if not labels:
        labels = sorted((focus_classes or set()).intersection(class_set))
        if false_class and false_class in class_set:
            labels.append(false_class)
    seen = set()
    resolved = []
    for label in labels:
        if label in class_set and label not in seen:
            resolved.append(label)
            seen.add(label)
    return resolved


def init_class_stats(n_classes: int) -> dict[str, np.ndarray | float]:
    return {
        "count": 0.0,
        "logit_sum": np.zeros(n_classes, dtype=np.float64),
        "prob_sum": np.zeros(n_classes, dtype=np.float64),
        "target_sum": np.zeros(n_classes, dtype=np.float64),
        "raw_loss_sum": np.zeros(n_classes, dtype=np.float64),
        "final_loss_sum": np.zeros(n_classes, dtype=np.float64),
        "pos_count": np.zeros(n_classes, dtype=np.float64),
        "ignored_neg_count": np.zeros(n_classes, dtype=np.float64),
    }


def update_class_stats(stats, logits, target, raw_loss, final_loss, diag_indices, ignore_mask=None) -> None:
    import torch

    if diag_indices is None or diag_indices.numel() == 0:
        return
    idx = diag_indices.to(device=logits.device, dtype=torch.long)
    with torch.no_grad():
        selected_logits = logits.float().index_select(1, idx)
        selected_target = target.float().index_select(1, idx)
        selected_raw = raw_loss.float().index_select(1, idx)
        selected_final = final_loss.float().index_select(1, idx)
        selected_prob = torch.sigmoid(selected_logits)
        stats["count"] += float(selected_logits.shape[0])
        stats["logit_sum"] += selected_logits.sum(dim=0).detach().cpu().numpy()
        stats["prob_sum"] += selected_prob.sum(dim=0).detach().cpu().numpy()
        stats["target_sum"] += selected_target.sum(dim=0).detach().cpu().numpy()
        stats["raw_loss_sum"] += selected_raw.sum(dim=0).detach().cpu().numpy()
        stats["final_loss_sum"] += selected_final.sum(dim=0).detach().cpu().numpy()
        stats["pos_count"] += (selected_target > 0).sum(dim=0).detach().cpu().numpy()
        if ignore_mask is not None and ignore_mask.numel() > 0:
            selected_ignore = ignore_mask.to(device=logits.device, dtype=torch.bool).index_select(0, idx).unsqueeze(0)
            ignored = selected_ignore & (selected_target <= 0)
            stats["ignored_neg_count"] += ignored.sum(dim=0).detach().cpu().numpy()


def format_class_stats(prefix: str, epoch: int, total_epochs: int, labels: list[str], stats) -> str:
    count = max(float(stats.get("count", 0.0)), 1.0)
    parts = []
    for pos, label in enumerate(labels):
        parts.append(
            f"{label}:logit={stats['logit_sum'][pos] / count:.3f},"
            f"prob={stats['prob_sum'][pos] / count:.5f},"
            f"target={stats['target_sum'][pos] / count:.5f},"
            f"pos={int(stats['pos_count'][pos])},"
            f"raw_loss={stats['raw_loss_sum'][pos] / count:.5f},"
            f"final_loss={stats['final_loss_sum'][pos] / count:.5f},"
            f"ignored_neg={int(stats['ignored_neg_count'][pos])}"
        )
    return f"diagnostic_{prefix} epoch={epoch}/{total_epochs} rows={int(count)} " + " | ".join(parts)


def frame_class_count_summary(frame: pd.DataFrame, classes: list[str], labels: list[str], name: str) -> str:
    if frame.empty or not labels:
        return f"diagnostic_frame name={name} rows={len(frame)}"
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    counts = {label: 0 for label in labels}
    weighted = {label: 0.0 for label in labels}
    source_counts = frame["source"].astype(str).value_counts().to_dict() if "source" in frame.columns else {}
    for _, row in frame.iterrows():
        sample_weight = float(row.get("sample_weight", 1.0))
        if not np.isfinite(sample_weight):
            sample_weight = 1.0
        target = row.get("target", None)
        if isinstance(target, np.ndarray):
            active = [label for label in labels if label in class_to_idx and target[class_to_idx[label]] > 0]
        else:
            active_set = set(split_labels(row.get("primary_label", "")))
            active = [label for label in labels if label in active_set]
        for label in active:
            counts[label] += 1
            weighted[label] += sample_weight
    details = " | ".join(f"{label}:pos={counts[label]},weighted={weighted[label]:.2f}" for label in labels)
    return f"diagnostic_frame name={name} rows={len(frame)} sources={source_counts} {details}"


def format_probe_diagnostics(
    epoch: int,
    total_epochs: int,
    raw_logits: np.ndarray,
    sparse_pred: np.ndarray,
    target: np.ndarray,
    row_ids: list[str],
    classes: list[str],
    labels: list[str],
    temperature: float,
    bias: float,
    topk: int,
    log_examples: int,
) -> list[str]:
    if raw_logits.size == 0 or sparse_pred.size == 0 or not labels:
        return []
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx = [class_to_idx[label] for label in labels if label in class_to_idx]
    idx_labels = [label for label in labels if label in class_to_idx]
    dense = 1.0 / (1.0 + np.exp(-((raw_logits.astype(np.float32) - float(bias)) / max(float(temperature), 1e-6))))
    row_sum = sparse_pred.sum(axis=1)
    sparse_top = np.argsort(-sparse_pred, axis=1)[:, : max(1, min(int(topk), sparse_pred.shape[1]))]
    dense_rank = np.argsort(np.argsort(-dense, axis=1), axis=1) + 1
    target_rank = np.argsort(np.argsort(-target, axis=1), axis=1) + 1
    lines = [
        "diagnostic_probe_shape "
        f"epoch={epoch}/{total_epochs} rows={sparse_pred.shape[0]} "
        f"sparse_mean={float(sparse_pred.mean()):.6f} sparse_std={float(sparse_pred.std()):.6f} "
        f"dense_mean={float(dense.mean()):.6f} dense_std={float(dense.std()):.6f} "
        f"active_gt05={int((sparse_pred > 0.05).sum())} "
        f"row_sum_mean={float(row_sum.mean()):.6f} row_sum_std={float(row_sum.std()):.6f}"
    ]
    parts = []
    for label, class_idx in zip(idx_labels, idx):
        parts.append(
            f"{label}:raw={float(raw_logits[:, class_idx].mean()):.3f},"
            f"dense={float(dense[:, class_idx].mean()):.5f},"
            f"sparse={float(sparse_pred[:, class_idx].mean()):.5f},"
            f"target={float(target[:, class_idx].mean()):.5f},"
            f"active={int((sparse_pred[:, class_idx] > 0.05).sum())},"
            f"dense_rank={float(dense_rank[:, class_idx].mean()):.2f},"
            f"target_rank={float(target_rank[:, class_idx].mean()):.2f}"
        )
    lines.append(f"diagnostic_probe_classes epoch={epoch}/{total_epochs} " + " | ".join(parts))
    for row_idx in range(min(max(int(log_examples), 0), sparse_pred.shape[0])):
        top_labels = [
            f"{classes[class_idx]}:{float(sparse_pred[row_idx, class_idx]):.5f}"
            for class_idx in sparse_top[row_idx]
        ]
        lines.append(
            f"diagnostic_probe_row epoch={epoch}/{total_epochs} "
            f"row={row_ids[row_idx] if row_idx < len(row_ids) else row_idx} "
            f"top={','.join(top_labels)}"
        )
    return lines


def probability_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, dtype=np.float32), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float32), eps, 1.0)
    return float(np.mean(p * (np.log(p) - np.log(q)) + (1.0 - p) * (np.log1p(-p) - np.log1p(-q))))


def format_base_preservation_diagnostics(
    epoch: int,
    total_epochs: int,
    base_pred: np.ndarray,
    current_pred: np.ndarray,
    classes: list[str],
    labels: list[str],
    topk: int,
) -> list[str]:
    if base_pred.size == 0 or current_pred.size == 0:
        return []
    topk = max(1, min(int(topk), base_pred.shape[1], current_pred.shape[1]))
    base_top = np.argsort(-base_pred, axis=1)[:, :topk]
    current_top = np.argsort(-current_pred, axis=1)[:, :topk]
    overlap = np.mean([
        len(set(base_top[row].tolist()).intersection(set(current_top[row].tolist()))) / float(topk)
        for row in range(base_pred.shape[0])
    ])
    base_top1 = base_top[:, 0]
    current_top1 = current_top[:, 0]
    delta = current_pred - base_pred
    lines = [
        "diagnostic_base_preservation "
        f"epoch={epoch}/{total_epochs} "
        f"kl={probability_kl(base_pred, current_pred):.6f} "
        f"mae={float(np.mean(np.abs(delta))):.6f} "
        f"max_abs_delta={float(np.max(np.abs(delta))):.6f} "
        f"top1_flip_rate={float(np.mean(base_top1 != current_top1)):.4f} "
        f"top{topk}_overlap={float(overlap):.4f}"
    ]
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    parts = []
    for label in labels:
        class_idx = class_to_idx.get(label)
        if class_idx is None:
            continue
        class_delta = delta[:, class_idx]
        parts.append(
            f"{label}:base={float(base_pred[:, class_idx].mean()):.5f},"
            f"current={float(current_pred[:, class_idx].mean()):.5f},"
            f"delta={float(class_delta.mean()):.5f},"
            f"max_abs_delta={float(np.max(np.abs(class_delta))):.5f}"
        )
    if parts:
        lines.append(f"diagnostic_base_preservation_classes epoch={epoch}/{total_epochs} " + " | ".join(parts))
    return lines


def add_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["fold"] = -1
    rng = np.random.default_rng(seed)
    n_folds = max(int(n_folds), 2)
    for _label, group in frame.groupby(frame["primary_label"].astype(str), sort=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        if len(indices) == 1:
            frame.loc[indices, "fold"] = 0
            continue
        for offset, idx in enumerate(indices):
            frame.loc[idx, "fold"] = offset % min(n_folds, len(indices))
    return frame


def choose_validation_fold(frame: pd.DataFrame, requested_fold: int) -> int:
    available = sorted(int(fold) for fold in frame["fold"].dropna().unique())
    if requested_fold in available:
        return requested_fold
    if not available:
        return 0
    return available[0]


def split_train_val(frame: pd.DataFrame, requested_fold: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    validation_fold = choose_validation_fold(frame, requested_fold)
    val_frame = frame[frame["fold"] == validation_fold].reset_index(drop=True)
    train_frame = frame[frame["fold"] != validation_fold].reset_index(drop=True)
    if train_frame.empty:
        train_frame = frame.reset_index(drop=True)
    if val_frame.empty:
        val_frame = frame.reset_index(drop=True)
    return train_frame, val_frame, validation_fold


class AudioDataset:
    def __init__(self, frame, classes, args, training: bool):
        self.frame = frame.reset_index(drop=True)
        self.classes = classes
        self.class_to_idx = {label: idx for idx, label in enumerate(classes)}
        self.args = args
        self.training = training
        self.n_samples = int(round(args.sr * args.window_seconds))
        self.audio_cache_max_bytes = max(int(args.audio_cache_mb), 0) * 1024 * 1024
        self.audio_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.audio_cache_bytes = 0

    def __len__(self):
        return len(self.frame)

    def _load_audio(self, path: Path):
        cache_key = str(path)
        cached = self.audio_cache.get(cache_key)
        if cached is not None:
            self.audio_cache.move_to_end(cache_key)
            return cached
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.args.sr:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.args.sr).astype(np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
        if self.audio_cache_max_bytes > 0 and audio.nbytes <= self.audio_cache_max_bytes:
            self.audio_cache[cache_key] = audio
            self.audio_cache_bytes += audio.nbytes
            while self.audio_cache_bytes > self.audio_cache_max_bytes and self.audio_cache:
                _, evicted = self.audio_cache.popitem(last=False)
                self.audio_cache_bytes -= evicted.nbytes
        return audio

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        audio = self._load_audio(Path(row["path"]))
        if len(audio) >= self.n_samples:
            max_start = len(audio) - self.n_samples
            if "start_seconds" in row and pd.notna(row.get("start_seconds")):
                start = int(round(float(row["start_seconds"]) * self.args.sr))
                start = min(max(start, 0), max_start)
            else:
                start = random.randint(0, max_start) if self.training and max_start > 0 else max_start // 2
            clip = audio[start : start + self.n_samples]
        else:
            clip = np.zeros(self.n_samples, dtype=np.float32)
            clip[: len(audio)] = audio
        if "target" in row and isinstance(row.get("target"), np.ndarray):
            target = row["target"].astype(np.float32)
            target = np.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
            target = np.clip(target, 0.0, 1.0)
        else:
            target = np.zeros(len(self.classes), dtype=np.float32)
            label = str(row["primary_label"])
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0
        weight = float(row.get("sample_weight", 1.0))
        if not np.isfinite(weight):
            weight = 1.0
        return clip, target, weight


def collate_audio(batch):
    import torch

    audio = torch.from_numpy(np.stack([item[0] for item in batch]).astype(np.float32))
    target = torch.from_numpy(np.stack([item[1] for item in batch]).astype(np.float32))
    weight = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return audio, target, weight


class MelFrontend:
    def __init__(self, args, device):
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            f_min=args.fmin,
            f_max=args.fmax,
            power=2.0,
        ).to(device)

    def __call__(self, wave):
        import torch

        wave = torch.nan_to_num(wave.float(), nan=0.0, posinf=0.0, neginf=0.0)
        wave = wave - wave.mean(dim=1, keepdim=True)
        mel = self.mel(wave)
        mel = mel.clamp_min(1e-6).log()
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
        return ((mel - mean) / std).unsqueeze(1)


def model_name_candidates(model_name: str) -> list[str]:
    candidates = [model_name]
    if "." in model_name:
        candidates.append(model_name.split(".", 1)[0])
    return list(dict.fromkeys(candidates))


def build_model(
    model_name: str,
    num_classes: int,
    pretrained_checkpoint: str | None = None,
    timm_pretrained: bool = False,
):
    import timm
    import torch

    last_error = None
    for candidate_name in model_name_candidates(model_name):
        try:
            model = timm.create_model(
                candidate_name,
                pretrained=timm_pretrained and pretrained_checkpoint is None,
                in_chans=1,
                num_classes=num_classes,
            )
            model_name = candidate_name
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        raise last_error if last_error is not None else RuntimeError(f"could not create model {model_name}")
    if pretrained_checkpoint is None:
        if timm_pretrained:
            print("No BirdCLEF 2025pre checkpoint supplied; initializing backbone from timm pretrained weights.")
        else:
            print("No BirdCLEF 2025pre or timm pretrained weights supplied; training starts from random backbone weights.")
    if pretrained_checkpoint:
        ckpt = torch.load(pretrained_checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=False)
    return model


def configure_trainable_scope(model, scope: str) -> tuple[int, int]:
    if scope == "all":
        total = sum(param.numel() for param in model.parameters())
        return total, total
    head_markers = ("classifier", "fc", "head")
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        is_trainable = any(marker in name.lower() for marker in head_markers)
        param.requires_grad = is_trainable
        if is_trainable:
            trainable += param.numel()
    return trainable, total


def choose_torch_device():
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        device = torch.device("cuda")
        _ = torch.zeros(1, device=device) + 1
        torch.cuda.synchronize()
        return device
    except Exception as exc:
        print(f"CUDA unavailable for this torch/image/GPU combination; falling back to CPU: {exc}")
        return torch.device("cpu")


def load_perch_embeddings(path: str | None):
    """Load precomputed Perch embedding sequences.

    Expects an ``.npz`` with arrays ``E`` of shape ``(F, T, D)`` (F files, T windows,
    D embedding dim) and ``filenames`` of length F. Returns ``(E_tensor, filenames)``
    or ``(None, None)`` if ``path`` is falsy.
    """
    import numpy as np
    import torch

    if not path:
        return None, None
    data = np.load(path, allow_pickle=True)
    if "E" not in data:
        raise ValueError(f"{path}: missing 'E' array (expected shape (F,T,D))")
    embeddings = torch.from_numpy(np.asarray(data["E"], dtype=np.float32))
    filenames = [str(name) for name in data["filenames"]] if "filenames" in data else None
    return embeddings, filenames


def load_soundscape_label_targets(path, filenames, classes, n_windows, window_seconds=5.0):
    """Build host-side per-window hard targets (n_files,n_windows,C) from a labeled
    train_soundscapes CSV (columns: filename, start_sec/start_seconds, then class codes,
    or a long primary_label form). Returns an all-zero tensor when ``path`` is absent so
    the supervised term is simply skipped (round-0 becomes teacher-free no-op until
    pseudo-labels arrive)."""
    import numpy as np
    import torch

    targets = torch.zeros(len(filenames or []), n_windows, len(classes))
    if not path or filenames is None or not Path(path).exists():
        return targets
    df = pd.read_csv(path)
    idx_by_name = {name: i for i, name in enumerate(filenames)}
    cls_idx = {c: j for j, c in enumerate(classes)}
    start_col = "start_sec" if "start_sec" in df.columns else (
        "start_seconds" if "start_seconds" in df.columns else None
    )
    wide_cls = [c for c in df.columns if c in cls_idx]
    for row in df.itertuples(index=False):
        d = row._asdict()
        fn = str(d.get("filename", ""))
        fi = idx_by_name.get(fn)
        if fi is None:
            continue
        wi = int(round(float(d.get(start_col, 0.0)) / max(window_seconds, 1e-6))) if start_col else 0
        if not (0 <= wi < n_windows):
            continue
        if wide_cls:
            for c in wide_cls:
                v = float(d.get(c, 0.0) or 0.0)
                if v > 0:
                    targets[fi, wi, cls_idx[c]] = 1.0
        else:
            lbl = str(d.get("primary_label", d.get("label", "")))
            if lbl in cls_idx:
                targets[fi, wi, cls_idx[lbl]] = 1.0
    return targets


def compute_class_mean_embeddings(embeddings, labeled_targets, num_classes):
    """Mean embedding per class over labeled windows -> (C, D) for prototype seeding.
    Classes with no labeled support stay all-zero (seed_prototypes leaves them at random
    init, the intended behaviour for the 28 zero-train classes)."""
    import torch

    n_files, n_windows, _ = embeddings.shape
    flat_E = embeddings.reshape(n_files * n_windows, -1)
    flat_y = labeled_targets.reshape(n_files * n_windows, num_classes)
    if float(flat_y.sum()) <= 0:
        return None
    # (C, NW) @ (NW, D) summed, normalized by per-class window count
    counts = flat_y.sum(dim=0).clamp_min(1.0)               # (C,)
    summed = flat_y.t().to(flat_E.dtype) @ flat_E           # (C, D)
    means = summed / counts.unsqueeze(1)
    means[flat_y.sum(dim=0) <= 0] = 0.0
    return means


def train_ssm_noisy_student(args: argparse.Namespace, classes: list[str], output_dir: Path) -> Path:
    """T2-B SSM head training, optionally inside the T2-A noisy-student loop.

    Operates on precomputed Perch embedding sequences (``--perch-embeddings``). Labeled
    targets come from the soundscape-label CSV / pseudo parquet aligned by filename; the
    Yao-probe selection score (LB-correlated) chooses the best round. When
    ``--noisy-student-rounds <= 0`` it trains a single supervised round.

    NOTE: the real run requires ``--perch-embeddings``; this function raises if absent so
    a misconfigured GPU launch fails fast rather than training on noise. The CPU smoke
    path is :mod:`run_ssm_smoke`, which synthesizes embeddings in-process.
    """
    import numpy as np
    import torch

    from ssm_head import build_ssm_head, summarize_head, param_count, flops_per_file
    from noisy_student import (
        NoiseSchedule,
        apply_embedding_noise,
        ensemble_teacher_probs,
        make_soft_targets,
        mixup_embeddings,
        run_noisy_student,
    )

    if not args.perch_embeddings:
        raise ValueError(
            "--ssm-head requires --perch-embeddings (precomputed Perch sequences .npz). "
            "For a dependency-free CPU check run run_ssm_smoke.py instead."
        )
    device = choose_torch_device()

    embeddings, filenames = load_perch_embeddings(args.perch_embeddings)
    n_files, n_windows, embed_dim = embeddings.shape

    head_config = build_ssm_head_config(args)
    head_config.num_classes = len(classes)
    # The cached Perch embeddings define D; adapt the head so the launch never silently
    # trains a (D=1280) head against (D=1536) Perch-v2 embeddings or vice versa.
    if int(head_config.embed_dim) != int(embed_dim):
        print(
            f"ssm_head: adapting embed_dim {head_config.embed_dim} -> {embed_dim} "
            f"(from cached Perch embeddings {args.perch_embeddings})"
        )
        head_config.embed_dim = int(embed_dim)
    print(summarize_head(head_config, n_windows=n_windows))
    p, f = param_count(head_config), flops_per_file(head_config, n_windows=n_windows)
    print(f"ssm_head params_total={p['total']} MFLOPs_per_file={f['total'] / 1e6:.3f}")

    # GPU-efficiency config. The SSM head is tiny and frozen-embedding-fed, so the
    # bottleneck is moving the (F,T,D) tensor and running teacher+student forwards.
    # Keep the full embedding tensor pinned on the host and stream minibatches to the
    # GPU (non_blocking) so we never hold 2x the tensor (teacher + student) on-device,
    # and run the forward/backward under autocast (bf16 if the GPU supports it).
    use_cuda = device.type == "cuda"
    amp_enabled = bool(getattr(args, "amp", False)) and use_cuda
    amp_dtype = torch.float32
    if amp_enabled:
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(
            f"Using CUDA device: {torch.cuda.get_device_name(0)}; "
            f"amp={amp_enabled} amp_dtype={amp_dtype} n_files={n_files} "
            f"n_windows={n_windows} embed_dim={embed_dim}"
        )
    else:
        print(f"Using CPU; amp=False n_files={n_files} embed_dim={embed_dim}")
    ssm_batch = max(int(getattr(args, "ssm_batch_size", 256) or 256), 1)
    if use_cuda:
        embeddings = embeddings.pin_memory()

    # labeled per-window targets from the labeled-soundscape CSV (aligned by filename);
    # pseudo soft targets from B1's parquet. Rows without labels stay all-zero (treated
    # as unlabeled by the soft-target mixer). Targets live on the HOST and are streamed
    # to the GPU per minibatch (matching the pinned embedding tensor).
    labeled_targets = load_soundscape_label_targets(
        args.soundscape_labels_csv, filenames, classes, n_windows,
        window_seconds=float(args.window_seconds),
    )
    n_labeled_windows = int((labeled_targets.sum(dim=-1) > 0).sum().item())
    print(f"ssm labeled windows with >=1 positive: {n_labeled_windows}/{n_files * n_windows}")
    pseudo_meta, pseudo_soft = load_pseudo_parquet(args.pseudo_parquet, classes)
    if pseudo_meta is not None and filenames is not None:
        idx_by_name = {name: i for i, name in enumerate(filenames)}
        soft_seq = torch.zeros(n_files, n_windows, len(classes))
        for (filename, start_sec, _conf), row in zip(
            pseudo_meta.itertuples(index=False, name=None), pseudo_soft
        ):
            fi = idx_by_name.get(str(filename))
            if fi is None:
                continue
            wi = int(round(float(start_sec) / max(float(args.window_seconds), 1e-6)))
            if 0 <= wi < n_windows:
                soft_seq[fi, wi] = torch.from_numpy(row)
        pseudo_targets = soft_seq
    else:
        pseudo_targets = torch.zeros_like(labeled_targets)

    # Seed class prototypes from mean labeled embeddings so the cosine head starts near
    # the data manifold (zero-train classes keep random init -> lifted by sonotype mirror).
    class_mean = compute_class_mean_embeddings(embeddings, labeled_targets, len(classes))

    # Yao-probe selection: reuse the existing probe CSV machinery for the LB-correlated
    # score. The probe targets are per-window soft scores over the probe files.
    probe_target = pseudo_targets.mean(dim=1).detach().cpu().numpy()

    ns_cfg = build_noisy_student_config(args)
    rounds = max(int(args.noisy_student_rounds), 1)
    ns_cfg.n_rounds = rounds

    def _iter_batches(shuffle):
        order = torch.randperm(n_files) if shuffle else torch.arange(n_files)
        for s in range(0, n_files, ssm_batch):
            yield order[s : s + ssm_batch]

    def init_student(_round_idx):
        head = build_ssm_head(head_config).to(device)
        if class_mean is not None:
            head.seed_prototypes_from_embeddings(class_mean.to(device))
        return head

    def train_one_round(student, teacher, noise: NoiseSchedule, _round_idx, use_tss):
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
        # Cache the (frozen) teacher soft targets ONCE per round instead of recomputing a
        # full-set teacher forward every epoch (the teacher does not change within a round).
        soft_cache = None
        if teacher is not None:
            teacher.eval()
            soft_chunks = []
            with torch.no_grad():
                for idx in _iter_batches(shuffle=False):
                    eb = embeddings[idx].to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                        tlog = teacher(eb)
                    probs = ensemble_teacher_probs([tlog.float()])
                    soft = make_soft_targets(
                        probs, pseudo_targets[idx].to(device), ns_cfg, use_tss=use_tss
                    )
                    soft_chunks.append(soft.detach().cpu())
            soft_cache = torch.cat(soft_chunks, dim=0)
        student.train()
        for _epoch in range(max(int(args.epochs), 1)):
            for idx in _iter_batches(shuffle=True):
                opt.zero_grad(set_to_none=True)
                eb = embeddings[idx].to(device, non_blocking=True)
                yb_host = labeled_targets[idx].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    xb = apply_embedding_noise(eb, noise)
                    xb, yb = mixup_embeddings(xb, yb_host, noise)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        student(xb), yb
                    )
                    if soft_cache is not None:
                        soft = soft_cache[idx].to(device, non_blocking=True)
                        xu = apply_embedding_noise(eb, noise)
                        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                            student(xu), soft
                        )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite SSM noisy-student loss")
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            print(f"  ssm round epoch {_epoch + 1}/{max(int(args.epochs), 1)} loss={float(loss):.4f}", flush=True)
        return student

    def evaluate(student):
        student.eval()
        preds = []
        with torch.no_grad():
            for idx in _iter_batches(shuffle=False):
                eb = embeddings[idx].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    out = student(eb)
                preds.append(out.float().sigmoid().mean(dim=1).detach().cpu().numpy())
        import numpy as np
        pred = np.concatenate(preds, axis=0)
        return compute_yao_probe_metrics(pred, probe_target, classes, topk=args.yao_probe_topk)

    best = run_noisy_student(
        config=ns_cfg,
        init_student=init_student,
        train_one_round=train_one_round,
        evaluate=evaluate,
        selection_score_fn=lambda m: compute_yao_selection_score(m, val_loss=None),
        teacher_init=None,
    )
    best_path = output_dir / "g124_ssm_fold1_fp16.pt"
    torch.save(
        {
            "state_dict": best.state_dict,
            "classes": classes,
            "config": {
                "head": "BiSSDProtoHead",
                "ssm_head_config": head_config.__dict__,
                "noisy_student": ns_cfg.__dict__,
                "best_round": best.round_idx,
                "is_tss": best.is_tss,
                "selection_score": best.selection_score,
                "param_count": p,
            },
        },
        best_path,
    )
    print(f"saved {best_path} best_round={best.round_idx + 1} selection_score={best.selection_score:.5f}")
    return best_path


def train(args: argparse.Namespace) -> Path:
    import torch
    from torch.utils.data import DataLoader

    seed_everything(args.seed)
    competition_dir = Path(args.competition_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = load_classes(competition_dir)
    if args.ssm_head:
        # T2-B: bidirectional SSM / prototype head over Perch embedding sequences,
        # optionally wrapped in the T2-A noisy-student loop. Distinct from the
        # EfficientNet mel pipeline below; returns its own checkpoint.
        return train_ssm_noisy_student(args, classes, output_dir)
    frame = build_train_audio_frame(competition_dir, classes, max_files=args.max_train_files)
    frame = add_folds(frame, args.n_folds, args.seed)
    train_frame, val_frame, validation_fold = split_train_val(frame, args.fold)
    focus_classes = parse_class_list(args.focus_classes)
    pseudo_frame = build_pseudo_frame(args.pseudo_csv, competition_dir, classes, args.pseudo_weight)
    if not pseudo_frame.empty:
        train_frame = pd.concat([train_frame, pseudo_frame], ignore_index=True)
    soundscape_label_frame = build_soundscape_label_frame(
        args.soundscape_labels_csv,
        competition_dir,
        classes,
        args.soundscape_label_weight,
        focus_classes=focus_classes,
        focus_weight=args.focus_class_weight,
    )
    if not soundscape_label_frame.empty:
        train_frame = pd.concat([train_frame, soundscape_label_frame], ignore_index=True)
    probe_frame, yao_probe_target, yao_probe_row_ids = build_yao_probe_frame(args.yao_probe_csv, competition_dir, classes)
    yao_probe_true_sets = build_probe_true_label_sets(args.soundscape_labels_csv, yao_probe_row_ids)
    print(
        f"Stage={args.stage} train={len(train_frame)} val={len(val_frame)} "
        f"pseudo={len(pseudo_frame)} labeled_soundscape={len(soundscape_label_frame)} "
        f"yao_probe={len(probe_frame)} classes={len(classes)}"
    )

    device = choose_torch_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {gpu_name}; amp={args.amp}; audio_cache_mb_per_worker={args.audio_cache_mb}")
    else:
        print(f"Using CPU; amp=False; audio_cache_mb_per_worker={args.audio_cache_mb}")
    model = build_model(args.model_name, len(classes), args.pretrained_checkpoint, args.timm_pretrained).to(device)
    model = model.to(memory_format=torch.channels_last)
    frontend = MelFrontend(args, device)
    trainable_params, total_params = configure_trainable_scope(model, args.trainable_scope)
    print(
        "trainable_scope "
        f"scope={args.trainable_scope} trainable_params={trainable_params} total_params={total_params} "
        f"fraction={trainable_params / max(total_params, 1):.6f}"
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    class_weight = torch.tensor(
        build_focus_class_weights(classes, focus_classes, args.focus_class_weight),
        dtype=torch.float32,
        device=device,
    )
    loss_ignore_negative_classes = parse_class_list(args.loss_ignore_negative_classes)
    negative_loss_ignore_mask = torch.tensor(
        build_negative_loss_ignore_mask(classes, loss_ignore_negative_classes),
        dtype=torch.bool,
        device=device,
    )
    distill_classes = parse_class_list(args.yao_distill_classes)
    focus_probe_labels = set(distill_classes) if distill_classes else set(focus_classes)
    if args.yao_probe_false_class:
        focus_probe_labels.add(str(args.yao_probe_false_class))
    focus_probe_indices = torch.tensor(
        [idx for idx, label in enumerate(classes) if label in focus_probe_labels],
        dtype=torch.long,
        device=device,
    )
    diagnostic_labels = resolve_diagnostic_labels(
        classes,
        parse_class_list(args.diagnostic_classes),
        focus_probe_labels,
        args.yao_probe_false_class,
    )
    diagnostic_indices = torch.tensor(
        [classes.index(label) for label in diagnostic_labels],
        dtype=torch.long,
        device=device,
    )
    print(
        "diagnostic_config "
        f"classes={','.join(diagnostic_labels)} "
        f"loss_ignore_negative_classes={','.join(sorted(loss_ignore_negative_classes)) or 'none'} "
        f"probe_topk={args.yao_probe_topk} diagnostic_topk={args.diagnostic_topk}"
    )
    if diagnostic_labels:
        print(frame_class_count_summary(train_frame, classes, diagnostic_labels, "train"))
        print(frame_class_count_summary(val_frame, classes, diagnostic_labels, "val"))
        if yao_probe_target.size:
            probe_frame_counts = pd.DataFrame(
                {
                    "target": [row.astype(np.float32) for row in yao_probe_target],
                    "sample_weight": 1.0,
                    "source": "yao_probe_target",
                }
            )
            print(frame_class_count_summary(probe_frame_counts, classes, diagnostic_labels, "yao_probe_target"))

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        AudioDataset(train_frame, classes, args, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_audio,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        AudioDataset(val_frame, classes, args, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_audio,
        **loader_kwargs,
    )
    probe_loader = None
    if not probe_frame.empty:
        probe_loader = DataLoader(
            AudioDataset(probe_frame, classes, args, training=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_audio,
            **loader_kwargs,
        )

    base_probe_pred = np.empty((0, len(classes)), dtype=np.float32)
    base_probe_dense = np.empty((0, len(classes)), dtype=np.float32)
    if probe_loader is not None:
        model.eval()
        base_probe_preds = []
        base_probe_dense_preds = []
        with torch.inference_mode():
            for wave, _target, _weight in probe_loader:
                wave = wave.to(device, non_blocking=True)
                x = frontend(wave).to(memory_format=torch.channels_last)
                logits = model(x)
                dense_probs = torch.sigmoid(
                    (logits.float() - float(args.yao_probe_bias)) / max(float(args.yao_probe_temperature), 1e-6)
                )
                probs = calibrate_probe_logits(
                    logits,
                    temperature=args.yao_probe_temperature,
                    bias=args.yao_probe_bias,
                    keep_topk=args.yao_probe_topk,
                    prob_floor=args.yao_probe_prob_floor,
                )
                base_probe_preds.append(probs.detach().cpu().numpy().astype(np.float32))
                base_probe_dense_preds.append(dense_probs.detach().cpu().numpy().astype(np.float32))
        if base_probe_preds:
            base_probe_pred = np.concatenate(base_probe_preds, axis=0)
            base_probe_dense = np.concatenate(base_probe_dense_preds, axis=0)
            base_metrics = compute_yao_probe_metrics(
                base_probe_pred,
                yao_probe_target,
                classes,
                true_label_sets=yao_probe_true_sets,
                focus_classes=focus_classes,
                topk=args.yao_probe_topk,
                false_class=args.yao_probe_false_class,
            )
            if base_metrics:
                print(
                    "diagnostic_base_probe "
                    f"corr={float(base_metrics['yao_corr']):.4f} "
                    f"rank_corr={float(base_metrics['yao_rank_corr']):.4f} "
                    f"rank_mae={float(base_metrics['yao_rank_mae']):.5f} "
                    f"mae={float(base_metrics['yao_mae']):.5f} "
                    f"top{args.yao_probe_topk}_overlap={float(base_metrics['topk_overlap']):.4f} "
                    f"sonotype_recall={float(base_metrics['sonotype_recall']):.4f} "
                    f"active_gt05={int(base_metrics['active_gt05'])} "
                    f"false_{args.yao_probe_false_class}_mean={float(base_metrics['false_517063_mean']):.5f} "
                    f"top{args.yao_probe_topk}_first={base_metrics['topk_first']}"
                )

    best_score = -float("inf")
    best_path = output_dir / "g124_fold1_fp16.pt"
    total_epochs = int(args.total_epochs) if args.total_epochs is not None else int(args.start_epoch) + int(args.epochs)
    for epoch in range(args.epochs):
        display_epoch = int(args.start_epoch) + epoch + 1
        t0 = time.time()
        model.train()
        train_losses = []
        ced_epoch_parts = []
        train_diag_stats = init_class_stats(len(diagnostic_labels))
        optimizer_steps = 0
        for wave, target, weight in train_loader:
            wave = wave.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            x = frontend(wave).to(memory_format=torch.channels_last)
            if not torch.isfinite(x).all():
                raise FloatingPointError("non-finite mel features detected")
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
                masked_loss = apply_negative_loss_ignore(raw_loss, target, negative_loss_ignore_mask)
                weighted_loss = masked_loss * class_weight.unsqueeze(0)
                loss = (weighted_loss.mean(dim=1) * weight).mean()
                if args.ced_loss_weight > 0:
                    ced_mask = build_ced_time_mask(
                        x,
                        keep_fraction=args.ced_keep_fraction,
                        min_width=args.ced_min_width,
                    )
                    x_drop = apply_ced_time_mask(x, ced_mask).to(memory_format=torch.channels_last)
                    dropped_logits = model(x_drop)
                    ced_loss, ced_parts = counterfactual_evidence_loss(
                        logits,
                        dropped_logits,
                        target,
                        drop_margin=args.ced_drop_margin,
                        context_weight=args.ced_context_weight,
                        preserve_weight=args.ced_preserve_weight,
                        positive_threshold=args.ced_positive_threshold,
                    )
                    loss = loss + float(args.ced_loss_weight) * ced_loss
                    ced_epoch_parts.append(
                        {
                            "loss": float(ced_loss.detach().cpu()),
                            "evidence_drop": float(ced_parts["evidence_drop"].detach().cpu()),
                            "context_suppression": float(ced_parts["context_suppression"].detach().cpu()),
                            "preserve": float(ced_parts["preserve"].detach().cpu()),
                            "active_time_frac": float(ced_mask.float().mean().detach().cpu()),
                        }
                    )
                update_class_stats(
                    train_diag_stats,
                    logits,
                    target,
                    raw_loss,
                    weighted_loss,
                    diagnostic_indices,
                    ignore_mask=negative_loss_ignore_mask,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite train loss at epoch={display_epoch}; "
                    f"logits finite={bool(torch.isfinite(logits).all())} "
                    f"target range=({float(target.min()):.4g}, {float(target.max()):.4g}) "
                    f"wave range=({float(wave.min()):.4g}, {float(wave.max()):.4g})"
                )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            train_losses.append(float(loss.detach().cpu()))

        if probe_loader is not None and (
            args.yao_rank_loss_weight > 0
            or args.yao_value_loss_weight > 0
            or args.base_preserve_loss_weight > 0
        ):
            model.train()
            distill_steps = max(1, int(args.yao_distill_steps))
            for distill_step in range(distill_steps):
                optimizer.zero_grad(set_to_none=True)
                probe_logits = []
                for wave, _target, _weight in probe_loader:
                    wave = wave.to(device, non_blocking=True)
                    x = frontend(wave).to(memory_format=torch.channels_last)
                    with torch.autocast(device_type=device.type, enabled=False):
                        probe_logits.append(model(x.float()))
                if probe_logits:
                    logits_probe = torch.cat(probe_logits, dim=0)
                    target_probe = torch.tensor(yao_probe_target, dtype=torch.float32, device=device)
                    base_dense_probe = torch.tensor(base_probe_dense, dtype=torch.float32, device=device)
                    adjusted_logits = (logits_probe - float(args.yao_probe_bias)) / max(float(args.yao_probe_temperature), 1e-6)
                    dense_probe = torch.sigmoid(adjusted_logits.float())
                    distill_loss = adjusted_logits.sum() * 0.0
                    rank_loss_value = 0.0
                    value_loss_value = 0.0
                    preserve_loss_value = 0.0
                    if args.yao_rank_loss_weight > 0:
                        rank_loss = pairwise_rank_distillation_loss(adjusted_logits, target_probe, focus_probe_indices)
                        distill_loss = distill_loss + float(args.yao_rank_loss_weight) * rank_loss
                        rank_loss_value = float(rank_loss.detach().cpu())
                    if args.yao_value_loss_weight > 0:
                        _calibrated_probe = calibrate_probe_logits(
                            logits_probe,
                            temperature=args.yao_probe_temperature,
                            bias=args.yao_probe_bias,
                            keep_topk=args.yao_probe_topk,
                            prob_floor=args.yao_probe_prob_floor,
                        )
                        value_loss = focus_value_distillation_loss(
                            logits_probe,
                            target_probe,
                            focus_probe_indices,
                            temperature=args.yao_probe_temperature,
                            bias=args.yao_probe_bias,
                        )
                        distill_loss = distill_loss + float(args.yao_value_loss_weight) * value_loss
                        value_loss_value = float(value_loss.detach().cpu())
                    if args.base_preserve_loss_weight > 0 and base_dense_probe.numel() > 0:
                        preserve_loss = torch.nn.functional.mse_loss(dense_probe, base_dense_probe)
                        distill_loss = distill_loss + float(args.base_preserve_loss_weight) * preserve_loss
                        preserve_loss_value = float(preserve_loss.detach().cpu())
                    if not torch.isfinite(distill_loss):
                        raise FloatingPointError("non-finite Yao probe distillation loss")
                    scaler.scale(distill_loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_steps += 1
                    print(
                        "yao_distill "
                        f"epoch={display_epoch}/{total_epochs} "
                        f"step={distill_step + 1}/{distill_steps} "
                        f"rank_loss={rank_loss_value:.5f} "
                        f"value_loss={value_loss_value:.5f} "
                        f"preserve_loss={preserve_loss_value:.5f} "
                        f"weighted={float(distill_loss.detach().cpu()):.5f}"
                    )
                    if diagnostic_labels and (distill_step == 0 or distill_step + 1 == distill_steps):
                        dense_probe = torch.sigmoid(adjusted_logits.float()).detach().cpu().numpy()
                        target_np = target_probe.detach().cpu().numpy()
                        details = []
                        for label, class_idx in zip(diagnostic_labels, diagnostic_indices.detach().cpu().numpy().tolist()):
                            details.append(
                                f"{label}:dense={float(dense_probe[:, class_idx].mean()):.5f},"
                                f"target={float(target_np[:, class_idx].mean()):.5f},"
                                f"raw_logit={float(logits_probe[:, class_idx].float().mean().detach().cpu()):.3f}"
                            )
                        print(
                            "diagnostic_distill "
                            f"epoch={display_epoch}/{total_epochs} "
                            f"step={distill_step + 1}/{distill_steps} "
                            + " | ".join(details)
                        )

        model.eval()
        val_losses = []
        val_diag_stats = init_class_stats(len(diagnostic_labels))
        with torch.inference_mode():
            for wave, target, _weight in val_loader:
                wave = wave.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                x = frontend(wave).to(memory_format=torch.channels_last)
                logits = model(x)
                val_raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
                update_class_stats(
                    val_diag_stats,
                    logits,
                    target,
                    val_raw_loss,
                    val_raw_loss,
                    diagnostic_indices,
                )
                val_loss_batch = loss_fn(logits, target)
                if not torch.isfinite(val_loss_batch):
                    raise FloatingPointError(
                        f"non-finite validation loss at epoch={display_epoch}; "
                        f"logits finite={bool(torch.isfinite(logits).all())}"
                    )
                val_losses.append(float(val_loss_batch.detach().cpu()))
        yao_metrics = {}
        yao_pred = np.empty((0, len(classes)), dtype=np.float32)
        yao_logits = np.empty((0, len(classes)), dtype=np.float32)
        if probe_loader is not None:
            probe_preds = []
            probe_logits_raw = []
            with torch.inference_mode():
                for wave, _target, _weight in probe_loader:
                    wave = wave.to(device, non_blocking=True)
                    x = frontend(wave).to(memory_format=torch.channels_last)
                    logits = model(x)
                    probs = calibrate_probe_logits(
                        logits,
                        temperature=args.yao_probe_temperature,
                        bias=args.yao_probe_bias,
                        keep_topk=args.yao_probe_topk,
                        prob_floor=args.yao_probe_prob_floor,
                    )
                    probe_preds.append(probs.detach().cpu().numpy().astype(np.float32))
                    probe_logits_raw.append(logits.detach().cpu().numpy().astype(np.float32))
            if probe_preds:
                yao_pred = np.concatenate(probe_preds, axis=0)
                yao_logits = np.concatenate(probe_logits_raw, axis=0)
                yao_metrics = compute_yao_probe_metrics(
                    yao_pred,
                    yao_probe_target,
                    classes,
                    true_label_sets=yao_probe_true_sets,
                    focus_classes=focus_classes,
                    topk=args.yao_probe_topk,
                    false_class=args.yao_probe_false_class,
                )
        if optimizer_steps > 0:
            scheduler.step()
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(
            f"epoch={display_epoch}/{total_epochs} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} time={time.time() - t0:.1f}s"
        )
        if args.ced_loss_weight > 0 and ced_epoch_parts:
            print(
                "ced_train "
                f"epoch={display_epoch}/{total_epochs} "
                f"loss={np.mean([part['loss'] for part in ced_epoch_parts]):.5f} "
                f"evidence_drop={np.mean([part['evidence_drop'] for part in ced_epoch_parts]):.5f} "
                f"context_suppression={np.mean([part['context_suppression'] for part in ced_epoch_parts]):.5f} "
                f"preserve={np.mean([part['preserve'] for part in ced_epoch_parts]):.5f} "
                f"active_time_frac={np.mean([part['active_time_frac'] for part in ced_epoch_parts]):.5f}"
            )
        if diagnostic_labels:
            print(format_class_stats("train", display_epoch, total_epochs, diagnostic_labels, train_diag_stats))
            print(format_class_stats("val", display_epoch, total_epochs, diagnostic_labels, val_diag_stats))
        if yao_metrics:
            print(
                "yao_probe "
                f"epoch={display_epoch}/{total_epochs} "
                f"corr={float(yao_metrics['yao_corr']):.4f} "
                f"rank_corr={float(yao_metrics['yao_rank_corr']):.4f} "
                f"rank_mae={float(yao_metrics['yao_rank_mae']):.5f} "
                f"mae={float(yao_metrics['yao_mae']):.5f} "
                f"top{args.yao_probe_topk}_overlap={float(yao_metrics['topk_overlap']):.4f} "
                f"sonotype_recall={float(yao_metrics['sonotype_recall']):.4f} "
                f"active_gt05={int(yao_metrics['active_gt05'])} "
                f"false_{args.yao_probe_false_class}_mean={float(yao_metrics['false_517063_mean']):.5f} "
                f"top{args.yao_probe_topk}_first={yao_metrics['topk_first']}"
            )
            for line in format_probe_diagnostics(
                display_epoch,
                total_epochs,
                yao_logits,
                yao_pred,
                yao_probe_target,
                yao_probe_row_ids,
                classes,
                diagnostic_labels,
                temperature=args.yao_probe_temperature,
                bias=args.yao_probe_bias,
                topk=args.diagnostic_topk,
                log_examples=args.diagnostic_log_examples,
            ):
                print(line)
            for line in format_base_preservation_diagnostics(
                display_epoch,
                total_epochs,
                base_probe_pred,
                yao_pred,
                classes,
                diagnostic_labels,
                topk=args.diagnostic_topk,
            ):
                print(line)
        selection_score = compute_yao_selection_score(yao_metrics, val_loss) if yao_metrics else -val_loss
        if yao_metrics:
            print(f"yao_selection epoch={display_epoch}/{total_epochs} score={selection_score:.5f}")
        if selection_score > best_score:
            best_score = selection_score
            state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": state,
                    "classes": classes,
                    "config": {
                        "model_name": args.model_name,
                        "stage": args.stage,
                        "fold": args.fold,
                        "sr": args.sr,
                        "window_seconds": args.window_seconds,
                        "focus_classes": sorted(focus_classes),
                        "focus_class_weight": args.focus_class_weight,
                        "soundscape_label_weight": args.soundscape_label_weight,
                        "yao_probe_metrics": yao_metrics,
                        "yao_selection_score": selection_score,
                        "yao_rank_loss_weight": args.yao_rank_loss_weight,
                        "yao_value_loss_weight": args.yao_value_loss_weight,
                        "yao_distill_steps": args.yao_distill_steps,
                        "yao_distill_classes": sorted(distill_classes),
                        "base_preserve_loss_weight": args.base_preserve_loss_weight,
                        "loss_ignore_negative_classes": sorted(loss_ignore_negative_classes),
                        "trainable_scope": args.trainable_scope,
                    },
                },
                best_path,
            )
            print(f"saved {best_path} val_loss={val_loss:.5f} selection_score={selection_score:.5f}")
    return best_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = train(args)
    print(f"Best checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
