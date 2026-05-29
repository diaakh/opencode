from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf

from g124_assets import apply_sidecar_profile
from g124_assets import list_fixed_windows
from g124_assets import rank_blend_diagnostics
from g124_assets import SIDECAR_PROFILES
from g124_assets import sidecar_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G124 EfficientNetV2-S BirdCLEF sidecar inference")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--tta-shifts", type=int, default=0)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--smooth-weight", type=float, default=0.0)
    parser.add_argument("--logit-temperature", type=float, default=1.0)
    parser.add_argument("--logit-bias", type=float, default=0.0)
    parser.add_argument("--keep-topk", type=int, default=0)
    parser.add_argument("--prob-floor", type=float, default=0.0)
    parser.add_argument("--sidecar-profile", default="none", choices=sorted(SIDECAR_PROFILES))
    parser.add_argument("--evidence-promote-classes", default="")
    parser.add_argument("--evidence-promote-n-chunks", type=int, default=8)
    parser.add_argument("--evidence-promote-min-prob", type=float, default=0.20)
    parser.add_argument("--evidence-promote-min-delta", type=float, default=0.08)
    parser.add_argument("--evidence-promote-min-concentration", type=float, default=0.45)
    parser.add_argument("--evidence-promote-min-value", type=float, default=0.06)
    parser.add_argument("--evidence-promote-max-value", type=float, default=0.25)
    parser.add_argument("--disable-context-postprocess", action="store_true")
    parser.add_argument("--fast-fixed-60s", action="store_true")
    parser.add_argument("--assume-sr", type=int, default=32000)
    parser.add_argument("--assume-duration", type=float, default=60.0)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--model-name", default="tf_efficientnetv2_s.in21ft1k")
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=16000.0)
    return parser


def load_label_columns(data_dir: Path) -> list[str]:
    sample_path = data_dir / "sample_submission.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path, nrows=1).columns[1:].astype(str).tolist()
    taxonomy_path = data_dir / "taxonomy.csv"
    if taxonomy_path.exists():
        return pd.read_csv(taxonomy_path)["primary_label"].astype(str).tolist()
    raise FileNotFoundError(f"sample_submission.csv or taxonomy.csv not found under {data_dir}")


def _read_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def _extract_windows_for_file(path: Path, starts: list[float], window_seconds: float, sr: int) -> np.ndarray:
    audio = _read_audio(path, sr)
    n_samples = int(round(window_seconds * sr))
    out = np.zeros((len(starts), n_samples), dtype=np.float32)
    for idx, start_seconds in enumerate(starts):
        start = int(round(start_seconds * sr))
        end = start + n_samples
        if start >= len(audio):
            continue
        clip = audio[start:min(end, len(audio))]
        out[idx, : len(clip)] = clip
    return out


def _build_model(model_name: str, num_classes: int, checkpoint_path: Path):
    import timm
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model_name = cfg.get("model_name", model_name)
    last_error = None
    for candidate_name in model_name_candidates(model_name):
        try:
            model = timm.create_model(candidate_name, pretrained=False, in_chans=1, num_classes=num_classes)
            model_name = candidate_name
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        raise last_error if last_error is not None else RuntimeError(f"could not create model {model_name}")
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in checkpoint:
                state = checkpoint[key]
                break
    if isinstance(state, dict):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded {checkpoint_path.name}: model={model_name}, missing={len(missing)}, unexpected={len(unexpected)}",
        flush=True,
    )
    return model


def model_name_candidates(model_name: str) -> list[str]:
    candidates = [model_name]
    if "." in model_name:
        candidates.append(model_name.split(".", 1)[0])
    return list(dict.fromkeys(candidates))


def _make_mel_transform(args, device):
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=args.assume_sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=args.fmin,
        f_max=args.fmax,
        power=2.0,
    ).to(device)


def _wave_to_model_input(wave_batch, mel_transform, device):
    import torch

    wave = torch.from_numpy(wave_batch).to(device)
    wave = wave - wave.mean(dim=1, keepdim=True)
    mel = mel_transform(wave)
    mel = torch.log(torch.clamp(mel, min=1e-6))
    mean = mel.mean(dim=(1, 2), keepdim=True)
    std = mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
    mel = (mel - mean) / std
    return mel.unsqueeze(1)


def _calibrate_dense_probs(logits, args) -> np.ndarray:
    import torch

    temperature = max(float(args.logit_temperature), 1e-6)
    bias = float(args.logit_bias)
    return torch.sigmoid((logits - bias) / temperature).detach().cpu().numpy().astype(np.float32)


def _sparsify_topk_probs(probs: np.ndarray, keep_topk: int, prob_floor: float) -> np.ndarray:
    keep_topk = int(keep_topk)
    if keep_topk > 0 and keep_topk < probs.shape[1]:
        floor = float(np.clip(prob_floor, 0.0, 1.0))
        sparse = np.full_like(probs, floor, dtype=np.float32)
        top_idx = np.argpartition(probs, -keep_topk, axis=1)[:, -keep_topk:]
        rows = np.arange(probs.shape[0])[:, None]
        sparse[rows, top_idx] = probs[rows, top_idx]
        probs = sparse
    return probs


def _calibrate_batch_probs(logits, args) -> np.ndarray:
    return _sparsify_topk_probs(_calibrate_dense_probs(logits, args), args.keep_topk, args.prob_floor)


def _parse_class_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).replace(";", ",").split(",") if item.strip()]


def _resolve_evidence_class_indices(label_cols: list[str], class_text: str) -> tuple[np.ndarray, list[str]]:
    labels = _parse_class_list(class_text)
    label_to_idx = {label: idx for idx, label in enumerate(label_cols)}
    resolved = [(label_to_idx[label], label) for label in labels if label in label_to_idx]
    if not resolved:
        return np.asarray([], dtype=np.int64), []
    indices, names = zip(*resolved)
    return np.asarray(indices, dtype=np.int64), list(names)


def _apply_sparse_evidence_promotions(
    sparse_probs: np.ndarray,
    dense_probs: np.ndarray,
    class_indices: np.ndarray,
    evidence_mask: np.ndarray,
    keep_topk: int,
    prob_floor: float,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    values = np.asarray(sparse_probs, dtype=np.float32).copy()
    dense = np.asarray(dense_probs, dtype=np.float32)
    if class_indices.size == 0 or evidence_mask.size == 0:
        return values
    if evidence_mask.shape != (values.shape[0], class_indices.size):
        raise ValueError(f"evidence_mask shape {evidence_mask.shape} incompatible with classes {class_indices.size}")
    floor = np.float32(np.clip(prob_floor, 0.0, 1.0))
    lo = np.float32(np.clip(min_value, 0.0, 1.0))
    hi = np.float32(np.clip(max_value, float(lo), 1.0))
    forced = np.zeros_like(values, dtype=bool)
    rows = np.arange(values.shape[0])[:, None]
    promoted_values = np.clip(dense[rows, class_indices[None, :]], lo, hi)
    values[rows, class_indices[None, :]] = np.where(
        evidence_mask,
        np.maximum(values[rows, class_indices[None, :]], promoted_values),
        values[rows, class_indices[None, :]],
    )
    forced[rows, class_indices[None, :]] = evidence_mask

    keep = int(keep_topk)
    if keep <= 0 or keep >= values.shape[1]:
        return values
    for row_idx in range(values.shape[0]):
        active = np.flatnonzero(values[row_idx] > floor)
        if active.size <= keep:
            continue
        forced_idx = np.flatnonzero(forced[row_idx])
        if forced_idx.size >= keep:
            order = forced_idx[np.argsort(values[row_idx, forced_idx])[::-1][:keep]]
            keep_idx = order
        else:
            remaining = np.setdiff1d(active, forced_idx, assume_unique=False)
            remaining = remaining[np.argsort(values[row_idx, remaining])[::-1]]
            keep_idx = np.concatenate([forced_idx, remaining[: keep - forced_idx.size]])
        row = np.full(values.shape[1], floor, dtype=np.float32)
        row[keep_idx] = values[row_idx, keep_idx]
        values[row_idx] = row
    return values


def _time_chunks(time_bins: int, n_chunks: int) -> list[tuple[int, int]]:
    n_chunks = min(max(int(n_chunks), 1), int(time_bins))
    edges = np.rint(np.linspace(0, int(time_bins), n_chunks + 1)).astype(int)
    return [(int(start), max(int(start) + 1, int(end))) for start, end in zip(edges[:-1], edges[1:])]


def _mask_mel_time_chunk(x, start: int, end: int):
    fill = x.mean(dim=-1, keepdim=True)
    masked = x.clone()
    masked[..., int(start) : int(end)] = fill
    return masked


def _evidence_mask_for_batch(model, x, dense_probs: np.ndarray, class_indices: np.ndarray, args) -> np.ndarray:
    if class_indices.size == 0:
        return np.zeros((x.shape[0], 0), dtype=bool)
    deltas = []
    for start, end in _time_chunks(int(x.shape[-1]), int(args.evidence_promote_n_chunks)):
        masked_logits = model(_mask_mel_time_chunk(x, start, end))
        masked_dense = _calibrate_dense_probs(masked_logits, args)[:, class_indices]
        deltas.append(dense_probs[:, class_indices] - masked_dense)
    delta = np.stack(deltas, axis=2).astype(np.float32)
    positive_delta = np.clip(delta, 0.0, None)
    max_delta = positive_delta.max(axis=2)
    total_delta = positive_delta.sum(axis=2)
    concentration = max_delta / np.maximum(total_delta, 1e-6)
    return (
        (dense_probs[:, class_indices] >= float(args.evidence_promote_min_prob))
        & (max_delta >= float(args.evidence_promote_min_delta))
        & (concentration >= float(args.evidence_promote_min_concentration))
    )


def run_inference(args: argparse.Namespace) -> pd.DataFrame:
    import torch

    data_dir = Path(args.data_dir)
    input_dir = Path(args.input_dir)
    label_cols = load_label_columns(data_dir)
    windows = list_fixed_windows(
        input_dir,
        window_seconds=args.window_seconds,
        assume_duration=args.assume_duration,
        limit_files=args.limit_files,
    )
    if not windows:
        raise FileNotFoundError(f"no audio files found under {input_dir}")

    grouped: dict[Path, list[int]] = defaultdict(list)
    for idx, window in enumerate(windows):
        grouped[window.path].append(idx)

    n_rows = len(windows)
    n_classes = len(label_cols)
    all_probs = np.zeros((n_rows, n_classes), dtype=np.float32)
    evidence_class_indices, evidence_class_names = _resolve_evidence_class_indices(
        label_cols, args.evidence_promote_classes
    )
    evidence_counts = np.zeros(evidence_class_indices.shape[0], dtype=np.int64)
    if evidence_class_names:
        print(
            "Evidence promotion enabled: "
            f"classes={','.join(evidence_class_names)} n_chunks={args.evidence_promote_n_chunks} "
            f"min_prob={args.evidence_promote_min_prob:.4f} min_delta={args.evidence_promote_min_delta:.4f}",
            flush=True,
        )
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    mel_transform = _make_mel_transform(args, device)

    def load_group(path_and_indices):
        path, indices = path_and_indices
        starts = [windows[idx].start_seconds for idx in indices]
        return indices, _extract_windows_for_file(path, starts, args.window_seconds, args.assume_sr)

    items = list(grouped.items())
    max_workers = max(1, int(args.num_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        loaded_groups = list(pool.map(load_group, items))

    row_indices = []
    wave_chunks = []
    for indices, wave in loaded_groups:
        row_indices.extend(indices)
        wave_chunks.append(wave)
    wave_all = np.concatenate(wave_chunks, axis=0)
    row_indices_np = np.array(row_indices, dtype=int)

    with torch.inference_mode():
        for checkpoint in args.checkpoint:
            model = _build_model(args.model_name, n_classes, Path(checkpoint)).to(device).eval()
            probs = np.zeros((n_rows, n_classes), dtype=np.float32)
            for start in range(0, len(wave_all), args.batch_size):
                end = min(start + args.batch_size, len(wave_all))
                x = _wave_to_model_input(wave_all[start:end], mel_transform, device)
                logits = model(x)
                dense_probs = _calibrate_dense_probs(logits, args)
                batch_probs = _sparsify_topk_probs(dense_probs, args.keep_topk, args.prob_floor)
                if evidence_class_indices.size:
                    evidence_mask = _evidence_mask_for_batch(model, x, dense_probs, evidence_class_indices, args)
                    evidence_counts += evidence_mask.sum(axis=0).astype(np.int64)
                    batch_probs = _apply_sparse_evidence_promotions(
                        batch_probs,
                        dense_probs,
                        evidence_class_indices,
                        evidence_mask,
                        args.keep_topk,
                        args.prob_floor,
                        args.evidence_promote_min_value,
                        args.evidence_promote_max_value,
                    )
                probs[row_indices_np[start:end]] = batch_probs
            all_probs += probs / max(len(args.checkpoint), 1)
            del model

    all_probs = np.clip(all_probs, 0.0, 1.0)
    if evidence_class_names:
        print(
            "Evidence promotion counts: "
            + ", ".join(f"{label}={int(count)}" for label, count in zip(evidence_class_names, evidence_counts)),
            flush=True,
        )
    out = pd.DataFrame(all_probs, columns=label_cols)
    out.insert(0, "row_id", [window.row_id for window in windows])
    raw_out = out.copy()
    out = apply_sidecar_profile(out, label_cols, profile_name=args.sidecar_profile)
    if args.sidecar_profile != "none":
        diagnostics = sidecar_diagnostics(out, label_cols)
        rank_diagnostics = rank_blend_diagnostics(raw_out, out, label_cols)
        print(
            "Sidecar profile "
            f"{args.sidecar_profile}: min={diagnostics['min']:.6f} "
            f"max={diagnostics['max']:.6f} mean={diagnostics['mean']:.6f} "
            f"std={diagnostics['std']:.6f} active_gt05={diagnostics['active_gt05']}",
            flush=True,
        )
        print(
            "Sidecar rank diagnostics "
            f"global_rank_corr={rank_diagnostics['global_rank_corr']:.6f} "
            f"median_class_rank_corr={rank_diagnostics['median_class_rank_corr']:.6f} "
            f"mean_abs_rank_delta={rank_diagnostics['mean_abs_rank_delta']:.6f} "
            f"max_abs_rank_delta={rank_diagnostics['max_abs_rank_delta']:.6f} "
            f"top3_overlap={rank_diagnostics['top3_overlap']:.6f} "
            f"top8_overlap={rank_diagnostics['top8_overlap']:.6f} "
            f"top10_overlap={rank_diagnostics['top10_overlap']:.6f}",
            flush=True,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = run_inference(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    values = out.iloc[:, 1:].to_numpy(dtype=np.float32)
    print(
        f"Wrote {output}: rows={len(out)}, classes={values.shape[1]}, "
        f"min={values.min():.6f}, max={values.max():.6f}, mean={values.mean():.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
