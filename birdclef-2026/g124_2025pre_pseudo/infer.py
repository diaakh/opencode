from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf

from g124_assets import list_fixed_windows


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
                batch_probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                probs[row_indices_np[start:end]] = batch_probs
            all_probs += probs / max(len(args.checkpoint), 1)
            del model

    all_probs = np.clip(all_probs, 0.0, 1.0)
    out = pd.DataFrame(all_probs, columns=label_cols)
    out.insert(0, "row_id", [window.row_id for window in windows])
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
