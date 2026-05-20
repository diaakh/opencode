from __future__ import annotations

import argparse
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
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--timm-pretrained", action="store_true", help="Allow timm pretrained weight loading when internet/cache is available")
    return parser


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


def build_pseudo_frame(pseudo_csv: str | None, competition_dir: Path, classes: list[str], weight: float) -> pd.DataFrame:
    if not pseudo_csv:
        return pd.DataFrame()
    pseudo_path = Path(pseudo_csv)
    if not pseudo_path.exists():
        raise FileNotFoundError(pseudo_path)
    df = pd.read_csv(pseudo_path)
    if "row_id" not in df.columns:
        raise ValueError(f"{pseudo_path} has no row_id column")
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


def add_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold

    frame = frame.copy()
    frame["fold"] = -1
    counts = frame["primary_label"].astype(str).value_counts()
    effective_folds = min(int(n_folds), len(frame), int(counts.min()))
    if effective_folds < 2:
        frame["fold"] = 0
        return frame
    n_folds = effective_folds
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = frame["primary_label"].astype(str)
    for fold, (_, val_idx) in enumerate(splitter.split(frame, y)):
        frame.loc[frame.index[val_idx], "fold"] = fold
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

    def __len__(self):
        return len(self.frame)

    def _load_audio(self, path: Path):
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.args.sr:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.args.sr).astype(np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)

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


def train(args: argparse.Namespace) -> Path:
    import torch
    from torch.utils.data import DataLoader

    seed_everything(args.seed)
    competition_dir = Path(args.competition_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = load_classes(competition_dir)
    frame = build_train_audio_frame(competition_dir, classes, max_files=args.max_train_files)
    frame = add_folds(frame, args.n_folds, args.seed)
    train_frame, val_frame, validation_fold = split_train_val(frame, args.fold)
    pseudo_frame = build_pseudo_frame(args.pseudo_csv, competition_dir, classes, args.pseudo_weight)
    if not pseudo_frame.empty:
        train_frame = pd.concat([train_frame, pseudo_frame], ignore_index=True)
    print(
        f"Stage={args.stage} train={len(train_frame)} val={len(val_frame)} "
        f"pseudo={len(pseudo_frame)} classes={len(classes)}"
    )

    device = choose_torch_device()
    model = build_model(args.model_name, len(classes), args.pretrained_checkpoint, args.timm_pretrained).to(device)
    model = model.to(memory_format=torch.channels_last)
    frontend = MelFrontend(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        AudioDataset(train_frame, classes, args, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_audio,
        drop_last=True,
    )
    val_loader = DataLoader(
        AudioDataset(val_frame, classes, args, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_audio,
    )

    best_val = float("inf")
    best_path = output_dir / "g124_fold1_fp16.pt"
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_losses = []
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
                loss = (raw_loss.mean(dim=1) * weight).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite train loss at epoch={epoch + 1}; "
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

        model.eval()
        val_losses = []
        with torch.inference_mode():
            for wave, target, _weight in val_loader:
                wave = wave.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                x = frontend(wave).to(memory_format=torch.channels_last)
                logits = model(x)
                val_loss_batch = loss_fn(logits, target)
                if not torch.isfinite(val_loss_batch):
                    raise FloatingPointError(
                        f"non-finite validation loss at epoch={epoch + 1}; "
                        f"logits finite={bool(torch.isfinite(logits).all())}"
                    )
                val_losses.append(float(val_loss_batch.detach().cpu()))
        if optimizer_steps > 0:
            scheduler.step()
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f} time={time.time() - t0:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
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
                    },
                },
                best_path,
            )
            print(f"saved {best_path} val_loss={best_val:.5f}")
    return best_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = train(args)
    print(f"Best checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
