"""BirdMAE-finetune inference on labeled OOF (train_soundscapes_labels).

Uses adkasd/birdmae-finetune-234 (1GB checkpoint, 234 BC2026 classes).
Bird-MAE is a Masked Autoencoder pretrained on bird vocalizations, SOTA on
BirdSet benchmark. Validation macro-AUC 0.9775 on its own training set.

Output: birdmae_labeled_oof.npz with per-(window, class) predictions.
"""
import os, sys, re, time, subprocess, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import glob as _glob
import torch
from torch.utils.data import DataLoader, Dataset

# Locate BirdMAE package
PKG_ROOT = None
for cand in [Path("/kaggle/input/birdmae-finetune-234"),
             Path("/kaggle/input/datasets/adkasd/birdmae-finetune-234")]:
    if cand.exists() and (cand / "checkpoints" / "best.pt").exists():
        PKG_ROOT = cand; break
assert PKG_ROOT, "BirdMAE package not found"
print(f"BirdMAE package: {PKG_ROOT}")

REPO_DIR = PKG_ROOT / "repo" / "Bird-MAE"
SCRIPT_DIR = PKG_ROOT / "scripts"
CKPT = PKG_ROOT / "checkpoints" / "best.pt"

# Import the finetune classifier
sys.path.insert(0, str(SCRIPT_DIR.resolve()))
sys.path.insert(0, str(REPO_DIR.resolve()))
from birdclef_finetune import BirdMAEClassifier  # noqa: E402

# Load encoder
encoder_path = REPO_DIR / "models" / "mae" / "encoder.py"
spec = importlib.util.spec_from_file_location("birdmae_encoder", encoder_path)
enc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc_mod)
MAE_Encoder = enc_mod.MAE_Encoder

encoder = MAE_Encoder(
    img_size_x=512, img_size_y=128, patch_size=16, in_chans=1,
    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
    norm_layer=torch.nn.LayerNorm, pos_trainable=False,
)

# Load classifier head
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
classes = ckpt.get("classes")
if classes is None:
    # try metadata
    meta = json.load(open(PKG_ROOT / "metadata" / "finetune_result.json"))
    classes = meta["classes"]
print(f"BirdMAE classes: {len(classes)}")

n_cls = len(classes)
model = BirdMAEClassifier(encoder=encoder, num_classes=n_cls, pool="mean", drop=0.1)
state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
try:
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded; missing={len(missing)}, unexpected={len(unexpected)}")
except Exception as e:
    print(f"  load error: {e}")
    raise

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device).eval()

# Load labels
COMP = None
for cand in ["/kaggle/input/birdclef-2026", "/kaggle/input/competitions/birdclef-2026"]:
    if Path(cand).exists():
        COMP = Path(cand); break
labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)
print(f"Labels: {len(labels)} rows, {labels['filename'].nunique()} files")
tax = pd.read_csv(COMP / "taxonomy.csv")
bc_classes = tax["primary_label"].astype(str).tolist()
N = len(labels); C = len(bc_classes)

# Build Y matrix
cls_idx = {c: i for i, c in enumerate(bc_classes)}
Y = np.zeros((N, C), dtype=np.float32)
for i, row in labels.iterrows():
    for c in str(row["primary_label"]).split(";"):
        c = c.strip()
        if c in cls_idx: Y[i, cls_idx[c]] = 1.0
HOUR_RE = re.compile(r"_\d{8}_(\d{2})\d{4}")
hours = np.array([int(HOUR_RE.search(f).group(1)) if HOUR_RE.search(f) else -1 for f in labels["filename"]])

# Map BirdMAE classes → BC2026 indices
birdmae_to_bc = {}
for bi, bc_lbl in enumerate(classes):
    if str(bc_lbl) in cls_idx:
        birdmae_to_bc[bi] = cls_idx[str(bc_lbl)]
print(f"BirdMAE → BC2026 mapping: {len(birdmae_to_bc)} classes")

# Process: load audio, extract 5s windows, run BirdMAE forward
SR = 32000
WIN_SAMP = 5 * SR
N_WIN = 12

ts_dir = COMP / "train_soundscapes"
file_to_idx = {}
for i, f in enumerate(labels["filename"]):
    file_to_idx.setdefault(f, []).append(i)

print(f"\nProcessing {len(file_to_idx)} files...")
P_birdmae = np.zeros((N, C), dtype=np.float32)

# Preprocessing: 32kHz audio → mel-spectrogram (128 mel × 512 time = 5s × 32k / hop)
import torchaudio
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=2048, win_length=2048, hop_length=314,
    n_mels=128, f_min=0, f_max=16000, power=2.0,
).to(device)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(device)

t0 = time.time()
n_done = 0
BATCH = 16
with torch.no_grad():
    for f, idx_list in file_to_idx.items():
        fpath = ts_dir / f
        if not fpath.exists():
            for ext in [".ogg", ".wav", ".flac"]:
                if (ts_dir / (f + ext)).exists():
                    fpath = ts_dir / (f + ext); break
        try:
            audio, sr = sf.read(str(fpath))
            if audio.ndim > 1: audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
        except: continue
        
        # Build batch of windows for this file
        win_batch = []
        for row_i in idx_list:
            st_val = labels.iloc[row_i]["start"]
            if isinstance(st_val, str) and ":" in st_val:
                h, m, s = st_val.split(":")
                start_sec = int(h) * 3600 + int(m) * 60 + int(s)
            else:
                start_sec = int(float(st_val))
            start_samp = start_sec * SR
            end_samp = start_samp + WIN_SAMP
            if end_samp <= len(audio):
                clip = audio[start_samp:end_samp]
            else:
                clip = np.zeros(WIN_SAMP, dtype=np.float32)
                avail = max(0, len(audio) - start_samp)
                if avail > 0:
                    clip[:avail] = audio[start_samp:start_samp+avail]
            win_batch.append(clip)
        if not win_batch: continue
        
        # Batched forward
        x = torch.tensor(np.stack(win_batch), dtype=torch.float32, device=device)
        # mel-spec: (B, n_mels, T)
        spec = mel(x)
        spec = amplitude_to_db(spec)
        # Bird-MAE expects (B, 1, n_mels, T) → resize to (B, 1, 128, 512)
        spec = spec.unsqueeze(1)
        if spec.shape[-1] != 512:
            spec = torch.nn.functional.interpolate(spec, size=(128, 512), mode="bilinear", align_corners=False)
        # Forward
        logits = model(spec)
        probs = torch.sigmoid(logits).cpu().numpy()
        # Place into BC2026 columns
        for j, row_i in enumerate(idx_list):
            for bi, ci in birdmae_to_bc.items():
                P_birdmae[row_i, ci] = probs[j, bi]
        n_done += 1
        if n_done % 10 == 0:
            el = time.time() - t0
            print(f"  [{n_done}/{len(file_to_idx)}] {el:.0f}s rate={n_done/el:.2f}/s")

print(f"\nTotal: {time.time()-t0:.0f}s")

# Save
out_path = "/kaggle/working/birdmae_labeled_oof.npz"
np.savez(out_path,
         P_birdmae=P_birdmae.astype(np.float32),
         Y=Y, classes=np.array(bc_classes),
         row_filename=labels["filename"].to_numpy(),
         row_hour=hours)
print(f"Saved: {out_path}")
print(f"P_birdmae stats: min={P_birdmae.min():.4f}, max={P_birdmae.max():.4f}")

# Quick AUC
from sklearn.metrics import roc_auc_score
aucs = []
for c in range(C):
    if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
    if P_birdmae[:, c].max() == 0: continue
    try: aucs.append(roc_auc_score(Y[:, c], P_birdmae[:, c]))
    except: pass
print(f"\nBirdMAE macro-AUC on labeled OOF: {np.mean(aucs):.4f} ({len(aucs)} classes)")
