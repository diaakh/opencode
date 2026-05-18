"""F11 AnuraSet model inference on labeled OOF.

F11 is EfficientNet-B1 trained on AnuraSet (42 South American frog species).
ALL 4 BC2026 bottleneck chorus frogs are in F11's training set:
  65380 Dwarf Tree Frog        → DENNAN (Dendropsophus nanus)
  517063 Orange-legged Frog    → PITAZU (Pithecopus azureus)
  555146 Chaco Tree Frog       → BOARAN (Boana raniceps)
  24279 Lesser Snouted Frog    → SCINAS (Scinax nasicus)
  22961 Pointedbelly Frog      → LEPPOD (Leptodactylus podicipinus)

Since F11 was trained on a DIFFERENT dataset (AnuraSet, not BC2026 train_audio)
with a DIFFERENT model class (EfficientNet vs Bruce's Ridge-on-Perch), its
predictions are genuinely orthogonal to our entire stack.

Output: f11_labeled_oof.npz with per-(window, frog_class) predictions mapped
to BC2026 class indices.
"""
import os, sys, glob, time, re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import json
import torch
import torch.nn.functional as F

import librosa

# Find F11 model
F11_HITS = sorted(glob.glob("/kaggle/input/**/f11_ep10.pth", recursive=True))
assert F11_HITS, "Attach lvweibin/birdclef-2026-f11-ep10-pytorch"
F11_PATH = F11_HITS[0]
SPECIES_HITS = sorted(glob.glob("/kaggle/input/**/species_codes.json", recursive=True))
assert SPECIES_HITS
SPECIES_CODES = json.load(open(SPECIES_HITS[0]))
CONFIG_HITS = sorted(glob.glob("/kaggle/input/**/birdset_config.json", recursive=True))
config = json.load(open(CONFIG_HITS[0])) if CONFIG_HITS else None
print(f"F11 weights: {F11_PATH}")
print(f"F11 species codes: {len(SPECIES_CODES)} classes")
print(f"Config: arch={config.get('architectures') if config else '?'}")

# Find competition
COMP = None
for cand in ["/kaggle/input/birdclef-2026", "/kaggle/input/competitions/birdclef-2026"]:
    if Path(cand).exists():
        COMP = Path(cand); break
assert COMP

# Load model — EfficientNet-B1 from transformers (BirdSet)
print("Loading F11 model...")
try:
    from transformers import EfficientNetForImageClassification, EfficientNetConfig
    cfg = EfficientNetConfig.from_dict(config) if config else EfficientNetConfig()
    cfg.num_labels = len(SPECIES_CODES)
    cfg.id2label = {i: c for i, c in enumerate(SPECIES_CODES)}
    cfg.label2id = {c: i for i, c in enumerate(SPECIES_CODES)}
    model = EfficientNetForImageClassification(cfg)
    state = torch.load(F11_PATH, map_location="cpu", weights_only=True)
    # Handle key prefixes
    if "model" in state: state = state["model"]
    elif "state_dict" in state: state = state["state_dict"]
    # Try to load — may have prefix mismatch
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  loaded; missing={len(missing)}, unexpected={len(unexpected)}")
        if missing: print(f"  first missing: {missing[:3]}")
        if unexpected: print(f"  first unexpected: {unexpected[:3]}")
    except Exception as e:
        print(f"  load failed: {e}")
        # Try removing common prefixes
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "").replace("model.", "")
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  device: {device}")
except Exception as e:
    print(f"Failed transformers route: {e}")
    raise

# Inspect input expectations
# Print removed — attribute does not exist on transformers EfficientNet

# AnuraSet/BirdSet uses mel spectrogram inputs typically. Compute mel for each window.
# Standard BirdSet preprocessing: 32kHz, 5s windows, mel-spectrogram (128 bins).
SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC
N_FFT = 2048
HOP = 320  # gives 501 frames for 5s window
N_MELS = 128
F_MIN = 16
F_MAX = 16000

def mel_for_clip(audio):
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=SR,
        n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=F_MIN, fmax=F_MAX, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    # Normalize to [0, 1]
    mel_norm = (mel_db + 80.0) / 80.0
    return mel_norm.astype(np.float32)

# Test mel shape
test_audio = np.zeros(WIN_SAMPLES, dtype=np.float32)
mel = mel_for_clip(test_audio)
print(f"Mel shape: {mel.shape}")  # expect (128, T)

# EfficientNet-B1 input typically 240x240 but adapt to mel size
# BirdSet model: input is 3-channel (RGB-style) or 1-channel grayscale mel
# Default to 3 channels for transformers EfficientNet
in_ch = 3
print(f"Conv expects {in_ch} channels")

# Load labels
labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)
print(f"Labels: {len(labels)} rows, {labels['filename'].nunique()} files")
tax = pd.read_csv(COMP / "taxonomy.csv")
bc_classes = tax["primary_label"].astype(str).tolist()
N = len(labels)
C = len(bc_classes)

# Build mapping from F11 species code → BC2026 class index
def normalize_sci(name):
    """Normalize scientific name to compare with F11 codes."""
    parts = str(name).split()
    if len(parts) >= 2:
        return parts[0][:3].upper() + parts[1][:3].upper()
    return None

bc_sci_map = {}  # 6-letter code → bc_index
for i, row in tax.iterrows():
    sci = row.get("scientific_name", "")
    code = normalize_sci(sci)
    if code:
        bc_sci_map[code] = i

f11_to_bc = {}
for f_idx, code in enumerate(SPECIES_CODES):
    if code in bc_sci_map:
        f11_to_bc[f_idx] = bc_sci_map[code]
print(f"F11→BC2026 mapping: {len(f11_to_bc)}/{len(SPECIES_CODES)} F11 species map to BC2026")
for f_idx, bc_idx in sorted(f11_to_bc.items())[:10]:
    print(f"  {SPECIES_CODES[f_idx]} → {bc_classes[bc_idx]} ({tax.iloc[bc_idx]['common_name']})")

# Hour parsing
HOUR_RE = re.compile(r"_\d{8}_(\d{2})\d{4}")
hours = np.array([int(HOUR_RE.search(f).group(1)) if HOUR_RE.search(f) else -1 for f in labels["filename"]])

# Build Y
cls_idx = {c: i for i, c in enumerate(bc_classes)}
Y = np.zeros((N, C), dtype=np.float32)
for i, row in labels.iterrows():
    for c in str(row["primary_label"]).split(";"):
        c = c.strip()
        if c in cls_idx: Y[i, cls_idx[c]] = 1.0

# Process files
ts_dir = COMP / "train_soundscapes"
file_to_idx = {}
for i, f in enumerate(labels["filename"]):
    file_to_idx.setdefault(f, []).append(i)

print(f"\nProcessing {len(file_to_idx)} files...")
P_f11_42 = np.zeros((N, len(SPECIES_CODES)), dtype=np.float32)  # raw 42-class predictions
P_f11_bc = np.zeros((N, C), dtype=np.float32)  # mapped to BC2026

t0 = time.time()
n_done = 0
for f, idx_list in file_to_idx.items():
    fpath = ts_dir / f
    if not fpath.exists():
        for ext in [".ogg", ".wav", ".flac"]:
            if (ts_dir / (f + ext)).exists(): fpath = ts_dir / (f + ext); break
    audio, _ = librosa.load(str(fpath), sr=SR, mono=True)
    for row_i in idx_list:
        st_val = labels.iloc[row_i]["start"]
        if isinstance(st_val, str) and ":" in st_val:
            h, m, s = st_val.split(":")
            start_sec = int(h) * 3600 + int(m) * 60 + int(s)
        else:
            start_sec = int(float(st_val))
        start_samp = start_sec * SR
        end_samp = start_samp + WIN_SAMPLES
        if end_samp <= len(audio):
            clip = audio[start_samp:end_samp]
        else:
            clip = np.zeros(WIN_SAMPLES, dtype=np.float32)
            avail = max(0, len(audio) - start_samp)
            if avail > 0:
                clip[:avail] = audio[start_samp:start_samp+avail]
        mel = mel_for_clip(clip)
        # Resize/format for EfficientNet input
        # B1 typical input 240x240. Resize mel to required dim.
        # PyTorch needs (B, C, H, W)
        mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, T)
        if in_ch == 3:
            mel_t = mel_t.repeat(1, 3, 1, 1)
        # Resize to 240x240
        mel_t = F.interpolate(mel_t, size=(240, 240), mode="bilinear", align_corners=False)
        mel_t = mel_t.to(device)
        with torch.no_grad():
            out = model(pixel_values=mel_t)
            logits = out.logits.cpu().numpy()[0]  # (42,) or (num_labels,)
            probs = 1 / (1 + np.exp(-logits))
        P_f11_42[row_i] = probs[:len(SPECIES_CODES)]
        for f_idx, bc_idx in f11_to_bc.items():
            P_f11_bc[row_i, bc_idx] = P_f11_42[row_i, f_idx]
    n_done += 1
    if n_done % 10 == 0:
        el = time.time() - t0
        print(f"  [{n_done}/{len(file_to_idx)}] {el:.0f}s rate={n_done/el:.2f}/s")

print(f"\nTotal: {time.time()-t0:.0f}s")

# Save
out_path = "/kaggle/working/f11_labeled_oof.npz"
np.savez(out_path,
         P_f11_42=P_f11_42.astype(np.float32),
         P_f11_bc=P_f11_bc.astype(np.float32),
         Y=Y,
         species_codes=np.array(SPECIES_CODES),
         classes=np.array(bc_classes),
         row_filename=labels["filename"].to_numpy(),
         row_hour=hours,
         f11_to_bc_keys=np.array(list(f11_to_bc.keys()), dtype=np.int32),
         f11_to_bc_values=np.array(list(f11_to_bc.values()), dtype=np.int32))
print(f"Saved: {out_path}")
print(f"P_f11_bc stats: min={P_f11_bc.min():.4f}, max={P_f11_bc.max():.4f}")

# Quick AUC check for mapped classes
from sklearn.metrics import roc_auc_score
print("\nF11 per-class AUC on mapped BC2026 classes:")
for f_idx, bc_idx in f11_to_bc.items():
    if Y[:, bc_idx].sum() < 2 or Y[:, bc_idx].sum() == N: continue
    try:
        auc = roc_auc_score(Y[:, bc_idx], P_f11_bc[:, bc_idx])
        cls_label = bc_classes[bc_idx]
        common = tax.iloc[bc_idx]["common_name"]
        print(f"  {SPECIES_CODES[f_idx]:10s} → {cls_label:10s} ({common[:30]:<30s}): AUC={auc:.4f}")
    except: pass
