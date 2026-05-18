"""Handcrafted audio features on labeled OOF windows.

Different signal source than Perch:
- MFCCs (13 coeffs × T frames → stats)
- Spectral centroid, bandwidth, rolloff, flatness, contrast
- Chroma features (12 pitch classes)
- Zero-crossing rate
- RMS energy envelope stats
- Onset density (event count)

These features capture acoustic STRUCTURE that Perch's deep model abstracts.
Use as input to Ridge/LR/LGB classifiers, blend with the existing stack.
"""
import os, sys, glob, re, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from concurrent.futures import ThreadPoolExecutor

COMP = None
for cand in ["/kaggle/input/birdclef-2026", "/kaggle/input/competitions/birdclef-2026"]:
    if Path(cand).exists():
        COMP = Path(cand); break
assert COMP

# Load labels
labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)
tax = pd.read_csv(COMP / "taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
N = len(labels)
C = len(classes)

# Build Y + file map
HOUR_RE = re.compile(r"_\d{8}_(\d{2})\d{4}")
hours = np.array([int(HOUR_RE.search(f).group(1)) if HOUR_RE.search(f) else -1 for f in labels["filename"]])
cls_idx = {c: i for i, c in enumerate(classes)}
Y = np.zeros((N, C), dtype=np.float32)
for i, row in labels.iterrows():
    for c in str(row["primary_label"]).split(";"):
        c = c.strip()
        if c in cls_idx: Y[i, cls_idx[c]] = 1.0

SR = 32000
WIN_SAMPLES = 5 * SR  # 160000
N_FFT = 2048
HOP = 512

# Feature extraction per 5-sec clip
def extract_features(audio):
    """Returns flat feature vector ~80 dims."""
    if len(audio) < WIN_SAMPLES:
        audio = np.pad(audio, (0, WIN_SAMPLES - len(audio)))
    audio = audio[:WIN_SAMPLES].astype(np.float32)
    feats = []
    # MFCC (13 coeffs × 313 frames → mean, std, max, min per coeff)
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
    feats.extend(mfcc.mean(axis=1))  # 13
    feats.extend(mfcc.std(axis=1))   # 13
    # Spectral stats
    centroid = librosa.feature.spectral_centroid(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP)
    feats.append(centroid.mean()); feats.append(centroid.std())
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP)
    feats.append(bandwidth.mean()); feats.append(bandwidth.std())
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP)
    feats.append(rolloff.mean()); feats.append(rolloff.std())
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP)
    feats.append(flatness.mean()); feats.append(flatness.std())
    contrast = librosa.feature.spectral_contrast(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP)
    feats.extend(contrast.mean(axis=1))  # 7
    feats.extend(contrast.std(axis=1))   # 7
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP)
    feats.extend(chroma.mean(axis=1))  # 12
    feats.extend(chroma.std(axis=1))   # 12
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=N_FFT, hop_length=HOP)
    feats.append(zcr.mean()); feats.append(zcr.std())
    # RMS
    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP)
    feats.append(rms.mean()); feats.append(rms.std()); feats.append(rms.max())
    # Onset density (events per second)
    try:
        onsets = librosa.onset.onset_detect(y=audio, sr=SR, hop_length=HOP, units='time')
        feats.append(len(onsets) / 5.0)  # per second
    except: feats.append(0)
    # Mel-spectrogram stats (high-freq energy)
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Per-frequency-band mean
    band_means = mel_db.mean(axis=1)  # 64
    feats.extend(band_means)
    return np.array(feats, dtype=np.float32)

# Process all 739 windows
ts_dir = COMP / "train_soundscapes"
file_to_idx = {}
for i, f in enumerate(labels["filename"]):
    file_to_idx.setdefault(f, []).append(i)

print(f"Extracting features from {len(file_to_idx)} files × {N} windows...")
features = np.zeros((N, 168), dtype=np.float32)  # approx 168 feats
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
        feats = extract_features(clip)
        if features.shape[1] != len(feats):
            features = np.zeros((N, len(feats)), dtype=np.float32)
        features[row_i] = feats
    n_done += 1
    if n_done % 10 == 0:
        el = time.time() - t0
        rate = n_done / el
        print(f"  [{n_done}/{len(file_to_idx)}] {el:.0f}s rate={rate:.2f}/s eta={(len(file_to_idx)-n_done)/max(rate,0.01):.0f}s")

print(f"\nTotal: {time.time()-t0:.0f}s")
print(f"Features shape: {features.shape}")

# Save
out_path = "/kaggle/working/audio_features_oof.npz"
np.savez(out_path,
         features=features,
         Y=Y,
         classes=np.array(classes),
         row_filename=labels["filename"].to_numpy(),
         row_hour=hours)
print(f"Saved {out_path}")

# Quick AUC: train Ridge per class via file-grouped CV
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(features)

file_codes = pd.Categorical(labels["filename"]).codes
gkf = GroupKFold(n_splits=5)
P_hand = np.zeros((N, C), dtype=np.float32)
print("\nFile-grouped CV Ridge on handcrafted features...")
t0 = time.time()
for fold, (tr, va) in enumerate(gkf.split(np.zeros(N), Y, groups=file_codes)):
    for c in range(C):
        if Y[tr, c].sum() < 3 or Y[tr, c].sum() == len(tr): continue
        try:
            m = Ridge(alpha=5.0)
            m.fit(X[tr], Y[tr, c])
            P_hand[va, c] = m.predict(X[va])
        except: pass
print(f"  {time.time()-t0:.0f}s")

aucs = []
for c in range(C):
    if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
    try: aucs.append(roc_auc_score(Y[:, c], P_hand[:, c]))
    except: pass
print(f"Handcrafted-features OOF macro-AUC: {np.mean(aucs):.4f} ({len(aucs)} classes)")

# Save predictions
np.savez("/kaggle/working/handcrafted_oof_preds.npz", P_hand=P_hand, features=features)
