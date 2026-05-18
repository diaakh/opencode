"""Run V73 (5-fold mel CNN) on a sample of train_soundscapes UNLABELED files.

Saves: v73_unlabeled_predictions.npz with (windows, 234) sigmoid probs.

This gives a genuinely orthogonal retrieval DB:
- V73 has different inductive bias than Bruce (mel-spectrogram CNN vs Ridge-on-Perch)
- V73 trained on different data (was Aves-focused)
- Predictions on unlabeled audio that Bruce also predicted on
- For each window, we now have BOTH Bruce-pseudo-label AND V73-pseudo-label

Local consumer can build a "V73-based KNN DB" with V73-predicted labels and
use it as a retrieval source completely independent of Bruce.
"""
import os, sys, glob, time, re, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

# Install onnxruntime-gpu if available, else cpu
ort_whls = sorted(glob.glob("/kaggle/input/**/onnxruntime*.whl", recursive=True))
if ort_whls:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", ort_whls[0]])
import onnxruntime as ort

import librosa

V73_HITS = sorted(glob.glob("/kaggle/input/**/v73_fold*.onnx", recursive=True))
print(f"V73 ONNX files: {V73_HITS}")
assert V73_HITS, "Attach adkasd/v73-fold0-onnx"

COMP = None
for cand in ["/kaggle/input/birdclef-2026", "/kaggle/input/competitions/birdclef-2026"]:
    if Path(cand).exists():
        COMP = Path(cand); break

ts_dir = COMP / "train_soundscapes"
all_files = sorted(ts_dir.glob("*.ogg"))
print(f"train_soundscapes total: {len(all_files)} files")

# Exclude labeled files
labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
labeled_files = set(labels["filename"].unique())
unlabeled_files = [f for f in all_files if f.name not in labeled_files]
print(f"Unlabeled files: {len(unlabeled_files)}")

# Sample 400 files stratified by site (extract site from filename)
SITE_RE = re.compile(r"_(S\d+)_")
file_sites = []
for f in unlabeled_files:
    m = SITE_RE.search(f.name)
    file_sites.append(m.group(1) if m else "UNK")

df_files = pd.DataFrame({"path": unlabeled_files, "site": file_sites})
# Stratified sample
N_SAMPLE = 400
sample_per_site = max(1, N_SAMPLE // df_files["site"].nunique())
sample_df = df_files.groupby("site").apply(lambda g: g.sample(min(sample_per_site, len(g)), random_state=42)).reset_index(drop=True)
print(f"Stratified sample: {len(sample_df)} files across {sample_df['site'].nunique()} sites")
print(sample_df["site"].value_counts().head(20))

# Mel params
SR = 32000
N_FFT = 2048
HOP = 512
N_MELS = 128
F_MIN = 20
F_MAX = 15000
WIN_SAMPLES = 5 * SR
N_WINDOWS = 12

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Try GPU
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
v73_sessions = []
for p in V73_HITS:
    try:
        s = ort.InferenceSession(p, sess_options=so, providers=providers)
        v73_sessions.append(s)
    except Exception as e:
        print(f"  Fall back to CPU for {p}: {e}")
        s = ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"])
        v73_sessions.append(s)
print(f"Loaded {len(v73_sessions)} V73 sessions, providers: {v73_sessions[0].get_providers()}")

def mel_for_clip(audio):
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=SR,
        n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=F_MIN, fmax=F_MAX, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    mel_norm = (mel_db + 80.0) / 80.0
    if mel_norm.shape[1] < 313:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, 313 - mel_norm.shape[1])))
    else:
        mel_norm = mel_norm[:, :313]
    return mel_norm[None, :, :].astype(np.float32)

# Process each file with full 12 windows
N = len(sample_df)
W = N_WINDOWS
P_v73 = np.zeros((N * W, 234), dtype=np.float32)
file_names = []
window_idx = []
sites = []
t0 = time.time()
for fi, row in sample_df.iterrows():
    fpath = row["path"]
    audio, _ = librosa.load(str(fpath), sr=SR, mono=True)
    for w in range(W):
        start = w * WIN_SAMPLES
        end = start + WIN_SAMPLES
        if end <= len(audio):
            clip = audio[start:end]
        else:
            clip = np.zeros(WIN_SAMPLES, dtype=np.float32)
            avail = max(0, len(audio) - start)
            if avail > 0:
                clip[:avail] = audio[start:start+avail]
        mel = mel_for_clip(clip)
        # Run all folds, average
        all_probs = []
        for sess in v73_sessions:
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            mel_batch = mel[None, :, :, :]  # (1, 1, 128, 313)
            logits = sess.run([out_name], {in_name: mel_batch})[0]
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            all_probs.append(probs)
        mean_probs = np.mean(np.concatenate(all_probs, axis=0), axis=0)
        idx = fi * W + w
        P_v73[idx] = mean_probs
        file_names.append(fpath.name)
        window_idx.append(w)
        sites.append(row["site"])
    if (fi + 1) % 20 == 0 or fi == N - 1:
        el = time.time() - t0
        rate = (fi + 1) / el
        eta = (N - (fi + 1)) / max(rate, 0.01)
        print(f"  [{fi+1}/{N}] elapsed={el:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")

print(f"\nTotal: {time.time()-t0:.0f}s")

# Save
out_path = "/kaggle/working/v73_unlabeled_predictions.npz"
np.savez(out_path,
         P_v73=P_v73.astype(np.float32),
         file_names=np.array(file_names),
         window_idx=np.array(window_idx, dtype=np.int32),
         sites=np.array(sites),
         taxonomy=pd.read_csv(COMP / "taxonomy.csv")["primary_label"].astype(str).to_numpy())
print(f"Saved {out_path}: P_v73 shape={P_v73.shape}")
print(f"Stats: min={P_v73.min():.4f}, max={P_v73.max():.4f}, mean={P_v73.mean():.4f}")
