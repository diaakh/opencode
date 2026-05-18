"""V73 fold0 ONNX inference on labeled train_soundscapes.

V73 is a mel-spectrogram CNN at LB 0.941 (strongest single public model).
We have Bruce (Ridge on Perch features) and Perch raw — V73 is the missing
non-Perch-derived signal in our OOF stack.

With TTA: 3 shifts × 1 fold = 3 predictions averaged per window.
Output: v73_labeled_oof.npz with (N, 234) sigmoid probabilities + filenames.
"""
import os, re, sys, glob, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

# Install onnxruntime offline
ort_whls = glob.glob("/kaggle/input/**/onnxruntime*.whl", recursive=True)
if ort_whls:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", ort_whls[0]])
import onnxruntime as ort

import librosa

# Find V73 ONNX (only fold0 — adkasd/v73-fold0-onnx)
V73_HITS = sorted(glob.glob("/kaggle/input/**/v73_fold*.onnx", recursive=True))
print(f"V73 ONNX files: {V73_HITS}")
assert V73_HITS, "Attach adkasd/v73-fold0-onnx"

# Find competition data
COMP = None
for cand in ["/kaggle/input/birdclef-2026", "/kaggle/input/competitions/birdclef-2026"]:
    if Path(cand).exists():
        COMP = Path(cand); break
assert COMP, "competition not mounted"
print(f"COMP: {COMP}")

# Load labels
labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)
print(f"Labels: {len(labels)} rows, {labels['filename'].nunique()} files")
tax = pd.read_csv(COMP / "taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
N, C = len(labels), len(classes)

# Build Y matrix + per-row hour/file metadata
cls_idx = {c: i for i, c in enumerate(classes)}
Y = np.zeros((N, C), dtype=np.float32)
for i, row in labels.iterrows():
    for c in str(row["primary_label"]).split(";"):
        c = c.strip()
        if c in cls_idx:
            Y[i, cls_idx[c]] = 1.0

# Hour parsing from filename: SXX_YYYYMMDD_HHMMSS_*.ogg
HOUR_RE = re.compile(r"_(\d{8})_(\d{2})\d{4}")
hours = np.array([int(HOUR_RE.search(f).group(2)) if HOUR_RE.search(f) else -1 for f in labels["filename"]])

# Load all sessions (likely only fold0)
SR = 32000
N_FFT = 2048
HOP = 512
N_MELS = 128
F_MIN = 20
F_MAX = 15000
WIN_SAMPLES = 5 * SR  # 160000
TTA_SHIFT = int(2.5 * SR)  # 80000

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
v73_sessions = []
for p in V73_HITS:
    v73_sessions.append(ort.InferenceSession(p, sess_options=so, providers=["CPUExecutionProvider"]))
print(f"Loaded {len(v73_sessions)} V73 ONNX session(s)")

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

def get_clip(audio, start, length):
    end = start + length
    if end <= len(audio):
        return audio[start:end]
    clip = np.zeros(length, dtype=np.float32)
    avail = max(0, len(audio) - start)
    if avail > 0:
        clip[:avail] = audio[start:start+avail]
    return clip

# Process file by file
ts_dir = COMP / "train_soundscapes"
test_paths = sorted(ts_dir.glob("*.ogg"))
unique_files = labels["filename"].unique()
print(f"\nProcessing {len(unique_files)} labeled files")

P_v73 = np.zeros((N, C), dtype=np.float32)
file_to_indices = {}
for i, f in enumerate(labels["filename"]):
    file_to_indices.setdefault(f, []).append(i)

t0 = time.time()
n_done = 0
for f, idx_list in file_to_indices.items():
    fpath = ts_dir / f
    if not fpath.exists():
        # filename might already include extension
        for ext in [".ogg", ".wav", ".flac"]:
            if (ts_dir / (f + ext)).exists():
                fpath = ts_dir / (f + ext); break
    if not fpath.exists():
        print(f"  MISSING: {f}")
        continue
    audio, _ = librosa.load(str(fpath), sr=SR, mono=True)
    # 12 windows expected per file
    for row_i in idx_list:
        # Each row has a start time
        win_start = int(labels.iloc[row_i]["start"]) * SR
        # TTA: center, left-2.5s, right-2.5s
        starts = [
            win_start,
            max(0, win_start - TTA_SHIFT),
            min(max(0, len(audio) - WIN_SAMPLES), win_start + TTA_SHIFT),
        ]
        tta_mels = []
        for s in starts:
            clip = get_clip(audio, s, WIN_SAMPLES)
            tta_mels.append(mel_for_clip(clip))
        tta_batch = np.concatenate(tta_mels, axis=0)  # (3, 128, 313)
        tta_batch = tta_batch[:, None, :, :]  # add channel dim: (3, 1, 128, 313)
        all_probs = []
        for sess in v73_sessions:
            try:
                logits = sess.run(["clipwise"], {"mel": tta_batch})[0]  # (3, 234)
            except Exception as e:
                # Try without "mel" name
                in_name = sess.get_inputs()[0].name
                out_name = sess.get_outputs()[0].name
                logits = sess.run([out_name], {in_name: tta_batch})[0]
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            all_probs.append(probs)
        # Average across folds AND TTA
        mean_probs = np.mean(np.concatenate(all_probs, axis=0), axis=0)
        P_v73[row_i] = mean_probs
    n_done += 1
    if n_done % 10 == 0:
        el = time.time() - t0
        rate = n_done / el
        eta = (len(file_to_indices) - n_done) / max(rate, 0.01)
        print(f"  [{n_done}/{len(file_to_indices)}] dt={el:.1f}s rate={rate:.2f}/s eta={eta:.0f}s")

print(f"\nTotal time: {time.time()-t0:.1f}s, mean per file: {(time.time()-t0)/len(file_to_indices):.2f}s")

# Save
out_path = "/kaggle/working/v73_labeled_oof.npz"
np.savez(out_path,
         P_v73=P_v73,
         Y=Y,
         classes=np.array(classes),
         row_filename=labels["filename"].to_numpy(),
         row_hour=hours,
         row_start=labels["start"].to_numpy())
print(f"Saved: {out_path}")
print(f"P_v73 stats: min={P_v73.min():.4f}, max={P_v73.max():.4f}, mean={P_v73.mean():.4f}")

# Quick AUC check
from sklearn.metrics import roc_auc_score
aucs = []
for c in range(C):
    if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
    try: aucs.append(roc_auc_score(Y[:, c], P_v73[:, c]))
    except: pass
print(f"\nV73 standalone macro-AUC on labeled OOF: {np.mean(aucs):.4f} ({len(aucs)} classes)")
