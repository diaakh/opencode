"""Compute Perch embeddings on BC2026 train_audio (focal single-species recordings).

train_audio has one folder per species, each with multiple focal recordings.
These are SINGLE-SPECIES recordings — clean prototype data for each class.
Unlike our current KNN DB (train_soundscapes + Bruce-pseudo which can have
chorus contamination), train_audio gives uncontaminated per-species signatures.

Output: train_audio_perch.npz with per-class prototype embeddings.
"""
import os, sys, glob, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

# Install onnxruntime offline
ort_whls = sorted(glob.glob("/kaggle/input/**/onnxruntime*.whl", recursive=True))
if ort_whls:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", ort_whls[0]])
import onnxruntime as ort

PERCH_HITS = sorted(glob.glob("/kaggle/input/**/perch_v2_no_dft.onnx", recursive=True))
assert PERCH_HITS
PERCH_PATH = PERCH_HITS[0]

COMP = None
for cand in ["/kaggle/input/birdclef-2026", "/kaggle/input/competitions/birdclef-2026"]:
    if Path(cand).exists():
        COMP = Path(cand); break
TRAIN_AUDIO = COMP / "train_audio"
print(f"COMP: {COMP}")
print(f"train_audio: {TRAIN_AUDIO}")

# Load taxonomy
tax = pd.read_csv(COMP / "taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
print(f"Classes: {len(classes)}")

# Find which classes have train_audio folders
class_dirs = {}
for c in classes:
    d = TRAIN_AUDIO / c
    if d.exists():
        files = sorted(d.glob("*.ogg")) + sorted(d.glob("*.wav"))
        if files:
            class_dirs[c] = files
print(f"Classes with train_audio: {len(class_dirs)}/{len(classes)}")

# Setup Perch
so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(PERCH_PATH, sess_options=so, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
print(f"Perch input: {in_name}")
print(f"Perch outputs: {[o.name for o in sess.get_outputs()]}")

SR = 32000
WIN_SAMPLES = 5 * SR
N_WINDOWS = 12

def load_and_chunk(fp):
    """Load audio, chunk to up to 12 × 5s windows."""
    try:
        y, sr = sf.read(str(fp))
        if y.ndim > 1:
            y = y.mean(axis=1)
        # Resample if needed (assume 32k already from spec)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        n_full = len(y) // WIN_SAMPLES
        n_use = min(n_full, N_WINDOWS) if n_full > 0 else 1
        if n_full == 0:
            # Pad to one full window
            padded = np.zeros(WIN_SAMPLES, dtype=np.float32)
            padded[:len(y)] = y
            return padded.reshape(1, WIN_SAMPLES)
        return y[:n_use * WIN_SAMPLES].reshape(n_use, WIN_SAMPLES).astype(np.float32)
    except Exception as e:
        return None

# Process: for each class, take up to 15 files, get Perch embeddings, average
MAX_FILES_PER_CLASS = 12
N_total_est = sum(min(MAX_FILES_PER_CLASS, len(v)) for v in class_dirs.values()) * 3  # ~3 windows avg per file
print(f"Estimated rows: ~{N_total_est}")

all_emb = []
all_y = []
all_class = []
all_file = []

t0 = time.time()
n_done = 0
for c_label, files in class_dirs.items():
    c_idx = classes.index(c_label)
    sel_files = files[:MAX_FILES_PER_CLASS]
    for fp in sel_files:
        chunks = load_and_chunk(fp)
        if chunks is None: continue
        try:
            emb = sess.run(["output_0"], {in_name: chunks})[0] if "output_0" in [o.name for o in sess.get_outputs()] else sess.run(None, {in_name: chunks})[0]
            for i in range(emb.shape[0]):
                all_emb.append(emb[i])
                y_vec = np.zeros(len(classes), dtype=np.float32)
                y_vec[c_idx] = 1.0
                all_y.append(y_vec)
                all_class.append(c_label)
                all_file.append(fp.name)
        except Exception as e:
            pass
    n_done += 1
    if n_done % 30 == 0 or n_done == len(class_dirs):
        el = time.time() - t0
        rate = n_done / el
        print(f"  [{n_done}/{len(class_dirs)}] {el:.0f}s rate={rate:.2f}cls/s eta={(len(class_dirs)-n_done)/max(rate,0.01):.0f}s rows={len(all_emb)}")

print(f"\nTotal: {time.time()-t0:.0f}s, embeddings: {len(all_emb)}")
emb_arr = np.array(all_emb, dtype=np.float32)
y_arr = np.array(all_y, dtype=np.float32)
print(f"emb shape: {emb_arr.shape}")
print(f"y shape: {y_arr.shape}")
print(f"Classes covered: {y_arr.sum(axis=0).astype(int).tolist()[:20]}...")

# Save
out_path = "/kaggle/working/train_audio_perch.npz"
np.savez(out_path,
         emb=emb_arr,
         Y=y_arr,
         classes=np.array(classes),
         file_names=np.array(all_file),
         class_labels=np.array(all_class))
print(f"Saved: {out_path}")
print(f"Class coverage: {(y_arr.sum(axis=0) > 0).sum()}/{len(classes)} classes")
