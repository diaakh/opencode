"""Perch 2.0 CPU on labeled OOF — uses TF 2.21 installed offline.

Kaggle's bundled TF 2.18 can't deserialize Perch 2.0's XLA modules. Solution:
install TF 2.21 (which we used locally and confirmed works) via offline wheel.
"""
import os, sys, subprocess, glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Find wheels — try multiple paths
WHEEL_PATHS = [
    "/kaggle/input/tensorflow-221-py312-wheel",
    "/kaggle/input/datasets/adkasd/tensorflow-221-py312-wheel",
]
TF_WHEEL_DIR = None
for p in WHEEL_PATHS:
    if os.path.exists(p):
        TF_WHEEL_DIR = p; break
assert TF_WHEEL_DIR, f"TF 2.21 wheels not attached: {WHEEL_PATHS}"
print(f"TF wheels at: {TF_WHEEL_DIR}")

# Install TF 2.21 CPU + deps
print("Installing TF 2.21 CPU + deps offline...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "--no-index", "--find-links", TF_WHEEL_DIR,
    f"{TF_WHEEL_DIR}/numpy-2.4.5-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
    f"{TF_WHEEL_DIR}/protobuf-7.34.1-py3-none-any.whl",
    f"{TF_WHEEL_DIR}/ml_dtypes-0.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
    f"{TF_WHEEL_DIR}/keras-3.14.1-py3-none-any.whl",
    f"{TF_WHEEL_DIR}/tensorflow_cpu-2.21.0-cp312-cp312-manylinux_2_27_x86_64.whl",
])
print("TF install done")




# Import everything we need
import re, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
tf.config.optimizer.set_jit(False)
print(f"TF version: {tf.__version__}")

# Find Perch 2.0 SavedModel
# Kaggle Models mount under /kaggle/input/<owner>-<model>/<framework>/<variation>/<version>/
PERCH20_ROOT = None
for cand in glob.glob("/kaggle/input/**/saved_model.pb", recursive=True):
    PERCH20_ROOT = Path(cand).parent
    print(f"Found saved_model: {PERCH20_ROOT}")
    break

if not PERCH20_ROOT:
    # Search for Perch variations
    print("Searching /kaggle/input for Perch 2.0 model...")
    for p in Path("/kaggle/input").rglob("*"):
        if "perch" in str(p).lower() and p.is_dir():
            print(f"  {p}")

assert PERCH20_ROOT, "Perch 2.0 SavedModel not found"

import tensorflow as tf
tf.config.optimizer.set_jit(False)
print(f"TF version: {tf.__version__}")
print(f"Loading: {PERCH20_ROOT}")
model = tf.saved_model.load(str(PERCH20_ROOT))
print(f"Model signatures: {list(model.signatures.keys())}")
sig = model.signatures.get("serving_default") or list(model.signatures.values())[0]
print(f"Inputs: {sig.structured_input_signature}")
print(f"Outputs: {list(sig.structured_outputs.keys())}")

# Load labels
COMP = None
for cand in ["/kaggle/input/birdclef-2026", "/kaggle/input/competitions/birdclef-2026"]:
    if Path(cand).exists():
        COMP = Path(cand); break
labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)
tax = pd.read_csv(COMP / "taxonomy.csv")
bc_classes = tax["primary_label"].astype(str).tolist()
N = len(labels); C = len(bc_classes)
print(f"\nLabels: {N} rows, {labels['filename'].nunique()} files")
cls_idx = {c: i for i, c in enumerate(bc_classes)}
Y = np.zeros((N, C), dtype=np.float32)
for i, row in labels.iterrows():
    for c in str(row["primary_label"]).split(";"):
        c = c.strip()
        if c in cls_idx: Y[i, cls_idx[c]] = 1.0

# Process files
SR = 32000
WIN_SAMP = 5 * SR
ts_dir = COMP / "train_soundscapes"
file_to_idx = {}
for i, f in enumerate(labels["filename"]):
    file_to_idx.setdefault(f, []).append(i)

# Look at output shapes to understand mapping
# Probe with one window
print(f"\nProbing one window...")
test_audio = np.zeros((1, WIN_SAMP), dtype=np.float32)
input_key = list(sig.structured_input_signature[1].keys())[0]
test_in = tf.constant(test_audio)
out = sig(**{input_key: test_in})
for k, v in out.items():
    print(f"  output {k}: shape={v.shape}, dtype={v.dtype}")

# Figure out which output is the classification logits/probabilities
# Likely "logits" or "predictions" or similar
LOGIT_KEY = None
EMB_KEY = None
for k, v in out.items():
    if v.shape[-1] is not None and v.shape[-1] > 1000:  # 14795 classes
        LOGIT_KEY = k
    elif v.shape[-1] is not None and v.shape[-1] >= 512 and v.shape[-1] < 2000:
        EMB_KEY = k

print(f"Logit key: {LOGIT_KEY}, Emb key: {EMB_KEY}")

# Load Perch labels/species mapping
# Perch 2.0 has its own label file — search for it
PERCH_LABELS = None
for f in PERCH20_ROOT.rglob("*"):
    name = f.name.lower()
    if "label" in name or "class" in name or "taxa" in name:
        print(f"  found {f}")
        PERCH_LABELS = f
        break

# Even without label mapping, we can use embeddings as RAG
# For now just save embeddings + logits and we'll map later

print("\n=== Processing 66 files ===")
all_emb = []
all_logits = []
all_meta = []
import time
t0 = time.time()
n_done = 0
for f, idx_list in file_to_idx.items():
    fpath = ts_dir / f
    if not fpath.exists():
        for ext in [".ogg", ".wav", ".flac"]:
            if (ts_dir / (f + ext)).exists(): fpath = ts_dir / (f + ext); break
    try:
        audio, sr = sf.read(str(fpath))
        if audio.ndim > 1: audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
    except: continue
    
    win_batch = []
    for row_i in idx_list:
        st_val = labels.iloc[row_i]["start"]
        if isinstance(st_val, str) and ":" in st_val:
            h, m, s = st_val.split(":")
            ss = int(h)*3600 + int(m)*60 + int(s)
        else:
            ss = int(float(st_val))
        s_samp = ss * SR
        e_samp = s_samp + WIN_SAMP
        if e_samp <= len(audio):
            clip = audio[s_samp:e_samp]
        else:
            clip = np.zeros(WIN_SAMP, dtype=np.float32)
            avail = max(0, len(audio) - s_samp)
            if avail > 0: clip[:avail] = audio[s_samp:s_samp+avail]
        win_batch.append(clip)
    
    if not win_batch: continue
    batch_in = tf.constant(np.stack(win_batch).astype(np.float32))
    out = sig(**{input_key: batch_in})
    for j, row_i in enumerate(idx_list):
        all_meta.append(row_i)
        if EMB_KEY: all_emb.append(out[EMB_KEY].numpy()[j])
        if LOGIT_KEY: all_logits.append(out[LOGIT_KEY].numpy()[j])
    n_done += 1
    if n_done % 10 == 0:
        el = time.time() - t0
        print(f"  [{n_done}/{len(file_to_idx)}] {el:.0f}s rate={n_done/el:.2f}/s")

emb_arr = np.array(all_emb, dtype=np.float32) if all_emb else None
logits_arr = np.array(all_logits, dtype=np.float32) if all_logits else None
meta_arr = np.array(all_meta, dtype=np.int32)
print(f"\nTotal: {time.time()-t0:.0f}s")
if emb_arr is not None: print(f"  Embeddings: {emb_arr.shape}")
if logits_arr is not None: print(f"  Logits: {logits_arr.shape}")

# Save
np.savez("/kaggle/working/perch20_labeled_oof.npz",
         emb=emb_arr if emb_arr is not None else np.zeros((0, 0)),
         logits=logits_arr if logits_arr is not None else np.zeros((0, 0)),
         meta=meta_arr,
         Y=Y, classes=np.array(bc_classes),
         row_filename=labels["filename"].to_numpy())
print("Saved: /kaggle/working/perch20_labeled_oof.npz")
