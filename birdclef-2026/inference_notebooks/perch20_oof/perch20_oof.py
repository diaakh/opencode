"""Perch 2.0 via perch-hoplite — full inference on labeled OOF."""
import os, sys, subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "perch-hoplite"])

import re, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
print(f"TF: {tf.__version__}")

from perch_hoplite.zoo import model_configs
ModelConfigName = model_configs.ModelConfigName

# Load Perch 2.0 CPU
print("Loading Perch 2.0 CPU via perch-hoplite...")
model = model_configs.load_model_by_name(ModelConfigName.PERCH_V2_CPU)
print(f"Model: {type(model).__name__}")
print(f"Methods: {[m for m in dir(model) if not m.startswith('_')]}")

# Test forward
test_audio = np.zeros(160000, dtype=np.float32)
out = model.embed(test_audio[np.newaxis])
print(f"\nForward works!")
print(f"Output type: {type(out)}")
if hasattr(out, '_asdict'):
    for k, v in out._asdict().items():
        if v is not None:
            try: print(f"  {k}: shape={v.shape if hasattr(v, 'shape') else type(v).__name__}")
            except: print(f"  {k}: {type(v).__name__}")
elif hasattr(out, '__dict__'):
    for k, v in vars(out).items():
        if v is not None:
            try: print(f"  {k}: shape={v.shape if hasattr(v, 'shape') else type(v).__name__}")
            except: print(f"  {k}: {type(v).__name__}")

# BC2026 setup
COMP = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP.exists(): COMP = Path("/kaggle/input/birdclef-2026")
labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)
tax = pd.read_csv(COMP / "taxonomy.csv")
bc_classes = tax["primary_label"].astype(str).tolist()
N = len(labels); C = len(bc_classes)
cls_idx = {c: i for i, c in enumerate(bc_classes)}
Y = np.zeros((N, C), dtype=np.float32)
for i, row in labels.iterrows():
    for c in str(row["primary_label"]).split(";"):
        c = c.strip()
        if c in cls_idx: Y[i, cls_idx[c]] = 1.0

# Find Perch class labels
class_list_names = ['class_list_names', 'taxonomy', 'class_lists', 'classes']
class_list = None
for n in class_list_names:
    if hasattr(model, n):
        class_list = getattr(model, n)
        print(f"\nFound classes via {n}: {type(class_list)}")
        if hasattr(class_list, '__len__'): print(f"  count: {len(class_list)}")
        break

# Process files
SR = 32000; WIN_SAMP = 5 * SR
ts_dir = COMP / "train_soundscapes"
file_to_idx = {}
for i, f in enumerate(labels["filename"]):
    file_to_idx.setdefault(f, []).append(i)

# Process first file to get output shape
all_logits = {}  # row_i -> 1D array of logits
all_emb = {}     # row_i -> 1D embedding
print(f"\nProcessing {len(file_to_idx)} files...")
t0 = time.time(); nd = 0
for f, idx_list in file_to_idx.items():
    fpath = ts_dir / f
    if not fpath.exists():
        for ext in [".ogg", ".wav", ".flac"]:
            if (ts_dir / (f + ext)).exists(): fpath = ts_dir / (f + ext); break
    try:
        audio, _ = sf.read(str(fpath))
        if audio.ndim > 1: audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
    except: continue
    wins = []
    for row_i in idx_list:
        st = labels.iloc[row_i]["start"]
        if isinstance(st, str) and ":" in st:
            h, m, s = st.split(":"); ss = int(h)*3600+int(m)*60+int(s)
        else: ss = int(float(st))
        s_samp = ss * SR
        clip = np.zeros(WIN_SAMP, dtype=np.float32)
        avail = max(0, len(audio) - s_samp)
        if avail >= WIN_SAMP: clip = audio[s_samp:s_samp+WIN_SAMP]
        elif avail > 0: clip[:avail] = audio[s_samp:s_samp+avail]
        wins.append(clip)
    if not wins: continue
    batch = np.stack(wins).astype(np.float32)
    out = model.embed(batch)
    # Access logits + embedding
    logits_arr = None; emb_arr = None
    if hasattr(out, 'logits'):
        logits_dict = out.logits
        if isinstance(logits_dict, dict) and logits_dict:
            logits_arr = next(iter(logits_dict.values()))
            if hasattr(logits_arr, 'numpy'): logits_arr = logits_arr.numpy()
    if hasattr(out, 'embeddings') and out.embeddings is not None:
        emb_arr = out.embeddings
        if hasattr(emb_arr, 'numpy'): emb_arr = emb_arr.numpy()
    elif hasattr(out, 'embedding') and out.embedding is not None:
        emb_arr = out.embedding
        if hasattr(emb_arr, 'numpy'): emb_arr = emb_arr.numpy()
    # logits_arr expected (B, N_classes); emb_arr (B, D)
    if logits_arr is not None and logits_arr.ndim == 3:
        # might be (B, T, C) — pool
        logits_arr = logits_arr.mean(axis=1)
    if emb_arr is not None and emb_arr.ndim == 3:
        emb_arr = emb_arr.mean(axis=1)
    for j, row_i in enumerate(idx_list):
        if logits_arr is not None and j < logits_arr.shape[0]:
            all_logits[row_i] = logits_arr[j]
        if emb_arr is not None and j < emb_arr.shape[0]:
            all_emb[row_i] = emb_arr[j]
    nd += 1
    if nd % 10 == 0:
        el = time.time()-t0
        print(f"  [{nd}/{len(file_to_idx)}] {el:.0f}s rate={nd/el:.2f}/s")
print(f"Total: {time.time()-t0:.0f}s")

# Save raw outputs
if all_logits:
    L0 = next(iter(all_logits.values()))
    P = np.zeros((N, L0.shape[-1]), dtype=np.float32)
    for ri, v in all_logits.items():
        P[ri] = 1.0/(1.0+np.exp(-v))
    print(f"Logits → P shape: {P.shape}")
else:
    P = np.zeros((N, 0), dtype=np.float32)

if all_emb:
    E0 = next(iter(all_emb.values()))
    E = np.zeros((N, E0.shape[-1]), dtype=np.float32)
    for ri, v in all_emb.items():
        E[ri] = v
    print(f"Embeddings shape: {E.shape}")
else:
    E = np.zeros((N, 0), dtype=np.float32)

np.savez("/kaggle/working/perch20_labeled_oof.npz",
    P_perch20=P, emb=E, Y=Y,
    classes=np.array(bc_classes),
    row_filename=labels["filename"].to_numpy())
print("Saved.")

# Class mapping then quick AUC on mapped
if P.shape[1] > 0 and class_list is not None:
    # class_list may have ebird codes or scientific names
    # We'd need to inspect first - print a few
    if hasattr(class_list, '__getitem__'):
        try: print(f"First 5 classes: {list(class_list[:5])}")
        except: pass
