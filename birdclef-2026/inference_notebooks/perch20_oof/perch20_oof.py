"""Perch 2.0 CPU inference on labeled OOF — uses internet to install TF 2.20+.

Internet ON: pip install --upgrade tensorflow tensorflow-cpu directly.
This unblocks the XLA v10 / ABI compatibility issue we hit with offline wheels.
"""
import os, sys, subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Install TF 2.20+ from pypi (internet enabled in kernel metadata)
print("Installing TF 2.20+ from pypi...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "tensorflow-cpu>=2.20",
    "keras>=3.5",
])
print("Install done")

import re, time, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
print(f"TF version: {tf.__version__}")

# Locate Perch 2.0 CPU SavedModel
PERCH20_ROOT = None
for cand in glob.glob("/kaggle/input/**/saved_model.pb", recursive=True):
    p = Path(cand).parent
    # Prefer perch_v2_cpu variant
    if "cpu" in str(p):
        PERCH20_ROOT = p; print(f"Found Perch 2.0 CPU: {p}"); break
    if PERCH20_ROOT is None:
        PERCH20_ROOT = p
assert PERCH20_ROOT

print(f"Loading: {PERCH20_ROOT}")
model = tf.saved_model.load(str(PERCH20_ROOT))
sig = model.signatures["serving_default"]
print(f"Inputs: {sig.structured_input_signature}")
print(f"Outputs: {list(sig.structured_outputs.keys())}")

# Test forward
test = tf.constant(np.zeros((1, 160000), dtype=np.float32))
input_key = list(sig.structured_input_signature[1].keys())[0]
out = sig(**{input_key: test})
print("\nForward works!")
for k, v in out.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

# Load Perch labels (14795 inat2024 species names)
LABELS_CSV = None
for p in PERCH20_ROOT.rglob("*"):
    if p.name == "labels.csv":
        LABELS_CSV = p; break
perch_labels = pd.read_csv(LABELS_CSV) if LABELS_CSV else None
EBIRD_CSV = None
for p in PERCH20_ROOT.rglob("*"):
    if "ebird" in p.name and p.name.endswith(".csv"):
        EBIRD_CSV = p; break
ebird_labels = pd.read_csv(EBIRD_CSV) if EBIRD_CSV else None
print(f"\nPerch labels: {len(perch_labels) if perch_labels is not None else '?'}")
print(f"Ebird codes: {len(ebird_labels) if ebird_labels is not None else '?'}")

# Load BC2026 + comp data
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
HOUR_RE = re.compile(r"_\d{8}_(\d{2})\d{4}")
hours = np.array([int(HOUR_RE.search(f).group(1)) if HOUR_RE.search(f) else -1 for f in labels["filename"]])

# Map BC2026 → Perch indices via scientific name OR ebird code OR taxon ID
perch_sci_to_idx = {}  # scientific name lower → perch index
perch_ebird_to_idx = {}  # ebird code → perch index
if perch_labels is not None:
    col = perch_labels.columns[0]
    for pi, name in enumerate(perch_labels[col]):
        perch_sci_to_idx[str(name).strip().lower()] = pi
if ebird_labels is not None:
    col = ebird_labels.columns[0]
    for pi, name in enumerate(ebird_labels[col]):
        perch_ebird_to_idx[str(name).strip().lower()] = pi

# BC2026 has both numeric iNat IDs (e.g. "65380") and 6-letter ebird codes (e.g. "ashgre1")
bc_to_perch = {}
for bi, bc_lbl in enumerate(bc_classes):
    sci = str(tax.iloc[bi].get("scientific_name", "")).strip().lower()
    if sci and sci in perch_sci_to_idx:
        bc_to_perch[bi] = perch_sci_to_idx[sci]; continue
    if str(bc_lbl).lower() in perch_ebird_to_idx:
        bc_to_perch[bi] = perch_ebird_to_idx[str(bc_lbl).lower()]; continue
print(f"\nBC2026 → Perch 2.0 mapping: {len(bc_to_perch)}/{C} classes mapped")

# Process labeled files
SR = 32000
WIN_SAMP = 5 * SR
ts_dir = COMP / "train_soundscapes"
file_to_idx = {}
for i, f in enumerate(labels["filename"]):
    file_to_idx.setdefault(f, []).append(i)
print(f"\nProcessing {len(file_to_idx)} files...")

P_perch20 = np.zeros((N, C), dtype=np.float32)
emb20 = np.zeros((N, 1536), dtype=np.float32)
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
        st = labels.iloc[row_i]["start"]
        if isinstance(st, str) and ":" in st:
            h, m, s = st.split(":"); ss = int(h)*3600+int(m)*60+int(s)
        else:
            ss = int(float(st))
        s_samp = ss * SR
        clip = np.zeros(WIN_SAMP, dtype=np.float32)
        avail = max(0, len(audio) - s_samp)
        if avail >= WIN_SAMP:
            clip = audio[s_samp:s_samp+WIN_SAMP]
        elif avail > 0:
            clip[:avail] = audio[s_samp:s_samp+avail]
        win_batch.append(clip)
    if not win_batch: continue
    batch = tf.constant(np.stack(win_batch).astype(np.float32))
    out = sig(**{input_key: batch})
    logits = out["label"].numpy()  # (B, 14795)
    embs = out["embedding"].numpy()  # (B, 1536)
    sigmoid_probs = 1.0 / (1.0 + np.exp(-logits))
    for j, row_i in enumerate(idx_list):
        emb20[row_i] = embs[j]
        for bc_i, perch_i in bc_to_perch.items():
            P_perch20[row_i, bc_i] = sigmoid_probs[j, perch_i]
    n_done += 1
    if n_done % 10 == 0:
        el = time.time() - t0
        print(f"  [{n_done}/{len(file_to_idx)}] {el:.0f}s rate={n_done/el:.2f}/s")

print(f"\nTotal: {time.time()-t0:.0f}s")

# Save
np.savez("/kaggle/working/perch20_labeled_oof.npz",
         P_perch20=P_perch20, emb=emb20, Y=Y,
         classes=np.array(bc_classes),
         row_filename=labels["filename"].to_numpy())
print("Saved.")

# Quick AUC
from sklearn.metrics import roc_auc_score
aucs = []
for c in range(C):
    if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
    if P_perch20[:, c].max() == 0: continue
    try: aucs.append(roc_auc_score(Y[:, c], P_perch20[:, c]))
    except: pass
print(f"\nPerch 2.0 macro-AUC on labeled OOF: {np.mean(aucs):.4f} ({len(aucs)} mapped classes)")
