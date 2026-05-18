"""Perch 2.0 (multi-taxa Aug 2025) inference via ONNX — bypasses TF/XLA issues.

Uses nina2025/perch-v2-no-dft-v3-onnx which is actually Perch 2.0 ONNX export
(413MB, 14,795 classes — matches Perch 2.0 not v1's 10,932).

Outputs:
  - embedding (1536d)
  - spatial_embedding (16, 4, 1536)
  - spectrogram (500, 128)
  - label (14,795 logits)
"""
import os, sys, glob, re, time, subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Install onnxruntime from the bundled wheel in nina's dataset
ort_wheel = None
for p in glob.glob("/kaggle/input/**/onnxruntime-*.whl", recursive=True):
    if "1.24" in p: ort_wheel = p; break
if ort_wheel:
    print(f"Installing {ort_wheel}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", ort_wheel])

import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
print(f"onnxruntime {ort.__version__}")

# Find Perch 2.0 ONNX (nina2025's, distinguishable by 14795 classes)
ONNX_PATH = None
for p in glob.glob("/kaggle/input/**/perch_v2_no_dft.onnx", recursive=True):
    # Check if size matches Perch 2.0 (~413MB)
    sz = os.path.getsize(p) / 1e6
    if 350 < sz < 450:
        ONNX_PATH = p
        print(f"Perch 2.0 ONNX: {p} ({sz:.0f}MB)")
        break
assert ONNX_PATH
from pathlib import Path

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"])
print(f"Inputs: {[(i.name, i.shape) for i in sess.get_inputs()]}")
print(f"Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")
input_name = sess.get_inputs()[0].name

# Quick forward test
test_audio = np.zeros((2, 160000), dtype=np.float32)
out = sess.run(None, {input_name: test_audio})
out_names = [o.name for o in sess.get_outputs()]
for n, v in zip(out_names, out):
    print(f"  {n}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

# Locate label index for 14795 outputs (perch 2.0)
LABEL_IDX = out_names.index("label") if "label" in out_names else None
EMB_IDX = out_names.index("embedding") if "embedding" in out_names else None
print(f"label idx: {LABEL_IDX}, emb idx: {EMB_IDX}")

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

# Find Perch 2.0 class labels (search for any csv with 14795 rows in /kaggle/input)
PERCH_LABELS = None
PERCH_EBIRD = None
for p in glob.glob("/kaggle/input/**/*.csv", recursive=True):
    sz = os.path.getsize(p)
    if 100_000 < sz < 500_000:
        try:
            df = pd.read_csv(p)
            if len(df) > 14000 and len(df) < 15000:
                cn = df.columns[0].lower()
                if "ebird" in cn or "ebird" in p.lower():
                    PERCH_EBIRD = df; print(f"Perch ebird labels: {p}")
                else:
                    PERCH_LABELS = df; print(f"Perch sci labels: {p}")
        except: pass

print(f"\nPerch labels: {len(PERCH_LABELS) if PERCH_LABELS is not None else 0}")
print(f"Perch ebird: {len(PERCH_EBIRD) if PERCH_EBIRD is not None else 0}")

# Build BC2026 → Perch 2.0 mapping
sci_to_pi = {}
ebird_to_pi = {}
if PERCH_LABELS is not None:
    for pi, s in enumerate(PERCH_LABELS.iloc[:, 0]):
        sci_to_pi[str(s).strip().lower()] = pi
if PERCH_EBIRD is not None:
    for pi, s in enumerate(PERCH_EBIRD.iloc[:, 0]):
        ebird_to_pi[str(s).strip().lower()] = pi

bc_to_perch = {}
for bi, bc_lbl in enumerate(bc_classes):
    sci = str(tax.iloc[bi].get("scientific_name", "")).strip().lower()
    if sci in sci_to_pi: bc_to_perch[bi] = sci_to_pi[sci]; continue
    if str(bc_lbl).lower() in ebird_to_pi: bc_to_perch[bi] = ebird_to_pi[str(bc_lbl).lower()]; continue
print(f"BC2026 → Perch 2.0 mapping: {len(bc_to_perch)}/{C}")

# Process labeled files
SR = 32000; WIN_SAMP = 5 * SR
ts_dir = COMP / "train_soundscapes"
file_to_idx = {}
for i, f in enumerate(labels["filename"]):
    file_to_idx.setdefault(f, []).append(i)
P_p20 = np.zeros((N, C), dtype=np.float32)
emb_p20 = np.zeros((N, 1536), dtype=np.float32)
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
    out = sess.run(None, {input_name: batch})
    logits = out[LABEL_IDX]  # (B, 14795)
    embs = out[EMB_IDX]  # (B, 1536)
    probs = 1.0 / (1.0 + np.exp(-logits))
    for j, row_i in enumerate(idx_list):
        emb_p20[row_i] = embs[j]
        for bi, pi in bc_to_perch.items():
            P_p20[row_i, bi] = probs[j, pi]
    nd += 1
    if nd % 10 == 0:
        el = time.time()-t0
        print(f"  [{nd}/{len(file_to_idx)}] {el:.0f}s rate={nd/el:.2f}/s")
print(f"\nTotal: {time.time()-t0:.0f}s")

np.savez("/kaggle/working/perch20_labeled_oof.npz",
    P_perch20=P_p20.astype(np.float32),
    emb=emb_p20.astype(np.float32),
    Y=Y,
    classes=np.array(bc_classes),
    row_filename=labels["filename"].to_numpy())
print("Saved.")

from sklearn.metrics import roc_auc_score
aucs = []
for c in range(C):
    if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
    if P_p20[:, c].max() == 0: continue
    try: aucs.append(roc_auc_score(Y[:, c], P_p20[:, c]))
    except: pass
print(f"\nPerch 2.0 macro-AUC on labeled OOF: {np.mean(aucs):.4f} ({len(aucs)} mapped classes)")
