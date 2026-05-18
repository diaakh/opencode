"""Run long_convnextv2-tiny SED model on all 10,592 unlabeled train_soundscapes
files, save predictions for context-aware pseudo-labeling.

Output: /kaggle/working/long_convnext_on_unlabeled.npz
"""
import os, re, glob, sys, time, subprocess, gc
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

# Install onnxruntime (offline wheel)
ort_whls = sorted(glob.glob("/kaggle/input/**/onnxruntime*.whl", recursive=True))
if ort_whls:
    print(f"Installing {ort_whls[0]}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", ort_whls[0]])
import onnxruntime as ort

# Find paths
print("=== /kaggle/input ===")
for p in sorted(glob.glob("/kaggle/input/*"))[:30]:
    print(" ", p)

# Locate long_convnextv2 ONNX (file is named fold0.onnx etc.)
onnx_cands = sorted(glob.glob("/kaggle/input/**/long-convnextv2-tiny-onnx/fold*.onnx", recursive=True))
if not onnx_cands:
    onnx_cands = sorted(glob.glob("/kaggle/input/**/fold*.onnx", recursive=True))
print(f"ONNX candidates: {onnx_cands}")
assert onnx_cands, "no fold*.onnx found under long-convnextv2-tiny-onnx"
ONNX_PATH = onnx_cands[0]  # fold0 only; ensemble would 5x runtime
print(f"Using fold: {ONNX_PATH}")

# Locate competition dir
COMP_CANDS = sorted(glob.glob("/kaggle/input/**/birdclef-2026", recursive=False))
if not COMP_CANDS:
    COMP_CANDS = [p for p in sorted(glob.glob("/kaggle/input/competitions/*")) if "birdclef" in p]
COMP = Path(COMP_CANDS[0])
print(f"Competition dir: {COMP}")
SOUNDSCAPES = COMP / "train_soundscapes"

# Load taxonomy
tax = pd.read_csv(COMP / "taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
N_CLS = len(classes)
print(f"Classes: {N_CLS}")

# Load ONNX
so = ort.SessionOptions()
so.intra_op_num_threads = 4
sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=["CPUExecutionProvider"])
inputs = sess.get_inputs()
outputs = sess.get_outputs()
print(f"Inputs: {[(i.name, i.shape) for i in inputs]}")
print(f"Outputs: {[(o.name, o.shape) for o in outputs]}")

# Long-format models accept 60s waveform → output (12, 234)
# Find SR (assume 32000)
SR = 32_000
N_SAMPLES = SR * 60

# List all unlabeled train_soundscape files
all_files = sorted(SOUNDSCAPES.glob("*.ogg"))
print(f"Total train_soundscape files: {len(all_files)}")

# Exclude the 66 labeled ones (we already have labeled preds elsewhere)
try:
    labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
    labeled_names = set(labels["filename"].unique())
    unlabeled_files = [p for p in all_files if p.name not in labeled_names]
    print(f"Unlabeled files: {len(unlabeled_files)}")
except Exception as e:
    print(f"WARN: couldn't read labels CSV, processing all {len(all_files)} files: {e}")
    unlabeled_files = all_files

# Inference loop
N_FILES = len(unlabeled_files)
N_WIN = 12
# Storage: per-window predictions
all_preds = np.full((N_FILES * N_WIN, N_CLS), np.nan, dtype=np.float32)
all_row_ids = []
all_stems = []
t0 = time.time()
fail = 0
input_name = inputs[0].name

# Determine expected input length
exp_len = inputs[0].shape[-1] if isinstance(inputs[0].shape[-1], int) else N_SAMPLES
print(f"Expected input length: {exp_len} (will use {N_SAMPLES})")

# Some long-format models expect (B, T); others (B, 1, T). Check first.
expects_3d = len(inputs[0].shape) == 3
print(f"Expects 3D input: {expects_3d}")

for fi, fp in enumerate(unlabeled_files):
    try:
        y, sr = sf.read(str(fp), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        # Pad / truncate to 60s
        if len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)))
        else:
            y = y[:N_SAMPLES]
        x = y.reshape(1, 1, N_SAMPLES) if expects_3d else y.reshape(1, N_SAMPLES)
        out = sess.run(None, {input_name: x.astype(np.float32)})
        # Find the (1, 12, 234) output
        preds = None
        for o in out:
            a = np.asarray(o).squeeze(0)
            if a.ndim == 2 and a.shape[0] == N_WIN and a.shape[1] == N_CLS:
                preds = a
                break
            elif a.ndim == 2 and a.shape == (N_WIN, N_CLS):
                preds = a
                break
        if preds is None:
            # Fallback: take first 2D output and reshape
            preds = np.asarray(out[0]).reshape(N_WIN, N_CLS)
        # Apply sigmoid if logits
        if preds.max() > 1.0 or preds.min() < 0.0:
            preds = 1.0 / (1.0 + np.exp(-preds))
        stem = fp.stem
        for w in range(N_WIN):
            row_idx = fi * N_WIN + w
            all_preds[row_idx] = preds[w]
            all_row_ids.append(f"{stem}_{(w+1)*5}")
            all_stems.append(stem)
    except Exception as e:
        fail += 1
        if fail < 5:
            print(f"  fail {fp.name}: {e}")

    if (fi + 1) % 500 == 0 or fi == N_FILES - 1:
        dt = time.time() - t0
        rate = (fi + 1) / dt
        eta = (N_FILES - fi - 1) / rate / 60.0
        print(f"  [{fi+1}/{N_FILES}] dt={dt/60:.1f}min rate={rate:.1f}/s eta={eta:.1f}min  fails={fail}")

print(f"Done. Total: {time.time()-t0:.1f}s, fails={fail}")

# Save
out_path = "/kaggle/working/long_convnext_on_unlabeled.npz"
np.savez_compressed(
    out_path,
    P=all_preds,
    row_ids=np.array(all_row_ids),
    stems=np.array(all_stems),
    classes=np.array(classes),
)
print(f"Saved {out_path}: shape={all_preds.shape}")
