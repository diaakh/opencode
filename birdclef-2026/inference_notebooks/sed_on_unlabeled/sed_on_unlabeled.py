"""GPU-accelerated long_convnextv2 on all 10,592 unlabeled train_soundscapes.

Speedups over v2 (CPU, 0.3 files/s, 9h ETA):
  - CUDAExecutionProvider for ONNX
  - File-batched ONNX inference (batch=8)
  - ThreadPoolExecutor prefetch (load next batch's audio while GPU runs current)
  - Sigmoid in numpy vectorized

Target: 50-100 files/s → ~3-5 min total
"""
import os, re, glob, sys, time, subprocess, gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import soundfile as sf

# ---------- Install onnxruntime-gpu if available, else onnxruntime ----------
gpu_whls = sorted(glob.glob("/kaggle/input/**/onnxruntime[-_]gpu*.whl", recursive=True))
cpu_whls = sorted(glob.glob("/kaggle/input/**/onnxruntime-1*.whl", recursive=True))
for whl in (gpu_whls + cpu_whls):
    try:
        print(f"Installing {whl}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", whl])
        break
    except Exception as e:
        print(f"  failed: {e}")
import onnxruntime as ort
print(f"ONNX runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

# ---------- Locate files ----------
print("\n=== /kaggle/input ===")
for p in sorted(glob.glob("/kaggle/input/*"))[:30]:
    print(" ", p)

onnx_cands = sorted(glob.glob("/kaggle/input/**/long-convnextv2-tiny-onnx/fold*.onnx", recursive=True))
if not onnx_cands:
    onnx_cands = sorted(glob.glob("/kaggle/input/**/fold*.onnx", recursive=True))
ONNX_PATH = onnx_cands[0]
print(f"ONNX: {ONNX_PATH}")

COMP_CANDS = sorted(glob.glob("/kaggle/input/competitions/birdclef-2026"))
COMP = Path(COMP_CANDS[0]) if COMP_CANDS else Path("/kaggle/input/birdclef-2026")
SOUNDSCAPES = COMP / "train_soundscapes"
print(f"Competition dir: {COMP}")

tax = pd.read_csv(COMP / "taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
N_CLS = len(classes)

# ---------- Load ONNX with GPU if possible ----------
so = ort.SessionOptions()
so.intra_op_num_threads = 4
preferred = []
avail = ort.get_available_providers()
if "CUDAExecutionProvider" in avail:
    preferred.append("CUDAExecutionProvider")
elif "TensorrtExecutionProvider" in avail:
    preferred.append("TensorrtExecutionProvider")
preferred.append("CPUExecutionProvider")
sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=preferred)
print(f"ONNX providers in use: {sess.get_providers()}")
print(f"Inputs: {[(i.name, i.shape) for i in sess.get_inputs()]}")
print(f"Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
# Expected: (batch, 1920000) for 60s @ 32kHz
SR = 32_000
N_SAMPLES = SR * 60
N_WIN = 12

# Test if batching is supported (dynamic batch dim)
try:
    test = np.zeros((4, N_SAMPLES), dtype=np.float32)
    out = sess.run(None, {input_name: test})
    BATCH = 8
    print(f"Batching supported, will use BATCH={BATCH}")
except Exception as e:
    BATCH = 1
    print(f"Batching not supported, using BATCH=1: {e}")

# ---------- Determine files to process ----------
all_files = sorted(SOUNDSCAPES.glob("*.ogg"))
try:
    labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
    labeled_names = set(labels["filename"].unique())
    files = [p for p in all_files if p.name not in labeled_names]
except Exception:
    files = all_files
N_FILES = len(files)
print(f"Files to process: {N_FILES}")


def load_audio(fp):
    try:
        y, sr = sf.read(str(fp), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)))
        else:
            y = y[:N_SAMPLES]
        return y, fp.stem, None
    except Exception as e:
        return None, fp.stem, str(e)


# ---------- Inference loop ----------
all_preds = np.full((N_FILES * N_WIN, N_CLS), np.nan, dtype=np.float32)
all_row_ids = [None] * (N_FILES * N_WIN)
all_stems = [None] * (N_FILES * N_WIN)
t0 = time.time()
fail = 0

with ThreadPoolExecutor(max_workers=4) as pool:
    # Prefetch first batch
    next_idx = 0
    next_futs = []
    while next_idx < min(BATCH, N_FILES):
        next_futs.append(pool.submit(load_audio, files[next_idx]))
        next_idx += 1

    completed = 0
    while completed < N_FILES:
        # Collect current batch's audio
        batch_audio = []
        batch_stems = []
        batch_orig_idx = []
        for fi in range(completed, min(completed + BATCH, N_FILES)):
            fut = next_futs[fi - completed]
            audio, stem, err = fut.result()
            if audio is None:
                fail += 1
                batch_audio.append(np.zeros(N_SAMPLES, dtype=np.float32))
                batch_stems.append(stem)
                batch_orig_idx.append(fi)
            else:
                batch_audio.append(audio)
                batch_stems.append(stem)
                batch_orig_idx.append(fi)

        # Prefetch next batch
        new_futs = []
        for fi in range(completed + BATCH, min(completed + 2 * BATCH, N_FILES)):
            new_futs.append(pool.submit(load_audio, files[fi]))
        next_futs = new_futs

        # Run ONNX on current batch
        x = np.stack(batch_audio, axis=0)
        try:
            out = sess.run(None, {input_name: x})
            preds = np.asarray(out[0])  # (B, 12, 234) or similar
            if preds.ndim == 4:
                preds = preds.squeeze(1)
            if preds.shape != (len(batch_audio), N_WIN, N_CLS):
                # Try other shape interpretations
                preds = preds.reshape(len(batch_audio), N_WIN, N_CLS)
            # Sigmoid
            if preds.max() > 1.0 or preds.min() < 0.0:
                preds = 1.0 / (1.0 + np.exp(-preds))
            # Store
            for b, (stem, orig_fi) in enumerate(zip(batch_stems, batch_orig_idx)):
                for w in range(N_WIN):
                    row = orig_fi * N_WIN + w
                    all_preds[row] = preds[b, w]
                    all_row_ids[row] = f"{stem}_{(w+1)*5}"
                    all_stems[row] = stem
        except Exception as e:
            fail += len(batch_audio)
            print(f"Batch failed: {e}")

        completed += BATCH
        if completed % (BATCH * 50) == 0 or completed >= N_FILES:
            dt = time.time() - t0
            rate = completed / dt
            eta_min = (N_FILES - completed) / rate / 60.0
            print(f"  [{completed:5d}/{N_FILES}] dt={dt/60:.1f}min rate={rate:.1f}/s eta={eta_min:.1f}min fails={fail}")

print(f"\nDone. Total time: {(time.time()-t0)/60:.1f}min  fails={fail}")

# Save
out_path = "/kaggle/working/long_convnext_on_unlabeled.npz"
# Trim None entries (failed files)
valid_mask = np.array([rid is not None for rid in all_row_ids])
np.savez_compressed(
    out_path,
    P=all_preds[valid_mask],
    row_ids=np.array([r for r in all_row_ids if r is not None]),
    stems=np.array([s for s in all_stems if s is not None]),
    classes=np.array(classes),
)
print(f"Saved {out_path}: {valid_mask.sum()} rows, shape={all_preds[valid_mask].shape}")
