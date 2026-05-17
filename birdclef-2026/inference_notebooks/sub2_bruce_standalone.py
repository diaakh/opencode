# ============================================================================
# SUBMISSION 2 / 5 — STANDALONE Bruce-Ridge + hour_prior (w=3.0)
#
# Full self-contained inference notebook (not a post-processing cell).
# Uses Bruce Wu's clip_student_bundle.pkl + Perch v2 ONNX + my pseudo_hour_priors.
#
# Verified on Bruce's 739-row labeled OOF: macro-AUC 0.9586.
# Independent path from exp019 — useful for ensembling.
#
# REQUIRED Kaggle datasets attached:
#   - birdclef-2026 (competition data, auto-attached)
#   - brucewu1200/birdclef-2026-cvlb-assets-0911 (clip_student_bundle.pkl
#     + perch_v2_no_dft.onnx)
#   - <your-username>/birdclef-2026-priors-research (pseudo_hour_priors.csv)
#
# Expected runtime on Kaggle CPU: ~30 min for 600 test files.
# ============================================================================

import gc
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# Install ONNX runtime from bundled wheel
import subprocess

ASSETS_DIR_CANDIDATES = list(Path("/kaggle/input").rglob("clip_student_bundle.pkl"))
assert ASSETS_DIR_CANDIDATES, "Attach brucewu1200/birdclef-2026-cvlb-assets-0911"
ASSETS_DIR = ASSETS_DIR_CANDIDATES[0].parent

# Kaggle base image does NOT ship onnxruntime — install from bundled wheel
try:
    import onnxruntime as ort  # noqa
except ImportError:
    whl_candidates = list(Path("/kaggle/input").rglob("onnxruntime-*.whl"))
    assert whl_candidates, "No onnxruntime wheel found in /kaggle/input — attach perch-v2-no-dft-onnx"
    print(f"Installing onnxruntime from {whl_candidates[0]}")
    subprocess.check_call(["pip", "install", "-q", str(whl_candidates[0])])
    import onnxruntime as ort  # noqa

EPS = 1e-7

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP_DIR.exists():
    COMP_DIR = Path("/kaggle/input/birdclef-2026")
TEST_DIR = COMP_DIR / "test_soundscapes"
SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"

BUNDLE_PATH = ASSETS_DIR / "clip_student_bundle.pkl"
PERCH_ONNX_PATH = list(Path("/kaggle/input").rglob("perch_v2_no_dft.onnx"))
assert PERCH_ONNX_PATH, "Could not find perch_v2_no_dft.onnx anywhere in /kaggle/input"
PERCH_ONNX_PATH = PERCH_ONNX_PATH[0]

PRIORS_DIR_HITS = list(Path("/kaggle/input").rglob("pseudo_hour_priors.csv"))
assert PRIORS_DIR_HITS, "Attach <your-username>/birdclef-2026-priors-research"
PRIORS_DIR = PRIORS_DIR_HITS[0].parent

print(f"COMP_DIR: {COMP_DIR}")
print(f"BUNDLE: {BUNDLE_PATH}")
print(f"PERCH: {PERCH_ONNX_PATH}")
print(f"PRIORS: {PRIORS_DIR}")

# ----------------------------------------------------------------------------
# Load Bruce's pipeline
# ----------------------------------------------------------------------------
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)

scaler = bundle["clip_bundle"]["emb_scaler"]
pca = bundle["clip_bundle"]["pca"]
fscaler = bundle["clip_bundle"]["feature_scaler"]
ridge = bundle["clip_bundle"]["model"]
primary_labels = bundle["primary_labels"]
class_to_bc = bundle["class_to_bc"]
print(f"Loaded Bruce bundle: ridge={type(ridge).__name__}, {len(primary_labels)} classes")

# Load Perch ONNX
so = ort.SessionOptions()
so.intra_op_num_threads = 4
perch_sess = ort.InferenceSession(str(PERCH_ONNX_PATH), so, providers=["CPUExecutionProvider"])
perch_inputs = [i.name for i in perch_sess.get_inputs()]
perch_outputs = [o.name for o in perch_sess.get_outputs()]
print(f"Perch ONNX: in={perch_inputs}, out={perch_outputs}")

# ----------------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------------
hour_prior_df = pd.read_csv(PRIORS_DIR / "pseudo_hour_priors.csv").set_index("hour")

# ----------------------------------------------------------------------------
# Audio loader
# ----------------------------------------------------------------------------
SR = 32000
WIN_SEC = 5
SAMPS = SR * WIN_SEC  # 160000


def load_audio(path):
    y, sr = sf.read(path, dtype="float32")
    if sr != SR:
        # Should not happen for competition data, but fall back
        import scipy.signal
        y = scipy.signal.resample_poly(y, SR, sr).astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1).astype(np.float32)
    return y


def chunk_5sec_windows(y, expected=12):
    """Return (expected, SAMPS) shape array. Pad if short."""
    total = y.shape[0]
    target = expected * SAMPS
    if total < target:
        y = np.pad(y, (0, target - total))
    y = y[:target]
    return y.reshape(expected, SAMPS)


# ----------------------------------------------------------------------------
# Inference — batched across files (12 windows × BATCH_FILES per ONNX call)
# ----------------------------------------------------------------------------
import concurrent.futures

N_WINDOWS = 12
BATCH_FILES = 8   # 8 files * 12 windows = 96-sample ONNX batch per call


def perch_predict_batch(x):
    """x: (N, 160000) float32 → returns (N, 1536) emb, (N, 234) logits."""
    outs = perch_sess.run(None, {perch_inputs[0]: x})
    # Detect which output is embedding vs logits by last dim
    emb, logit = None, None
    for o in outs:
        a = np.asarray(o)
        if a.ndim == 2 and a.shape[1] == 1536:
            emb = a.astype(np.float32)
        elif a.ndim == 2 and a.shape[1] == 234:
            logit = a.astype(np.float32)
    if emb is None:
        emb = np.asarray(outs[0], dtype=np.float32).reshape(x.shape[0], -1)[:, :1536]
    if logit is None:
        # Some Perch v2 ONNX outputs (1, 234) per single sample, or (N, 1536) only
        # If no 234-class output, build zeros (Ridge alone covers it)
        logit = np.zeros((x.shape[0], 234), dtype=np.float32)
    return emb, logit


def bruce_pipeline_predict(emb, logit):
    """Apply scaler -> PCA -> concat with logits -> scaler -> ridge."""
    emb_s = scaler.transform(emb)
    emb_p = pca.transform(emb_s)
    feat = np.concatenate([emb_p, logit], axis=1)
    feat = fscaler.transform(feat)
    out = ridge.predict(feat)  # (n, 234) logits
    return out


# ----------------------------------------------------------------------------
# Main loop — concurrent audio IO + batched Perch ONNX
# ----------------------------------------------------------------------------
samp = pd.read_csv(SAMPLE_SUB_PATH)
class_cols = [c for c in samp.columns if c != "row_id"]
assert len(class_cols) == 234

test_files = sorted(TEST_DIR.glob("*.ogg"))
if len(test_files) == 0:
    print("No test files found — dry run, emitting all-zero submission")
    out_df = samp.copy()
    out_df.iloc[:, 1:] = 0.0
    out_df.to_csv("submission.csv", index=False)
    sys.exit(0)

print(f"Processing {len(test_files)} test soundscapes (batch_files={BATCH_FILES})...")

# Pre-compute per-hour prior arrays (24 × 234) aligned to class_cols
hour_prior_arr = np.zeros((24, 234), dtype=np.float64)
for h in range(24):
    if h in hour_prior_df.index:
        hour_prior_arr[h] = hour_prior_df.loc[h].reindex(class_cols).fillna(0.0).to_numpy()
hour_prior_arr = np.clip(hour_prior_arr, EPS, 1.0)
log_hour_prior = np.log(hour_prior_arr)  # (24, 234)

# Filename: BC2026_Test_0001_S05_20250227_010002 → take last 6-digit group as HHMMSS
ROW_RE = re.compile(r"_(\d{8})_(\d{6})$")


def load_and_chunk(path):
    y = load_audio(path)
    yw = chunk_5sec_windows(y, expected=N_WINDOWS)
    return path, yw


row_ids_all = []
prob_all = []
t0 = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    # Prefetch first batch
    next_batch = test_files[:BATCH_FILES]
    next_futures = [pool.submit(load_and_chunk, p) for p in next_batch]

    for start in range(0, len(test_files), BATCH_FILES):
        # Collect current batch's audio
        batch_results = [f.result() for f in next_futures]
        # Prefetch next batch in parallel with the ONNX call
        next_start = start + BATCH_FILES
        if next_start < len(test_files):
            next_batch = test_files[next_start:next_start + BATCH_FILES]
            next_futures = [pool.submit(load_and_chunk, p) for p in next_batch]

        # Stack into one big (B*12, 160000) tensor
        batch_n = len(batch_results)
        x = np.empty((batch_n * N_WINDOWS, SAMPS), dtype=np.float32)
        for bi, (_, yw) in enumerate(batch_results):
            x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = yw

        # One ONNX call for the whole batch
        emb, logit = perch_predict_batch(x)
        bruce_logits = bruce_pipeline_predict(emb, logit)  # (B*12, 234)

        # Per-file: hour prior + row_ids
        for bi, (fpath, _) in enumerate(batch_results):
            s = slice(bi * N_WINDOWS, (bi + 1) * N_WINDOWS)
            file_logits = bruce_logits[s]
            m = ROW_RE.search(fpath.stem)
            hour = int(m.group(2)[:2]) if m else 0
            if 0 <= hour < 24:
                file_logits = file_logits + 3.0 * log_hour_prior[hour][None, :]
            prob = 1.0 / (1.0 + np.exp(-file_logits))
            prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
            stem = fpath.stem
            for i in range(N_WINDOWS):
                row_ids_all.append(f"{stem}_{(i + 1) * WIN_SEC}")
            prob_all.append(prob)

        done = start + batch_n
        if done % (BATCH_FILES * 5) == 0 or done == len(test_files):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1.0)
            eta = (len(test_files) - done) / max(rate, 0.01)
            print(f"  [{done}/{len(test_files)}] elapsed={elapsed:.0f}s "
                  f"rate={rate:.1f} files/s eta={eta:.0f}s")

prob_all = np.concatenate(prob_all, axis=0)
print(f"Total inference time: {time.time() - t0:.0f}s "
      f"({(time.time() - t0) / max(len(test_files), 1):.2f}s/file)")

# Write submission
out_df = pd.DataFrame(prob_all, columns=class_cols)
out_df.insert(0, "row_id", row_ids_all)

# Align to sample_submission if rows match
sample_ids = samp["row_id"].astype(str).tolist()
if set(out_df["row_id"]) == set(sample_ids):
    out_df = out_df.set_index("row_id").loc[sample_ids].reset_index()
out_df.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv: {len(out_df)} rows × {out_df.shape[1]} cols, "
      f"min={out_df.iloc[:, 1:].min().min():.4f}, max={out_df.iloc[:, 1:].max().max():.4f}")
