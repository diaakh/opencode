# ============================================================================
# SUB V5: STANDALONE Bruce + TIME-SHIFT TTA + LABELED hour_prior
#
# Standalone Bruce pipeline (like sub2) but adds 3-shift TTA at inference:
#   - shift -2.5s, 0, +2.5s on each 60s window
#   - average predictions
#   - then apply labeled hour_prior post-processing
#
# Per BC2025 1st place, ±2.5s TTA gives +0.012 LB.
#
# Expected runtime: 3x the standalone Bruce baseline ~ 15-30 min for 600 files.
# Comfortable within 90-min code-submission budget.
#
# REQUIRED Kaggle datasets:
#   - birdclef-2026 (competition)
#   - brucewu1200/birdclef-2026-cvlb-assets-0911 (Bruce bundle)
#   - tuckerarrants/perch-v2-no-dft-onnx (Perch ONNX + wheel)
#   - adkasd/birdclef-2026-priors-research (labeled priors)
# ============================================================================
import gc
import pickle
import re
import sys
import subprocess
import time
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# Install ONNX runtime from bundled wheel
ASSETS_DIR_CANDIDATES = list(Path("/kaggle/input").rglob("clip_student_bundle.pkl"))
assert ASSETS_DIR_CANDIDATES, "Attach brucewu1200/birdclef-2026-cvlb-assets-0911"
ASSETS_DIR = ASSETS_DIR_CANDIDATES[0].parent

try:
    import onnxruntime as ort
except ImportError:
    whl = list(Path("/kaggle/input").rglob("onnxruntime-*.whl"))
    assert whl, "No onnxruntime wheel; attach perch-v2-no-dft-onnx"
    print(f"Installing onnxruntime from {whl[0]}")
    subprocess.check_call(["pip", "install", "-q", str(whl[0])])
    import onnxruntime as ort

EPS = 1e-7

# Paths
COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP_DIR.exists():
    COMP_DIR = Path("/kaggle/input/birdclef-2026")
TEST_DIR = COMP_DIR / "test_soundscapes"
SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"

BUNDLE_PATH = ASSETS_DIR / "clip_student_bundle.pkl"
PERCH_ONNX_PATH = list(Path("/kaggle/input").rglob("perch_v2_no_dft.onnx"))
assert PERCH_ONNX_PATH, "perch_v2_no_dft.onnx not found"
PERCH_ONNX_PATH = PERCH_ONNX_PATH[0]

PRIORS_DIR = list(Path("/kaggle/input").rglob("pseudo_hour_priors.csv"))
assert PRIORS_DIR, "Attach adkasd/birdclef-2026-priors-research"
PRIORS_DIR = PRIORS_DIR[0].parent

print(f"COMP: {COMP_DIR}")
print(f"BUNDLE: {BUNDLE_PATH}")
print(f"PERCH: {PERCH_ONNX_PATH}")
print(f"PRIORS: {PRIORS_DIR}")

# Load Bruce pipeline
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)
scaler = bundle["clip_bundle"]["emb_scaler"]
pca = bundle["clip_bundle"]["pca"]
fscaler = bundle["clip_bundle"]["feature_scaler"]
ridge = bundle["clip_bundle"]["model"]

# Load Perch
so = ort.SessionOptions()
so.intra_op_num_threads = 4
perch_sess = ort.InferenceSession(str(PERCH_ONNX_PATH), so, providers=["CPUExecutionProvider"])
perch_inputs = [i.name for i in perch_sess.get_inputs()]
print(f"Perch inputs: {perch_inputs}, outputs: {[o.name for o in perch_sess.get_outputs()]}")

# Load priors (with labeled fallback)
hour_df = pd.read_csv(PRIORS_DIR / "pseudo_hour_priors.csv").set_index("hour")
hourly_sum = hour_df.sum(axis=1)
covered = hourly_sum[hourly_sum > 0].index.tolist()
dead = [h for h in range(24) if h not in covered]
if dead:
    global_prior = hour_df.loc[covered].mean(axis=0)
    for h in dead:
        hour_df.loc[h] = global_prior
hour_df = hour_df.sort_index()

# Overlay labeled hourly prior where available
try:
    lab_hour = pd.read_csv(PRIORS_DIR / "hourly_species_priors.csv").set_index("hour")
    for h in range(24):
        if h not in lab_hour.index:
            lab_hour.loc[h] = lab_hour.mean(axis=0)
    lab_hour = lab_hour.sort_index()
    for c in lab_hour.columns:
        if c in hour_df.columns:
            hour_df[c] = lab_hour[c].reindex(hour_df.index).fillna(lab_hour[c].mean())
    print(f"[v5] Overlaid {len(lab_hour.columns)} labeled hour columns")
except FileNotFoundError:
    print("[v5] No labeled hour prior — using pseudo only")

# Audio config
SR = 32000
WIN_SEC = 5
SAMPS = SR * WIN_SEC  # 160000
N_WINDOWS = 12
BATCH_FILES = 8
TTA_SHIFTS = [0, int(2.5 * SR), -int(2.5 * SR)]  # samples


def load_audio(path):
    y, sr = sf.read(path, dtype="float32")
    if sr != SR:
        import scipy.signal
        y = scipy.signal.resample_poly(y, SR, sr).astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1).astype(np.float32)
    return y


def chunk_5sec_windows_shifted(y, shift_samples, expected=12):
    """Take 12 5-sec windows starting at offset `shift_samples`."""
    target = expected * SAMPS
    if shift_samples >= 0:
        # Take from shift_samples onward
        y_shifted = y[shift_samples:]
    else:
        # Pad at start
        y_shifted = np.concatenate([np.zeros(-shift_samples, dtype=np.float32), y])
    if y_shifted.shape[0] < target:
        y_shifted = np.pad(y_shifted, (0, target - y_shifted.shape[0]))
    y_shifted = y_shifted[:target]
    return y_shifted.reshape(expected, SAMPS)


def perch_predict_batch(x):
    outs = perch_sess.run(None, {perch_inputs[0]: x})
    emb = logit = None
    for o in outs:
        a = np.asarray(o)
        if a.ndim == 2 and a.shape[1] == 1536:
            emb = a.astype(np.float32)
        elif a.ndim == 2 and a.shape[1] == 234:
            logit = a.astype(np.float32)
    if emb is None:
        emb = np.asarray(outs[0], dtype=np.float32).reshape(x.shape[0], -1)[:, :1536]
    if logit is None:
        logit = np.zeros((x.shape[0], 234), dtype=np.float32)
    return emb, logit


def bruce_predict(emb, logit):
    emb_s = scaler.transform(emb)
    emb_p = pca.transform(emb_s)
    feat = np.concatenate([emb_p, logit], axis=1)
    feat = fscaler.transform(feat)
    return ridge.predict(feat)  # (n, 234) logits


samp = pd.read_csv(SAMPLE_SUB_PATH)
class_cols = [c for c in samp.columns if c != "row_id"]
test_files = sorted(TEST_DIR.glob("*.ogg"))
if not test_files:
    print("No test files — emitting all-zero")
    out_df = samp.copy()
    out_df.iloc[:, 1:] = 0.0
    out_df.to_csv("submission.csv", index=False)
    sys.exit(0)

# Pre-compute hour prior matrix aligned to class_cols
hp_arr = np.zeros((24, 234), dtype=np.float64)
for h in range(24):
    hp_arr[h] = hour_df.loc[h].reindex(class_cols).fillna(0.0).to_numpy()
hp_arr = np.clip(hp_arr, EPS, 1.0)
log_hp = np.log(hp_arr)
HOUR_W = 0.02   # labeled prior; tighter than the 0.05 used with pseudo

ROW_RE = re.compile(r"_(\d{8})_(\d{6})$")
print(f"[v5] Processing {len(test_files)} files with {len(TTA_SHIFTS)} TTA shifts (batch={BATCH_FILES})")

row_ids_all = []
prob_all = []
t0 = time.time()


def load_and_chunk(path, shift):
    y = load_audio(path)
    yw = chunk_5sec_windows_shifted(y, shift, N_WINDOWS)
    return path, yw


for start in range(0, len(test_files), BATCH_FILES):
    batch_paths = test_files[start:start + BATCH_FILES]
    batch_n = len(batch_paths)

    # Predict at each TTA shift and average
    avg_logits = np.zeros((batch_n * N_WINDOWS, 234), dtype=np.float32)
    for shift in TTA_SHIFTS:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(load_and_chunk, p, shift) for p in batch_paths]
            results = [f.result() for f in futs]
        x = np.empty((batch_n * N_WINDOWS, SAMPS), dtype=np.float32)
        for bi, (_, yw) in enumerate(results):
            x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = yw
        emb, logit = perch_predict_batch(x)
        b_logits = bruce_predict(emb, logit)
        avg_logits += b_logits / len(TTA_SHIFTS)

    # Apply hour_prior per-file
    for bi, fpath in enumerate(batch_paths):
        s = slice(bi * N_WINDOWS, (bi + 1) * N_WINDOWS)
        file_logits = avg_logits[s]
        m = ROW_RE.search(fpath.stem)
        hour = int(m.group(2)[:2]) if m else 0
        if 0 <= hour < 24:
            file_logits = file_logits + HOUR_W * log_hp[hour][None, :]
        prob = 1.0 / (1.0 + np.exp(-file_logits))
        prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
        for i in range(N_WINDOWS):
            row_ids_all.append(f"{fpath.stem}_{(i + 1) * WIN_SEC}")
        prob_all.append(prob)

    done = start + batch_n
    if done % (BATCH_FILES * 5) == 0 or done == len(test_files):
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1.0)
        eta = (len(test_files) - done) / max(rate, 0.01)
        print(f"  [{done}/{len(test_files)}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

prob_all = np.concatenate(prob_all, axis=0)
print(f"[v5] Total: {time.time() - t0:.0f}s ({(time.time() - t0)/len(test_files):.2f}s/file)")

out_df = pd.DataFrame(prob_all, columns=class_cols)
out_df.insert(0, "row_id", row_ids_all)
sample_ids = samp["row_id"].astype(str).tolist()
if set(out_df["row_id"]) == set(sample_ids):
    out_df = out_df.set_index("row_id").loc[sample_ids].reset_index()
out_df.to_csv("submission.csv", index=False)
print(f"[v5] Wrote submission.csv: {len(out_df)} rows × {out_df.shape[1]} cols, "
      f"min={out_df.iloc[:, 1:].min().min():.4f}, max={out_df.iloc[:, 1:].max().max():.4f}")
