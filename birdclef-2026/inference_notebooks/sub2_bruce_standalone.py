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

# Install ONNX runtime from bundled wheel (the assets bundle ships it)
import subprocess

ASSETS_DIR_CANDIDATES = list(Path("/kaggle/input").rglob("clip_student_bundle.pkl"))
assert ASSETS_DIR_CANDIDATES, "Attach brucewu1200/birdclef-2026-cvlb-assets-0911"
ASSETS_DIR = ASSETS_DIR_CANDIDATES[0].parent

import onnxruntime as ort  # noqa  (Kaggle base image already ships it)

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
# Inference
# ----------------------------------------------------------------------------
def perch_predict_file(y_windows):
    """Run Perch on each 5s window. Return (12, 1536), (12, 234)."""
    embs = []
    logits = []
    for w in y_windows:
        ins = {perch_inputs[0]: w[None, :].astype(np.float32)}
        outs = perch_sess.run(None, ins)
        # Outputs: embedding (1, 1536), logits (1, 234) usually
        emb_arr = outs[0].reshape(-1)
        logit_arr = outs[1].reshape(-1) if len(outs) > 1 else None
        embs.append(emb_arr[:1536])
        if logit_arr is None or logit_arr.shape[0] != 234:
            # Fallback: assume only embedding; logits zero
            logit_arr = np.zeros(234, dtype=np.float32)
        logits.append(logit_arr)
    return np.stack(embs), np.stack(logits)


def bruce_pipeline_predict(emb, logit):
    """Apply scaler -> PCA -> concat with logits -> scaler -> ridge."""
    emb_s = scaler.transform(emb)
    emb_p = pca.transform(emb_s)
    feat = np.concatenate([emb_p, logit], axis=1)
    feat = fscaler.transform(feat)
    out = ridge.predict(feat)  # (n, 234) logits
    return out


# ----------------------------------------------------------------------------
# Main loop
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

print(f"Processing {len(test_files)} test soundscapes...")
row_ids_all = []
prob_all = []

# Filename: BC2026_Test_0001_S05_20250227_010002 → take last 6-digit group as HHMMSS
ROW_RE = re.compile(r"_(\d{8})_(\d{6})$")

t0 = time.time()
for fi, fpath in enumerate(test_files):
    y = load_audio(fpath)
    yw = chunk_5sec_windows(y, expected=12)
    emb, logit = perch_predict_file(yw)
    bruce_logits = bruce_pipeline_predict(emb, logit)  # (12, 234)
    # hour from filename — match HHMMSS at end of stem
    m = ROW_RE.search(fpath.stem)
    hour = int(m.group(2)[:2]) if m else 0
    # apply hour prior in logit space
    if hour in hour_prior_df.index:
        prior = hour_prior_df.loc[hour].reindex(class_cols).fillna(0.0).to_numpy(dtype=np.float64)
        prior = np.clip(prior, EPS, 1.0)
        log_prior = np.log(prior)
        bruce_logits = bruce_logits + 3.0 * log_prior[None, :]
    prob = 1.0 / (1.0 + np.exp(-bruce_logits))
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    # row_ids
    stem = fpath.stem
    for i, w in enumerate(yw):
        end = (i + 1) * WIN_SEC
        row_ids_all.append(f"{stem}_{end}")
    prob_all.append(prob)

    if (fi + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (fi + 1) / max(elapsed, 1.0)
        eta = (len(test_files) - fi - 1) / max(rate, 0.01)
        print(f"  [{fi+1}/{len(test_files)}] elapsed={elapsed:.0f}s rate={rate:.1f} files/s eta={eta:.0f}s")

prob_all = np.concatenate(prob_all, axis=0)
print(f"Total inference time: {time.time() - t0:.0f}s")

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
