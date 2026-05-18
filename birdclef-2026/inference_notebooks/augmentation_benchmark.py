"""Real audio-augmentation benchmark using fold0.onnx + train_soundscapes + labels.

Tests at INFERENCE time:
  - Time-shift TTA (±2.5s)
  - Gain TTA (±3 dB)
  - Soft-clipping (tanh)
  - RMS normalization to test-domain target (~0.025)
  - Codec re-encoding (downsample→upsample as a cheap proxy for low-bitrate Ogg)
  - High-pass filter (remove low-freq noise)
  - All combinations: individual + stacked + averaged

Measures macro-AUC on Bruce's labeled 66 files (1478 5-sec windows) vs ground-truth.

fold0.onnx is a 60-sec waveform → 12-window-logits model. It's NOT Perch v2 but
serves as a proxy: relative augmentation effects should generalize.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/user/opencode/birdclef-2026")
SAMPLE_SUB = pd.read_csv(ROOT / "data/sample_submission.csv")
CLASS_COLS = [c for c in SAMPLE_SUB.columns if c != "row_id"]
N_CLASSES = len(CLASS_COLS)
assert N_CLASSES == 234

SR = 32000
WIN_SEC = 5
SAMPS = SR * WIN_SEC  # 160000
N_WIN = 12
TOTAL_SAMPS = SAMPS * N_WIN  # 1920000

# Load fold0.onnx model
ONNX_PATH = ROOT / "meta_corpus/datasets/fold0.onnx"
so = ort.SessionOptions()
so.intra_op_num_threads = 4
sess = ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name
print(f"Loaded {ONNX_PATH.name}: input={INPUT_NAME}, expects (batch, {TOTAL_SAMPS})")


def load_audio(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1).astype(np.float32)
    if sr != SR:
        import scipy.signal
        y = scipy.signal.resample_poly(y, SR, sr).astype(np.float32)
    if y.shape[0] < TOTAL_SAMPS:
        y = np.pad(y, (0, TOTAL_SAMPS - y.shape[0]))
    return y[:TOTAL_SAMPS]


# ========== Augmentations ==========
def aug_identity(y):
    return y


def aug_time_shift(samples):
    def fn(y):
        if samples >= 0:
            return np.concatenate([y[samples:], y[:samples]])  # circular shift
        return np.concatenate([y[samples:], y[:samples]])
    return fn


def aug_gain(db):
    g = 10 ** (db / 20.0)
    def fn(y):
        return np.clip(y * g, -1.0, 1.0).astype(np.float32)
    return fn


def aug_soft_clip(amount=0.7):
    def fn(y):
        # tanh-style soft clip; amount=0.7 → gentle, 0.95 → aggressive
        return np.tanh(y / amount).astype(np.float32) * amount
    return fn


def aug_rms_norm(target=0.025):
    def fn(y):
        cur = float(np.sqrt(np.mean(y ** 2)))
        if cur < 1e-6: return y
        return np.clip(y * (target / cur), -1.0, 1.0).astype(np.float32)
    return fn


def aug_codec_proxy():
    """Crude proxy for low-bitrate Ogg: downsample→upsample to lose HF content."""
    import scipy.signal as sps
    target_sr = 16000  # half rate
    def fn(y):
        dn = sps.resample_poly(y, target_sr, SR)
        up = sps.resample_poly(dn, SR, target_sr).astype(np.float32)
        if up.shape[0] < y.shape[0]:
            up = np.pad(up, (0, y.shape[0] - up.shape[0]))
        return up[:y.shape[0]]
    return fn


def aug_hpf(cutoff=200):
    import scipy.signal as sps
    sos = sps.butter(2, cutoff, btype="hp", fs=SR, output="sos")
    def fn(y):
        return sps.sosfiltfilt(sos, y).astype(np.float32)
    return fn


def aug_chain(funcs):
    def fn(y):
        for f in funcs:
            y = f(y)
        return y
    return fn


# ========== Load labeled data ==========
labs = pd.read_csv(ROOT / "data/train_soundscapes_labels.csv")
files = sorted(labs["filename"].unique())
file_paths = [ROOT / "data/train_soundscapes" / f for f in files]
file_paths = [p for p in file_paths if p.exists()]
print(f"Labeled files available locally: {len(file_paths)}")
# Subset for faster run (fold0.onnx is slow at batch=1)
N_BENCH_FILES = 30
file_paths = file_paths[:N_BENCH_FILES]
print(f"Using subset of {len(file_paths)} files for benchmark")

# Build y_true per file (12 windows × 234 classes)
class_idx = {c: i for i, c in enumerate(CLASS_COLS)}
y_true_all = np.zeros((len(file_paths) * N_WIN, N_CLASSES), dtype=np.int32)
for fi, p in enumerate(file_paths):
    fname = p.name
    file_labs = labs[labs["filename"] == fname]
    for _, row in file_labs.iterrows():
        # start "00:00:05" → end_sec 5 → window_idx 0; "00:00:10" → 1; ...
        start = row["start"]
        h, m, s = start.split(":")
        start_sec = int(h) * 3600 + int(m) * 60 + int(s)
        w_idx = start_sec // 5
        if w_idx < 0 or w_idx >= N_WIN: continue
        for tok in str(row["primary_label"]).split(";"):
            tok = tok.strip()
            if tok in class_idx:
                y_true_all[fi * N_WIN + w_idx, class_idx[tok]] = 1
print(f"y_true_all: shape {y_true_all.shape}, positives {y_true_all.sum()}, classes with positives {y_true_all.any(axis=0).sum()}")


def predict_under_aug(aug_fn, batch=1):  # fold0.onnx hardcodes batch=1 reshape
    """Run fold0 over all files with given augmentation, return (N*12, 234) logits."""
    out = np.zeros((len(file_paths) * N_WIN, N_CLASSES), dtype=np.float32)
    for start in range(0, len(file_paths), batch):
        batch_paths = file_paths[start:start + batch]
        x = np.zeros((len(batch_paths), TOTAL_SAMPS), dtype=np.float32)
        for bi, p in enumerate(batch_paths):
            y = load_audio(p)
            x[bi] = aug_fn(y)
        # ONNX
        outs = sess.run(None, {INPUT_NAME: x})
        logits = outs[0]  # (batch, 12, 234)
        for bi in range(len(batch_paths)):
            out[(start + bi) * N_WIN:(start + bi + 1) * N_WIN] = logits[bi]
    return out


def macro_auc(y, scores):
    aucs = []
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0: continue
        try:
            aucs.append(roc_auc_score(y[:, c], scores[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


# ========== Run all augmentations ==========
configs = [
    ("baseline (no aug)",           aug_identity),
    ("time_shift +0.5s",            aug_time_shift(int(0.5 * SR))),
    ("time_shift -0.5s",            aug_time_shift(-int(0.5 * SR))),
    ("time_shift +1.0s",            aug_time_shift(int(1.0 * SR))),
    ("time_shift -1.0s",            aug_time_shift(-int(1.0 * SR))),
    ("time_shift +2.5s",            aug_time_shift(int(2.5 * SR))),
    ("time_shift -2.5s",            aug_time_shift(-int(2.5 * SR))),
    ("gain +3dB",                   aug_gain(+3)),
    ("gain -3dB",                   aug_gain(-3)),
    ("gain +6dB",                   aug_gain(+6)),
    ("gain -6dB",                   aug_gain(-6)),
    ("soft_clip 0.7",               aug_soft_clip(0.7)),
    ("soft_clip 0.9",               aug_soft_clip(0.9)),
    ("rms_norm 0.025",              aug_rms_norm(0.025)),
    ("rms_norm 0.05",               aug_rms_norm(0.05)),
    ("rms_norm 0.10",               aug_rms_norm(0.10)),
    ("codec_proxy 16k",             aug_codec_proxy()),
    ("hpf 200Hz",                   aug_hpf(200)),
    ("hpf 500Hz",                   aug_hpf(500)),
]

print("\n" + "=" * 70)
print(" INDIVIDUAL AUGMENTATION BENCHMARK (fold0.onnx, 66 files)")
print("=" * 70)
results = {}
t0 = time.time()
for name, fn in configs:
    pred = predict_under_aug(fn)
    auc = macro_auc(y_true_all, pred)
    results[name] = (pred, auc)
    print(f"  {name:<25}  AUC={auc:.4f}  cumulative wall={time.time()-t0:.0f}s")

base_auc = results["baseline (no aug)"][1]
print(f"\nBaseline: {base_auc:.4f}")

# ========== Stack via averaging ==========
print("\n" + "=" * 70)
print(" TTA STACKS (averaged logits)")
print("=" * 70)
shift_keys = ["time_shift +2.5s", "time_shift -2.5s"]
shift_smaller = ["time_shift +0.5s", "time_shift -0.5s", "time_shift +1.0s", "time_shift -1.0s"]
gain_keys = ["gain +3dB", "gain -3dB"]
stacks = [
    ("3-shift TTA (0, ±2.5s)",        ["baseline (no aug)"] + shift_keys),
    ("5-shift TTA (0, ±0.5, ±1.0)",   ["baseline (no aug)"] + shift_smaller[:4]),
    ("7-shift TTA (0, ±0.5, ±1.0, ±2.5)", ["baseline (no aug)"] + shift_smaller[:4] + shift_keys),
    ("gain TTA (0, ±3dB)",            ["baseline (no aug)"] + gain_keys),
    ("shift + gain (5 paths)",        ["baseline (no aug)"] + shift_keys + gain_keys),
    ("rms_norm + 3-shift TTA",        ["rms_norm 0.05", "time_shift +2.5s", "time_shift -2.5s"]),
    ("hpf + 3-shift TTA",             ["hpf 200Hz", "time_shift +2.5s", "time_shift -2.5s"]),
    ("all (shift+gain+norm)",         ["baseline (no aug)"] + shift_keys + gain_keys + ["rms_norm 0.05"]),
]
for name, keys in stacks:
    sigs = [results[k][0] for k in keys]
    avg = np.mean(sigs, axis=0)
    auc = macro_auc(y_true_all, avg)
    delta = auc - base_auc
    marker = " ⭐" if delta > 0.001 else ""
    print(f"  {name:<35}  AUC={auc:.4f}  Δ={delta:+.4f}{marker}")

# ========== Rank-space blend instead of mean ==========
print("\n" + "=" * 70)
print(" RANK-SPACE TTA BLEND")
print("=" * 70)
from scipy.stats import rankdata
def rank_blend(preds):
    n = preds[0].shape[0]
    r = np.zeros_like(preds[0])
    for p in preds:
        for c in range(p.shape[1]):
            r[:, c] += rankdata(p[:, c])
    return r / (n * len(preds))

for name, keys in [
    ("3-shift TTA (rank)",      ["baseline (no aug)", "time_shift +2.5s", "time_shift -2.5s"]),
    ("7-shift TTA (rank)",      ["baseline (no aug)"] + shift_smaller[:4] + shift_keys),
    ("all paths (rank)",        ["baseline (no aug)"] + shift_keys + gain_keys + ["rms_norm 0.05", "hpf 200Hz"]),
]:
    preds = [results[k][0] for k in keys]
    blend = rank_blend(preds)
    auc = macro_auc(y_true_all, blend)
    delta = auc - base_auc
    marker = " ⭐" if delta > 0.001 else ""
    print(f"  {name:<35}  AUC={auc:.4f}  Δ={delta:+.4f}{marker}")

print(f"\nTotal wall time: {time.time() - t0:.0f}s")
print("=" * 70)
print(f" SUMMARY: baseline={base_auc:.4f}")
print("=" * 70)
