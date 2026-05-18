"""Kaggle kernel: run Perch v2 + Bruce CLIP-Ridge on the 66 train_soundscapes_labels
files and save per-window predictions as npz for offline analysis.

Run as a Kaggle CPU notebook (internet off). Attaches:
  - birdclef-2026 (competition)
  - brucewu1200/birdclef-2026-cvlb-assets-0911  (Bruce bundle: clip_student_bundle.pkl + perch_v2_no_dft.onnx)

Output: /kaggle/working/labeled_oof_perch_bruce.npz
"""
import os, re, sys, pickle, time, glob, gc, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

# Install onnxruntime from offline wheel (no internet)
import glob as _glob
print("=== /kaggle/input contents ===")
for root in sorted(_glob.glob("/kaggle/input/*"))[:30]:
    print(" ", root)
ort_whls = _glob.glob("/kaggle/input/**/onnxruntime*.whl", recursive=True)
print(f"Found onnxruntime wheels: {ort_whls}")
if ort_whls:
    whl = ort_whls[0]
    print(f"Installing {whl}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", whl])
else:
    # Last resort: pip already installed?
    try:
        import onnxruntime
        print("onnxruntime already importable")
    except ImportError:
        raise RuntimeError("No onnxruntime wheel found in /kaggle/input — attach a dataset that contains one")
import onnxruntime as ort

SR = 32_000
WIN_SEC = 5
N_WIN = 12  # 12 × 5s = 60s per file

# ---------- Paths ----------
COMP = Path("/kaggle/input/birdclef-2026")
BRUCE = Path("/kaggle/input/birdclef-2026-cvlb-assets-0911")

# Find Perch ONNX (Bruce's bundle has perch_v2_no_dft.onnx; fallback to rishikeshjani's)
PERCH_PATHS = [
    BRUCE / "perch_v2_no_dft.onnx",
    Path("/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2_no_dft.onnx"),
    Path("/kaggle/input/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx"),
]
PERCH_ONNX = next((p for p in PERCH_PATHS if p.exists()), None)
print(f"Perch ONNX: {PERCH_ONNX}")
assert PERCH_ONNX is not None, "Perch ONNX not found in any expected location"

CLIP_BUNDLE_PATH = BRUCE / "clip_student_bundle.pkl"
print(f"Bruce bundle: {CLIP_BUNDLE_PATH.exists()}")

# ---------- Labels + Y matrix ----------
labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)
print(f"Labeled rows: {len(labels)}, unique files: {labels['filename'].nunique()}")

tax = pd.read_csv(COMP / "taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
cls_idx = {c: i for i, c in enumerate(classes)}
N, C = len(labels), len(classes)
print(f"N={N}, C={C}")

Y = np.zeros((N, C), dtype=np.float32)
for i, lab in enumerate(labels["primary_label"]):
    for c in str(lab).split(";"):
        if c in cls_idx:
            Y[i, cls_idx[c]] = 1.0


# ---------- Parse hour, site from filename ----------
def parse(fn):
    m = re.match(r"BC2026_Train_\d+_(S\d+)_(\d{8})_(\d{2})(\d{2})(\d{2})", fn)
    return (m.group(1), m.group(2), int(m.group(3))) if m else (None, None, None)


labels[["site", "date", "hour"]] = labels["filename"].apply(
    lambda fn: pd.Series(parse(fn))
)


# ---------- Build (filename, start) → row index lookup ----------
# In labels CSV, start/end are "HH:MM:SS" strings (per OVERVIEW). Convert to end_sec.
def parse_time(t):
    if isinstance(t, (int, float)):
        return int(t)
    try:
        h, m, s = str(t).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None


labels["start_sec"] = labels["start"].apply(parse_time)
labels["end_sec"] = labels["end"].apply(parse_time)
# A 5-sec window ends at one of {5, 10, 15, ..., 60}. window_idx = (end-1)//5
labels["win_idx"] = labels["end_sec"].apply(lambda s: (s - 1) // WIN_SEC if s else None)
print("Per-row start_sec sample:", labels["start_sec"].head(3).tolist())
print("Per-row end_sec sample:", labels["end_sec"].head(3).tolist())
print("Per-row win_idx sample:", labels["win_idx"].head(3).tolist())

row_lookup = {
    (row["filename"], int(row["win_idx"])): i for i, row in labels.iterrows()
}


# ---------- Load Perch ONNX ----------
so = ort.SessionOptions()
so.intra_op_num_threads = 4
sess = ort.InferenceSession(str(PERCH_ONNX), sess_options=so, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
print(f"Perch input: name={input_name} shape={input_shape}")
print(f"Perch outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")


# ---------- Load Bruce CLIP-Ridge bundle ----------
with open(CLIP_BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)
print(f"Bundle keys: {list(bundle.keys()) if isinstance(bundle, dict) else type(bundle)}")
clip_bundle = bundle.get("clip_bundle", bundle)
print(f"clip_bundle keys: {list(clip_bundle.keys()) if isinstance(clip_bundle, dict) else type(clip_bundle)}")
emb_scaler = clip_bundle["emb_scaler"]
pca = clip_bundle["pca"]
feat_scaler = clip_bundle["feature_scaler"]
ridge = clip_bundle["model"]
ridge_classes = clip_bundle.get("classes")  # may be None — fall back to taxonomy order
print(f"PCA components: {pca.n_components_}")
print(f"Ridge classes: {None if ridge_classes is None else len(ridge_classes)}")


# ---------- Class column reordering between Bruce ridge and taxonomy ----------
# Bruce's ridge outputs may be in its own class order; reorder to taxonomy.
if ridge_classes is not None:
    bruce_to_tax = np.array([
        cls_idx[c] if c in cls_idx else -1 for c in [str(rc) for rc in ridge_classes]
    ])
    valid = bruce_to_tax >= 0
    print(f"Bruce → taxonomy mapping: {valid.sum()}/{len(ridge_classes)} matched")
else:
    bruce_to_tax = np.arange(C)
    valid = np.ones(C, dtype=bool)


# ---------- Iterate labeled files, run Perch + Bruce ----------
P_perch_logits = np.full((N, C), np.nan, dtype=np.float32)
P_bruce = np.full((N, C), np.nan, dtype=np.float32)
ALL_EMB = np.full((N, 1536), np.nan, dtype=np.float32)

labeled_filenames = sorted(labels["filename"].unique())
print(f"Processing {len(labeled_filenames)} labeled files")

t0 = time.time()
for fi, fname in enumerate(labeled_filenames):
    fp = COMP / "train_soundscapes" / fname
    if not fp.exists():
        print(f"  MISSING {fname}")
        continue
    try:
        y, sr = sf.read(str(fp), dtype="float32")
        if sr != SR:
            print(f"  {fname}: unexpected sr {sr}")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < SR * 60:
            y = np.pad(y, (0, SR * 60 - len(y)))
        else:
            y = y[: SR * 60]
    except Exception as e:
        print(f"  {fname}: load failed: {e}")
        continue

    # 12 windows of 5s
    windows = y.reshape(N_WIN, SR * WIN_SEC)

    # Batch through Perch — try (12, samples) first
    try:
        out = sess.run(None, {input_name: windows.astype(np.float32)})
    except Exception as e:
        # Fallback: per-window
        out_list = [sess.run(None, {input_name: w[None].astype(np.float32)}) for w in windows]
        out = [np.concatenate([o[i] for o in out_list], axis=0) for i in range(len(out_list[0]))]

    # Two outputs typically: emb (B,1536) and logits (B, 14795 or 234)
    emb = None
    logits = None
    for o in out:
        a = np.asarray(o)
        if a.ndim == 2 and a.shape[1] == 1536 and emb is None:
            emb = a
        elif a.ndim == 2 and a.shape[1] >= 234 and logits is None:
            logits = a

    if emb is None or logits is None:
        # Fallback: assume first output is emb, second is logits
        emb = np.asarray(out[0]).reshape(N_WIN, -1)
        logits = np.asarray(out[1]).reshape(N_WIN, -1)

    # Bruce pipeline: scale → PCA → concat with raw logits (subset to 234) → scale → ridge
    emb_s = emb_scaler.transform(emb)
    emb_pca = pca.transform(emb_s)

    # Subset logits to first 234 OR to taxonomy-mapped columns
    if logits.shape[1] == C:
        logits_for_bruce = logits
    else:
        # Perch outputs 14795 logits; the kernel needs to know which 234 to use.
        # Bruce's bundle should embed this mapping in `feature_scaler` input shape.
        # Try first 234 as a sanity fallback (will be re-mapped via ridge_classes).
        logits_for_bruce = logits[:, :C]
    features = np.concatenate([emb_pca, logits_for_bruce], axis=1)
    features = feat_scaler.transform(features)
    bruce_logits = ridge.predict(features)  # (N_WIN, C_bruce)

    # Reorder to taxonomy if needed
    bruce_remap = np.full((N_WIN, C), np.nan, dtype=np.float32)
    for src_idx, tax_idx in enumerate(bruce_to_tax):
        if tax_idx >= 0:
            bruce_remap[:, tax_idx] = bruce_logits[:, src_idx]
    bruce_prob = 1.0 / (1.0 + np.exp(-bruce_remap))  # sigmoid

    # Perch raw logits → first 234 cols (or remap)
    perch_remap = logits_for_bruce
    if perch_remap.shape[1] != C:
        perch_remap = np.full((N_WIN, C), np.nan, dtype=np.float32)

    # Store into N-row arrays by (filename, win_idx)
    for w in range(N_WIN):
        key = (fname, w)
        if key in row_lookup:
            i = row_lookup[key]
            P_perch_logits[i] = perch_remap[w]
            P_bruce[i] = bruce_prob[w]
            ALL_EMB[i] = emb[w]

    if (fi + 1) % 5 == 0 or fi == len(labeled_filenames) - 1:
        dt = time.time() - t0
        rate = (fi + 1) / dt
        eta = (len(labeled_filenames) - fi - 1) / rate
        print(f"  [{fi+1}/{len(labeled_filenames)}] dt={dt:.1f}s rate={rate:.2f}/s eta={eta:.1f}s")
    del y, windows, out, emb, logits, emb_s, emb_pca, features, bruce_logits, bruce_remap, bruce_prob
    gc.collect()

# ---------- Save ----------
out_path = "/kaggle/working/labeled_oof_perch_bruce.npz"
np.savez_compressed(
    out_path,
    Y=Y,
    P_perch_logits=P_perch_logits,
    P_bruce=P_bruce,
    embeddings=ALL_EMB,
    classes=np.array(classes),
    row_filename=labels["filename"].values,
    row_start_sec=labels["start_sec"].values,
    row_end_sec=labels["end_sec"].values,
    row_site=labels["site"].values,
    row_hour=labels["hour"].values.astype(np.int8),
)
print(f"Saved {out_path}")
print(f"Sizes: Y={Y.shape}, P_perch={P_perch_logits.shape}, P_bruce={P_bruce.shape}, emb={ALL_EMB.shape}")

# Quick sanity-AUC
from sklearn.metrics import roc_auc_score
def macro_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        if 0 < y[:, c].sum() < y.shape[0] and not np.all(np.isnan(p[:, c])):
            try:
                mask = ~np.isnan(p[:, c])
                if mask.sum() > 5 and 0 < y[mask, c].sum() < mask.sum():
                    aucs.append(roc_auc_score(y[mask, c], p[mask, c]))
            except Exception:
                pass
    return np.mean(aucs) if aucs else float("nan"), len(aucs)

print("Bruce macro-AUC :", macro_auc(Y, P_bruce))
print("Perch raw macro-AUC:", macro_auc(Y, P_perch_logits))
