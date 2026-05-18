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

# Glob for Perch ONNX anywhere under /kaggle/input
perch_onnx_candidates = sorted(_glob.glob("/kaggle/input/**/perch_v2_no_dft.onnx", recursive=True))
if not perch_onnx_candidates:
    perch_onnx_candidates = sorted(_glob.glob("/kaggle/input/**/perch_v2*.onnx", recursive=True))
print("Perch ONNX candidates:")
for p in perch_onnx_candidates:
    print(f"  {p} ({os.path.getsize(p)/1e6:.1f} MB)")
PERCH_ONNX = perch_onnx_candidates[0] if perch_onnx_candidates else None
print(f"Using Perch ONNX: {PERCH_ONNX}")
assert PERCH_ONNX is not None, "No Perch v2 ONNX found in /kaggle/input"

# Glob for Bruce CLIP-Ridge bundle
clip_bundle_candidates = sorted(_glob.glob("/kaggle/input/**/clip_student_bundle.pkl", recursive=True))
print("CLIP bundle candidates:")
for p in clip_bundle_candidates:
    print(f"  {p} ({os.path.getsize(p)/1e6:.2f} MB)")
CLIP_BUNDLE_PATH = clip_bundle_candidates[0] if clip_bundle_candidates else None
assert CLIP_BUNDLE_PATH is not None, "No clip_student_bundle.pkl found"
print(f"Using Bruce bundle: {CLIP_BUNDLE_PATH}")

# Also re-locate COMP (competition data) — script kernels mount it at /kaggle/input/competitions/<name>/
comp_candidates = sorted(_glob.glob("/kaggle/input/**/birdclef-2026", recursive=False))
if not comp_candidates:
    comp_candidates = sorted([p for p in _glob.glob("/kaggle/input/competitions/*") if 'birdclef' in p])
COMP = Path(comp_candidates[0]) if comp_candidates else Path("/kaggle/input/birdclef-2026")
print(f"Using competition dir: {COMP} (exists={COMP.exists()})")

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
print(f"PCA components: {pca.n_components_}")

# Bundle's primary_labels + class_to_bc give the Perch-14795 → BC2026-234 mapping
primary_labels_bundle = list(bundle.get("primary_labels"))  # 234 BC class strings, bundle/ridge output order
class_to_bc = bundle.get("class_to_bc")  # dict: bc_class_str -> Perch logit idx (NaN if unmapped)
print(f"primary_labels (bundle): {len(primary_labels_bundle)}")
print(f"class_to_bc keys sample: {list(class_to_bc.keys())[:5]}")
print(f"class_to_bc values sample: {[(k, class_to_bc[k]) for k in list(class_to_bc.keys())[:5]]}")

# Build (234,) array of Perch indices in bundle's primary_labels order; -1 = unmapped
perch_idx_for_bundle_cls = np.full(len(primary_labels_bundle), -1, dtype=np.int64)
n_unmapped = 0
for src_idx, bc_class in enumerate(primary_labels_bundle):
    bc_str = str(bc_class)
    v = class_to_bc.get(bc_str)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        n_unmapped += 1
        continue
    perch_idx_for_bundle_cls[src_idx] = int(v)
print(f"Perch-mapped bundle classes: {(perch_idx_for_bundle_cls >= 0).sum()}/{len(primary_labels_bundle)} "
      f"(unmapped: {n_unmapped})")
print(f"Perch idx range: {perch_idx_for_bundle_cls[perch_idx_for_bundle_cls >= 0].min()} – "
      f"{perch_idx_for_bundle_cls.max()}")

# Map from bundle's class order → taxonomy class order
bundle_to_tax = np.array([
    cls_idx.get(str(c), -1) for c in primary_labels_bundle
])
print(f"bundle → taxonomy mapping: {(bundle_to_tax >= 0).sum()}/{len(primary_labels_bundle)} matched")


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

    # Bruce pipeline: scale → PCA → concat with mapped Perch logits → scale → ridge
    emb_s = emb_scaler.transform(emb)
    emb_pca = pca.transform(emb_s)

    # Map Perch's 14795-d logits to BC2026 234-d (in bundle's class order).
    # Unmapped classes get 0 (neutral logit ≈ sigmoid 0.5).
    logits_for_bruce = np.zeros((N_WIN, len(primary_labels_bundle)), dtype=np.float32)
    mask = perch_idx_for_bundle_cls >= 0
    logits_for_bruce[:, mask] = logits[:, perch_idx_for_bundle_cls[mask].astype(int)]

    features = np.concatenate([emb_pca, logits_for_bruce], axis=1)
    features = feat_scaler.transform(features)
    bruce_logits = ridge.predict(features)  # (N_WIN, 234) in BUNDLE class order

    # Reorder bundle's class order → taxonomy class order
    bruce_remap = np.full((N_WIN, C), np.nan, dtype=np.float32)
    for src_idx, tax_idx in enumerate(bundle_to_tax):
        if tax_idx >= 0:
            bruce_remap[:, tax_idx] = bruce_logits[:, src_idx]
    bruce_prob = 1.0 / (1.0 + np.exp(-bruce_remap))  # sigmoid

    # Perch raw logits → taxonomy order (unmapped left as NaN to be excluded from AUC)
    perch_remap = np.full((N_WIN, C), np.nan, dtype=np.float32)
    for src_idx, tax_idx in enumerate(bundle_to_tax):
        if tax_idx >= 0 and perch_idx_for_bundle_cls[src_idx] >= 0:
            perch_remap[:, tax_idx] = logits_for_bruce[:, src_idx]

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
