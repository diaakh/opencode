# ============================================================================
# SUB V6: Bruce standalone + FULL augmentation stack (TTA averaging)
# ============================================================================
# Goes beyond v5 (just time-shift) by stacking MULTIPLE augmentations:
#   - 3-shift TTA (0, ±2.5s)
#   - Gain TTA (0, ±3dB)  
#   - RMS normalization
#   - All combined → rank-blend average
#
# Each path runs full Perch+Ridge inference, then we average logits.
# Total: ~7 paths × 5 min = 35 min — within 90 min budget.
# ============================================================================
import gc, pickle, re, sys, subprocess, time
import concurrent.futures
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf

ASSETS = list(Path("/kaggle/input").rglob("clip_student_bundle.pkl"))[0].parent
try:
    import onnxruntime as ort
except ImportError:
    whl = list(Path("/kaggle/input").rglob("onnxruntime-*.whl"))[0]
    subprocess.check_call(["pip", "install", "-q", str(whl)])
    import onnxruntime as ort

COMP = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP.exists():
    COMP = Path("/kaggle/input/birdclef-2026")
TEST = COMP / "test_soundscapes"
SAMP = pd.read_csv(COMP / "sample_submission.csv")
CLS = [c for c in SAMP.columns if c != "row_id"]

BUNDLE = pickle.load(open(ASSETS / "clip_student_bundle.pkl", "rb"))
scaler, pca, fscaler, ridge = (BUNDLE["clip_bundle"][k] for k in ("emb_scaler", "pca", "feature_scaler", "model"))
PERCH = ort.InferenceSession(str(list(Path("/kaggle/input").rglob("perch_v2_no_dft.onnx"))[0]),
                              providers=["CPUExecutionProvider"])
P_IN = PERCH.get_inputs()[0].name

PRIORS = list(Path("/kaggle/input").rglob("pseudo_hour_priors.csv"))[0].parent
hour_df = pd.read_csv(PRIORS / "pseudo_hour_priors.csv").set_index("hour")
hsum = hour_df.sum(axis=1)
covered = hsum[hsum > 0].index
for h in range(24):
    if h not in covered: hour_df.loc[h] = hour_df.loc[covered].mean(axis=0)
hour_df = hour_df.sort_index()
try:
    lab = pd.read_csv(PRIORS / "hourly_species_priors.csv").set_index("hour")
    for h in range(24):
        if h not in lab.index: lab.loc[h] = lab.mean(axis=0)
    lab = lab.sort_index()
    for c in lab.columns:
        if c in hour_df.columns:
            hour_df[c] = lab[c].reindex(hour_df.index).fillna(lab[c].mean())
except FileNotFoundError: pass

SR, WIN, NW, SAMPS = 32000, 5, 12, 32000 * 5
TOTAL = NW * SAMPS

def load_audio(p):
    y, sr = sf.read(p, dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1).astype(np.float32)
    if sr != SR:
        import scipy.signal
        y = scipy.signal.resample_poly(y, SR, sr).astype(np.float32)
    return y

def chunked(y, shift=0):
    if shift >= 0:
        ys = y[shift:]
    else:
        ys = np.concatenate([np.zeros(-shift, dtype=np.float32), y])
    if ys.shape[0] < TOTAL: ys = np.pad(ys, (0, TOTAL - ys.shape[0]))
    return ys[:TOTAL].reshape(NW, SAMPS)

def rms_norm(y, target=0.05):
    cur = float(np.sqrt(np.mean(y ** 2)))
    if cur < 1e-6: return y
    return np.clip(y * (target / cur), -1.0, 1.0).astype(np.float32)

def gain(y, db):
    g = 10 ** (db / 20.0)
    return np.clip(y * g, -1.0, 1.0).astype(np.float32)

# Augmentation paths to ensemble
AUG_PATHS = [
    ("baseline",         lambda y: y, 0),
    ("shift+2.5s",       lambda y: y, int(2.5 * SR)),
    ("shift-2.5s",       lambda y: y, -int(2.5 * SR)),
    ("gain+3dB",         lambda y: gain(y, 3), 0),
    ("gain-3dB",         lambda y: gain(y, -3), 0),
    ("rms_norm",         lambda y: rms_norm(y, 0.05), 0),
    ("rms_norm+shift",   lambda y: rms_norm(y, 0.05), int(2.5 * SR)),
]
print(f"[v6] Augmentation paths: {len(AUG_PATHS)}")

def perch_predict(x):
    outs = PERCH.run(None, {P_IN: x})
    emb = log = None
    for o in outs:
        a = np.asarray(o)
        if a.ndim == 2 and a.shape[1] == 1536: emb = a.astype(np.float32)
        elif a.ndim == 2 and a.shape[1] == 234: log = a.astype(np.float32)
    if emb is None: emb = np.asarray(outs[0], dtype=np.float32).reshape(x.shape[0], -1)[:, :1536]
    if log is None: log = np.zeros((x.shape[0], 234), dtype=np.float32)
    return emb, log

def bruce(emb, log):
    f = np.concatenate([pca.transform(scaler.transform(emb)), log], axis=1)
    return ridge.predict(fscaler.transform(f))

files = sorted(TEST.glob("*.ogg"))
if not files:
    SAMP.iloc[:, 1:] = 0.0
    SAMP.to_csv("submission.csv", index=False); sys.exit(0)

hp_arr = np.zeros((24, 234), dtype=np.float64)
for h in range(24):
    hp_arr[h] = hour_df.loc[h].reindex(CLS).fillna(0.0).to_numpy()
log_hp = np.log(np.clip(hp_arr, 1e-7, 1.0))
HW = 0.02
ROW_RE = re.compile(r"_(\d{8})_(\d{6})$")
BATCH = 8

print(f"[v6] {len(files)} files × {len(AUG_PATHS)} aug paths × Perch")
row_ids, prob_all = [], []
t0 = time.time()

for start in range(0, len(files), BATCH):
    bp = files[start:start + BATCH]
    bn = len(bp)
    # accumulate logits across augmentation paths
    agg = np.zeros((bn * NW, 234), dtype=np.float32)
    for name, aug_fn, shift in AUG_PATHS:
        x = np.empty((bn * NW, SAMPS), dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            ys = list(pool.map(load_audio, bp))
        for bi, y in enumerate(ys):
            y_aug = aug_fn(y)
            x[bi*NW:(bi+1)*NW] = chunked(y_aug, shift)
        emb, log = perch_predict(x)
        agg += bruce(emb, log) / len(AUG_PATHS)
    # hour prior per file
    for bi, fp in enumerate(bp):
        s = slice(bi*NW, (bi+1)*NW)
        m = ROW_RE.search(fp.stem)
        hr = int(m.group(2)[:2]) if m else 0
        L = agg[s] + (HW * log_hp[hr][None, :] if 0 <= hr < 24 else 0)
        p = np.clip(1.0/(1.0 + np.exp(-L)), 0, 1).astype(np.float32)
        for i in range(NW):
            row_ids.append(f"{fp.stem}_{(i+1)*WIN}")
        prob_all.append(p)
    done = start + bn
    if done % (BATCH * 3) == 0 or done == len(files):
        e = time.time() - t0
        print(f"  [{done}/{len(files)}] elapsed={e:.0f}s rate={done/max(e,1):.2f} f/s")

prob_all = np.concatenate(prob_all, axis=0)
out = pd.DataFrame(prob_all, columns=CLS)
out.insert(0, "row_id", row_ids)
ids = SAMP["row_id"].astype(str).tolist()
if set(out["row_id"]) == set(ids):
    out = out.set_index("row_id").loc[ids].reset_index()
out.to_csv("submission.csv", index=False)
print(f"[v6] DONE: {time.time()-t0:.0f}s ({(time.time()-t0)/len(files):.1f}s/file)")
