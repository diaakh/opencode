"""
============================================================================
SUBMISSION v8 — full 0.9613-OOF recipe as standalone notebook
============================================================================

Self-contained Bruce + Perch + megaKNN + Probe + combined_hour_prior pipeline.
Reproduces the 0.9613 honest macro-AUC recipe from RECIPE_AT_0961.md on the
test_soundscapes.

REQUIRED Kaggle datasets attached:
  - birdclef-2026 (auto)
  - brucewu1200/birdclef-2026-cvlb-assets-0911  (clip_student_bundle + Perch ONNX)
  - tuckerarrants/perch-v2-no-dft-onnx           (onnxruntime wheel)
  - adkasd/birdclef-2026-priors-research         (combined_hour_prior.csv)
  - adkasd/birdclef-2026-knn-index               (knn_index.pkl, 93MB)  ← upload before submitting
  - adkasd/birdclef-2026-probe-ridge             (probe_ridge.pkl)       ← upload before submitting

If knn_index / probe_ridge datasets aren't uploaded yet, the script falls
back gracefully and uses only Bruce+Perch+prior (still ~0.93 expected LB).

Math vs RECIPE_AT_0961.md (0.9613 OOF):
  - Bruce_smoothed (within-file texture/event kernels):   IDENTICAL
  - megaKNN multi-K {10, 20, 50}:                         IDENTICAL
  - Probe Ridge on Perch embedding:                       IDENTICAL
  - Combined hour prior (pseudo + iNat) at w=2.5:         IDENTICAL
  - Multi-distance + meta-augmented KNN variants:         OMITTED for CPU time
    (the simpler cosine-only KNN captures ~0.005-0.007 less than full megaKNN —
     expected OOF ~0.954-0.956 vs 0.9613, still well above Bruce-alone 0.867)

Expected runtime on Kaggle CPU (4 vCPU): 35-50 min for 600 test files
  — well under the 90-min budget for competition submissions.
"""

import gc, pickle, re, sys, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

EPS = 1e-7
N_WINDOWS = 12
SR = 32000
WIN_SEC = 5
SAMPS = SR * WIN_SEC
BATCH_FILES = 16  # 16*12 = 192 windows per ONNX call — sustained CPU throughput
KNN_KS = (10, 20, 50)  # multi-K from the 0.9613 recipe (matches OOF math exactly)

# ============================================================================
# Discover paths
# ============================================================================
COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP_DIR.exists():
    COMP_DIR = Path("/kaggle/input/birdclef-2026")
TEST_DIR = COMP_DIR / "test_soundscapes"
SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"

BUNDLE_HITS = list(Path("/kaggle/input").rglob("clip_student_bundle.pkl"))
assert BUNDLE_HITS, "Attach brucewu1200/birdclef-2026-cvlb-assets-0911"
BUNDLE_PATH = BUNDLE_HITS[0]

PERCH_HITS = list(Path("/kaggle/input").rglob("perch_v2_no_dft.onnx"))
assert PERCH_HITS, "Could not find perch_v2_no_dft.onnx"
PERCH_ONNX_PATH = PERCH_HITS[0]

PRIOR_HITS = list(Path("/kaggle/input").rglob("combined_hour_prior.csv"))
assert PRIOR_HITS, "Attach adkasd/birdclef-2026-priors-research (with combined_hour_prior.csv)"
PRIOR_PATH = PRIOR_HITS[0]

KNN_HITS = list(Path("/kaggle/input").rglob("knn_index.pkl"))
PROBE_HITS = list(Path("/kaggle/input").rglob("probe_ridge.pkl"))
HAS_KNN = bool(KNN_HITS)
HAS_PROBE = bool(PROBE_HITS)

print(f"COMP_DIR:   {COMP_DIR}")
print(f"BUNDLE:     {BUNDLE_PATH}")
print(f"PERCH:      {PERCH_ONNX_PATH}")
print(f"PRIOR:      {PRIOR_PATH}")
print(f"KNN INDEX:  {KNN_HITS[0] if HAS_KNN else '<MISSING — falls back without megaKNN>'}")
print(f"PROBE:      {PROBE_HITS[0] if HAS_PROBE else '<MISSING — falls back without Probe>'}")

# Install onnxruntime if missing
try:
    import onnxruntime as ort
except ImportError:
    whls = list(Path("/kaggle/input").rglob("onnxruntime-*.whl"))
    assert whls, "Attach perch-v2-no-dft-onnx with onnxruntime wheel"
    subprocess.check_call(["pip", "install", "-q", str(whls[0])])
    import onnxruntime as ort

# ============================================================================
# Load model artifacts
# ============================================================================
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)
scaler = bundle["clip_bundle"]["emb_scaler"]
pca = bundle["clip_bundle"]["pca"]
fscaler = bundle["clip_bundle"]["feature_scaler"]
ridge = bundle["clip_bundle"]["model"]
primary_labels = bundle["primary_labels"]
class_to_bc = bundle["class_to_bc"]
print(f"Bruce ridge loaded: {len(primary_labels)} primary classes")

so = ort.SessionOptions()
# Kaggle gives 4 vCPUs. Use all of them on ONNX matmul, leave IO threads to Python.
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
perch_sess = ort.InferenceSession(str(PERCH_ONNX_PATH), so, providers=["CPUExecutionProvider"])
perch_in = perch_sess.get_inputs()[0].name

# KNN
knn_db = None
if HAS_KNN:
    with open(KNN_HITS[0], "rb") as f:
        knn_db = pickle.load(f)
    print(f"KNN db: emb={knn_db['emb_db_n'].shape}, Y={knn_db['Y_db'].shape}")

# Probe
probe = None
if HAS_PROBE:
    with open(PROBE_HITS[0], "rb") as f:
        probe = pickle.load(f)
    print(f"Probe loaded: {type(probe).__name__}")

# Combined hour prior
hp_df = pd.read_csv(PRIOR_PATH).set_index("hour")
samp = pd.read_csv(SAMPLE_SUB_PATH)
class_cols = [c for c in samp.columns if c != "row_id"]
assert len(class_cols) == 234
hp_arr = hp_df.reindex(columns=class_cols).fillna(0).to_numpy(dtype=np.float64)
hp_arr = np.clip(hp_arr, EPS, 1.0)
log_hp = np.log(hp_arr)
print(f"Combined prior: shape={hp_arr.shape}, full 24h dense")

# ============================================================================
# Helpers
# ============================================================================
def load_audio(path):
    y, sr = sf.read(path, dtype="float32")
    if sr != SR:
        import scipy.signal
        y = scipy.signal.resample_poly(y, SR, sr).astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1).astype(np.float32)
    return y

def chunk_windows(y, expected=N_WINDOWS):
    target = expected * SAMPS
    if y.shape[0] < target:
        y = np.pad(y, (0, target - y.shape[0]))
    return y[:target].reshape(expected, SAMPS)

def rank_norm(P):
    from scipy.stats import rankdata
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(P.shape[1]):
        col = P[:, c]
        v = ~np.isnan(col)
        if v.sum() > 0:
            R[v, c] = rankdata(col[v]) / v.sum()
            R[~v, c] = 0.5
    return R

# Sonotype/Insecta/Amphibia detection — these get the texture smoothing kernel
SONOTYPE_PREFIXES = ("47158", "sonotype")
def is_texture_class(cls):
    return any(p in str(cls).lower() for p in ["son", "frog", "amphib"])
is_texture = np.array([is_texture_class(c) for c in class_cols])

def smooth_within_file(P_file, ker_event=(0.20, 0.60, 0.20), ker_tex=(0.35, 0.30, 0.35)):
    """P_file: (12, 234) -> smoothed (12, 234)."""
    out = P_file.copy()
    for c in range(P_file.shape[1]):
        ker = ker_tex if is_texture[c] else ker_event
        pad = np.pad(P_file[:, c], (1, 1), mode="edge")
        out[:, c] = pad[:-2] * ker[0] + pad[1:-1] * ker[1] + pad[2:] * ker[2]
    return out

def perch_predict(x):
    outs = perch_sess.run(None, {perch_in: x})
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
        logit = np.zeros((x.shape[0], 234), dtype=np.float32)
    return emb, logit

def bruce_predict(emb, logit):
    emb_s = scaler.transform(emb)
    emb_p = pca.transform(emb_s)
    feat = np.concatenate([emb_p, logit], axis=1)
    feat = fscaler.transform(feat)
    return ridge.predict(feat)

def knn_predict(emb_query):
    """Vectorized multi-K KNN cosine. Returns (n, 234) probability.

    Implements the same K in (10, 20, 50) ensemble as the 0.9613 OOF recipe.
    Trick: sort once, slice for each K — saves redundant argpartition calls.
    The np.einsum over (n, K, 234) handles all queries' weighted votes at once.
    """
    if knn_db is None:
        return np.zeros((emb_query.shape[0], 234), dtype=np.float32)
    from sklearn.preprocessing import normalize
    q = normalize(emb_query)
    db = knn_db["emb_db_n"]
    Y_db = knn_db["Y_db"]
    sims = q @ db.T  # (n, N_db)
    n_q = emb_query.shape[0]
    K_max = min(max(KNN_KS), sims.shape[1] - 1)
    # One sort gets the top-K_max — then slice for each smaller K
    idx_top_max = np.argpartition(-sims, K_max, axis=1)[:, :K_max]  # (n, K_max)
    row_ix = np.arange(n_q)[:, None]
    sims_top = sims[row_ix, idx_top_max]  # (n, K_max)
    # Sort top-K_max by sim desc so we can slice prefixes for each K
    order = np.argsort(-sims_top, axis=1)
    idx_sorted = np.take_along_axis(idx_top_max, order, axis=1)
    sims_sorted = np.take_along_axis(sims_top, order, axis=1)

    out = np.zeros((n_q, 234), dtype=np.float32)
    for K in KNN_KS:
        K_ = min(K, K_max)
        idx_k = idx_sorted[:, :K_]
        w = np.maximum(sims_sorted[:, :K_], 0)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum > 0, w_sum, 1.0)
        w_norm = w / w_sum  # (n, K_)
        Y_top = Y_db[idx_k]  # (n, K_, 234)
        out += np.einsum("nk,nkc->nc", w_norm, Y_top).astype(np.float32)
    return out / len(KNN_KS)

# ============================================================================
# Main inference
# ============================================================================
test_files = sorted(TEST_DIR.glob("*.ogg"))
if not test_files:
    print("No test files — emitting all-zero submission")
    out_df = samp.copy()
    out_df.iloc[:, 1:] = 0.0
    out_df.to_csv("submission.csv", index=False)
    sys.exit(0)

print(f"\nProcessing {len(test_files)} files (batch={BATCH_FILES})")
ROW_RE = re.compile(r"_(\d{8})_(\d{6})$")

all_bruce, all_perch, all_knn, all_probe = [], [], [], []
all_hours = []
row_ids_all = []
t0 = time.time()

import concurrent.futures
def load_and_chunk(p):
    return p, chunk_windows(load_audio(p))

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    next_batch = test_files[:BATCH_FILES]
    next_futures = [pool.submit(load_and_chunk, p) for p in next_batch]

    for start in range(0, len(test_files), BATCH_FILES):
        batch_results = [f.result() for f in next_futures]
        next_start = start + BATCH_FILES
        if next_start < len(test_files):
            next_batch = test_files[next_start:next_start + BATCH_FILES]
            next_futures = [pool.submit(load_and_chunk, p) for p in next_batch]

        bn = len(batch_results)
        x = np.empty((bn * N_WINDOWS, SAMPS), dtype=np.float32)
        for bi, (_, yw) in enumerate(batch_results):
            x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = yw

        emb, logit = perch_predict(x)
        bruce_logits = bruce_predict(emb, logit)
        bruce_probs = 1.0 / (1.0 + np.exp(-bruce_logits))
        perch_probs = 1.0 / (1.0 + np.exp(-logit))
        knn_probs = knn_predict(emb)
        probe_probs = (1.0 / (1.0 + np.exp(-probe.predict(emb)))
                       if probe is not None else np.zeros_like(bruce_probs))

        for bi, (fpath, _) in enumerate(batch_results):
            s = slice(bi * N_WINDOWS, (bi + 1) * N_WINDOWS)
            stem = fpath.stem
            m = ROW_RE.search(stem)
            hour = int(m.group(2)[:2]) if m else 0
            for i in range(N_WINDOWS):
                row_ids_all.append(f"{stem}_{(i + 1) * WIN_SEC}")
                all_hours.append(hour)

            # Within-file smoothing on Bruce
            bp_sm = smooth_within_file(bruce_probs[s])
            all_bruce.append(bp_sm)
            all_perch.append(perch_probs[s])
            all_knn.append(knn_probs[s])
            all_probe.append(probe_probs[s])

        done = start + bn
        if done % (BATCH_FILES * 5) == 0 or done == len(test_files):
            el = time.time() - t0
            rate = done / max(el, 1.0)
            eta = (len(test_files) - done) / max(rate, 0.01)
            print(f"  [{done}/{len(test_files)}] elapsed={el:.0f}s "
                  f"rate={rate:.2f} files/s eta={eta:.0f}s")

P_bruce_all = np.concatenate(all_bruce, axis=0)
P_perch_all = np.concatenate(all_perch, axis=0)
P_knn_all = np.concatenate(all_knn, axis=0)
P_probe_all = np.concatenate(all_probe, axis=0)
hours = np.array(all_hours, dtype=np.int32)
print(f"\nInference done in {time.time()-t0:.0f}s. "
      f"Bruce p1-p99: [{np.percentile(P_bruce_all,1):.3f}, {np.percentile(P_bruce_all,99):.3f}]")

# ============================================================================
# Cross-file rank-normalize + weighted blend + prior
# ============================================================================
print("Computing cross-file rank-norm...")
R_bruce = rank_norm(P_bruce_all)
R_perch = rank_norm(P_perch_all)
R_knn = rank_norm(P_knn_all) if HAS_KNN else None
R_probe = rank_norm(P_probe_all) if HAS_PROBE else None

# Blend (fall back gracefully if KNN/Probe missing)
if HAS_KNN and HAS_PROBE:
    blend = 0.30*R_bruce + 0.40*R_knn + 0.20*R_probe + 0.10*R_perch
    print("Full blend: 0.30 Bruce_sm + 0.40 KNN + 0.20 Probe + 0.10 Perch")
elif HAS_KNN:
    blend = 0.50*R_bruce + 0.40*R_knn + 0.10*R_perch
    print("Fallback blend (no Probe): 0.50 Bruce_sm + 0.40 KNN + 0.10 Perch")
elif HAS_PROBE:
    blend = 0.50*R_bruce + 0.30*R_probe + 0.20*R_perch
    print("Fallback blend (no KNN): 0.50 Bruce_sm + 0.30 Probe + 0.20 Perch")
else:
    blend = 0.70*R_bruce + 0.30*R_perch
    print("Minimum blend (Bruce + Perch only)")

# Apply combined hour prior
W_PRIOR = 2.5  # tested optimal for rank-blend on labeled OOF
logit_p = np.log(np.clip(blend, EPS, 1-EPS) / np.clip(1-blend, EPS, 1))
valid = (hours >= 0) & (hours < 24)
shift = np.zeros_like(blend, dtype=np.float64)
shift[valid] = W_PRIOR * log_hp[hours[valid]]
final = 1.0 / (1.0 + np.exp(-(logit_p + shift)))
final = np.clip(final, 0.0, 1.0).astype(np.float32)

out_df = pd.DataFrame(final, columns=class_cols)
out_df.insert(0, "row_id", row_ids_all)

sample_ids = samp["row_id"].astype(str).tolist()
if set(out_df["row_id"]) == set(sample_ids):
    out_df = out_df.set_index("row_id").loc[sample_ids].reset_index()
out_df.to_csv("submission.csv", index=False)
print(f"\nWrote submission.csv: {len(out_df)} rows × {out_df.shape[1]} cols, "
      f"min={final.min():.4f}, max={final.max():.4f}")
print(f"Recipe: Bruce_smoothed + KNN={'YES' if HAS_KNN else 'NO'} + "
      f"Probe={'YES' if HAS_PROBE else 'NO'} + Perch + combined_prior(w={W_PRIOR})")
