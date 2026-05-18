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
from sklearn.preprocessing import normalize

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
# BENCHMARK MODE: use train_soundscapes for runtime measurement
TEST_DIR = COMP_DIR / "train_soundscapes"
SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"
print(f"BENCHMARK MODE — running on train_soundscapes")

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
BAL_LR_HITS = list(Path("/kaggle/input").rglob("balanced_lr_bundle.pkl"))
META_HITS = list(Path("/kaggle/input").rglob("meta_stacker_bundle.pkl"))
HOUR_LR_HITS = list(Path("/kaggle/input").rglob("hour_lr_bundle.pkl"))
LGB_HITS = list(Path("/kaggle/input").rglob("lgb_meta_bundle.pkl"))
MLP_HITS = list(Path("/kaggle/input").rglob("mlp_5seed_bundle.pkl"))
PROTO_HITS = list(Path("/kaggle/input").rglob("prototype_bundle.pkl"))
HAS_KNN = bool(KNN_HITS)
HAS_PROBE = bool(PROBE_HITS)
HAS_BAL_LR = bool(BAL_LR_HITS)
HAS_META = bool(META_HITS)
HAS_HOUR_LR = bool(HOUR_LR_HITS)
HAS_LGB = bool(LGB_HITS)
HAS_MLP = bool(MLP_HITS)
HAS_PROTO = bool(PROTO_HITS)

print(f"COMP_DIR:   {COMP_DIR}")
print(f"BUNDLE:     {BUNDLE_PATH}")
print(f"PERCH:      {PERCH_ONNX_PATH}")
print(f"PRIOR:      {PRIOR_PATH}")
print(f"KNN INDEX:  {KNN_HITS[0] if HAS_KNN else '<MISSING — falls back without megaKNN>'}")
print(f"PROBE:      {PROBE_HITS[0] if HAS_PROBE else '<MISSING — falls back without Probe>'}")
print(f"BAL_LR:     {BAL_LR_HITS[0] if HAS_BAL_LR else '<MISSING — falls back without balanced LR (+0.0067 OOF)>'}")
print(f"META:       {META_HITS[0] if HAS_META else '<MISSING — falls back without meta-stacker (+0.0007 OOF)>'}")
print(f"HOUR_LR:    {HOUR_LR_HITS[0] if HAS_HOUR_LR else '<MISSING — falls back without hour-conditional LR (+0.0016 OOF)>'}")
print(f"LGB_META:   {LGB_HITS[0] if HAS_LGB else '<MISSING — falls back without LGB stacker (+0.0012 OOF)>'}")
print(f"MLP_5SEED:  {MLP_HITS[0] if HAS_MLP else '<MISSING — falls back without 5-seed MLP (+0.0033 OOF -> 0.9708)>'}")

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

# sklearn 1.6 compat: bundles pickled under sklearn 1.8 lose the
# `multi_class` attribute on LogisticRegression, breaking predict_proba on
# Kaggle's pinned 1.6.1. Restore it post-load.
def _sklearn_compat_fix(obj):
    from sklearn.linear_model import LogisticRegression as _LR
    seen = set()
    def walk(x):
        i = id(x)
        if i in seen: return
        seen.add(i)
        if isinstance(x, _LR) and not hasattr(x, "multi_class"):
            x.multi_class = "auto"
        if isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x: walk(v)
        elif hasattr(x, "__dict__"):
            for v in vars(x).values(): walk(v)
    walk(obj)
    return obj

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

# Balanced LR bundle (SVD + per-class class_weight='balanced' LogisticRegression)
bal_lr = None
if HAS_BAL_LR:
    with open(BAL_LR_HITS[0], "rb") as f:
        bal_lr = pickle.load(f)
    _sklearn_compat_fix(bal_lr)
    n_trained = sum(1 for m in bal_lr["lr_models"] if m is not None)
    print(f"Balanced LR: SVD={bal_lr['svd'].n_components}, {n_trained}/{len(bal_lr['classes'])} per-class models, "
          f"OOF AUC contribution -> {bal_lr.get('auc_oof', 'unknown')}")

# Meta-stacker bundle (per-class LR over 5 model rank scores)
meta_stacker = None
if HAS_META:
    with open(META_HITS[0], "rb") as f:
        meta_stacker = pickle.load(f)
    _sklearn_compat_fix(meta_stacker)
    n_trained = sum(1 for m in meta_stacker["meta_models"] if m is not None)
    print(f"Meta-stacker: {n_trained}/{len(meta_stacker['classes'])} per-class LR over "
          f"{len(meta_stacker['features'])} rank features, blend alpha={meta_stacker['blend_alpha']}")

# Hour-conditional LR bundle (per-(hour_bucket, class) balanced LR)
hour_lr = None
if HAS_HOUR_LR:
    with open(HOUR_LR_HITS[0], "rb") as f:
        hour_lr = pickle.load(f)
    _sklearn_compat_fix(hour_lr)
    n_trained = len(hour_lr["hr_models"])
    print(f"Hour-LR: {n_trained} per-(bucket, class) models across "
          f"{len(set(k[0] for k in hour_lr['hr_models']))} hour buckets")

# LightGBM meta-stacker bundle (per-class LGB over 6 rank features)
lgb_meta = None
if HAS_LGB:
    try:
        import lightgbm as _lgb  # ensure available at inference
        with open(LGB_HITS[0], "rb") as f:
            lgb_meta = pickle.load(f)
        _sklearn_compat_fix(lgb_meta)
        n_trained = sum(1 for m in lgb_meta["lgb_models"] if m is not None)
        print(f"LGB stacker: {n_trained}/{len(lgb_meta['classes'])} per-class models, "
              f"6 features, blend w_lgb={lgb_meta['blend_w_lgb']}, w_dist={lgb_meta['blend_w_dist']}")
    except ImportError:
        print("WARNING: lightgbm not installed at runtime — skipping LGB stacker")
        HAS_LGB = False
        lgb_meta = None

# MLP 5-seed ensemble bundle (per-class small MLPs, SVD-96 features)
mlp_bundle = None
if HAS_MLP:
    with open(MLP_HITS[0], "rb") as f:
        mlp_bundle = pickle.load(f)
    _sklearn_compat_fix(mlp_bundle)
    n_trained = sum(1 for m in mlp_bundle["mlp_ensembles"] if m is not None)
    print(f"MLP 5-seed: {n_trained}/{len(mlp_bundle['classes'])} per-class ensembles, "
          f"SVD={mlp_bundle['svd'].n_components}, blend alpha={mlp_bundle['blend_alpha']}, "
          f"OOF AUC={mlp_bundle.get('oof_auc', '?')}")

# Prototype bundle (per-class Perch-embedding pure-call prototypes)
# Adds prototype-similarity as a rank-space signal → +0.0023 OOF (0.9706→0.9729)
proto_bundle = None
if HAS_PROTO:
    with open(PROTO_HITS[0], "rb") as f:
        proto_bundle = pickle.load(f)
    prototypes_mat = proto_bundle["prototypes"]  # (234, 1536) L2-normalized
    n_with_proto = int((np.linalg.norm(prototypes_mat, axis=1) > 0.5).sum())
    print(f"Prototypes: {n_with_proto}/234 classes, blend alpha={proto_bundle.get('blend_alpha', 0.30)}")

def hour_bucket(h):
    if h <= 4: return 0
    elif h <= 7: return 1
    elif h <= 10: return 2
    elif h <= 17: return 3
    else: return 4

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
# BENCHMARK: cap to 30 files for runtime measurement
test_files = test_files[:30]
print(f"BENCHMARK: capped to {len(test_files)} files")
if not test_files:
    print("No test files — emitting all-zero submission")
    out_df = samp.copy()
    out_df.iloc[:, 1:] = 0.0
    out_df.to_csv("submission.csv", index=False)
    sys.exit(0)

print(f"\nProcessing {len(test_files)} files (batch={BATCH_FILES})")
ROW_RE = re.compile(r"_(\d{8})_(\d{6})$")

all_bruce, all_perch, all_knn, all_probe, all_bal_lr, all_hour_lr, all_mlp, all_proto = [], [], [], [], [], [], [], []
all_hours = []
row_ids_all = []
t0 = time.time()

def bal_lr_predict(emb_query):
    """Apply SVD then per-class LogisticRegression. Returns (n, 234)."""
    if bal_lr is None:
        return np.zeros((emb_query.shape[0], 234), dtype=np.float32)
    emb_red = bal_lr["svd"].transform(emb_query)
    out = np.full((emb_query.shape[0], 234), 0.5, dtype=np.float32)
    for ci, m in enumerate(bal_lr["lr_models"]):
        if m is None: continue
        out[:, ci] = m.predict_proba(emb_red)[:, 1].astype(np.float32)
    return out

def mlp_predict(emb_query):
    """5-seed MLP ensemble per class. Returns (n, 234) — average over seeds."""
    if mlp_bundle is None:
        return np.zeros((emb_query.shape[0], 234), dtype=np.float32)
    emb_red = mlp_bundle["svd"].transform(emb_query)
    out = np.full((emb_query.shape[0], 234), 0.5, dtype=np.float32)
    for ci, ens in enumerate(mlp_bundle["mlp_ensembles"]):
        if ens is None: continue
        preds = []
        for m in ens:
            try:
                preds.append(m.predict_proba(emb_red)[:, 1])
            except Exception:
                pass
        if preds:
            out[:, ci] = np.mean(preds, axis=0).astype(np.float32)
    return out

def hour_lr_predict(emb_query, hours_query):
    """Per-(hour_bucket, class) LogisticRegression. Returns (n, 234)."""
    if hour_lr is None:
        return np.zeros((emb_query.shape[0], 234), dtype=np.float32)
    emb_red = hour_lr["svd"].transform(emb_query)
    out = np.full((emb_query.shape[0], 234), 0.5, dtype=np.float32)
    buckets = np.array([hour_bucket(h) for h in hours_query])
    for hb in set(buckets.tolist()):
        rows = np.where(buckets == hb)[0]
        if len(rows) == 0: continue
        for ci in range(234):
            m = hour_lr["hr_models"].get((hb, ci))
            if m is None: continue
            try:
                out[rows, ci] = m.predict_proba(emb_red[rows])[:, 1].astype(np.float32)
            except Exception:
                pass
    return out

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
        bal_lr_probs = bal_lr_predict(emb)
        mlp_probs = mlp_predict(emb)
        if proto_bundle is not None:
            emb_n = normalize(emb)
            proto_sim = (emb_n @ prototypes_mat.T).astype(np.float32)  # (B*W, 234)
        else:
            proto_sim = np.zeros((emb.shape[0], 234), dtype=np.float32)
        # For hour-conditional LR we need the per-window hour
        per_win_hours = []
        for bi, (fpath, _) in enumerate(batch_results):
            m = ROW_RE.search(fpath.stem)
            h = int(m.group(2)[:2]) if m else 0
            per_win_hours.extend([h] * N_WINDOWS)
        hour_lr_probs = hour_lr_predict(emb, np.array(per_win_hours, dtype=np.int32))

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
            all_bal_lr.append(bal_lr_probs[s])
            all_hour_lr.append(hour_lr_probs[s])
            all_mlp.append(mlp_probs[s])
            all_proto.append(proto_sim[s])

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
P_bal_lr_all = np.concatenate(all_bal_lr, axis=0)
P_hour_lr_all = np.concatenate(all_hour_lr, axis=0)
P_mlp_all = np.concatenate(all_mlp, axis=0)
P_proto_all = np.concatenate(all_proto, axis=0) if proto_bundle is not None else None
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
R_bal_lr = rank_norm(P_bal_lr_all) if HAS_BAL_LR else None
R_hour_lr = rank_norm(P_hour_lr_all) if HAS_HOUR_LR else None
R_mlp = rank_norm(P_mlp_all) if HAS_MLP else None
R_proto = rank_norm(P_proto_all) if proto_bundle is not None else None

# Step 1: build the 4-model base rank-blend (matches RECIPE_AT_0961 exactly)
if HAS_KNN and HAS_PROBE:
    R_base = 0.30*R_bruce + 0.40*R_knn + 0.20*R_probe + 0.10*R_perch
    print("Base blend: 0.30 Bruce_sm + 0.40 KNN + 0.20 Probe + 0.10 Perch")
elif HAS_KNN:
    R_base = 0.50*R_bruce + 0.40*R_knn + 0.10*R_perch
    print("Fallback base (no Probe): 0.50 Bruce_sm + 0.40 KNN + 0.10 Perch")
elif HAS_PROBE:
    R_base = 0.50*R_bruce + 0.30*R_probe + 0.20*R_perch
    print("Fallback base (no KNN): 0.50 Bruce_sm + 0.30 Probe + 0.20 Perch")
else:
    R_base = 0.70*R_bruce + 0.30*R_perch
    print("Minimum base (Bruce + Perch only)")

# Step 2: blend balanced LR (alone or ensembled with hour-conditional LR) on top
# OOF measurements:
#   R_base + 0.55 * R_bal_lr                        = 0.9647 (+0.0067)
#   R_base + 0.55 * (R_bal_lr + R_hour_lr)/2        = 0.9663 (+0.0083) ⭐
if HAS_BAL_LR and HAS_HOUR_LR:
    ALPHA_LR = 0.55
    R_lr_combined = 0.5 * R_bal_lr + 0.5 * R_hour_lr
    R_blend_v1 = (1 - ALPHA_LR) * R_base + ALPHA_LR * R_lr_combined
    print(f"Added balanced+hour LR ensemble @ alpha={ALPHA_LR} (lifts OOF +0.0083 -> 0.9663)")
elif HAS_BAL_LR:
    ALPHA_LR = bal_lr.get("alpha", 0.55)
    R_blend_v1 = (1 - ALPHA_LR) * R_base + ALPHA_LR * R_bal_lr
    print(f"Added balanced LR @ alpha={ALPHA_LR} (lifts OOF +0.0067)")
else:
    R_blend_v1 = R_base
    print("No balanced LR available")

# Step 2.5: blend LightGBM meta-stacker on top (final +0.0012 OOF)
if HAS_LGB and HAS_BAL_LR and HAS_HOUR_LR:
    n_rows = R_bruce.shape[0]
    P_lgb_all = np.full((n_rows, 234), 0.5, dtype=np.float32)
    for ci, m in enumerate(lgb_meta["lgb_models"]):
        if m is None: continue
        # 6 features in this exact order: Bruce, KNN, Probe, Perch, BalLR, HourLR
        X = np.column_stack([R_bruce[:, ci],
                             R_knn[:, ci] if HAS_KNN else np.full(n_rows, 0.5),
                             R_probe[:, ci] if HAS_PROBE else np.full(n_rows, 0.5),
                             R_perch[:, ci], R_bal_lr[:, ci], R_hour_lr[:, ci]])
        try:
            P_lgb_all[:, ci] = m.predict_proba(X)[:, 1].astype(np.float32)
        except Exception:
            pass
    R_lgb = rank_norm(P_lgb_all)
    W_LGB = lgb_meta.get("blend_w_lgb", 0.10)
    R_blend_v1 = (1 - W_LGB) * R_blend_v1 + W_LGB * R_lgb
    print(f"Added LGB stacker @ w={W_LGB} (lifts OOF +0.0008 -> 0.9671)")

# Step 2.7: blend MLP 5-seed ensemble on top (+0.0033 OOF — biggest stacker win)
if HAS_MLP and R_mlp is not None:
    ALPHA_MLP = mlp_bundle.get("blend_alpha", 0.35)
    R_blend_v1 = (1 - ALPHA_MLP) * R_blend_v1 + ALPHA_MLP * R_mlp
    print(f"Added 5-seed MLP @ alpha={ALPHA_MLP} (lifts OOF +0.0033 -> 0.9708)")

# Step 2.8: blend pure-call prototype similarity (+0.0023 OOF — global alpha)
# Per-class Perch-embedding prototype from KNN-DB single-label rows.
# Adds within-chorus disambiguation signal that complements all other models.
if proto_bundle is not None and R_proto is not None:
    ALPHA_PROTO = proto_bundle.get("blend_alpha", 0.30)
    R_blend_v1 = (1 - ALPHA_PROTO) * R_blend_v1 + ALPHA_PROTO * R_proto
    print(f"Added prototype-sim @ alpha={ALPHA_PROTO} (lifts OOF +0.0023 -> 0.9729)")

# Step 3: blend meta-stacker on top (the +0.0007 OOF additive — per-class LR over rank features)
if HAS_META:
    # Build per-class meta predictions
    n_rows = R_bruce.shape[0]
    P_meta_all = np.full((n_rows, 234), 0.5, dtype=np.float32)
    for ci, m in enumerate(meta_stacker["meta_models"]):
        if m is None: continue
        # Stack 5 features for this class across all rows
        if HAS_BAL_LR:
            X = np.column_stack([R_bruce[:, ci], R_knn[:, ci] if HAS_KNN else np.full(n_rows, 0.5),
                                 R_probe[:, ci] if HAS_PROBE else np.full(n_rows, 0.5),
                                 R_perch[:, ci], R_bal_lr[:, ci]])
        else:
            # Mirror with zeros for missing models so the LR sees the same feature shape
            X = np.column_stack([R_bruce[:, ci], R_knn[:, ci] if HAS_KNN else np.full(n_rows, 0.5),
                                 R_probe[:, ci] if HAS_PROBE else np.full(n_rows, 0.5),
                                 R_perch[:, ci], np.full(n_rows, 0.5)])
        try:
            P_meta_all[:, ci] = m.predict_proba(X)[:, 1].astype(np.float32)
        except Exception:
            pass
    R_meta = rank_norm(P_meta_all)
    ALPHA_META = meta_stacker["blend_alpha"]
    blend = (1 - ALPHA_META) * R_blend_v1 + ALPHA_META * R_meta
    print(f"Added meta-stacker @ alpha={ALPHA_META} (lifts OOF by +0.0007 -> 0.9654)")
else:
    blend = R_blend_v1
    print("No meta-stacker available (would have added +0.0007 OOF)")

# Apply combined hour prior
W_PRIOR = 2.0 if HAS_MLP else 2.5  # MLP recipe plateau peaks at w=2.0; old recipe at 2.5
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
      f"Probe={'YES' if HAS_PROBE else 'NO'} + Perch + "
      f"BalancedLR={'YES' if HAS_BAL_LR else 'NO'} + "
      f"HourLR={'YES' if HAS_HOUR_LR else 'NO'} + "
      f"LGB={'YES' if HAS_LGB else 'NO'} + "
      f"MetaStacker={'YES' if HAS_META else 'NO'} + combined_prior(w={W_PRIOR})")
expected_oof = "0.9580"
if HAS_BAL_LR: expected_oof = "0.9647"
if HAS_BAL_LR and HAS_HOUR_LR: expected_oof = "0.9663"
if HAS_BAL_LR and HAS_HOUR_LR and HAS_LGB: expected_oof = "0.9671"
if HAS_BAL_LR and HAS_HOUR_LR and HAS_LGB and HAS_META: expected_oof = "0.9675"
if HAS_BAL_LR and HAS_HOUR_LR and HAS_MLP: expected_oof = "0.9708"
print(f"Expected OOF: {expected_oof}")
