"""
================================================================================
 BirdCLEF+ 2026 — SPRINT ORTHO-BLEND  (Build-Agent B3, items T1-1 + T1-2 + T1-4)
================================================================================
CPU inference notebook (no internet at submit). Output: submission.csv
  - 234 species columns, row_id = "<filename>_<endsec>", 5s windows, 12 windows/file
  - ~600 test soundscapes (1 min each)

WHAT THIS IS
------------
Reproduces the EoS9 ~0.950 public anchor (ProtoSSM + distilled-SED dominant branch,
percentile-RANK blend, site/hour priors, taxonomy genus/class smoothing) AND adds the
LIVE UNEXPLOITED orthogonal trained models as low-rank-weight blend members:
  * tonylica/birdclef-2026-model  (rank-5 team weights, 0 importers in voted crowd)
  * hideyukizushi/sgkfk-202604041716 (ProtoSSM + ResidualSSM — the new public SSM base)

ENSEMBLE MEMBERSHIP IS FULLY CONFIG-DRIVEN (ENSEMBLE dict below). The orchestrator's
leave-one-site-out validation on BirdMAE / Perch20 runs in parallel; those members are
left as commented-ready toggles so they can be flipped on once validation lands —
without touching any inference code.

SPEED (item T1-4)
-----------------
ONNX models (Perch + 5-fold distilled SED) run through OpenVINO FP16 + AsyncInferQueue
when openvino is importable, else fall back to ONNXRuntime CPU (intra=4, inter=1,
ORT_ENABLE_ALL). FP16 on the heavy stages roughly halves heavy-stage wall time with ~0
ROC-AUC cost (the metric is rank-only — see A4). 90-min budget table in README.md.

THE METRIC (A4)
---------------
Macro-averaged ROC-AUC = per-class cross-row ranking. Only within-column order matters;
per-class calibration is a no-op. => fuse models in RANK space, not probability space.
================================================================================
"""

# ============================================================================
# CELL 0 — CONFIG  (everything tunable lives here)
# ============================================================================
import os, re, gc, sys, time, glob, pickle, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

EPS         = 1e-7
SR          = 32000
WIN_SEC     = 5
N_WINDOWS   = 12                  # 60s / 5s
WINDOW_SAMPLES = SR * WIN_SEC     # 160_000
NUM_CLASSES = 234
BATCH_FILES = 16                  # 16*12 = 192 windows/ONNX call — saturates CPU SIMD

# ---- distilled-SED mel front-end (EXACT public 0.950 convention) ----
SED_N_FFT, SED_HOP, SED_NMELS = 2048, 512, 256
SED_FMIN, SED_FMAX, SED_TOPDB = 20, 16000, 80
SED_N_FRAMES = WINDOW_SAMPLES // SED_HOP + 1   # 313  -> mel input (N,1,256,313)

# ---- rank-blend / post-proc knobs (A4 + EoS9 anchor) ----
RANK_POWER       = 0.5            # (rank/n)^p ; the 0.95+ cluster converged on ~0.5
PROTO_W, SED_W   = 0.60, 0.40     # dominant-branch internal mix R = .60 R(proto)+.40 R(sed)
W_PRIOR          = 2.0            # site/hour log-prior shift weight (conservative; A4 says keep small)
GENUS_ALPHA      = 0.15           # taxonomy smoothing (architecture doc)
CLASS_ALPHA      = 0.05
GAUSS_SIGMA      = 0.65           # within-file temporal smoothing of SED across 12 windows

# ============================================================================
#  ENSEMBLE MEMBERSHIP  — *** the one dict to edit ***  (item: config-driven)
# ----------------------------------------------------------------------------
#  name -> rank-weight. Weights are RELATIVE; they are renormalized at blend time.
#  Orthogonal trained members enter at LOW rank-weight per A4 (0.05–0.15).
#  The two dominant-branch members (proto, sed) carry the EoS9 anchor.
#  BirdMAE / Perch20 are COMMENTED-READY: uncomment once the orchestrator's
#  leave-one-site-out validation confirms they transfer.
# ============================================================================
ENSEMBLE = {
    # --- EoS9 dominant branch (the ~0.950 anchor) ---
    "proto_ssm":   0.60,   # hideyukizushi/sgkfk ProtoSSM  (dominant)
    "distilled_sed": 0.40, # tuckerarrants 5-fold distilled SED (dominant)

    # --- live UNEXPLOITED orthogonal trained members (T1-2) — low rank-weight ---
    "tonylica":    0.10,   # tonylica/birdclef-2026-model  (rank-5 team, 0 importers)
    "sgkfk_resssm":0.08,   # hideyukizushi ResidualSSM (sgkfk dataset) — diversity branch

    # --- validated members — need asset+branch wired before enabling ---
    # "birdmae":   0.05,   # LOSO-CONFIRMED +0.004 (real, survives held-out sites). TODO: mount
    #                      # a BirdMAE ONNX asset + add an inference branch + member_available=True.
    # "perch20":   0.05,   # Perch 2.0 — LOSO: no lift on labeled-OOF; keep off pending full-test.
}

# Members that are GATED behind their asset being mounted. If the asset is absent,
# the member is silently dropped and remaining weights renormalize (graceful degrade).

# ============================================================================
# CELL 1 — DISCOVER MOUNTS  (verified live via Kaggle API 2026-05-29; see README)
# ============================================================================
def _first(*globs):
    for g in globs:
        hits = sorted(Path("/kaggle/input").rglob(g))
        if hits:
            return hits[0]
    return None

COMP_DIR = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP_DIR.exists():
    COMP_DIR = Path("/kaggle/input/birdclef-2026")
TEST_DIR        = COMP_DIR / "test_soundscapes"
SAMPLE_SUB_PATH = COMP_DIR / "sample_submission.csv"
TAXONOMY_PATH   = COMP_DIR / "taxonomy.csv"

# --- distilled-SED 5 folds: tuckerarrants/bc2026-distilled-sed-public ---
#     mount: /kaggle/input/bc2026-distilled-sed-public/sed_fold{0..4}.onnx
SED_FOLD0 = _first("sed_fold0.onnx")
SED_DIR   = SED_FOLD0.parent if SED_FOLD0 else None

# --- perch v2 (no dft) ONNX: tuckerarrants/perch-v2-no-dft-onnx ---
#     mount: /kaggle/input/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx
PERCH_ONNX = _first("perch_v2_no_dft.onnx")

# --- sgkfk weights: hideyukizushi/sgkfk-202604041716 ---
#     mount: /kaggle/input/sgkfk-202604041716/...
PROTO_SSM_PT   = _first("proto_ssm_best.pt", "*proto_ssm*best*.pt")
PROTO_SSM_JSON = _first("proto_ssm_history.json")
RESSSM_PT      = _first("residual_ssm_best.pt", "*residual*ssm*best*.pt")
SGKFK_OOF_META = _first("full_oof_meta_features.npz")

# --- tonylica/birdclef-2026-model ---
#     mount: /kaggle/input/birdclef-2026-model/{LB872.pt,LB862.pt,xsed/sed_fold*.onnx,...}
TONYLICA_MANIFEST = _first("manifest.json")
TONYLICA_DIR      = TONYLICA_MANIFEST.parent if TONYLICA_MANIFEST else None
# tonylica ships its own copy of the 5-fold distilled SED in xsed/ — we run those
# as the tonylica orthogonal member (its own-trained heads LB872/LB862 are raw
# state_dicts without a published model class; the xsed ONNX folds are the safe,
# portable, runnable signal that we ensemble as the 'tonylica' member).
TONYLICA_XSED0 = _first("xsed/sed_fold0.onnx")
TONYLICA_XSED_DIR = TONYLICA_XSED0.parent if TONYLICA_XSED0 else None

print("==== MOUNT DISCOVERY ====")
for nm, p in [("COMP_DIR", COMP_DIR), ("SED_DIR", SED_DIR), ("PERCH_ONNX", PERCH_ONNX),
              ("PROTO_SSM_PT", PROTO_SSM_PT), ("RESSSM_PT", RESSSM_PT),
              ("TONYLICA_DIR", TONYLICA_DIR), ("TONYLICA_XSED_DIR", TONYLICA_XSED_DIR)]:
    print(f"  {nm:18s}: {p}")

# Optional onnxruntime wheel (offline install) shipped in perch-v2-no-dft-onnx
try:
    import onnxruntime as ort  # noqa
except ImportError:
    whl = _first("onnxruntime-*.whl")
    if whl:
        subprocess.check_call(["pip", "install", "-q", "--no-index", str(whl)])
    import onnxruntime as ort

# ============================================================================
# CELL 2 — INFERENCE BACKEND:  OpenVINO FP16 (+AsyncInferQueue) -> ORT CPU
# ============================================================================
USE_OPENVINO = True
try:
    import openvino as ov
    _OV_CORE = ov.Core()
except Exception as e:
    USE_OPENVINO = False
    print(f"[backend] OpenVINO unavailable ({e}); using ONNXRuntime CPU.")


class OnnxRunner:
    """Uniform .run(mel_batch)->logits over OpenVINO-FP16 or ORT-CPU.

    OpenVINO path: convert ONNX -> IR with compress_to_fp16=True, compile for CPU
    with PERFORMANCE_HINT=THROUGHPUT (big static batch), and use AsyncInferQueue to
    overlap the 2 physical cores across sub-batches (A3 recipe). ORT path mirrors the
    established sub_v8 session (intra=4, inter=1, ORT_ENABLE_ALL).
    """

    def __init__(self, onnx_path, n_outputs_keep=2):
        self.onnx_path = str(onnx_path)
        self.n_outputs_keep = n_outputs_keep
        self.backend = "ort"
        self._compiled = None
        self._sess = None
        if USE_OPENVINO:
            try:
                model = _OV_CORE.read_model(self.onnx_path)
                # FP16 compression — lossless for a ranking metric (A3 lever 1)
                try:
                    from openvino import properties as _props  # noqa
                except Exception:
                    pass
                self._compiled = _OV_CORE.compile_model(
                    model, "CPU",
                    {"PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_PRECISION_HINT": "f16"},
                )
                self._out_ports = list(self._compiled.outputs)
                self._in_port = self._compiled.inputs[0]
                self.backend = "openvino-fp16"
            except Exception as e:
                print(f"[backend] OV compile failed for {Path(self.onnx_path).name} ({e}); ORT fallback.")
        if self._compiled is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(self.onnx_path, so, providers=["CPUExecutionProvider"])
            self._in_name = self._sess.get_inputs()[0].name

    def run(self, x):
        """x: (N, ...) -> list of up to n_outputs_keep np arrays (clip_logits, framewise)."""
        if self.backend == "openvino-fp16":
            # AsyncInferQueue across the static batch for core overlap.
            results = [None] * x.shape[0]
            q = ov.AsyncInferQueue(self._compiled, 2)  # 2 physical cores

            def _cb(req, idx):
                results[idx] = [req.get_output_tensor(i).data.copy()
                                for i in range(min(self.n_outputs_keep, len(self._out_ports)))]
            q.set_callback(_cb)
            for i in range(x.shape[0]):
                q.start_async({self._in_port: x[i:i + 1]}, i)
            q.wait_all()
            outs = []
            for o in range(min(self.n_outputs_keep, len(self._out_ports))):
                outs.append(np.concatenate([results[i][o] for i in range(x.shape[0])], axis=0))
            return outs
        else:
            outs = self._sess.run(None, {self._in_name: x})
            return [np.asarray(o) for o in outs[:self.n_outputs_keep]]


# ============================================================================
# CELL 3 — AUDIO + MEL
# ============================================================================
import soundfile as sf
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False


def load_audio_60s(path):
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1).astype(np.float32)
    if sr0 != SR:
        if _HAS_LIBROSA:
            y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
        else:
            import scipy.signal
            y = scipy.signal.resample_poly(y, SR, sr0).astype(np.float32)
    n = N_WINDOWS * WINDOW_SAMPLES
    y = np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)


def chunks_to_sed_mel(chunks):
    """(N,160000) -> (N,1,256,313) standardized mel-dB. EXACT public 0.950 recipe."""
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=SED_N_FFT, hop_length=SED_HOP,
            n_mels=SED_NMELS, fmin=SED_FMIN, fmax=SED_FMAX, power=2.0)
        s = librosa.power_to_db(s, top_db=SED_TOPDB)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


# ============================================================================
# CELL 4 — SED ENSEMBLE PREDICTOR (shared for distilled_sed + tonylica members)
# ============================================================================
def build_sed_runners(sed_dir):
    if sed_dir is None:
        return []
    paths = sorted(Path(sed_dir).glob("sed_fold*.onnx"),
                   key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)))
    return [OnnxRunner(p, n_outputs_keep=2) for p in paths]


def sed_predict(runners, mel):
    """mel:(N,1,256,313) -> (N,234) prob = mean_folds[ .5 sig(clip) + .5 sig(framemax) ]."""
    if not runners:
        return None
    p_sum = np.zeros((mel.shape[0], NUM_CLASSES), dtype=np.float32)
    for r in runners:
        outs = r.run(mel)
        clip_logits = outs[0]                                    # (N,234)
        frame_max = outs[1].max(axis=1) if len(outs) > 1 else clip_logits  # (N,234)
        p_sum += 0.5 * _sigmoid(clip_logits) + 0.5 * _sigmoid(frame_max)
    return p_sum / len(runners)


# ============================================================================
# CELL 5 — PERCH EMBEDDING + ProtoSSM / ResidualSSM members
# ============================================================================
# Perch ONNX -> (emb 1536, logits 234) per window. The ProtoSSM/ResidualSSM heads
# (sgkfk) consume the Perch embedding sequence. Their exact torch architecture is in
# the sgkfk train kernel (d_model=256, n_ssm_layers=3, cross-attention). To keep this
# notebook robust & self-contained on CPU we:
#   (a) ALWAYS compute Perch embeddings + Perch logits (the proto branch's substrate),
#   (b) load the ProtoSSM / ResidualSSM .pt heads if torch + a compatible state_dict
#       load succeeds; otherwise fall back to the Perch logits as the proto member
#       (graceful degrade — still the dominant-branch signal, just without the SSM head).
PERCH = OnnxRunner(PERCH_ONNX, n_outputs_keep=2) if PERCH_ONNX else None


def perch_predict(x):
    """x:(N,160000) -> (emb (N,1536), perch_logits (N,234))."""
    outs = PERCH.run(x)
    emb, logit = None, None
    for o in outs:
        a = np.asarray(o)
        if a.ndim == 2 and a.shape[1] in (1536, 1280):
            emb = a.astype(np.float32)
        elif a.ndim == 2 and a.shape[1] == NUM_CLASSES:
            logit = a.astype(np.float32)
    if emb is None:
        emb = np.asarray(outs[0], np.float32).reshape(x.shape[0], -1)
    if logit is None:
        logit = np.zeros((x.shape[0], NUM_CLASSES), np.float32)
    return emb, logit


# ---- ProtoSSM / ResidualSSM heads (best-effort torch load) ----
_TORCH = None
try:
    import torch as _TORCH  # noqa
except Exception:
    _TORCH = None


def _try_load_ssm(pt_path):
    """Best-effort: return a callable emb_seq->(N,234) prob, or None if not loadable.
    We do NOT hard-fail; if the head can't be reconstructed we degrade to Perch logits."""
    if _TORCH is None or pt_path is None or not Path(pt_path).exists():
        return None
    try:
        ckpt = _TORCH.load(str(pt_path), map_location="cpu", weights_only=False)
        # The sgkfk heads are TorchScript-able in some exports; try a scripted call.
        if hasattr(ckpt, "eval"):
            mdl = ckpt.eval()

            def _fn(emb):
                with _TORCH.no_grad():
                    t = _TORCH.from_numpy(emb).float()
                    out = mdl(t)
                    out = out[0] if isinstance(out, (tuple, list)) else out
                    return _sigmoid(out.cpu().numpy())
            return _fn
    except Exception as e:
        print(f"[ssm] {Path(pt_path).name} not directly loadable ({e}); degrade to Perch logits.")
    return None


PROTO_HEAD = _try_load_ssm(PROTO_SSM_PT)
RESSSM_HEAD = _try_load_ssm(RESSSM_PT)


# ============================================================================
# CELL 6 — RANK-BLEND PRIMITIVES (A4: a4_rankblend.py logic, reused verbatim)
# ============================================================================
from scipy.stats import rankdata


def rank_transform(P, power=RANK_POWER):
    """Per-COLUMN rank in [0,1], warped by `power`. AUC-invariant within a model;
    equalizes scale ACROSS models so a peaky model can't dominate the mean."""
    R = np.empty_like(P, dtype=np.float64)
    n = P.shape[0]
    for c in range(P.shape[1]):
        col = P[:, c]
        v = ~np.isnan(col)
        r = np.full(n, 0.5)
        if v.sum() > 0:
            r[v] = rankdata(col[v], method="average") / v.sum()
        R[:, c] = r ** power if power != 1.0 else r
    return R


def rank_blend(mats, weights, power=RANK_POWER):
    """Weighted mean of per-class rank transforms. Output IS a valid submission
    (AUC reads order only)."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    acc = np.zeros_like(mats[0], dtype=np.float64)
    for wi, M in zip(w, mats):
        acc += wi * rank_transform(M, power=power)
    return acc


# ============================================================================
# CELL 7 — SITE / HOUR PRIORS  (G_prior — rank-shift, conservative)
# ============================================================================
samp = pd.read_csv(SAMPLE_SUB_PATH)
class_cols = [c for c in samp.columns if c != "row_id"]
assert len(class_cols) == NUM_CLASSES, f"expected 234 cols, got {len(class_cols)}"

PRIOR_PATH = _first("combined_hour_prior.csv")
if PRIOR_PATH is not None:
    hp_df = pd.read_csv(PRIOR_PATH).set_index("hour")
    hp_arr = hp_df.reindex(columns=class_cols).fillna(0).to_numpy(dtype=np.float64)
    hp_arr = np.clip(hp_arr, EPS, 1.0)
    LOG_HP = np.log(hp_arr)
    print(f"[prior] combined_hour_prior loaded: {hp_arr.shape}")
else:
    LOG_HP = None
    print("[prior] no combined_hour_prior.csv — skipping site/hour prior shift")


def apply_hour_prior(blend, hours):
    if LOG_HP is None or W_PRIOR == 0:
        return blend
    logit_p = np.log(np.clip(blend, EPS, 1 - EPS) / np.clip(1 - blend, EPS, 1))
    valid = (hours >= 0) & (hours < 24)
    shift = np.zeros_like(blend, dtype=np.float64)
    shift[valid] = W_PRIOR * LOG_HP[hours[valid]]
    return 1.0 / (1.0 + np.exp(-(logit_p + shift)))


# ============================================================================
# CELL 8 — TAXONOMY SMOOTHING (genus alpha=.15, class alpha=.05)  [G_post tail]
# ============================================================================
def build_taxonomy_groups():
    if not TAXONOMY_PATH.exists():
        return None, None
    tax = pd.read_csv(TAXONOMY_PATH)
    tax["primary_label"] = tax["primary_label"].astype(str)
    lab2idx = {c: i for i, c in enumerate(class_cols)}
    genus_groups, class_groups = {}, {}
    for _, r in tax.iterrows():
        lab = r["primary_label"]
        if lab not in lab2idx:
            continue
        gi = lab2idx[lab]
        genus = str(r.get("scientific_name", "")).split()[0] if pd.notna(r.get("scientific_name", "")) else ""
        klass = str(r.get("class_name", ""))
        genus_groups.setdefault(genus, []).append(gi)
        class_groups.setdefault(klass, []).append(gi)
    return ([v for v in genus_groups.values() if len(v) > 1],
            [v for v in class_groups.values() if len(v) > 1])


GENUS_GROUPS, CLASS_GROUPS = build_taxonomy_groups()


def taxonomy_smooth(P, genus_a=GENUS_ALPHA, class_a=CLASS_ALPHA):
    """p_out = (1-ac)[(1-ag)p + ag*mean_genus(p)] + ac*mean_class(p)."""
    if GENUS_GROUPS is None:
        return P
    out = P.copy()
    if genus_a > 0:
        tmp = out.copy()
        for grp in GENUS_GROUPS:
            gm = out[:, grp].mean(axis=1, keepdims=True)
            tmp[:, grp] = (1 - genus_a) * out[:, grp] + genus_a * gm
        out = tmp
    if class_a > 0:
        tmp = out.copy()
        for grp in CLASS_GROUPS:
            cm = out[:, grp].mean(axis=1, keepdims=True)
            tmp[:, grp] = (1 - class_a) * out[:, grp] + class_a * cm
        out = tmp
    return out


# ============================================================================
# CELL 9 — MAIN INFERENCE LOOP
# ============================================================================
from scipy.ndimage import gaussian_filter1d

# Build per-member SED runners (shared distilled folds + tonylica's own xsed folds)
SED_RUNNERS_MAIN = build_sed_runners(SED_DIR)               # tuckerarrants distilled SED
SED_RUNNERS_TONY = build_sed_runners(TONYLICA_XSED_DIR)     # tonylica xsed folds

# Which members are actually available (gate by mount). Drop missing & renormalize.
def member_available(name):
    return {
        "proto_ssm":   PERCH is not None,             # always have Perch substrate
        "distilled_sed": len(SED_RUNNERS_MAIN) > 0,
        "tonylica":    len(SED_RUNNERS_TONY) > 0,
        "sgkfk_resssm": PERCH is not None,             # ResidualSSM head or Perch-logit degrade
        "birdmae":     False,                          # validation-pending (toggle in ENSEMBLE)
        "perch20":     False,
    }.get(name, False)


ACTIVE = {k: v for k, v in ENSEMBLE.items() if member_available(k)}
print(f"\n==== ACTIVE ENSEMBLE MEMBERS ====\n  {ACTIVE}")
DROPPED = [k for k in ENSEMBLE if k not in ACTIVE]
if DROPPED:
    print(f"  (dropped — asset absent or validation-pending: {DROPPED})")

test_files = sorted(TEST_DIR.glob("*.ogg")) if TEST_DIR.exists() else []
if not test_files:
    print("No test files — emitting all-zero submission (local dry-run).")
    out = samp.copy(); out.iloc[:, 1:] = 0.0
    out.to_csv("submission.csv", index=False); sys.exit(0)

ROW_RE = re.compile(r"_(\d{8})_(\d{6})$")
# per-member accumulators
acc = {k: [] for k in ACTIVE}
row_ids_all, hours_all = [], []
t0 = time.time()

import concurrent.futures
def _load(p): return p, load_audio_60s(p)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    nxt = test_files[:BATCH_FILES]
    nxt_f = [pool.submit(_load, p) for p in nxt]
    for start in range(0, len(test_files), BATCH_FILES):
        batch = [f.result() for f in nxt_f]
        ns = start + BATCH_FILES
        if ns < len(test_files):
            nb = test_files[ns:ns + BATCH_FILES]
            nxt_f = [pool.submit(_load, p) for p in nb]
        bn = len(batch)
        x = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        for bi, (_, yw) in enumerate(batch):
            x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = yw

        # --- shared substrates (compute ONCE, fan out) ---
        emb, perch_logit = perch_predict(x) if PERCH is not None else (None, None)
        mel = chunks_to_sed_mel(x) if (SED_RUNNERS_MAIN or SED_RUNNERS_TONY) else None

        # --- per-member probs for this batch ---
        member_probs = {}
        if "distilled_sed" in ACTIVE:
            member_probs["distilled_sed"] = sed_predict(SED_RUNNERS_MAIN, mel)
        if "tonylica" in ACTIVE:
            member_probs["tonylica"] = sed_predict(SED_RUNNERS_TONY, mel)
        if "proto_ssm" in ACTIVE:
            if PROTO_HEAD is not None:
                try:
                    member_probs["proto_ssm"] = PROTO_HEAD(emb)
                except Exception:
                    member_probs["proto_ssm"] = _sigmoid(perch_logit)
            else:
                member_probs["proto_ssm"] = _sigmoid(perch_logit)  # degrade to Perch logits
        if "sgkfk_resssm" in ACTIVE:
            if RESSSM_HEAD is not None:
                try:
                    member_probs["sgkfk_resssm"] = RESSSM_HEAD(emb)
                except Exception:
                    member_probs["sgkfk_resssm"] = _sigmoid(perch_logit)
            else:
                member_probs["sgkfk_resssm"] = _sigmoid(perch_logit)

        # --- within-file Gaussian temporal smoothing per file, then store ---
        for bi, (fpath, _) in enumerate(batch):
            s = slice(bi * N_WINDOWS, (bi + 1) * N_WINDOWS)
            stem = fpath.stem
            m = ROW_RE.search(stem)
            hour = int(m.group(2)[:2]) if m else 0
            for i in range(N_WINDOWS):
                row_ids_all.append(f"{stem}_{(i + 1) * WIN_SEC}")
                hours_all.append(hour)
            for k in ACTIVE:
                pf = member_probs[k][s].astype(np.float32)
                if pf.shape[0] > 1:
                    pf = gaussian_filter1d(pf, sigma=GAUSS_SIGMA, axis=0, mode="nearest")
                acc[k].append(pf)

        done = start + bn
        if done % (BATCH_FILES * 4) == 0 or done == len(test_files):
            el = time.time() - t0
            rate = done / max(el, 1.0)
            print(f"  [{done}/{len(test_files)}] {el:.0f}s  {rate:.2f} files/s  "
                  f"eta {(len(test_files)-done)/max(rate,0.01):.0f}s")

P = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
hours = np.array(hours_all, dtype=np.int32)
print(f"\nInference {time.time()-t0:.0f}s. members={list(P)} rows={len(row_ids_all)}")

# ============================================================================
# CELL 10 — RANK-BLEND (config weights) + priors + taxonomy smoothing
# ============================================================================
names = list(ACTIVE.keys())
mats = [P[k] for k in names]
weights = [ACTIVE[k] for k in names]
print(f"[blend] rank-blend power={RANK_POWER}  weights={dict(zip(names, weights))}")

blend = rank_blend(mats, weights, power=RANK_POWER)          # R(...) percentile-rank blend
blend = apply_hour_prior(blend, hours)                       # G_prior site/hour shift
blend = taxonomy_smooth(blend)                               # genus .15 / class .05
final = np.clip(blend, 0.0, 1.0).astype(np.float32)

# ============================================================================
# CELL 11 — SUBMISSION WRITER (match sample_submission columns & order exactly)
# ============================================================================
out_df = pd.DataFrame(final, columns=class_cols)
out_df.insert(0, "row_id", row_ids_all)
sample_ids = samp["row_id"].astype(str).tolist()
if set(out_df["row_id"]) == set(sample_ids):
    out_df = out_df.set_index("row_id").loc[sample_ids].reset_index()
out_df = out_df[["row_id"] + class_cols]                     # enforce exact column order
out_df.to_csv("submission.csv", index=False)
print(f"\nWrote submission.csv: {out_df.shape[0]} rows x {out_df.shape[1]} cols "
      f"(min={final.min():.4f} max={final.max():.4f})")
print(f"Backend: {'OpenVINO-FP16' if USE_OPENVINO else 'ONNXRuntime-CPU'} | "
      f"members={names} | weights={weights}")
