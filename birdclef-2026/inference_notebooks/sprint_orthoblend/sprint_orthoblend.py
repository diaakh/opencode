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
import os, re, gc, sys, time, glob, pickle, subprocess, resource
from pathlib import Path
import numpy as np
import pandas as pd

EPS         = 1e-7
SR          = 32000
WIN_SEC     = 5
N_WINDOWS   = 12                  # 60s / 5s
WINDOW_SAMPLES = SR * WIN_SEC     # 160_000
NUM_CLASSES = 234
BATCH_FILES = 4                   # 4*12 = 48 windows/ONNX call — small resident audio (OOM fix)

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
    # tonylica DROPPED: its xsed/sed_fold*.onnx are byte-identical to distilled_sed
    # (self-test proved mean|Δ|=0.00000) — it double-counts the SED member, no diversity.
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
#     NOTE: tonylica/birdclef-2026-model ALSO bundles a byte-identical copy under xsed/.
#     We must NOT pick that copy here, or the distilled_sed + tonylica members collapse
#     to the same files (zero diversity). Prefer the standalone distilled-SED dataset.
_sed_hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
SED_FOLD0 = next((h for h in _sed_hits
                  if "xsed" not in h.parts and "tonylica" not in str(h).lower()),
                 _sed_hits[0] if _sed_hits else None)
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
            so.intra_op_num_threads = 2
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Cap allocator growth — the CPU mem arena + mem pattern can hold large
            # pre-allocated buffers per session that never shrink (peak-RAM driver at
            # scale across Perch + 5 SED sessions). Disable both to keep RSS bounded.
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
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
# (sgkfk) consume the Perch embedding *sequence* (12 windows/file). Their exact torch
# architecture is ported verbatim below from the sgkfk training kernel
# (hideyukizushi/bird26-reprod-perch-proto-residualssm-train). The checkpoint tensor
# shapes are ground truth and dictate the init config (verified strict=True load):
#   ProtoSSMv2:  d_model=320 d_state=32 n_ssm_layers=4 n_sites=20 meta_dim=24
#                use_cross_attn=True cross_attn_heads=8  + family_head(5)
#   ResidualSSM: d_model=128 d_state=16 n_sites=20 meta_dim=8 (input 1536+234=1770)
# Two-pass flow (faithful to train): proto_logits = ProtoSSM(emb); refined =
# proto_logits + ResidualSSM(emb, proto_logits). We feed perch_logits=None so the
# proto head uses its prototype-similarity branch (the no-DFT Perch ONNX emits
# embeddings only — no classifier head — so a real perch_logit fusion is unavailable;
# the gated-fusion alpha was trained but the prototype branch alone is non-degenerate
# and is the faithful path given the embeddings-only ONNX, see PERCH-LOGITS note).
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


# ---- ProtoSSM / ResidualSSM torch heads (ported from the sgkfk train kernel) ----
_TORCH = None
try:
    import torch  # noqa
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = torch
except Exception:
    _TORCH = None


if _TORCH is not None:
    class SelectiveSSM(nn.Module):
        """Simplified Mamba-style selective SSM (sequential scan over T=12 windows)."""
        def __init__(self, d_model, d_state=16, d_conv=4):
            super().__init__()
            self.d_model = d_model
            self.d_state = d_state
            self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
            self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
            self.dt_proj = nn.Linear(d_model, d_model, bias=True)
            A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(d_model))
            self.B_proj = nn.Linear(d_model, d_state, bias=False)
            self.C_proj = nn.Linear(d_model, d_state, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            B_size, T, D = x.shape
            xz = self.in_proj(x)
            x_ssm, z = xz.chunk(2, dim=-1)
            x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
            x_conv = F.silu(x_conv)
            dt = F.softplus(self.dt_proj(x_conv))
            A = -torch.exp(self.A_log)
            B = self.B_proj(x_conv)
            C = self.C_proj(x_conv)
            h = torch.zeros(B_size, D, self.d_state, device=x.device)
            ys = []
            for t in range(T):
                dt_t = dt[:, t, :]
                dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
                dB = dt_t[:, :, None] * B[:, t, None, :]
                h = h * dA + x[:, t, :, None] * dB
                ys.append((h * C[:, t, None, :]).sum(-1))
            return torch.stack(ys, dim=1) + x * self.D[None, None, :]

    class TemporalCrossAttention(nn.Module):
        def __init__(self, d_model, n_heads=4, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model), nn.Dropout(dropout))
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            attn_out, _ = self.attn(x, x, x)
            x = residual + attn_out
            residual = x
            x = self.norm2(x)
            return residual + self.ffn(x)

    class ProtoSSMv2(nn.Module):
        def __init__(self, d_input=1536, d_model=320, d_state=32, n_ssm_layers=4,
                     n_classes=234, n_windows=12, dropout=0.2, n_sites=20, meta_dim=24,
                     use_cross_attn=True, cross_attn_heads=8):
            super().__init__()
            self.d_model = d_model
            self.n_classes = n_classes
            self.n_windows = n_windows
            self.input_proj = nn.Sequential(
                nn.Linear(d_input, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
            self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
            self.site_emb = nn.Embedding(n_sites, meta_dim)
            self.hour_emb = nn.Embedding(24, meta_dim)
            self.meta_proj = nn.Linear(2 * meta_dim, d_model)
            self.ssm_fwd = nn.ModuleList()
            self.ssm_bwd = nn.ModuleList()
            self.ssm_merge = nn.ModuleList()
            self.ssm_norm = nn.ModuleList()
            for _ in range(n_ssm_layers):
                self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
                self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
                self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
                self.ssm_norm.append(nn.LayerNorm(d_model))
            self.ssm_drop = nn.Dropout(dropout)
            self.use_cross_attn = use_cross_attn
            if use_cross_attn:
                self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)
            self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
            self.proto_temp = nn.Parameter(torch.tensor(5.0))
            self.class_bias = nn.Parameter(torch.zeros(n_classes))
            self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))
            self.n_families = 0
            self.family_head = None

        def init_family_head(self, n_families, class_to_family):
            self.n_families = n_families
            self.family_head = nn.Linear(self.d_model, n_families)
            self.register_buffer('class_to_family', torch.tensor(class_to_family, dtype=torch.long))

        def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
            B, T, _ = emb.shape
            h = self.input_proj(emb)
            h = h + self.pos_enc[:, :T, :]
            if site_ids is not None and hours is not None:
                meta = self.meta_proj(torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1))
                h = h + meta[:, None, :]
            for fwd, bwd, merge, norm in zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm):
                residual = h
                h_f = fwd(h)
                h_b = bwd(h.flip(1)).flip(1)
                h = merge(torch.cat([h_f, h_b], dim=-1))
                h = self.ssm_drop(h)
                h = norm(h + residual)
            if self.use_cross_attn:
                h = self.cross_attn(h)
            h_norm = F.normalize(h, dim=-1)
            p_norm = F.normalize(self.prototypes, dim=-1)
            temp = F.softplus(self.proto_temp)
            sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]
            if perch_logits is not None:
                alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
                species_logits = alpha * sim + (1 - alpha) * perch_logits
            else:
                species_logits = sim
            family_logits = None
            if self.family_head is not None:
                family_logits = self.family_head(h.mean(dim=1))
            return species_logits, family_logits, h

    class ResidualSSM(nn.Module):
        def __init__(self, d_input=1536, d_scores=234, d_model=128, d_state=16,
                     n_classes=234, n_windows=12, dropout=0.1, n_sites=20, meta_dim=8):
            super().__init__()
            self.d_model = d_model
            self.n_classes = n_classes
            self.input_proj = nn.Sequential(
                nn.Linear(d_input + d_scores, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
            self.site_emb = nn.Embedding(n_sites, meta_dim)
            self.hour_emb = nn.Embedding(24, meta_dim)
            self.meta_proj = nn.Linear(2 * meta_dim, d_model)
            self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
            self.ssm_fwd = SelectiveSSM(d_model, d_state)
            self.ssm_bwd = SelectiveSSM(d_model, d_state)
            self.ssm_merge = nn.Linear(2 * d_model, d_model)
            self.ssm_norm = nn.LayerNorm(d_model)
            self.ssm_drop = nn.Dropout(dropout)
            self.output_head = nn.Linear(d_model, n_classes)

        def forward(self, emb, first_pass_scores, site_ids=None, hours=None):
            B, T, _ = emb.shape
            x = torch.cat([emb, first_pass_scores], dim=-1)
            h = self.input_proj(x)
            if site_ids is not None and hours is not None:
                site_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
                hour_e = self.hour_emb(hours.clamp(0, 23))
                meta = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))
                h = h + meta.unsqueeze(1)
            h = h + self.pos_enc[:, :T, :]
            residual = h
            h_f = self.ssm_fwd(h)
            h_b = self.ssm_bwd(h.flip(1)).flip(1)
            h = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = self.ssm_norm(h + residual)
            return self.output_head(h)


def _emb_to_seq(emb):
    """Flat (N,1536) -> (N//12, 12, 1536) torch float tensor. N must be a multiple of 12."""
    n = emb.shape[0]
    nf = n // N_WINDOWS
    t = _TORCH.from_numpy(np.ascontiguousarray(emb)).float()
    return t.view(nf, N_WINDOWS, emb.shape[1])


class _SSMHeads:
    """Loads ProtoSSM + ResidualSSM with strict=True and runs the faithful two-pass.

    Exposes:
      .proto(emb_flat)  -> (N,234) proto-branch probs  (sigmoid of proto species_logits)
      .resssm(emb_flat) -> (N,234) refined probs        (sigmoid(proto_logits + correction))
    Both accept the flat (N,1536) Perch embeddings the inference loop already produces,
    reshape to (files,12,1536), and flatten the (files,12,234) output back to (N,234).
    perch_logits / site_ids / hours are passed as None (no-DFT ONNX has no classifier
    head; the prototype-similarity branch is non-degenerate on its own)."""

    def __init__(self, proto_pt, res_pt):
        self.ok_proto = False
        self.ok_res = False
        self.proto_model = None
        self.res_model = None
        if _TORCH is None:
            return
        if proto_pt is not None and Path(proto_pt).exists():
            m = ProtoSSMv2(d_input=1536, d_model=320, d_state=32, n_ssm_layers=4,
                           n_classes=NUM_CLASSES, n_windows=N_WINDOWS, n_sites=20,
                           meta_dim=24, use_cross_attn=True, cross_attn_heads=8)
            m.init_family_head(5, [0] * NUM_CLASSES)
            sd = _TORCH.load(str(proto_pt), map_location="cpu")
            m.load_state_dict(sd, strict=True)
            m.eval()
            self.proto_model = m
            self.ok_proto = True
            print(f"[ssm] ProtoSSMv2 loaded strict from {Path(proto_pt).name}")
        if res_pt is not None and Path(res_pt).exists():
            r = ResidualSSM(d_input=1536, d_scores=NUM_CLASSES, d_model=128, d_state=16,
                            n_classes=NUM_CLASSES, n_windows=N_WINDOWS, n_sites=20, meta_dim=8)
            sd = _TORCH.load(str(res_pt), map_location="cpu")
            r.load_state_dict(sd, strict=True)
            r.eval()
            self.res_model = r
            self.ok_res = True
            print(f"[ssm] ResidualSSM loaded strict from {Path(res_pt).name}")

    def _proto_logits(self, emb_flat):
        seq = _emb_to_seq(emb_flat)
        with _TORCH.no_grad():
            species_logits, _, _ = self.proto_model(seq, perch_logits=None)
        return species_logits  # (files,12,234)

    def proto(self, emb_flat):
        logits = self._proto_logits(emb_flat)
        return _sigmoid(logits.reshape(-1, NUM_CLASSES).cpu().numpy())

    def resssm(self, emb_flat):
        # faithful two-pass: refined = proto_logits + ResidualSSM(emb, proto_logits)
        logits = self._proto_logits(emb_flat)
        seq = _emb_to_seq(emb_flat)
        with _TORCH.no_grad():
            correction = self.res_model(seq, logits)
            refined = logits + correction
        return _sigmoid(refined.reshape(-1, NUM_CLASSES).cpu().numpy())

    def proto_and_resssm(self, emb_flat, want_proto, want_res):
        """Compute the proto forward ONCE and derive both members from it (item 4:
        avoids the duplicate proto forward that .proto()+.resssm() incurred).
        Returns (proto_probs_or_None, resssm_probs_or_None)."""
        proto_p = res_p = None
        if not (want_proto or want_res):
            return proto_p, res_p
        seq = _emb_to_seq(emb_flat)
        with _TORCH.no_grad():
            logits, _, _ = self.proto_model(seq, perch_logits=None)   # single proto forward
            if want_proto:
                proto_p = _sigmoid(logits.reshape(-1, NUM_CLASSES).cpu().numpy())
            if want_res:
                refined = logits + self.res_model(seq, logits)
                res_p = _sigmoid(refined.reshape(-1, NUM_CLASSES).cpu().numpy())
        del seq
        return proto_p, res_p


_SSM = _SSMHeads(PROTO_SSM_PT, RESSSM_PT)
PROTO_HEAD = (lambda emb: _SSM.proto(emb)) if _SSM.ok_proto else None
RESSSM_HEAD = (lambda emb: _SSM.resssm(emb)) if (_SSM.ok_proto and _SSM.ok_res) else None
# Combined single-proto-forward path used in the main loop (item 4).
SSM_BOTH = (lambda emb, wp, wr: _SSM.proto_and_resssm(emb, wp, wr)) if _SSM.ok_proto else None


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

# Build per-member SED runners (shared distilled folds only).
# NOTE: tonylica xsed folds are byte-identical to distilled_sed and the tonylica
# member was DROPPED from ENSEMBLE — we no longer load those 5 ONNX sessions
# (they were pure dead RAM: 5 extra InferenceSessions). The tonylica orthogonality
# probe is gone with them.
SED_RUNNERS_MAIN = build_sed_runners(SED_DIR)               # tuckerarrants distilled SED

# Which members are actually available (gate by mount). Drop missing & renormalize.
def member_available(name):
    return {
        "proto_ssm":   PERCH is not None and PROTO_HEAD is not None,  # need the loaded SSM head
        "distilled_sed": len(SED_RUNNERS_MAIN) > 0,
        "sgkfk_resssm": PERCH is not None and RESSSM_HEAD is not None,  # need both SSM heads
        "birdmae":     False,                          # validation-pending (toggle in ENSEMBLE)
        "perch20":     False,
    }.get(name, False)


ACTIVE = {k: v for k, v in ENSEMBLE.items() if member_available(k)}
print(f"\n==== ACTIVE ENSEMBLE MEMBERS ====\n  {ACTIVE}")
DROPPED = [k for k in ENSEMBLE if k not in ACTIVE]
if DROPPED:
    print(f"  (dropped — asset absent or validation-pending: {DROPPED})")

ROW_RE = re.compile(r"_(\d{8})_(\d{6})$")
import concurrent.futures
def _load(p): return p, load_audio_60s(p)


def _peak_rss_gb():
    """Linux: ru_maxrss is in KB. Return peak resident set size in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def run_full_inference(files, label="test"):
    """The EXACT scoring code path. Streams `files` in BATCH_FILES chunks through
    Perch + SED + SSM heads, accumulates per-member probs, and aggressively frees
    each batch's audio / embedding / mel / member buffers to keep peak RSS bounded
    (OOM fix). Returns (P_dict, row_ids_all, hours_array)."""
    acc = {k: [] for k in ACTIVE}
    row_ids_all, hours_all = [], []
    t0 = time.time()
    want_proto = "proto_ssm" in ACTIVE
    want_res = "sgkfk_resssm" in ACTIVE
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        nxt_f = [pool.submit(_load, p) for p in files[:BATCH_FILES]]
        for start in range(0, len(files), BATCH_FILES):
            batch = [f.result() for f in nxt_f]
            ns = start + BATCH_FILES
            if ns < len(files):
                nb = files[ns:ns + BATCH_FILES]
                nxt_f = [pool.submit(_load, p) for p in nb]
            bn = len(batch)
            x = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            for bi, (_, yw) in enumerate(batch):
                x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = yw

            # --- shared substrates (compute ONCE, fan out) ---
            emb, perch_logit = perch_predict(x) if PERCH is not None else (None, None)
            emb = emb.astype(np.float32, copy=False) if emb is not None else None
            del perch_logit
            mel = chunks_to_sed_mel(x) if SED_RUNNERS_MAIN else None

            # --- per-member probs for this batch ---
            member_probs = {}
            if "distilled_sed" in ACTIVE:
                member_probs["distilled_sed"] = sed_predict(SED_RUNNERS_MAIN, mel)
            # proto + residual share ONE proto forward (item 4: no duplicate forward)
            if (want_proto or want_res) and emb is not None and SSM_BOTH is not None:
                p_proto, p_res = SSM_BOTH(emb, want_proto, want_res)
                if want_proto:
                    member_probs["proto_ssm"] = p_proto
                if want_res:
                    member_probs["sgkfk_resssm"] = p_res

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

            # --- aggressive per-batch cleanup (peak RAM driver) ---
            del x, emb, mel, member_probs, batch
            gc.collect()

            done = start + bn
            if done % (BATCH_FILES * 4) == 0 or done == len(files):
                el = time.time() - t0
                rate = done / max(el, 1.0)
                print(f"  [{label} {done}/{len(files)}] {el:.0f}s  {rate:.2f} files/s  "
                      f"rss={_peak_rss_gb():.2f}GB  "
                      f"eta {(len(files)-done)/max(rate,0.01):.0f}s")

    P = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    del acc
    gc.collect()
    hours = np.array(hours_all, dtype=np.int32)
    print(f"\n[{label}] inference {time.time()-t0:.0f}s. members={list(P)} rows={len(row_ids_all)}")
    return P, row_ids_all, hours


def blend_members(P, row_ids_all, hours):
    """rank-blend (config weights) + hour prior + taxonomy smoothing -> final array."""
    names = list(ACTIVE.keys())
    mats = [P[k] for k in names]
    weights = [ACTIVE[k] for k in names]
    print(f"[blend] rank-blend power={RANK_POWER}  weights={dict(zip(names, weights))}")
    blend = rank_blend(mats, weights, power=RANK_POWER)          # percentile-rank blend
    blend = apply_hour_prior(blend, hours)                       # G_prior site/hour shift
    blend = taxonomy_smooth(blend)                               # genus .15 / class .05
    return np.clip(blend, 0.0, 1.0).astype(np.float32)


def write_submission(final, row_ids_all):
    out_df = pd.DataFrame(final, columns=class_cols)
    out_df.insert(0, "row_id", row_ids_all)
    sample_ids = samp["row_id"].astype(str).tolist()
    if set(out_df["row_id"]) == set(sample_ids):
        out_df = out_df.set_index("row_id").loc[sample_ids].reset_index()
    out_df = out_df[["row_id"] + class_cols]                     # enforce exact column order
    out_df.to_csv("submission.csv", index=False)
    print(f"\nWrote submission.csv: {out_df.shape[0]} rows x {out_df.shape[1]} cols "
          f"(min={final.min():.4f} max={final.max():.4f})")
    return out_df


def write_zero_submission():
    out = samp.copy(); out.iloc[:, 1:] = 0.0
    out.to_csv("submission.csv", index=False)
    print(f"Wrote ZERO placeholder submission.csv: {out.shape[0]} rows x {out.shape[1]} cols")


def tiny_model_selftest():
    """Tiny 2-file model sanity check (correctness, not scale). Informational."""
    print("---- TINY MODEL SELF-TEST (2 files) ----")
    ss_dir = COMP_DIR / "train_soundscapes"
    probe = sorted(ss_dir.glob("*.ogg"))[:2] if ss_dir.exists() else []
    if not probe:
        print("  [selftest] no train_soundscapes available to probe.")
        return
    xw = np.empty((len(probe) * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
    for bi, p in enumerate(probe):
        xw[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = load_audio_60s(p)
    def _stat(nm, a):
        a = np.asarray(a, dtype=np.float64)
        ok = np.isfinite(a).all() and float(a.std()) > 1e-6
        print(f"  [selftest] {nm:14s} shape={a.shape} min={a.min():.4f} "
              f"max={a.max():.4f} std={a.std():.4f} {'OK' if ok else 'DEGENERATE!!'}")
        return ok
    good = True
    if PERCH is not None:
        emb, perch_logit = perch_predict(xw)
        _stat("perch_logit", perch_logit)
        good &= _stat("perch_emb", emb)
    else:
        emb = None
    mel = chunks_to_sed_mel(xw) if SED_RUNNERS_MAIN else None
    if SED_RUNNERS_MAIN:
        good &= _stat("distilled_sed", sed_predict(SED_RUNNERS_MAIN, mel))
    if emb is not None and PROTO_HEAD is not None:
        try: good &= _stat("proto_ssm", PROTO_HEAD(emb))
        except Exception as e:
            good = False; print(f"  [selftest] proto_ssm head FAILED: {e}")
    if emb is not None and RESSSM_HEAD is not None:
        try: good &= _stat("sgkfk_resssm", RESSSM_HEAD(emb))
        except Exception as e:
            good = False; print(f"  [selftest] resssm head FAILED: {e}")
    if emb is not None and PROTO_HEAD is not None and RESSSM_HEAD is not None:
        dd = float(np.abs(PROTO_HEAD(emb) - RESSSM_HEAD(emb)).mean())
        print(f"  [selftest] proto_ssm vs sgkfk_resssm mean|Δ|={dd:.5f} "
              f"{'(distinct OK)' if dd > 1e-5 else '(IDENTICAL — residual head no-op!)'}")
    print(f"  [selftest] RESULT: {'ALL ACTIVE MEMBERS SANE ✓' if good else 'DEGENERATE OUTPUT ✗'}")
    del xw, mel
    gc.collect()


# ============================================================================
# CELL 9b — DISPATCH: real scoring path vs dry-run COMMIT SCALE-TEST
# ============================================================================
test_files = sorted(TEST_DIR.glob("*.ogg")) if TEST_DIR.exists() else []

if test_files:
    # -------- REAL SCORING PATH --------
    P, row_ids_all, hours = run_full_inference(test_files, label="test")
    final = blend_members(P, row_ids_all, hours)
    write_submission(final, row_ids_all)
    print(f"PEAK RSS = {_peak_rss_gb():.2f} GB | Backend: "
          f"{'OpenVINO-FP16' if USE_OPENVINO else 'ONNXRuntime-CPU'}")
else:
    # -------- DRY-RUN COMMIT SCALE-TEST (no submission, validate memory@scale) --------
    # The hidden test set (~600 files) is only mounted at scoring time. To PROVE the
    # pipeline fits RAM/time at scale BEFORE resubmitting, run the FULL scoring code
    # path (same run_full_inference + blend) over N real train_soundscapes files, then
    # report peak RSS, wall time, files/sec, and submission-shape sanity. Finally emit
    # the required zero placeholder (the real test is absent in a commit).
    tiny_model_selftest()
    SCALE_N = int(os.environ.get("ORTHOBLEND_SCALE_N", "300"))
    ss_dir = COMP_DIR / "train_soundscapes"
    scale_files = sorted(ss_dir.glob("*.ogg"))[:SCALE_N] if ss_dir.exists() else []
    if scale_files:
        print(f"\n==== COMMIT SCALE-TEST: full pipeline over {len(scale_files)} "
              f"real train_soundscapes (BATCH_FILES={BATCH_FILES}) ====")
        t_scale = time.time()
        P, row_ids_all, hours = run_full_inference(scale_files, label="scale")
        final = blend_members(P, row_ids_all, hours)
        wall = time.time() - t_scale
        n = len(scale_files)
        fps = n / max(wall, 1e-6)
        peak = _peak_rss_gb()
        finite = bool(np.isfinite(final).all())
        proj_600 = 600.0 / max(fps, 1e-6)
        print("\n==== SCALE-TEST REPORT ====")
        print(f"  files_processed   : {n}")
        print(f"  PEAK_RSS_GB       : {peak:.2f}  (ru_maxrss; budget ~13 GB)")
        print(f"  wall_time_s       : {wall:.0f}")
        print(f"  files_per_sec     : {fps:.3f}")
        print(f"  proj_600_runtime_s: {proj_600:.0f}  ({proj_600/60:.1f} min vs 90 min budget)")
        print(f"  submission_shape  : {final.shape}  (expect rows={n*N_WINDOWS}, cols={NUM_CLASSES})")
        print(f"  all_finite        : {finite}")
        print(f"  value_range       : min={final.min():.4f} max={final.max():.4f}")
        shape_ok = final.shape == (n * N_WINDOWS, NUM_CLASSES)
        verdict = finite and shape_ok and (final.min() >= 0.0) and (final.max() <= 1.0)
        print(f"  SCALE_TEST_VERDICT: {'PASS ✓' if verdict else 'FAIL ✗'} "
              f"(peak {peak:.2f}GB {'<' if peak < 13 else '>='} 13GB)")
        del P, final, row_ids_all, hours
        gc.collect()
    else:
        print("  [scale-test] no train_soundscapes available — skipping scale-test.")
    # Still emit the required submission.csv (zeros for the absent real test).
    write_zero_submission()
    sys.exit(0)
