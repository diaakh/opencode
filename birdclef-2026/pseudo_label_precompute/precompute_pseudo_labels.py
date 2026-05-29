# =============================================================================
# T1-5  PRE-COMPUTE SOFT PSEUDO-LABELS for the 10,592 unlabeled train_soundscapes
# =============================================================================
# Build-Agent B1 | BirdCLEF+ 2026 sprint | CPU-ONLY Kaggle utility kernel.
#
# WHY: Noisy-student self-distillation (A6 #1, +0.03-0.06 LB) is the moonshot
# T2-A play. It needs SOFT teacher labels on the unlabeled soundscapes. GPU is
# reset-locked, so we pre-compute the teacher targets NOW on CPU using the best
# PUBLIC ensemble, package them as a Kaggle dataset, and B2's training code reads
# them the instant GPU returns. See sprint_2026-05-29_architecture/00_MASTER_BOARD.md.
#
# ENSEMBLE (public, CPU-mountable, verified live HTTP 200 on 2026-05-29):
#   * Perch v2 branch  -> tuckerarrants/perch-v2-no-dft-onnx (perch_v2_no_dft.onnx)
#                         (also ships the onnxruntime CPU wheel). 234-class logit
#                         head + 1536-D embedding. Apache-2.0.
#                         Fallback: rishikeshjani/perch-onnx-for-birdclef-2026.
#   * Distilled-SED    -> tuckerarrants/bc2026-distilled-sed-public (sed_fold*.onnx)
#                         AttBlockV2 SED, 234-class clip + frame logits.
#   * (optional) perch-meta embedding cache -> jaejohn/perch-meta (not required here;
#                         we recompute embeddings live so the kernel is standalone).
#
# BLEND: the 0.950 public anchor's dominant branch (01_LATEST_PUBLIC_095.md):
#        z = G( 0.60 * R(p_perch) + 0.40 * R(p_sed) ),  R = class-wise percentile rank.
#        We keep it deliberately simple (rank-blend, no site/hour G_prior) because
#        the OUTPUT is a teacher target for distillation, not a leaderboard sub.
#
# SOFT-LABEL RULE (A6 / agents/A6_prior_winners.md, 5th-place verified):
#        PSEUDO_TH=0.3, PSEUDO_POWER=2
#        power_transform(p) = clamp( p*(p>0.3) + p**2 , 0, 1 )
#        The downstream MIXING with hard labels (alpha=0.7) happens in B2's
#        TRAINING loop (unlabeled rows have no hard label), so we save the
#        post-power-transform SOFT probabilities here. See README "consumption
#        contract". alpha=0.7 is documented but NOT applied to the artifact.
#
# OUTPUT: compact soft labels (parquet + npz), per-window 234 scores AND per-file
#         aggregated soft labels, plus a per-file confidence column.
#
# RUNTIME: CPU only. Expected ~2.0-3.0 h for 10,592 files (see README). Chunked,
#          resumable via shard checkpoints, progress logged.
# =============================================================================

from __future__ import annotations

import gc
import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# ----------------------------------------------------------------------------
# Config (all overridable via env for the smoke test / local dry-run)
# ----------------------------------------------------------------------------
SR = 32_000
WINDOW_SEC = 5
N_WINDOWS = 12                       # 12 x 5s = 60s, the repo convention
WINDOW_SAMPLES = SR * WINDOW_SEC     # 160_000
CLIP_SAMPLES = WINDOW_SAMPLES * N_WINDOWS  # 1_920_000 (60s)

# Distilled-SED mel frontend (verbatim from exp019_fast.py)
N_MELS_SED = 256
N_FFT_SED = 2048
HOP_SED = 512
FMIN_SED = 20
FMAX_SED = 16000
TOP_DB_SED = 80

# Ensemble blend weights = 0.950 anchor dominant branch (01_LATEST_PUBLIC_095.md)
W_PERCH = 0.60
W_SED = 0.40

# A6 soft-label transform (5th-place verified constants)
PSEUDO_TH = 0.30
PSEUDO_POWER = 2
PSEUDO_ALPHA = 0.70   # documented for B2; NOT applied to the saved artifact

# Runtime knobs
BATCH_FILES = int(os.environ.get("PL_BATCH_FILES", "8"))   # files per ONNX call
NUM_WORKERS = int(os.environ.get("PL_NUM_WORKERS", "4"))   # audio IO threads
SHARD_FILES = int(os.environ.get("PL_SHARD_FILES", "1000"))  # checkpoint every N files
LIMIT_FILES = os.environ.get("PL_LIMIT_FILES")              # smoke test cap
LIMIT_FILES = int(LIMIT_FILES) if LIMIT_FILES else None
ONLY_UNLABELED = os.environ.get("PL_ONLY_UNLABELED", "1") == "1"

OUT_DIR = Path(os.environ.get("PL_OUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-7


# ----------------------------------------------------------------------------
# onnxruntime (Kaggle base image lacks it -> install bundled wheel)
# ----------------------------------------------------------------------------
def ensure_onnxruntime():
    try:
        import onnxruntime as ort  # noqa
        return ort
    except ImportError:
        whls = sorted(glob.glob("/kaggle/input/**/onnxruntime-1*.whl", recursive=True))
        whls = [w for w in whls if "gpu" not in w.lower()]
        assert whls, "onnxruntime wheel not found — attach tuckerarrants/perch-v2-no-dft-onnx"
        print(f"Installing onnxruntime from {whls[0]}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", whls[0]])
        import onnxruntime as ort  # noqa
        return ort


# ----------------------------------------------------------------------------
# Asset discovery
# ----------------------------------------------------------------------------
def find_comp_dir() -> Path:
    for c in ("/kaggle/input/competitions/birdclef-2026", "/kaggle/input/birdclef-2026"):
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError("BirdCLEF-2026 competition data not found under /kaggle/input")


def find_perch_onnx() -> Path:
    for pat in ("perch_v2_no_dft.onnx", "*perch*.onnx"):
        hits = sorted(Path("/kaggle/input").rglob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "Perch ONNX not found — attach tuckerarrants/perch-v2-no-dft-onnx "
        "or rishikeshjani/perch-onnx-for-birdclef-2026"
    )


def find_sed_onnx() -> list[Path]:
    hits = sorted(
        Path("/kaggle/input").rglob("sed_fold*.onnx"),
        key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)),
    )
    return hits  # may be empty -> SED branch disabled, Perch-only fallback


def load_label_columns(comp_dir: Path) -> list[str]:
    samp = comp_dir / "sample_submission.csv"
    if samp.exists():
        cols = pd.read_csv(samp, nrows=1).columns[1:].astype(str).tolist()
        if cols:
            return cols
    tax = comp_dir / "taxonomy.csv"
    if tax.exists():
        return pd.read_csv(tax)["primary_label"].astype(str).tolist()
    raise FileNotFoundError("sample_submission.csv / taxonomy.csv not found")


# ----------------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------------
def load_clip_60s(path: Path) -> np.ndarray:
    """Load -> mono -> 32 kHz -> exactly 60 s (pad/trim)."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1).astype(np.float32)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR).astype(np.float32)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    else:
        y = y[:CLIP_SAMPLES]
    return y.astype(np.float32, copy=False)


def to_windows(y: np.ndarray) -> np.ndarray:
    """(1_920_000,) -> (12, 160_000)."""
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES)


# ----------------------------------------------------------------------------
# Perch branch
# ----------------------------------------------------------------------------
class PerchBranch:
    def __init__(self, ort, path: Path, n_classes: int):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.n_classes = n_classes
        self.has_logits = any(
            (o.shape and len(o.shape) == 2 and o.shape[-1] == n_classes)
            for o in self.sess.get_outputs()
        )
        outs = [(o.name, o.shape) for o in self.sess.get_outputs()]
        print(f"Perch ONNX outputs={outs} has_234_head={self.has_logits}", flush=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (N, 160_000) -> (N, n_classes) probs. Uses the 234-head if present."""
        outs = self.sess.run(None, {self.inp: x})
        logit = None
        for o in outs:
            a = np.asarray(o)
            if a.ndim == 2 and a.shape[1] == self.n_classes:
                logit = a.astype(np.float32)
                break
        if logit is None:
            # No species head in this ONNX export -> Perch branch contributes nothing;
            # caller will fall back to SED-only. Return NaN sentinel.
            return np.full((x.shape[0], self.n_classes), np.nan, dtype=np.float32)
        return _sigmoid(logit)


# ----------------------------------------------------------------------------
# SED branch
# ----------------------------------------------------------------------------
class SEDBranch:
    def __init__(self, ort, paths: list[Path], n_classes: int):
        self.n_classes = n_classes
        self.sessions = []
        for p in paths:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sessions.append(ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"]))
        print(f"SED folds loaded: {[p.name for p in paths]}", flush=True)

    def mel(self, windows: np.ndarray) -> np.ndarray:
        """windows: (N, 160_000) -> (N, 1, 256, T) log-mel, per-clip standardized."""
        import librosa
        mels = []
        for x in windows:
            s = librosa.feature.melspectrogram(
                y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
                n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0,
            )
            s = librosa.power_to_db(s, top_db=TOP_DB_SED)
            s = (s - s.mean()) / (s.std() + 1e-6)
            mels.append(s)
        return np.stack(mels)[:, None].astype(np.float32)

    def predict(self, mel_batch: np.ndarray) -> np.ndarray:
        """mel_batch: (N, 1, 256, T) -> (N, n_classes) probs, fold-averaged."""
        p_sum = np.zeros((mel_batch.shape[0], self.n_classes), dtype=np.float32)
        for sess in self.sessions:
            outs = sess.run(None, {sess.get_inputs()[0].name: mel_batch})
            clip_logits = outs[0]                       # (N, 234)
            frame_max = np.asarray(outs[1]).max(axis=1)  # (N, T, 234) -> (N, 234)
            p_sum += 0.5 * _sigmoid(clip_logits) + 0.5 * _sigmoid(frame_max)
        return (p_sum / len(self.sessions)).astype(np.float32)


# ----------------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


def class_rank(values: np.ndarray) -> np.ndarray:
    """Class-wise percentile rank over rows (R in the anchor). values: (rows, classes)."""
    return pd.DataFrame(values).rank(axis=0, pct=True).to_numpy(dtype=np.float32)


def power_transform(p: np.ndarray) -> np.ndarray:
    """A6 soft-label transform: p*(p>th) + p**power, clamped [0,1]."""
    p = np.asarray(p, dtype=np.float32)
    out = p * (p > PSEUDO_TH).astype(np.float32) + np.power(p, PSEUDO_POWER, dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def list_target_files(comp_dir: Path) -> list[Path]:
    soundscapes = comp_dir / "train_soundscapes"
    files = sorted(soundscapes.glob("*.ogg"))
    if ONLY_UNLABELED:
        for cand in ("train_soundscapes_labels.csv", "train_soundscape_labels.csv"):
            lp = comp_dir / cand
            if lp.exists():
                labeled = set(pd.read_csv(lp)["filename"].astype(str).unique())
                before = len(files)
                files = [p for p in files if p.name not in labeled]
                print(f"Excluded {before - len(files)} already-labeled files", flush=True)
                break
    if LIMIT_FILES:
        files = files[:LIMIT_FILES]
    return files


def blend_rank(p_perch: np.ndarray, p_sed: np.ndarray) -> np.ndarray:
    """Rank-blend the two branches over the given rows, then A6 power-transform.

    z = W_PERCH*R(p_perch) + W_SED*R(p_sed), R=class-wise percentile rank.
    Renormalizes to a single branch if the other is absent (all-NaN sentinel).
    """
    have_perch = p_perch.size and np.isfinite(p_perch).all()
    have_sed = p_sed.size and np.isfinite(p_sed).all()
    if have_perch and have_sed:
        z = W_PERCH * class_rank(p_perch) + W_SED * class_rank(p_sed)
    elif have_perch:
        z = class_rank(p_perch)
    elif have_sed:
        z = class_rank(p_sed)
    else:
        z = np.zeros_like(p_perch)
    # z is a percentile-rank score in [0,1]; A6 transform keeps the metric-relevant
    # ordering (see A6 "Why soft (not hard) pseudo-labels").
    return power_transform(z)


def _save_table(df: pd.DataFrame, stem: str) -> str:
    """Write parquet if an engine is available, else CSV. Returns the filename written."""
    try:
        path = OUT_DIR / f"{stem}.parquet"
        df.to_parquet(path, index=False)
        return path.name
    except Exception as exc:  # pragma: no cover - depends on Kaggle image
        path = OUT_DIR / f"{stem}.csv"
        df.to_csv(path, index=False)
        print(f"  (parquet engine unavailable: {type(exc).__name__}; wrote CSV instead)", flush=True)
        return path.name


def write_outputs(soft, filenames, start_secs, label_cols, used_perch, used_sed):
    """Per-window parquet + npz, per-file aggregate parquet, manifest JSON."""
    import json
    n_classes = len(label_cols)
    start_arr = np.asarray(start_secs, dtype=np.int32)

    win = pd.DataFrame(soft, columns=label_cols)
    win.insert(0, "filename", filenames)
    win.insert(1, "start_sec", start_arr)
    win.insert(2, "window_idx", start_arr // WINDOW_SEC)
    win.insert(3, "row_id", [f"{Path(f).stem}_{int(s) + WINDOW_SEC}"
                             for f, s in zip(filenames, start_secs)])
    win.insert(4, "win_confidence",
               soft.max(axis=1) if len(soft) else np.zeros(0, np.float32))

    win_name = _save_table(win, "pseudo_soft_per_window")
    print(f"Wrote {OUT_DIR / win_name}: {win.shape}", flush=True)

    npz_path = OUT_DIR / "pseudo_soft_per_window.npz"
    np.savez_compressed(
        npz_path,
        soft=soft.astype(np.float32),
        filename=np.asarray(filenames),
        start_sec=start_arr,
        window_idx=start_arr // WINDOW_SEC,
        classes=np.asarray(label_cols),
    )
    print(f"Wrote {npz_path}", flush=True)

    if len(soft):
        agg = win.groupby("filename", sort=False)[label_cols].max().reset_index()
        conf = (win.groupby("filename", sort=False)["win_confidence"].mean()
                .reset_index().rename(columns={"win_confidence": "file_confidence"}))
        agg = agg.merge(conf, on="filename")
        agg = agg[["filename", "file_confidence"] + label_cols]
    else:
        agg = pd.DataFrame(columns=["filename", "file_confidence"] + label_cols)
    file_name = _save_table(agg, "pseudo_soft_per_file")
    print(f"Wrote {OUT_DIR / file_name}: {agg.shape}", flush=True)

    manifest = {
        "artifact_version": "1.0",
        "produced_by": "pseudo_label_precompute/precompute_pseudo_labels.py (Build-Agent B1)",
        "n_classes": n_classes,
        "n_windows_per_file": N_WINDOWS,
        "window_sec": WINDOW_SEC,
        "sr": SR,
        "blend": {"w_perch": W_PERCH, "w_sed": W_SED, "rank_blend": True,
                  "used_perch_head": bool(used_perch), "used_sed": bool(used_sed)},
        "soft_label_rule": {"pseudo_th": PSEUDO_TH, "pseudo_power": PSEUDO_POWER,
                            "transform": "clamp(p*(p>th)+p**power,0,1)"},
        "downstream_mixing": {"pseudo_alpha": PSEUDO_ALPHA,
                              "rule": "label = alpha*soft + (1-alpha)*hard ; NOT applied here",
                              "note": "unlabeled rows have hard=0 -> label=alpha*soft"},
        "files": {"per_window_table": win_name,
                  "per_window_npz": "pseudo_soft_per_window.npz",
                  "per_file_table": file_name},
    }
    mpath = OUT_DIR / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {mpath}", flush=True)


def main() -> int:
    t_start = time.time()
    ort = ensure_onnxruntime()
    print(f"onnxruntime {ort.__version__} providers={ort.get_available_providers()}", flush=True)

    comp_dir = find_comp_dir()
    label_cols = load_label_columns(comp_dir)
    n_classes = len(label_cols)
    print(f"Comp dir: {comp_dir}  classes={n_classes}", flush=True)

    perch = PerchBranch(ort, find_perch_onnx(), n_classes)
    sed_paths = find_sed_onnx()
    sed = SEDBranch(ort, sed_paths, n_classes) if sed_paths else None
    if sed is None:
        print("WARNING: no SED folds found -> Perch-only pseudo labels", flush=True)
    if not perch.has_logits and sed is None:
        raise RuntimeError("Neither a Perch 234-head nor SED folds available — cannot build labels")

    files = list_target_files(comp_dir)
    n_files = len(files)
    if n_files == 0:
        print("No soundscape files found — emitting empty artifact (dry run).", flush=True)
        write_outputs(np.zeros((0, n_classes), np.float32), [], [], label_cols,
                      perch.has_logits, sed is not None)
        return 0
    print(f"Processing {n_files} files  batch_files={BATCH_FILES}  workers={NUM_WORKERS}", flush=True)

    n_rows = n_files * N_WINDOWS
    soft = np.zeros((n_rows, n_classes), dtype=np.float32)
    perch_buf = np.full((n_rows, n_classes), np.nan, dtype=np.float32)
    sed_buf = np.full((n_rows, n_classes), np.nan, dtype=np.float32)
    row_filename = np.empty(n_rows, dtype=object)
    row_start_sec = np.zeros(n_rows, dtype=np.int32)

    def load_one(p):
        return p, to_windows(load_clip_60s(p))

    processed = 0           # files done
    shard_anchor = 0        # first file index of the current un-flushed shard
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        next_futs = [pool.submit(load_one, p) for p in files[:BATCH_FILES]]
        for start in range(0, n_files, BATCH_FILES):
            results = [f.result() for f in next_futs]
            nxt = start + BATCH_FILES
            if nxt < n_files:
                next_futs = [pool.submit(load_one, p) for p in files[nxt:nxt + BATCH_FILES]]

            bn = len(results)
            windows = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            for bi, (_, w) in enumerate(results):
                windows[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = w

            p_perch = (perch.predict(windows) if perch.has_logits
                       else np.full((bn * N_WINDOWS, n_classes), np.nan, np.float32))
            p_sed = (sed.predict(sed.mel(windows)) if sed is not None
                     else np.full((bn * N_WINDOWS, n_classes), np.nan, np.float32))

            base = processed * N_WINDOWS
            perch_buf[base:base + bn * N_WINDOWS] = p_perch
            sed_buf[base:base + bn * N_WINDOWS] = p_sed
            for bi, (fpath, _) in enumerate(results):
                for w_i in range(N_WINDOWS):
                    ridx = base + bi * N_WINDOWS + w_i
                    row_filename[ridx] = fpath.name
                    row_start_sec[ridx] = w_i * WINDOW_SEC
            processed += bn

            if processed % (BATCH_FILES * 10) == 0 or processed >= n_files:
                el = time.time() - t_start
                rate = processed / max(el, 1.0)
                eta = (n_files - processed) / max(rate, 1e-3)
                print(f"  [{processed}/{n_files}] el={el/60:.1f}min "
                      f"rate={rate:.2f} files/s eta={eta/60:.1f}min", flush=True)

            # flush a shard (rank-blend is computed over the shard's rows)
            if (processed - shard_anchor) >= SHARD_FILES or processed >= n_files:
                lo, hi = shard_anchor * N_WINDOWS, processed * N_WINDOWS
                soft[lo:hi] = blend_rank(perch_buf[lo:hi], sed_buf[lo:hi])
                shard_anchor = processed
                gc.collect()

    soft = np.clip(soft, 0.0, 1.0).astype(np.float32)
    write_outputs(soft, [str(x) for x in row_filename.tolist()],
                  row_start_sec.tolist(), label_cols, perch.has_logits, sed is not None)

    el = time.time() - t_start
    print(f"DONE in {el/60:.1f} min  rows={n_rows} files={n_files} "
          f"soft[min={soft.min():.4f} max={soft.max():.4f} mean={soft.mean():.4f}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
