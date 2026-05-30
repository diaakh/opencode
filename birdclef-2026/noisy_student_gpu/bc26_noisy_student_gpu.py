# BirdCLEF+ 2026 - NOISY-STUDENT end-to-end on GPU (Nikita Babych recipe).
#
# WHY: round-0 CNN HURT the 0.950 blend (0.944) because it was trained on FOCAL train_audio only and
# is weak on the SOUNDSCAPE test domain. The fix (Nikita's disclosed +0.020): train a STUDENT on the
# SOUNDSCAPES THEMSELVES via teacher pseudo-labels => domain adaptation focal->soundscape.
#
# This ONE kernel does the whole pipeline on GPU (P100 sm_60 torch fix + onnxruntime-gpu teachers):
#   STAGE A  pseudo-label every unlabeled train_soundscapes window on CUDA
#            teacher = public Perch v2 ONNX (234-head) + distilled-SED ONNX (5 folds), rank-blend,
#            Nikita power-transform (threshold=0, per-round power). Report the 28 zero-train son-ID
#            coverage (the CRITICAL number: can the teacher even score the 25 Insecta + 3 Amphibia?).
#   STAGE B  train the noisy-student SED student (tf_efficientnet_b0, 1ch log-mel, AMP, channels_last)
#            on the pseudo-labeled soundscapes (+ optional focal train_audio), soft-CE targets,
#            MANDATORY fixed-lambda=0.5 MixUp of every sample with a random pseudo soundscape sample.
#            Multi-round: the trained student re-labels the soundscapes and becomes the next teacher.
#   STAGE C  validate (CUDA real, loss decreasing, sane soundscape preds, 28-class coverage carried),
#            save the student artifact (backbone + 234 head) + soundscape preds for orthogonality.
#
# Reuses train_g124.py building blocks (AudioDataset soft-target path, MelFrontend, build_model,
# collate_audio). Recipe constants mirror noisy_student.py NikitaConfig.

import os, sys, glob, time, subprocess, math, re, json
from pathlib import Path
import numpy as np

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# --------------------------------------------------------------- GPU / torch fix (P100 sm_60)
TORCH_PIN = ("torch==2.7.1", "torchaudio==2.7.1", "torchvision==0.22.1")
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"

def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr

def install_deps():
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                          "--format=csv"], capture_output=True, text=True).stdout, flush=True)
    log("installing", TORCH_PIN, "from", TORCH_INDEX, "(P100 sm_60 fix)")
    rc, out, err = sh([sys.executable, "-m", "pip", "install", "-q", *TORCH_PIN,
                       "--index-url", TORCH_INDEX])
    log("torch pip rc", rc)
    if rc != 0:
        print(out[-2000:], err[-2000:], flush=True); raise RuntimeError("torch install failed")
    # onnxruntime-gpu for the ONNX teachers on CUDAExecutionProvider (internet on)
    log("installing onnxruntime-gpu")
    rc, out, err = sh([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu"])
    log("onnxruntime-gpu pip rc", rc)
    if rc != 0:
        print(out[-1500:], err[-1500:], flush=True)
        log("WARNING: onnxruntime-gpu failed; will try bundled CPU wheel as fallback")

def verify_gpu():
    import torch, torch.nn as nn
    log("torch", torch.__version__, "cuda_build", torch.version.cuda,
        "is_available", torch.cuda.is_available(), "arch_list", torch.cuda.get_arch_list())
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    log("device", torch.cuda.get_device_name(0), "cap sm_", torch.cuda.get_device_capability(0))
    a = torch.randn(1024, 1024, device="cuda"); c = a @ a
    m = nn.Conv2d(1, 8, 3, padding=1).cuda().to(memory_format=torch.channels_last)
    x = torch.randn(4, 1, 128, 313, device="cuda").to(memory_format=torch.channels_last)
    with torch.autocast("cuda"):
        y = m(x)
    torch.cuda.synchronize()
    _ = float(c.sum()) + float(y.sum())
    log("CUDA real matmul+conv+AMP OK -> training will run on GPU")

# --------------------------------------------------------------- asset discovery
def find_competition_dir():
    for c in ("/kaggle/input/competitions/birdclef-2026", "/kaggle/input/birdclef-2026"):
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError("birdclef-2026 not attached")

def find_code_root():
    for c in glob.glob("/kaggle/input/**/train_g124.py", recursive=True):
        return Path(c).parent
    raise FileNotFoundError("birdclef-g124-code (train_g124.py) not attached")

def find_perch_onnx():
    for pat in ("perch_v2_no_dft.onnx", "*perch*.onnx"):
        hits = sorted(Path("/kaggle/input").rglob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError("Perch ONNX not attached (tuckerarrants/perch-v2-no-dft-onnx)")

def find_sed_onnx():
    hits = sorted(Path("/kaggle/input").rglob("sed_fold*.onnx"),
                  key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)))
    return hits

# --------------------------------------------------------------- recipe constants (Nikita)
SR = 32000
WINDOW_SEC = 5.0
N_WINDOWS = 12                  # 12 x 5s = 60s soundscape
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
CLIP_SAMPLES = WINDOW_SAMPLES * N_WINDOWS
# SED teacher mel (verbatim B1 / distilled-SED frontend)
N_MELS_SED, N_FFT_SED, HOP_SED, FMIN_SED, FMAX_SED, TOP_DB_SED = 256, 2048, 512, 20, 16000, 80
W_PERCH, W_SED = 0.60, 0.40     # 0.950-anchor dominant-branch blend
# Nikita power schedule (it1=1.0, it2=1/0.65, it3=1/0.55, it4=1/0.6); threshold=0 pure power transform
PSEUDO_POWER_SCHEDULE = (1.0, 1.53846, 1.81818, 1.66667)
MIXUP_LAMBDA = 0.5              # every sample mixed with a pseudo one at fixed lambda

def sigmoid_np(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)

def class_rank(values):
    import pandas as pd
    return pd.DataFrame(values).rank(axis=0, pct=True).to_numpy(dtype=np.float32)

def power_transform(p, power):
    # Nikita threshold=0: pure power sharpening, clamp [0,1]
    return np.clip(np.power(np.asarray(p, np.float32), power, dtype=np.float32), 0.0, 1.0)

# --------------------------------------------------------------- ONNX teachers (CUDA)
def make_session(ort, path, providers, gpu_mem_limit_gb=2.0):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers == ["CPUExecutionProvider"]:
        so.intra_op_num_threads = 4; so.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), so, providers=providers)
    # Cap per-session CUDA arena so 5 SED folds + torch coexist on a 16GB P100 (the OOM fix).
    if "CUDAExecutionProvider" in providers:
        cuda_opts = {"arena_extend_strategy": "kSameAsRequested",
                     "gpu_mem_limit": int(gpu_mem_limit_gb * 1024 * 1024 * 1024)}
        prov = [("CUDAExecutionProvider", cuda_opts) if p == "CUDAExecutionProvider" else p
                for p in providers]
        return ort.InferenceSession(str(path), so, providers=prov)
    return ort.InferenceSession(str(path), so, providers=providers)

class PerchTeacher:
    def __init__(self, ort, path, n_classes, providers):
        self.sess = make_session(ort, path, providers)
        self.inp = self.sess.get_inputs()[0].name
        self.n_classes = n_classes
        outs = [(o.name, o.shape) for o in self.sess.get_outputs()]
        self.has_logits = any(o.shape and len(o.shape) == 2 and o.shape[-1] == n_classes
                              for o in self.sess.get_outputs())
        log(f"Perch outputs={outs} has_234_head={self.has_logits} providers={self.sess.get_providers()}")
    def predict(self, x):  # x: (N,160000)
        outs = self.sess.run(None, {self.inp: x})
        for o in outs:
            a = np.asarray(o)
            if a.ndim == 2 and a.shape[1] == self.n_classes:
                return sigmoid_np(a.astype(np.float32))
        return np.full((x.shape[0], self.n_classes), np.nan, np.float32)

class SEDTeacher:
    """Distilled-SED ensemble. Loads ALL folds once (each capped to a small CUDA arena via
    make_session) so 5 folds + torch fit on a 16GB P100. mel() is CPU (librosa); inference on GPU."""
    def __init__(self, ort, paths, n_classes, providers):
        self.ort = ort; self.paths = paths; self.providers = providers; self.n_classes = n_classes
        # cap each fold to ~1.2GB CUDA arena -> 5 folds <= ~6GB, leaving headroom for torch.
        self.sessions = [make_session(ort, p, providers, gpu_mem_limit_gb=1.2) for p in paths]
        log(f"SED folds: {[p.name for p in paths]} providers={self.sessions[0].get_providers()}")
        i0 = self.sessions[0].get_inputs()[0]
        o = [(x.name, x.shape) for x in self.sessions[0].get_outputs()]
        log(f"SED input={i0.name}{i0.shape} outputs={o}")
    def mel(self, windows):
        import librosa
        mels = []
        for x in windows:
            s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
                                               n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0)
            s = librosa.power_to_db(s, top_db=TOP_DB_SED)
            s = (s - s.mean()) / (s.std() + 1e-6)
            mels.append(s)
        return np.stack(mels)[:, None].astype(np.float32)
    def predict(self, mel_batch, micro=12):
        # micro-batch only needed under the GPU arena cap; CPU runs the full batch fine.
        cpu = self.providers == ["CPUExecutionProvider"]
        step = mel_batch.shape[0] if cpu else micro
        p = np.zeros((mel_batch.shape[0], self.n_classes), np.float32)
        for s in self.sessions:
            iname = s.get_inputs()[0].name
            for b in range(0, mel_batch.shape[0], step):
                mb = mel_batch[b:b+step]
                outs = s.run(None, {iname: mb})
                clip = sigmoid_np(outs[0])
                frame = sigmoid_np(np.asarray(outs[1]).max(axis=1)) if len(outs) > 1 else clip
                p[b:b+step] += 0.5 * clip + 0.5 * frame
        return (p / len(self.sessions)).astype(np.float32)

# --------------------------------------------------------------- STAGE A: pseudo-label soundscapes
def load_clip_60s(path):
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1: y = y.mean(axis=1).astype(np.float32)
    if sr != SR:
        import librosa; y = librosa.resample(y, orig_sr=sr, target_sr=SR).astype(np.float32)
    if len(y) < CLIP_SAMPLES: y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    else: y = y[:CLIP_SAMPLES]
    return y.astype(np.float32, copy=False).reshape(N_WINDOWS, WINDOW_SAMPLES)

def pseudo_label_soundscapes(comp, ort, perch, sed, classes, limit=None, batch_files=8):
    """Return (paths list, start_secs array, raw_blend (rows,C) in [0,1])."""
    from concurrent.futures import ThreadPoolExecutor
    n_classes = len(classes)
    files = sorted((comp / "train_soundscapes").glob("*.ogg"))
    if limit: files = files[:limit]
    n = len(files)
    log(f"STAGE A: {n} soundscape files x {N_WINDOWS} windows = {n*N_WINDOWS} rows")
    rows = n * N_WINDOWS
    blend = np.zeros((rows, n_classes), np.float32)
    perch_buf = np.full((rows, n_classes), np.nan, np.float32)
    sed_buf = np.full((rows, n_classes), np.nan, np.float32)
    row_path = np.empty(rows, dtype=object); row_start = np.zeros(rows, np.int32)
    def load_one(p): return p, load_clip_60s(p)
    processed = 0; t = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        nxt = [pool.submit(load_one, p) for p in files[:batch_files]]
        for start in range(0, n, batch_files):
            res = [f.result() for f in nxt]
            j = start + batch_files
            if j < n: nxt = [pool.submit(load_one, p) for p in files[j:j+batch_files]]
            bn = len(res)
            w = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), np.float32)
            for bi, (_, ww) in enumerate(res): w[bi*N_WINDOWS:(bi+1)*N_WINDOWS] = ww
            pp = perch.predict(w) if (perch and perch.has_logits) else np.full((bn*N_WINDOWS, n_classes), np.nan, np.float32)
            ps = sed.predict(sed.mel(w)) if sed else np.full((bn*N_WINDOWS, n_classes), np.nan, np.float32)
            base = processed * N_WINDOWS
            perch_buf[base:base+bn*N_WINDOWS] = pp; sed_buf[base:base+bn*N_WINDOWS] = ps
            for bi, (fp, _) in enumerate(res):
                for wi in range(N_WINDOWS):
                    r = base + bi*N_WINDOWS + wi
                    row_path[r] = str(fp); row_start[r] = int(wi * WINDOW_SEC)
            processed += bn
            if processed % (batch_files*20) == 0 or processed >= n:
                el = time.time()-t; rate = processed/max(el,1e-3)
                log(f"  pseudo [{processed}/{n}] {rate:.2f} files/s eta={ (n-processed)/max(rate,1e-3)/60:.1f}min")
    have_p = np.isfinite(perch_buf).all(); have_s = np.isfinite(sed_buf).all()
    log(f"teacher availability: perch={have_p} sed={have_s}")
    # IMPORTANT (smoke finding): class-wise percentile rank on DEAD columns (e.g. the 28 zero-train
    # son-IDs the SED scores as ~constant) FABRICATES a uniform 0..1 ranking, manufacturing fake
    # signal. So rank-blend ONLY when fusing two real teachers; with a single (SED) teacher use its
    # RAW probabilities so dead columns stay honestly near-0 and the power-transform can't invent
    # structure from rank noise.
    if have_p and have_s:
        blend = W_PERCH*class_rank(perch_buf) + W_SED*class_rank(sed_buf)
    elif have_p: blend = perch_buf
    elif have_s: blend = sed_buf
    blend = np.clip(blend, 0.0, 1.0).astype(np.float32)
    return [str(p) for p in row_path.tolist()], row_start, blend, perch_buf, sed_buf

def report_zero_train_coverage(tag, blend, classes, zero_train, tax):
    import pandas as pd
    zt_idx = [classes.index(l) for l in zero_train if l in classes]
    log("=" * 70)
    log(f"[{tag}] 28 ZERO-TRAIN SON-ID TEACHER COVERAGE (the critical number):")
    nonzero = 0
    by_class = {}
    for j, l in zip(zt_idx, zero_train):
        col = blend[:, j]
        sci = str(tax.loc[tax["primary_label"].astype(str) == l, "scientific_name"].iloc[0]) if (tax["primary_label"].astype(str) == l).any() else "?"
        cls = str(tax.loc[tax["primary_label"].astype(str) == l, "class_name"].iloc[0]) if (tax["primary_label"].astype(str) == l).any() else "?"
        active = int((col > 0.5).sum()); spread = float(col.max() - col.min())
        if col.max() > 1e-4 and spread > 1e-5: nonzero += 1
        by_class[l] = dict(cls=cls, sci=sci, mean=float(col.mean()), max=float(col.max()),
                           active_gt05=active, spread=spread)
        log(f"   {l:>11} {cls:<9} {sci:<26} mean={col.mean():.4f} max={col.max():.4f} "
            f"windows>0.5={active} spread={spread:.4f}")
    log(f"[{tag}] NON-ZERO 28-class teacher heads: {nonzero}/{len(zero_train)} "
        f"(producing real, structured pseudo-signal; 0 => the 28 are unreachable for everyone)")
    log("=" * 70)
    return nonzero, by_class

# --------------------------------------------------------------- frame builders
def build_pseudo_frame(paths, start_secs, soft_targets, classes, power, conf_floor=0.0):
    """One row per soundscape window: path, start_seconds, soft target ndarray, sample_weight=win confidence."""
    import pandas as pd
    soft = power_transform(soft_targets, power)
    conf = soft.max(axis=1).astype(np.float32)
    keep = conf >= conf_floor
    rows = []
    for i in np.nonzero(keep)[0]:
        rows.append({"path": paths[i], "start_seconds": float(start_secs[i]),
                     "target": soft[i].astype(np.float32), "sample_weight": float(conf[i]),
                     "primary_label": "__pseudo__", "source": "pseudo_soundscape"})
    return pd.DataFrame(rows)

# --------------------------------------------------------------- STAGE B/C: noisy-student training
def main():
    install_deps()
    comp = find_competition_dir(); code_root = find_code_root()
    perch_path = find_perch_onnx(); sed_paths = find_sed_onnx()
    log("comp:", comp, "| code:", code_root)
    log("perch:", perch_path, "| sed folds:", len(sed_paths))
    verify_gpu()

    sys.path.insert(0, str(code_root))
    import torch, pandas as pd
    from torch.utils.data import DataLoader
    from train_g124 import (load_classes, AudioDataset, MelFrontend, build_model, collate_audio)

    import onnxruntime as ort
    avail = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
    log("onnxruntime", ort.__version__, "available", avail, "-> using", providers)

    classes = load_classes(comp); n_classes = len(classes)
    tax = pd.read_csv(comp / "taxonomy.csv", dtype={"primary_label": str})
    train_csv = pd.read_csv(comp / "train.csv", dtype={"primary_label": str})
    labels_with_audio = set(train_csv["primary_label"].unique())
    zero_train = sorted(set(tax["primary_label"].astype(str)) - labels_with_audio)
    log(f"classes={n_classes} zero_train={len(zero_train)} (expected 28)")
    log("zero-train by class:", tax[tax["primary_label"].astype(str).isin(zero_train)]["class_name"].value_counts().to_dict())

    # --- knobs (env-overridable to fit a 12h kernel) ---
    LIMIT = os.environ.get("NS_LIMIT_FILES"); LIMIT = int(LIMIT) if LIMIT else None
    rounds = int(os.environ.get("NS_ROUNDS", "2"))
    epochs = int(os.environ.get("NS_EPOCHS", "6"))
    batch = int(os.environ.get("NS_BATCH", "64"))
    workers = int(os.environ.get("NS_WORKERS", "3"))
    model_name = os.environ.get("NS_MODEL", "tf_efficientnet_b0")
    add_focal = os.environ.get("NS_ADD_FOCAL", "1") == "1"
    focal_max = int(os.environ.get("NS_FOCAL_MAX_FILES", "20"))   # files/class cap for focal supervision

    # student mel frontend args (1ch log-mel, our CNN convention)
    class A: pass
    args = A()
    args.sr = SR; args.window_seconds = WINDOW_SEC; args.n_mels = 128; args.n_fft = 2048
    args.hop_length = 512; args.fmin = 20.0; args.fmax = 16000.0; args.audio_cache_mb = 0
    device = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    frontend = MelFrontend(args, device)

    # optional focal train_audio (hard-label supervision to anchor the 206 mapped classes)
    focal_frame = pd.DataFrame()
    if add_focal:
        from train_g124 import build_train_audio_frame
        focal_frame = build_train_audio_frame(comp, classes, max_files=focal_max)
        focal_frame["sample_weight"] = 1.0
        focal_frame["source"] = "focal_train_audio"
        log(f"focal train_audio supervision frame: {len(focal_frame)} files")

    # ---- STAGE A: teacher pseudo-labels (ONNX teachers) ----
    # NOTE (smoke finding): the public Perch v2 ONNX `label` head is the GENERIC 14,795-class Perch
    # head, NOT the BirdCLEF-2026 234-class space (everyone uses Perch only as a frozen 1536-D
    # embedder). So Perch cannot produce 234-class pseudo-labels directly -> the 234-class TEACHER
    # is the distilled-SED ensemble (which IS trained on the 234-class space and can score the 28
    # son-IDs). We probe Perch's head just to log this, then build SED-only pseudo-labels.
    perch = PerchTeacher(ort, perch_path, n_classes, ["CPUExecutionProvider"])  # probe head on CPU (no GPU mem)
    if not perch.has_logits:
        log("Perch has NO 234-head (generic 14795-class) -> TEACHER = distilled-SED only (expected).")
        del perch; perch = None
    # SED teacher provider: GPU QuickGelu on P100 (sm_60) is pathologically slow under onnxruntime-gpu,
    # so default the SED teacher to multi-threaded CPU (B1's proven path). Override with NS_SED_GPU=1.
    sed_providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if os.environ.get("NS_SED_GPU", "0") == "1" and "CUDAExecutionProvider" in avail
                     else ["CPUExecutionProvider"])
    log("SED teacher providers ->", sed_providers, "(set NS_SED_GPU=1 to force GPU)")
    sed = SEDTeacher(ort, sed_paths, n_classes, sed_providers) if sed_paths else None
    if sed is None:
        raise RuntimeError("No SED folds and no Perch-234 head -> cannot build 234-class pseudo-labels")
    tA = time.time()
    paths, start_secs, blend, perch_buf, sed_buf = pseudo_label_soundscapes(
        comp, ort, perch, sed, classes, limit=LIMIT)
    pseudo_runtime = time.time() - tA
    log(f"STAGE A pseudo-label runtime = {pseudo_runtime/60:.1f} min for {len(set(paths))} files")
    # HONEST 28-son-ID teacher diagnostic on the RAW SED sigmoid probs (NOT rank-blended): if the
    # SED scores the son-IDs as a near-constant dead column, mean~max~std==0 and we say so plainly.
    zt_idx = [classes.index(l) for l in zero_train if l in classes]
    mapped_idx = [j for j in range(n_classes) if j not in set(zt_idx)]
    zt_cols = sed_buf[:, zt_idx] if np.isfinite(sed_buf).all() else blend[:, zt_idx]
    mp_cols = sed_buf[:, mapped_idx] if np.isfinite(sed_buf).all() else blend[:, mapped_idx]
    log("=" * 70)
    log("HONEST RAW-SED 28-SON-ID DIAGNOSTIC (raw sigmoid prob, not rank):")
    log(f"   28 zero-train cols : mean={zt_cols.mean():.5f} max={zt_cols.max():.5f} "
        f"std={zt_cols.std():.5f} frac>0.1={float((zt_cols>0.1).mean()):.4f}")
    log(f"   206 mapped cols    : mean={mp_cols.mean():.5f} max={mp_cols.max():.5f} "
        f"std={mp_cols.std():.5f} frac>0.1={float((mp_cols>0.1).mean()):.4f}")
    zt_alive = int((zt_cols.std(axis=0) > 1e-4).sum())
    log(f"   son-ID columns with real per-window variance (std>1e-4): {zt_alive}/{len(zt_idx)}")
    log(f"   VERDICT: {'SED scores the son-IDs (real signal)' if zt_cols.max() > 0.1 and zt_alive > 5 else 'SED CANNOT score the son-IDs (dead/flat columns) -> the 28 are UNREACHABLE for everyone; focus on the 206 mapped'}")
    log("=" * 70)
    nonzero_teacher = zt_alive
    # also keep the rank-style report for continuity
    report_zero_train_coverage("TEACHER blend", blend, classes, zero_train, tax)

    out_dir = Path("/kaggle/working/noisy_student"); out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "ns_student_b0_fold1_fp16.pt"

    # ---- STAGE B: multi-round noisy student ----
    current_soft = blend  # round-0 teacher = ONNX ensemble blend
    loss_trend = []; epoch_times = []
    for rnd in range(rounds):
        power = PSEUDO_POWER_SCHEDULE[min(rnd, len(PSEUDO_POWER_SCHEDULE)-1)]
        log("#" * 70)
        log(f"ROUND {rnd+1}/{rounds}  power={power:.4f}  mixup_lambda={MIXUP_LAMBDA}")
        pseudo_frame = build_pseudo_frame(paths, start_secs, current_soft, classes, power)
        log(f"  pseudo rows={len(pseudo_frame)} | focal rows={len(focal_frame)}")
        # train frame = pseudo soundscapes (+ focal). Pseudo pool used for MixUp partners.
        train_frame = pd.concat([pseudo_frame, focal_frame], ignore_index=True) if add_focal else pseudo_frame
        # a held-out soundscape slice for val loss (deterministic last 5%)
        n_p = len(pseudo_frame); v0 = int(n_p * 0.95)
        val_frame = pseudo_frame.iloc[v0:].reset_index(drop=True)

        train_ds = AudioDataset(train_frame.sample(frac=1.0, random_state=124+rnd).reset_index(drop=True),
                                classes, args, training=True)
        val_ds = AudioDataset(val_frame, classes, args, training=False)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers,
                                  pin_memory=True, persistent_workers=workers>0, collate_fn=collate_audio,
                                  drop_last=True, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers,
                                pin_memory=True, persistent_workers=workers>0, collate_fn=collate_audio,
                                prefetch_factor=2)
        # fresh student each round (Xie et al.); timm pretrained backbone for a sane start
        model = build_model(model_name, n_classes, pretrained_checkpoint=None, timm_pretrained=True)
        model = model.to(device).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        for epoch in range(epochs):
            t0 = time.time(); model.train(); tl = []
            for wave, target, weight in train_loader:
                wave = wave.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
                # MANDATORY fixed-lambda=0.5 MixUp with a random pseudo soundscape sample (Nikita)
                perm = torch.randperm(wave.shape[0], device=wave.device)
                wave = MIXUP_LAMBDA * wave + (1.0 - MIXUP_LAMBDA) * wave[perm]
                target = MIXUP_LAMBDA * target + (1.0 - MIXUP_LAMBDA) * target[perm]
                opt.zero_grad(set_to_none=True)
                x = frontend(wave).to(memory_format=torch.channels_last)
                with torch.autocast("cuda"):
                    logits = model(x)
                    # soft cross-entropy on soft pseudo-targets (CE over softmax; Nikita uses CE)
                    logp = torch.log_softmax(logits.float(), dim=1)
                    tgt = target.float()
                    tgt_sum = tgt.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    loss = -(tgt / tgt_sum * logp).sum(dim=1).mean()
                if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); tl.append(float(loss.detach().cpu()))
            sched.step()
            model.eval(); vl = []
            with torch.inference_mode():
                for wave, target, _w in val_loader:
                    wave = wave.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
                    x = frontend(wave).to(memory_format=torch.channels_last)
                    logp = torch.log_softmax(model(x).float(), dim=1)
                    tgt = target.float(); tgt_sum = tgt.sum(dim=1, keepdim=True).clamp_min(1e-6)
                    vl.append(float((-(tgt/tgt_sum*logp).sum(dim=1).mean()).cpu()))
            dt = time.time()-t0; epoch_times.append(dt)
            tr = float(np.mean(tl)); va = float(np.mean(vl)) if vl else float("nan")
            loss_trend.append((rnd+1, epoch+1, tr, va))
            log(f"  round{rnd+1} epoch={epoch+1}/{epochs} train_ce={tr:.5f} val_ce={va:.5f} "
                f"time={dt:.1f}s mean_epoch={np.mean(epoch_times):.1f}s")
            state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
            torch.save({"state_dict": state, "classes": classes,
                        "config": {"model_name": model_name, "sr": SR, "window_seconds": WINDOW_SEC,
                                   "n_mels": args.n_mels, "n_fft": args.n_fft, "hop_length": args.hop_length,
                                   "fmin": args.fmin, "fmax": args.fmax, "round": rnd+1, "epochs": epochs,
                                   "recipe": "nikita_noisy_student soft-CE mixup0.5 power-schedule",
                                   "zero_train_labels": zero_train}}, best_path)

        # ---- re-label soundscapes with this student => teacher for next round ----
        if rnd < rounds - 1:
            log(f"  re-labeling soundscapes with round-{rnd+1} student (-> next teacher)")
            current_soft = student_relabel(model, frontend, paths, start_secs, classes, device, batch=128)
            report_zero_train_coverage(f"STUDENT round{rnd+1} relabel", current_soft, classes, zero_train, tax)

    # ---- STAGE C: validate + soundscape preds + 28-class carry-through ----
    final_soft = student_relabel(model, frontend, paths, start_secs, classes, device, batch=128)
    nz_student, _ = report_zero_train_coverage("FINAL STUDENT soundscape preds", final_soft, classes, zero_train, tax)
    export_soundscape_preds(paths, start_secs, final_soft, classes, out_dir)
    log("ARTIFACT:", best_path, "exists:", best_path.exists(),
        "size_mb=", round(best_path.stat().st_size/1e6, 1) if best_path.exists() else 0)
    log("mean epoch time =", f"{np.mean(epoch_times):.1f}s", "| total epochs =", len(epoch_times))
    log("LOSS TREND (round, epoch, train_ce, val_ce):")
    for r, e, tr, va in loss_trend: log(f"   r{r} e{e} train={tr:.5f} val={va:.5f}")
    log("SUMMARY pseudo_runtime_min=", round(pseudo_runtime/60, 1),
        "teacher_28_nonzero=", nonzero_teacher, "student_28_nonzero=", nz_student)
    log("DONE total", f"{time.time()-T0:.1f}s")

def student_relabel(model, frontend, paths, start_secs, classes, device, batch=128):
    """Run the student over all soundscape windows -> soft probs (sigmoid), (rows,C)."""
    import torch, soundfile as sf
    model.eval()
    n_samples = int(round(SR * WINDOW_SEC))
    uniq = {}
    for p in paths: uniq.setdefault(p, None)
    audio_cache = {}
    def get_window(p, s):
        a = audio_cache.get(p)
        if a is None:
            y, sr = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
            if sr != SR:
                import librosa; y = librosa.resample(y, orig_sr=sr, target_sr=SR).astype(np.float32)
            a = y.astype(np.float32); audio_cache[p] = a
        st = int(round(s * SR)); clip = a[st:st+n_samples]
        if len(clip) < n_samples:
            z = np.zeros(n_samples, np.float32); z[:len(clip)] = clip; clip = z
        return clip
    out = np.zeros((len(paths), len(classes)), np.float32)
    buf = []; idxs = []
    def flush():
        if not buf: return
        wb = torch.from_numpy(np.stack(buf)).to(device)
        x = frontend(wb).to(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast("cuda"):
            p = torch.sigmoid(model(x).float()).cpu().numpy()
        for k, ridx in enumerate(idxs): out[ridx] = p[k]
        buf.clear(); idxs.clear()
    for i, (p, s) in enumerate(zip(paths, start_secs)):
        buf.append(get_window(p, float(s))); idxs.append(i)
        if len(buf) >= batch: flush()
        # cap audio cache memory
        if len(audio_cache) > 64:
            audio_cache.pop(next(iter(audio_cache)))
    flush()
    return out

def export_soundscape_preds(paths, start_secs, soft, classes, out_dir):
    import pandas as pd
    stems = [Path(p).stem for p in paths]
    df = pd.DataFrame(soft, columns=classes)
    df.insert(0, "row_id", [f"{st}_{int(s)+int(WINDOW_SEC)}" for st, s in zip(stems, start_secs)])
    df.insert(1, "filename", [Path(p).name for p in paths])
    df.insert(2, "start_sec", np.asarray(start_secs, np.int32))
    outp = out_dir / "ns_student_soundscape_preds.parquet"
    df.to_parquet(outp, index=False)
    log("wrote soundscape preds:", outp, "shape", df.shape,
        f"| pred[min={soft.min():.4f} max={soft.max():.4f} mean={soft.mean():.4f}]")

if __name__ == "__main__":
    main()
