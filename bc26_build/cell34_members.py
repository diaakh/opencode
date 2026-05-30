# =====================================================================
# SOUNDSCAPE-ADAPTED STUDENT ENSEMBLE  (config-driven MEMBERS)
# Each member is an independent 1ch log-mel CNN, loaded via
#   timm.create_model(backbone, pretrained=False, in_chans=1, num_classes=234)
# run over the SAME test windows as the base, sigmoid -> 234 probs, and
# rank-blended into the V237 base at its own (low) weight in cell below.
# CPU, internet OFF: weights come from mounted Kaggle datasets.
# Robust: a member whose dataset/checkpoint is missing is SKIPPED.
# =====================================================================
import os, glob, time, json, resource
from pathlib import Path as _MPath
import numpy as _m_np
import pandas as _m_pd

# ---- TOP-OF-FILE CONFIG: ensemble members ---------------------------------
# Default weights are LOW (these students distill the SED already in the base;
# measure, do not assume).  Weights are renormalized at blend time.
# To add a member: append a dict here once its dataset is published.
MEMBERS = [
    {"name": "ns_student_b0",    "dataset_slug": "adkasd/bc26-ns-student-b0",
     "checkpoint_file": None, "backbone": "tf_efficientnet_b0",   "weight": 0.06},
    {"name": "ns_student_nfnet", "dataset_slug": "adkasd/bc26-ns-student-nfnet",
     "checkpoint_file": None, "backbone": "eca_nfnet_l0",         "weight": 0.06},
    {"name": "ns_student_effv2s","dataset_slug": "adkasd/bc26-ns-student-effv2s",
     "checkpoint_file": None, "backbone": "tf_efficientnetv2_s",  "weight": 0.06},
    # ---- LIVE BUILD-TIME member (exists today) ----------------------------
    # The orthogonal effv2s CNN from the validated base; included so the COMMIT
    # scale-test exercises the FULL member pipeline + budget on real data RIGHT
    # NOW (before the ns_student_* artifacts publish).  weight=0 -> it measures
    # cost but does NOT alter the submission until intentionally weighted.
    {"name": "g124_effv2s",      "dataset_slug": "adkasd/bc26-g124-cnn-effv2s",
     "checkpoint_file": "g124_fold1_fp16.pt", "backbone": "tf_efficientnetv2_s.in21k", "weight": 0.0},
]
# "checkpoint_file" None  -> auto-pick the single/first *.pt[h]/*.ckpt in the dataset.
# "checkpoint_file" "x.pt"-> match exactly that filename anywhere under /kaggle/input.

# COMMIT scale-test: when running over train_soundscapes (no real hidden test),
# run the FULL member pipeline over up to this many real files to measure peak
# RSS + projected 600-file runtime.  Must fit <=90 min CPU and ~13GB RAM.
_M_SCALETEST_MAX_FILES = 300

# ---- mel / audio params -- MUST match training (128 / 2048 / 512, 5s@32k) --
_M_SR      = 32000
_M_NMELS   = 128
_M_NFFT    = 2048
_M_HOP     = 512
_M_FMIN    = 20
_M_FMAX    = 16000
_M_CHUNK_S = 5
_M_CHUNK_N = _M_SR * _M_CHUNK_S          # 160000
_M_NCLS    = 234


def _m_peak_rss_gb():
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _m_find_ckpt(member):
    """Resolve a member's checkpoint path under /kaggle/input, or None if absent."""
    slug_leaf = member["dataset_slug"].split("/")[-1]
    cf = member.get("checkpoint_file")
    norm_leaf = slug_leaf.replace("-", "").replace("_", "").lower()
    if cf:
        # Exact filename, but ONLY if it lives under this member's own dataset
        # dir (so two members that share a checkpoint name can't collide).
        hits = sorted(_MPath("/kaggle/input").rglob(cf))
        own = [h for h in hits if norm_leaf in str(h.parent).replace("-", "").replace("_", "").lower()]
        pick = own or hits
        return str(pick[0]) if pick else None
    # auto-pick: ONLY checkpoints whose path belongs to THIS member's dataset.
    # Do NOT fall back to unrelated checkpoints -- a missing member must skip,
    # never silently borrow another member's (e.g. g124) weights.
    cands = []
    for ext in ("*.pt", "*.pth", "*.ckpt"):
        cands.extend(_MPath("/kaggle/input").rglob(ext))
    own = [p for p in cands if norm_leaf in str(p).replace("-", "").replace("_", "").lower()]
    pick = sorted(own)
    return str(pick[0]) if pick else None


# ---- locate test audio aligned to the base subm_74.csv row order ----------
_m_t0 = time.time()
_m_base_df = _m_pd.read_csv("subm_74.csv")
_m_base_df["row_id"] = _m_base_df["row_id"].astype(str)
_m_row_ids = _m_base_df["row_id"].tolist()

_m_file_order = []          # stems in first-seen order
_m_file_rows = {}           # stem -> [(end_sec, row_idx), ...]
for _ri, _rid in enumerate(_m_row_ids):
    _stem, _end = _rid.rsplit("_", 1)
    if _stem not in _m_file_rows:
        _m_file_rows[_stem] = []
        _m_file_order.append(_stem)
    _m_file_rows[_stem].append((int(_end), _ri))

_m_audio_roots = [
    _MPath("/kaggle/input/competitions/birdclef-2026/test_soundscapes"),
    _MPath("/kaggle/input/birdclef-2026/test_soundscapes"),
    _MPath("/kaggle/input/competitions/birdclef-2026/train_soundscapes"),
    _MPath("/kaggle/input/birdclef-2026/train_soundscapes"),
]
_m_path_for = {}
for _root in _m_audio_roots:
    if _root.is_dir():
        for _f in _root.glob("*.ogg"):
            _m_path_for.setdefault(_f.stem, _f)

_m_have_audio = (len(_m_file_order) > 0) and all(s in _m_path_for for s in _m_file_order)
print(f"[members] base rows={len(_m_row_ids)} files={len(_m_file_order)} "
      f"audio_resolved={_m_have_audio}")

# Decide run mode:
#   real test  -> populate _m_member_probs for the blend.
#   commit/dry -> base rows not audio-resolvable: run a SCALE-TEST over real
#                 train_soundscapes (no blend; submission untouched).
_m_scaletest_only = False
_m_run = True
if not _m_have_audio:
    _m_scaletest_only = True
    _m_scaletest_files = []
    for _root in _m_audio_roots:
        if _root.is_dir() and "train_soundscapes" in str(_root):
            _m_scaletest_files = sorted(_root.glob("*.ogg"))[:_M_SCALETEST_MAX_FILES]
            break
    if not _m_scaletest_files:
        print("[members] no train_soundscapes for scale-test -- skipping members.")
        _m_run = False
    else:
        print(f"[members][scaletest] base rows not audio-resolvable -- running FULL "
              f"member pipeline over {len(_m_scaletest_files)} real train_soundscapes; "
              f"submission left unchanged.")

# Holds {name: (probs_ndarray, weight)} for the blend cell.  Real-test only.
_m_member_probs = {}

if _m_run:
    import torch as _m_torch
    import timm as _m_timm
    import librosa as _m_librosa
    _m_torch.set_num_threads(4)

    # Replicate the STUDENT's training MelFrontend EXACTLY (train_g124.MelFrontend):
    #   torchaudio.MelSpectrogram(power=2.0)  ->  clamp_min(1e-6).log()  (NATURAL log)
    #   -> per-sample standardize over (freq,time).  Also DC-removes the waveform.
    # This must match training; do NOT substitute librosa dB-scaling (10*log10/top_db).
    import torchaudio as _m_ta
    _m_melspec = _m_ta.transforms.MelSpectrogram(
        sample_rate=_M_SR, n_fft=_M_NFFT, hop_length=_M_HOP, n_mels=_M_NMELS,
        f_min=float(_M_FMIN), f_max=float(_M_FMAX), power=2.0)

    def _m_audio_to_mel(chunks):
        wave = _m_torch.from_numpy(_m_np.ascontiguousarray(chunks)).float()  # (nwin, samples)
        wave = _m_torch.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
        wave = wave - wave.mean(dim=1, keepdim=True)
        mel = _m_melspec(wave)                       # (nwin, n_mels, frames)
        mel = mel.clamp_min(1e-6).log()
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
        mel = ((mel - mean) / std).unsqueeze(1)      # (nwin, 1, n_mels, frames)
        return mel.numpy().astype(_m_np.float32)

    def _m_load_audio(path):
        try:
            import soundfile as _sf
            y, sr0 = _sf.read(str(path), dtype="float32", always_2d=False)
            if getattr(y, "ndim", 1) == 2:
                y = y.mean(axis=1)
            if sr0 != _M_SR:
                y = _m_librosa.resample(y, orig_sr=sr0, target_sr=_M_SR)
        except Exception:
            y, _ = _m_librosa.load(str(path), sr=_M_SR, mono=True)
        return y.astype(_m_np.float32)

    def _m_file_to_chunks(path, n_win):
        y = _m_load_audio(path)
        need = n_win * _M_CHUNK_N
        if len(y) < need:
            y = _m_np.pad(y, (0, need - len(y)))
        else:
            y = y[:need]
        return y.reshape(n_win, _M_CHUNK_N).astype(_m_np.float32)

    def _m_build_model(member, ckpt_path):
        ck = _m_torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        # strip common prefixes
        clean = {}
        for k, v in state.items():
            nk = k
            for pre in ("module.", "model.", "net.", "backbone."):
                if nk.startswith(pre):
                    nk = nk[len(pre):]
            clean[nk] = v.float() if hasattr(v, "float") else v
        model = _m_timm.create_model(
            member["backbone"], pretrained=False, in_chans=1, num_classes=_M_NCLS)
        miss, unexp = model.load_state_dict(clean, strict=False)
        model.eval()
        return model, len(miss), len(unexp), list(miss)[:5], list(unexp)[:5]

    def _m_infer_file(model, path, n_win):
        chunks = _m_file_to_chunks(path, n_win)
        mel = _m_audio_to_mel(chunks)
        with _m_torch.no_grad():
            p = _m_torch.sigmoid(model(_m_torch.from_numpy(mel))).numpy().astype(_m_np.float32)
        return p

    # ---- iterate configured members -------------------------------------
    _m_loaded, _m_skipped = [], []
    for _mem in MEMBERS:
        _w = float(_mem.get("weight", 0.0))
        # In REAL test, a zero-weight member would never affect the blend, so
        # skip its (costly) inference.  In SCALE-TEST we still run it to measure
        # the full per-member budget.
        if (not _m_scaletest_only) and _w <= 0.0:
            print(f"[members] SKIP {_mem['name']}: weight=0 (no effect on submission).")
            continue
        _ckpt = _m_find_ckpt(_mem)
        if _ckpt is None:
            print(f"[members] SKIP {_mem['name']}: no checkpoint for "
                  f"dataset '{_mem['dataset_slug']}' (artifact not attached).")
            _m_skipped.append(_mem["name"])
            continue
        try:
            _model, _nm, _nu, _ms, _us = _m_build_model(_mem, _ckpt)
        except Exception as _e:
            print(f"[members] SKIP {_mem['name']}: load failed on backbone "
                  f"'{_mem['backbone']}' ({type(_e).__name__}: {_e}).")
            _m_skipped.append(_mem["name"])
            continue
        print(f"[members] LOADED {_mem['name']} <- {os.path.basename(_ckpt)} "
              f"backbone={_mem['backbone']} w={_w:.3f} missing={_nm} unexpected={_nu}")
        if _nm:
            print(f"[members]   missing(sample)={_ms}")
        if _nu:
            print(f"[members]   unexpected(sample)={_us}")

        if _m_scaletest_only:
            # FULL pipeline over real train_soundscapes; measure budget. No blend.
            _ti = time.time()
            _acc = []
            for _f in _m_scaletest_files:
                _acc.append(_m_infer_file(_model, _f, 12))   # 12 windows = 60s file
            _arr = _m_np.concatenate(_acc, axis=0)
            _dt = time.time() - _ti
            _per = _dt / max(len(_m_scaletest_files), 1)
            print(f"[members][scaletest] {_mem['name']}: ran {len(_m_scaletest_files)} files "
                  f"in {_dt:.1f}s ({_per:.3f}s/file) "
                  f"probs mean={_arr.mean():.5f} std={_arr.std():.5f} "
                  f"min={_arr.min():.5f} max={_arr.max():.5f} "
                  f"frac>0.5={float((_arr>0.5).mean()):.5f}")
            assert _arr.std() > 1e-6, f"[members] degenerate output for {_mem['name']}"
            _m_loaded.append((_mem["name"], _per))
            del _acc, _arr, _model
        else:
            # REAL test: aligned (n_rows, 234) array for this member.
            _probs = _m_np.zeros((len(_m_row_ids), _M_NCLS), dtype=_m_np.float32)
            _nfiles = len(_m_file_order)
            for _fi, _stem in enumerate(_m_file_order, 1):
                _rows = _m_file_rows[_stem]
                _nwin = len(_rows)
                _p = _m_infer_file(_model, _m_path_for[_stem], _nwin)
                _order = sorted(range(_nwin), key=lambda j: _rows[j][0])
                for _wi, _j in enumerate(_order):
                    _probs[_rows[_j][1]] = _p[_wi]
                if _fi == 1 or _fi % 50 == 0 or _fi == _nfiles:
                    _el = time.time() - _m_t0
                    print(f"[members] {_mem['name']} {_fi}/{_nfiles} files {_el:.1f}s")
            assert _probs.std() > 1e-6, f"[members] degenerate output for {_mem['name']}"
            print(f"[members] {_mem['name']} probs: mean={_probs.mean():.5f} "
                  f"std={_probs.std():.5f} min={_probs.min():.5f} max={_probs.max():.5f} "
                  f"frac>0.5={float((_probs>0.5).mean()):.5f}")
            _m_member_probs[_mem["name"]] = (_probs, _w)
            del _model

    # ---- budget summary --------------------------------------------------
    _m_peak = _m_peak_rss_gb()
    if _m_scaletest_only:
        _m_total_per = sum(p for _, p in _m_loaded)          # sum s/file over members
        _m_proj_min = (_m_total_per * 600) / 60.0
        print("=" * 64)
        print(f"[members][scaletest] members loaded: {[n for n,_ in _m_loaded]}")
        print(f"[members][scaletest] members skipped: {_m_skipped}")
        print(f"[members][scaletest] per-file (all members summed)="
              f"{_m_total_per:.3f}s -> ALL-MEMBER 600-file projection ~{_m_proj_min:.1f} min")
        print(f"[members][scaletest] PEAK RSS so far = {_m_peak:.2f} GB "
              f"(budget: <=90 min CPU, ~13 GB RAM)")
        print(f"[members][scaletest] NOTE: base SED/Perch pipeline runs separately; "
              f"this measures the member branch contribution on top of the base.")
        print("=" * 64)
    else:
        print(f"[members] real-test: blended-in members = {list(_m_member_probs.keys())}, "
              f"skipped = {_m_skipped}, peak RSS = {_m_peak:.2f} GB")
