# BirdCLEF+ 2026 - Nikita Babych "zero-train-class specialist" (the 28 classes with NO train_audio).
#
# GOAL (his disclosed +0.005-0.015 lever): a dedicated EfficientNet-B0, 234-class, trained on
# focal train_audio PLUS Xeno-Canto extra-data clips that cover the zero-train species, so the
# specialist produces NON-ZERO predictions for the 28 classes that the main pipeline cannot
# learn. Its preds get slotted into a zero matrix and rank-blended.
#
# CRITICAL MAPPING FINDING (resolved offline against taxonomy.csv + the extra-data CSV):
#   The 28 zero-train classes split 25 Insecta + 3 Amphibia.
#     - 25 Insecta = anonymized 2026 SONOTYPES `47158sonNN` (all share iNat id 47158, scientific
#       name literally "Insect sonNN"). They have NO resolvable species name, so NO Xeno-Canto
#       clip can be mapped to them. Nikita's 16,218 grasshopper clips are European named species
#       (Gryllotalpa vineae, ...) that do not correspond to these Neotropical sonotypes. -> 0/25.
#     - 3 Amphibia (Adenomera guarani, Chiasmocleis mehelyi, Pithecopus azureus) have NO exact
#       extra-data match, but the extra-data DOES carry GENUS siblings (Adenomera andreae/heyeri,
#       Chiasmocleis haddadi, Pithecopus rohdei). We map those genus-proxy clips to the zero-train
#       label as a best-effort acoustic prior. -> up to 3/28 get (proxy) real frog audio.
#   So the honest, achievable coverage is reported per-class at runtime. This kernel does the
#   training + the zero-matrix-slot sanity regardless, so the mechanism is proven and the exact
#   coverage (and the Insecta-sonotype blocker) is logged.
#
# Reuses train_g124.py building blocks (AudioDataset, MelFrontend, build_model, collate_audio).
# GPU + internet on. P100 sm_60 torch fix applied before any torch import (same as bc26-cnn-train-gpu).

import os, sys, glob, time, subprocess, random, re
from pathlib import Path
import numpy as np

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# --------------------------------------------------------------- GPU / torch fix (P100 sm_60)
TORCH_PIN = ("torch==2.7.1", "torchaudio==2.7.1", "torchvision==0.22.1")
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"

def install_torch():
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                          "--format=csv"], capture_output=True, text=True).stdout, flush=True)
    log("installing", TORCH_PIN, "from", TORCH_INDEX, "(P100 sm_60 fix)")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *TORCH_PIN,
                        "--index-url", TORCH_INDEX], capture_output=True, text=True)
    log("pip rc", r.returncode)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        raise RuntimeError("torch 2.7.1+cu126 install failed")

def verify_gpu():
    import torch, torch.nn as nn
    log("torch", torch.__version__, "cuda_build", torch.version.cuda,
        "is_available", torch.cuda.is_available(), "arch_list", torch.cuda.get_arch_list())
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False - no GPU visible to torch")
    log("device", torch.cuda.get_device_name(0), "cap sm_", torch.cuda.get_device_capability(0))
    a = torch.randn(1024, 1024, device="cuda"); c = a @ a
    m = nn.Conv2d(1, 8, 3, padding=1).cuda().to(memory_format=torch.channels_last)
    x = torch.randn(4, 1, 128, 313, device="cuda").to(memory_format=torch.channels_last)
    with torch.autocast("cuda"):
        y = m(x)
    torch.cuda.synchronize()
    _ = float(c.sum()) + float(y.sum())
    log("CUDA real matmul+conv+AMP OK -> training will run on GPU")


def find_competition_dir():
    for c in ("/kaggle/input/competitions/birdclef-2026", "/kaggle/input/birdclef-2026"):
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError("birdclef-2026 competition not attached")

def find_code_root():
    for c in glob.glob("/kaggle/input/**/train_g124.py", recursive=True):
        return Path(c).parent
    raise FileNotFoundError("birdclef-g124-code (train_g124.py) not attached")

def find_extra_data_root():
    # nikitababich/birdclef2025-1st-place-extra-data mounted under /kaggle/input/<slug>/
    for c in glob.glob("/kaggle/input/**/birdclef2025_extra_species_data.csv", recursive=True):
        return Path(c).parent
    return None


# --------------------------------------------------------------- extra-data -> 234-label mapping
def build_extra_frame(extra_root, tax_df, zero_train_labels, max_per_species=200, max_clip_seconds=60.0):
    """Map Xeno-Canto extra-data clips to the 234-class space.

    Strategy (all by scientific name from taxonomy.csv):
      1. EXACT species match  -> map clip to that label (augments classes that already have
         train_audio; also catches any zero-train exact match if one exists).
      2. GENUS proxy for the zero-train AMPHIBIA only -> map a same-genus extra-data clip to the
         zero-train label, so those zero-train heads receive a real acoustic prior. Insecta
         sonotypes have no resolvable genus, so they get nothing here (logged).
    Returns (rows list, coverage dict label->(n_clips, mode)).
    """
    import pandas as pd
    csv = extra_root / "birdclef2025_extra_species_data.csv"
    xc = pd.read_csv(csv)
    xc["sci"] = (xc["gen"].astype(str).str.strip() + " " + xc["sp"].astype(str).str.strip()).str.lower()
    xc["genus"] = xc["gen"].astype(str).str.strip().str.lower()
    # clip length filter (<60s) using the "length" mm:ss column
    def secs(v):
        try:
            p = str(v).split(":")
            return float(p[0]) * 60 + float(p[1]) if len(p) == 2 else float(v)
        except Exception:
            return 0.0
    xc["len_s"] = xc["length"].map(secs)
    xc = xc[(xc["len_s"] > 0) & (xc["len_s"] <= max_clip_seconds)].copy()

    tax = tax_df.copy()
    tax["sci"] = tax["scientific_name"].str.lower().str.strip()
    tax["genus"] = tax["scientific_name"].str.split().str[0].str.lower()
    sci_to_label = dict(zip(tax["sci"], tax["primary_label"].astype(str)))
    zero_set = set(zero_train_labels)

    rows = []
    coverage = {}  # label -> (count, mode)

    def add(label, sub, mode):
        sub = sub.head(max_per_species)
        n = 0
        for fp in sub["filepath"].astype(str):
            path = extra_root / fp
            if path.exists():
                rows.append({"path": str(path), "primary_label": str(label),
                             "source": f"xc_{mode}"})
                n += 1
        if n:
            prev = coverage.get(label, (0, mode))
            coverage[label] = (prev[0] + n, mode if prev[0] == 0 else prev[1])
        return n

    # 1. exact species matches
    for sci, sub in xc.groupby("sci"):
        if sci in sci_to_label:
            add(sci_to_label[sci], sub, "exact")

    # 2. genus proxy ONLY for zero-train amphibia (skip if exact already covered it)
    zt = tax[tax["primary_label"].astype(str).isin(zero_set)]
    for _, r in zt.iterrows():
        label = str(r["primary_label"])
        if label in coverage:        # already got exact clips
            continue
        if str(r["class_name"]) != "Amphibia":
            continue                 # insecta sonotypes: no resolvable genus, skip
        genus = r["genus"]
        sub = xc[xc["genus"] == genus]
        if len(sub):
            # cap proxy clips modestly so a genus prior does not dominate
            add(label, sub, "genus_proxy")
    return rows, coverage


def main():
    install_torch()  # before any torch import
    comp = find_competition_dir()
    code_root = find_code_root()
    extra_root = find_extra_data_root()
    log("competition:", comp, "| code:", code_root, "| extra_data:", extra_root)
    verify_gpu()

    sys.path.insert(0, str(code_root))
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    from train_g124 import (build_parser, load_classes, AudioDataset, MelFrontend,
                            build_model, collate_audio, add_folds, split_train_val,
                            build_train_audio_frame)

    classes = load_classes(comp)
    tax = pd.read_csv(comp / "taxonomy.csv", dtype={"primary_label": str, "inat_taxon_id": str})
    train_csv = pd.read_csv(comp / "train.csv", dtype={"primary_label": str})
    labels_with_audio = set(train_csv["primary_label"].unique())
    zero_train = sorted(set(tax["primary_label"].astype(str)) - labels_with_audio)
    zt_idx = [classes.index(l) for l in zero_train if l in classes]
    log(f"classes={len(classes)} zero_train={len(zero_train)} (expected 28)")
    ztd = tax[tax["primary_label"].astype(str).isin(zero_train)]
    log("zero-train by class:", ztd["class_name"].value_counts().to_dict())

    # ----- build training frame -----
    config = dict(sr=32000, window_seconds=5.0, n_mels=128, n_fft=2048, hop_length=512,
                  fmin=20.0, fmax=16000.0)
    max_files = int(os.environ.get("SPEC_MAX_TRAIN_FILES", "120"))  # files/class cap for train_audio
    epochs = int(os.environ.get("SPEC_EPOCHS", "18"))               # scaled to fit 12h (Nikita 40)
    batch = int(os.environ.get("SPEC_BATCH", "128"))
    workers = int(os.environ.get("SPEC_WORKERS", "3"))
    model_name = os.environ.get("SPEC_MODEL", "tf_efficientnet_b0")
    max_per_species = int(os.environ.get("SPEC_XC_PER_SPECIES", "200"))

    ta = build_train_audio_frame(comp, classes, max_files=max_files)
    ta["source"] = "train_audio"
    log(f"train_audio frame: {len(ta)} files over {ta['primary_label'].nunique()} classes")

    extra_rows, coverage = ([], {})
    if extra_root is not None:
        extra_rows, coverage = build_extra_frame(extra_root, tax, zero_train,
                                                  max_per_species=max_per_species)
        log(f"extra-data frame: {len(extra_rows)} clips mapped to {len(coverage)} labels")
    else:
        log("WARNING: extra-data dataset NOT mounted; specialist falls back to train_audio only")

    extra_df = pd.DataFrame(extra_rows) if extra_rows else pd.DataFrame(columns=ta.columns)
    frame = pd.concat([ta, extra_df], ignore_index=True)

    # ----- CRITICAL: how many of the 28 zero-train classes got REAL audio -----
    zt_real = {}
    for l in zero_train:
        n = int((frame["primary_label"] == l).sum())
        if n > 0:
            mode = coverage.get(l, (n, "?"))[1]
            zt_real[l] = (n, mode)
    log("=" * 60)
    log(f"ZERO-TRAIN AUDIO COVERAGE: {len(zt_real)}/{len(zero_train)} of the 28 got real/proxy clips")
    for l in zero_train:
        sci = tax.loc[tax["primary_label"].astype(str) == l, "scientific_name"].iloc[0]
        cls = tax.loc[tax["primary_label"].astype(str) == l, "class_name"].iloc[0]
        if l in zt_real:
            log(f"   {l:>10} {cls:<9} {sci:<25} -> {zt_real[l][0]} clips ({zt_real[l][1]})")
        else:
            log(f"   {l:>10} {cls:<9} {sci:<25} -> 0 clips (UNMAPPABLE)")
    log("=" * 60)

    # min-1-sample/class is naturally satisfied for the 206 + whatever zero-train got clips.
    frame = add_folds(frame, n_folds=5, seed=124)
    train_frame, val_frame, vfold = split_train_val(frame, requested_fold=1)
    log(f"train={len(train_frame)} val={len(val_frame)} val_fold={vfold}")

    # ----- model / frontend -----
    class A:  # lightweight args namespace for AudioDataset / MelFrontend
        pass
    args = A()
    for k, v in config.items():
        setattr(args, k, v)
    args.sr = config["sr"]
    args.audio_cache_mb = 0

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = build_model(model_name, len(classes), pretrained_checkpoint=None, timm_pretrained=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    frontend = MelFrontend(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    train_loader = DataLoader(AudioDataset(train_frame, classes, args, training=True),
                              batch_size=batch, shuffle=True, num_workers=workers,
                              pin_memory=True, persistent_workers=workers > 0,
                              collate_fn=collate_audio, drop_last=True, prefetch_factor=2)
    val_loader = DataLoader(AudioDataset(val_frame, classes, args, training=False),
                            batch_size=batch, shuffle=False, num_workers=workers,
                            pin_memory=True, persistent_workers=workers > 0,
                            collate_fn=collate_audio, prefetch_factor=2)

    out_dir = Path("/kaggle/working/zerotrain_specialist")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "zerotrain_specialist_b0_fold1_fp16.pt"

    # ----- train -----
    epoch_times = []
    for epoch in range(epochs):
        t0 = time.time(); model.train(); tl = []
        for wave, target, weight in train_loader:
            wave = wave.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            x = frontend(wave).to(memory_format=torch.channels_last)
            with torch.autocast("cuda"):
                logits = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite train loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            tl.append(float(loss.detach().cpu()))
        scheduler.step()
        # val
        model.eval(); vl = []
        with torch.inference_mode():
            for wave, target, _w in val_loader:
                wave = wave.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
                x = frontend(wave).to(memory_format=torch.channels_last)
                logits = model(x)
                vl.append(float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target).cpu()))
        dt = time.time() - t0; epoch_times.append(dt)
        eta = (epochs - epoch - 1) * float(np.mean(epoch_times))
        log(f"epoch={epoch+1}/{epochs} train_loss={np.mean(tl):.5f} val_loss={np.mean(vl):.5f} "
            f"time={dt:.1f}s eta_remaining={eta/60:.1f}min")
        # save each epoch (last = artifact)
        state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
        torch.save({"state_dict": state, "classes": classes,
                    "config": {"model_name": model_name, **config,
                               "zero_train_labels": zero_train,
                               "zero_train_real_coverage": {k: v[0] for k, v in zt_real.items()},
                               "epochs": epochs, "batch_size": batch}},
                   best_path)
    log("TRAINED artifact:", best_path, "exists:", best_path.exists())
    log(f"mean epoch time = {np.mean(epoch_times):.1f}s; total train ~ {sum(epoch_times)/60:.1f}min")

    # ----- 28-class prediction sanity + orthogonality export -----
    sanity_and_export(comp, model, frontend, classes, args, zero_train, zt_idx, out_dir, device, tax)
    log("DONE total", f"{time.time()-T0:.1f}s")


def sanity_and_export(comp, model, frontend, classes, args, zero_train, zt_idx, out_dir, device, tax):
    """Run the trained model on a deterministic train_audio sample; report (a) per-class positive
    coverage for the 28 + logit stats (proving NON-ZERO preds), and (b) save a preds parquet for
    orthogonality checking."""
    import torch, pandas as pd, soundfile as sf
    model.eval()
    n_samples = int(round(args.sr * args.window_seconds))
    train_audio = comp / "train_audio"
    per_class = 3
    waves, meta = [], []
    for label in classes:
        d = train_audio / label
        if not d.exists():
            continue
        for f in sorted(d.glob("*.ogg"))[:per_class]:
            try:
                y, sr = sf.read(str(f), dtype="float32", always_2d=False)
            except Exception:
                continue
            if y.ndim > 1: y = y.mean(axis=1)
            if sr != args.sr:
                import librosa; y = librosa.resample(y, orig_sr=sr, target_sr=args.sr).astype(np.float32)
            if len(y) < n_samples:
                clip = np.zeros(n_samples, dtype=np.float32); clip[:len(y)] = y
            else:
                s = (len(y) - n_samples) // 2; clip = y[s:s + n_samples]
            waves.append(clip.astype(np.float32)); meta.append({"filename": f.name, "true_label": label})
    log(f"sanity/export: {len(waves)} windows")
    preds = []
    with torch.inference_mode():
        for i in range(0, len(waves), 128):
            wb = torch.from_numpy(np.stack(waves[i:i+128])).to(device)
            x = frontend(wb).to(memory_format=torch.channels_last)
            with torch.autocast("cuda"):
                logits = model(x)
            preds.append(logits.float().cpu().numpy())
    L = np.concatenate(preds, axis=0)          # raw logits
    P = 1.0 / (1.0 + np.exp(-L))               # probs

    # --- 28-class sanity: these windows are 206-class true labels (the 28 are never the truth
    # here), so a healthy specialist should give them LOW-but-NONZERO probs, with spread (not a
    # dead constant). We report per-class mean/max prob + #windows >0.05. ---
    log("=" * 60)
    log("28 ZERO-TRAIN CLASS PREDICTION SANITY (on train_audio sample; truth is never one of the 28):")
    nonzero_heads = 0
    for j, l in zip(zt_idx, zero_train):
        col = P[:, j]
        sci = tax.loc[tax["primary_label"].astype(str) == l, "scientific_name"].iloc[0]
        active = int((col > 0.05).sum())
        spread = float(col.max() - col.min())
        if col.max() > 1e-4 and spread > 1e-5:
            nonzero_heads += 1
        log(f"   {l:>10} {sci:<24} logit[mean={L[:,j].mean():+.3f} max={L[:,j].max():+.3f}] "
            f"prob[mean={col.mean():.5f} max={col.max():.5f}] active>0.05={active}")
    log(f"NON-ZERO 28-class heads: {nonzero_heads}/{len(zero_train)} (heads producing real spread, "
        f"i.e. NOT a dead all-zero column when slotted into the zero matrix)")
    log("=" * 60)

    df = pd.DataFrame(P, columns=classes)
    df.insert(0, "true_label", [m["true_label"] for m in meta])
    df.insert(0, "filename", [m["filename"] for m in meta])
    outp = out_dir / "zerotrain_specialist_trainaudio_preds.parquet"
    df.to_parquet(outp, index=False)
    top1 = (P.argmax(1) == np.array([classes.index(m["true_label"]) for m in meta])).mean()
    log("wrote preds:", outp, "shape", df.shape, f"| single-window top1={top1:.4f}")


if __name__ == "__main__":
    main()
