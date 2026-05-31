import os, sys, subprocess

# =====================================================================
# Self-bootstrapping P100 sm_60 fix (proven g124 launcher+subprocess pattern).
# First invocation: pip-install torch 2.7.1 cu126, then RE-EXEC this same file
# as a FRESH subprocess. Installing torch in-process leaves a mismatched
# torchvision binary that crashes timm's import (operator torchvision::nms
# does not exist), so we must do all torch/timm imports in the child process.
# =====================================================================
if os.environ.get("SED_BOOTSTRAPPED") != "1" and os.path.isdir("/kaggle"):
    print("[bootstrap] installing torch 2.7.1 cu126 (P100 sm_60 fix)...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                    "torch==2.7.1", "torchaudio==2.7.1", "torchvision==0.22.1",
                    "--index-url", "https://download.pytorch.org/whl/cu126"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.15"], check=False)
    env = os.environ.copy()
    env["SED_BOOTSTRAPPED"] = "1"
    env.setdefault("SED_BACKBONE", "eca_nfnet_l0")
    env.setdefault("SED_RUN", "nfnet")
    env.setdefault("SED_SEED", "1337")
    print("[bootstrap] re-exec trainer as fresh subprocess", flush=True)
    sys.exit(subprocess.run([sys.executable, os.path.abspath(__file__)], env=env).returncode)

import math, time, json, random, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import librosa

# =====================================================================
# Phase-1 independent focal SED backbone (Nikita method, AttBlockV2)
# Backbone + run name are injected by writing CFG_BACKBONE / CFG_RUN
# at the top of this file before pushing (see build step).
# =====================================================================
CFG_BACKBONE = os.environ.get("SED_BACKBONE", "eca_nfnet_l0")
CFG_RUN      = os.environ.get("SED_RUN", "nfnet")
CFG_SEED     = int(os.environ.get("SED_SEED", "1337"))

class CFG:
    seed = CFG_SEED
    sr = 32000
    n_fft = 2048
    hop = 512
    n_mels = 128
    fmin = 20
    fmax = 16000
    win_sec = 5.0
    samples = int(sr * win_sec)
    n_classes = None
    backbone = CFG_BACKBONE
    run = CFG_RUN
    in_chans = 1
    batch_size = 24
    epochs = 12
    lr = 1e-3
    weight_decay = 1e-2
    warmup_epochs = 1
    num_workers = 4
    data_root = "/kaggle/input/birdclef-2026"
    out_dir = "/kaggle/working"
    use_amp = True
    label_col = "primary_label"
    audio_dir = None
    label_smoothing = 0.05
    mixup_alpha = 0.5
    mixup_p = 0.5
    specaug_p = 0.7
    time_mask_w = 40
    freq_mask_w = 24
    gauss_p = 0.5
    gauss_std = 0.02
    max_per_class = 200   # cap for balanced oversampling weight
    val_frac = 0.04

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------------- Data ----------------
class BirdDataset(Dataset):
    def __init__(self, df, labels, cfg, train=True):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.cfg = cfg
        self.train = train
        self.label_to_idx = {l: i for i, l in enumerate(labels)}

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path):
        try:
            y, _ = librosa.load(path, sr=self.cfg.sr, mono=True)
        except Exception:
            y = np.zeros(self.cfg.samples, dtype=np.float32)
        if y is None or len(y) == 0:
            y = np.zeros(self.cfg.samples, dtype=np.float32)
        return y

    def _crop_or_pad(self, y):
        n = self.cfg.samples
        if len(y) >= n:
            start = random.randint(0, len(y) - n) if self.train else 0
            y = y[start:start+n]
        else:
            y = np.pad(y, (0, n - len(y)), mode="constant")
        return y

    def _melspec(self, y):
        m = librosa.feature.melspectrogram(
            y=y, sr=self.cfg.sr, n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop, n_mels=self.cfg.n_mels,
            fmin=self.cfg.fmin, fmax=self.cfg.fmax, power=2.0)
        m = librosa.power_to_db(m, ref=np.max)
        m = (m - m.mean()) / (m.std() + 1e-6)
        return m.astype(np.float32)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = os.path.join(self.cfg.audio_dir, row["filename"])
        y = self._load_audio(path)
        # waveform gaussian noise aug
        if self.train and random.random() < self.cfg.gauss_p:
            y = y + np.random.randn(len(y)).astype(np.float32) * self.cfg.gauss_std
        y = self._crop_or_pad(y)
        m = self._melspec(y)
        x = torch.from_numpy(m).unsqueeze(0)  # (1, mel, T)
        label = self.label_to_idx[row[self.cfg.label_col]]
        return x, label

# ---------------- SED head (AttBlockV2) ----------------
class AttBlockV2(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.att = nn.Conv1d(in_features, out_features, 1, bias=True)
        self.cla = nn.Conv1d(in_features, out_features, 1, bias=True)

    def forward(self, x):
        # x: (B, C, T)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = torch.sigmoid(self.cla(x))
        clipwise = torch.sum(norm_att * cla, dim=-1)
        return clipwise, norm_att, cla

class BirdSEDModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(cfg.n_mels)
        self.encoder = timm.create_model(
            cfg.backbone, pretrained=True, in_chans=cfg.in_chans,
            num_classes=0, global_pool="")
        feat = self.encoder.num_features
        self.fc1 = nn.Linear(feat, feat, bias=True)
        self.att_block = AttBlockV2(feat, cfg.n_classes)

    def forward(self, x):
        # x: (B, 1, mel, T)
        # bn over mel bins
        x = x.transpose(1, 2)            # (B, mel, 1, T)
        x = self.bn0(x)
        x = x.transpose(1, 2)            # (B, 1, mel, T)
        feat = self.encoder(x)           # (B, C, H, W)
        feat = torch.mean(feat, dim=2)   # pool freq -> (B, C, T)
        x1 = F.relu(self.fc1(feat.transpose(1, 2)).transpose(1, 2))  # (B, C, T)
        clipwise, _, _ = self.att_block(x1)  # (B, n_classes) in [0,1]
        return clipwise

def mixup_data(x, y_onehot, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return mixed_x, mixed_y

def spec_augment(x, cfg):
    # x: (B,1,mel,T)
    B, _, M, T = x.shape
    for _ in range(2):
        if random.random() < cfg.specaug_p:
            w = random.randint(0, cfg.freq_mask_w)
            f0 = random.randint(0, max(0, M - w))
            x[:, :, f0:f0+w, :] = 0
        if random.random() < cfg.specaug_p:
            w = random.randint(0, cfg.time_mask_w)
            t0 = random.randint(0, max(0, T - w))
            x[:, :, :, t0:t0+w] = 0
    return x

def main():
    set_seed(CFG.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("RUN", CFG.run, "BACKBONE", CFG.backbone, "SEED", CFG.seed, flush=True)
    print("torch", torch.__version__, "cuda avail", torch.cuda.is_available(), flush=True)
    # ---- P100 sm_60 real sanity check ----
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0), flush=True)
        a = torch.randn(128, 128, device="cuda"); b = torch.randn(128, 128, device="cuda")
        c = a @ b
        conv = nn.Conv2d(1, 8, 3).cuda(); t = torch.randn(2, 1, 32, 32, device="cuda")
        with torch.cuda.amp.autocast():
            o = conv(t)
        torch.cuda.synchronize()
        print("CUDA SANITY OK", float(c.sum().item() == c.sum().item()), c.shape, o.shape, flush=True)
    else:
        print("WARNING: no CUDA", flush=True)

    # ---- robust competition-data root detection (mount path varies) ----
    roots = [
        CFG.data_root,
        "/kaggle/input/birdclef-2026",
        "/kaggle/input/competitions/birdclef-2026",
    ]
    for g in sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*")):
        if os.path.isfile(os.path.join(g, "train.csv")):
            roots.append(g)
    chosen = None
    for r in roots:
        if r and os.path.isfile(os.path.join(r, "train.csv")):
            chosen = r; break
    if chosen is None:
        print("DEBUG /kaggle/input listing:", flush=True)
        for g in glob.glob("/kaggle/input/*"):
            print("  ", g, os.listdir(g)[:8] if os.path.isdir(g) else "(file)", flush=True)
        raise FileNotFoundError("train.csv not found under any /kaggle/input root")
    CFG.data_root = chosen
    print("data_root", CFG.data_root, flush=True)
    df = pd.read_csv(os.path.join(CFG.data_root, "train.csv"))
    print("train rows", len(df), "cols", list(df.columns), flush=True)
    labels = sorted(df[CFG.label_col].unique().tolist())
    CFG.n_classes = len(labels)
    print("n_classes", CFG.n_classes, flush=True)

    for cand in ["train_audio", "train_audio_2026", "audio"]:
        p = os.path.join(CFG.data_root, cand)
        if os.path.isdir(p):
            CFG.audio_dir = p; break
    print("audio_dir", CFG.audio_dir, flush=True)

    from sklearn.model_selection import train_test_split
    vc = df[CFG.label_col].value_counts()
    strat = df[CFG.label_col] if vc.min() > 1 else None
    tr_df, va_df = train_test_split(df, test_size=CFG.val_frac, random_state=CFG.seed, stratify=strat)
    print("train/val", len(tr_df), len(va_df), flush=True)

    tr_ds = BirdDataset(tr_df, labels, CFG, train=True)
    va_ds = BirdDataset(va_df, labels, CFG, train=False)

    # balanced sampler: weight = 1/sqrt(class_count) capped
    cls_counts = tr_df[CFG.label_col].map(tr_df[CFG.label_col].value_counts())
    weights = (1.0 / np.sqrt(np.minimum(cls_counts.values, CFG.max_per_class))).astype(np.float64)
    sampler = WeightedRandomSampler(weights, num_samples=len(tr_df), replacement=True)

    tr_dl = DataLoader(tr_ds, batch_size=CFG.batch_size, sampler=sampler,
                       num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False,
                       num_workers=CFG.num_workers, pin_memory=True)

    model = BirdSEDModel(CFG).to(device).to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    steps = len(tr_dl); total = steps * CFG.epochs; warm = steps * CFG.warmup_epochs
    def lr_lambda(s):
        if s < warm: return s / max(1, warm)
        prog = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_amp)

    def bce_ls(pred, target):
        # pred in [0,1] (already sigmoid'd in AttBlock). BCE on soft multi-hot.
        target = target * (1 - CFG.label_smoothing) + CFG.label_smoothing / CFG.n_classes
        pred = pred.clamp(1e-7, 1 - 1e-7)
        return F.binary_cross_entropy(pred, target)

    eye = torch.eye(CFG.n_classes, device=device)
    best_val = 1e9
    t_start = time.time()
    for epoch in range(CFG.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for it, (x, y) in enumerate(tr_dl):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            yoh = eye[y]
            if random.random() < CFG.mixup_p:
                x, yoh = mixup_data(x, yoh, CFG.mixup_alpha)
            x = spec_augment(x, CFG)
            with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                out = model(x)
                loss = bce_ls(out, yoh)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if it % 50 == 0:
                el = time.time() - t_start
                done = epoch * steps + it + 1
                eta = el / done * (total - done) / 3600.0
                print(f"ep{epoch} it{it}/{steps} loss{loss.item():.4f} lr{sched.get_last_lr()[0]:.2e} eta{eta:.2f}h", flush=True)
        tr_loss = run / max(1, steps)

        model.eval(); vrun = 0.0; n = 0; correct = 0
        with torch.no_grad():
            for x, y in va_dl:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                yoh = eye[y]
                with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                    out = model(x); loss = bce_ls(out, yoh)
                vrun += loss.item() * len(y); n += len(y)
                correct += (out.argmax(1) == y).sum().item()
        val_loss = vrun / max(1, n); val_acc = correct / max(1, n)
        print(f"== ep{epoch} tr{tr_loss:.4f} val{val_loss:.4f} acc{val_acc:.3f} time{time.time()-t0:.0f}s", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "labels": labels,
                        "backbone": CFG.backbone, "cfg": {k: v for k, v in vars(CFG).items() if not k.startswith('__') and isinstance(v, (int, float, str, bool, type(None)))}},
                       os.path.join(CFG.out_dir, f"sed_{CFG.run}_best.pt"))
            print("saved best", round(best_val, 5), flush=True)

    # ---- final fp16 checkpoint ----
    sd = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"model": sd, "labels": labels, "backbone": CFG.backbone},
               os.path.join(CFG.out_dir, f"sed_{CFG.run}_fp16.pt"))
    print("saved fp16", flush=True)

    # ---- train_audio preds (for orthogonality checks) on val subset + sample of train ----
    model.eval()
    pred_rows = []
    pred_df = va_df.copy()
    # also sample up to 4000 train rows for correlation analysis later
    extra = tr_df.sample(min(4000, len(tr_df)), random_state=0)
    pred_df = pd.concat([pred_df, extra]).drop_duplicates(subset=["filename"])
    pred_ds = BirdDataset(pred_df, labels, CFG, train=False)
    pred_dl = DataLoader(pred_ds, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=True)
    all_p = []
    with torch.no_grad():
        for x, _ in pred_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=CFG.use_amp):
                out = model(x)
            all_p.append(out.float().cpu().numpy())
    all_p = np.concatenate(all_p, 0).astype(np.float16)
    np.save(os.path.join(CFG.out_dir, f"sed_{CFG.run}_trainpreds.npy"), all_p)
    pred_df[["filename", "primary_label"]].reset_index(drop=True).to_csv(
        os.path.join(CFG.out_dir, f"sed_{CFG.run}_predindex.csv"), index=False)
    with open(os.path.join(CFG.out_dir, f"sed_{CFG.run}_labels.json"), "w") as f:
        json.dump(labels, f)
    print("saved trainpreds", all_p.shape, "best_val", round(best_val, 5), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
