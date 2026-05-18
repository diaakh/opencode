"""BirdMAE-finetune standalone LB submission for BirdCLEF 2026.

Self-contained: loads adkasd/birdmae-finetune-234, runs inference on
test_soundscapes, writes submission.csv.

Validated: macro-AUC 0.9721 on labeled OOF (per-bottleneck-frog gains
of +0.12 to +0.18 vs Bruce-derived stack).

Expected runtime: ~47 min for 600 files on Kaggle CPU.
"""
import os, sys, re, time, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

# Locate BirdMAE package
PKG_ROOT = None
for cand in [Path("/kaggle/input/birdmae-finetune-234"),
             Path("/kaggle/input/datasets/adkasd/birdmae-finetune-234")]:
    if cand.exists() and (cand / "checkpoints" / "best.pt").exists():
        PKG_ROOT = cand; break
assert PKG_ROOT, "BirdMAE package not found"
print(f"BirdMAE package: {PKG_ROOT}")

REPO_DIR = PKG_ROOT / "repo" / "Bird-MAE"
SCRIPT_DIR = PKG_ROOT / "scripts"
CKPT = PKG_ROOT / "checkpoints" / "best.pt"

sys.path.insert(0, str(SCRIPT_DIR.resolve()))
sys.path.insert(0, str(REPO_DIR.resolve()))
from birdclef_finetune import BirdMAEClassifier

encoder_path = REPO_DIR / "models" / "mae" / "encoder.py"
spec = importlib.util.spec_from_file_location("birdmae_encoder", encoder_path)
enc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc_mod)
MAE_Encoder = enc_mod.MAE_Encoder

encoder = MAE_Encoder(
    img_size_x=512, img_size_y=128, patch_size=16, in_chans=1,
    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
    norm_layer=torch.nn.LayerNorm, pos_trainable=False,
)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
classes = ckpt.get("classes") or json.load(open(PKG_ROOT / "metadata" / "finetune_result.json"))["classes"]
n_cls = len(classes)
print(f"BirdMAE classes: {n_cls}")

model = BirdMAEClassifier(encoder=encoder, num_classes=n_cls, pool="mean", dropout=0.1)
state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=False)
device = "cpu"  # Kaggle CUDA arch mismatch
model = model.to(device).eval()
print(f"Device: {device}")

# Comp data
COMP = None
for cand in ["/kaggle/input/birdclef-2026", "/kaggle/input/competitions/birdclef-2026"]:
    if Path(cand).exists():
        COMP = Path(cand); break
TEST_DIR = COMP / "test_soundscapes"
SAMPLE_SUB = COMP / "sample_submission.csv"
samp = pd.read_csv(SAMPLE_SUB)
print(f"COMP: {COMP}, test: {TEST_DIR}, sample: {samp.shape}")

class_cols = [c for c in samp.columns if c != "row_id"]
assert len(class_cols) == 234
# Map BirdMAE class index → BC2026 column index
bc_idx = {c: i for i, c in enumerate(class_cols)}
birdmae_to_bc = {bi: bc_idx[str(c)] for bi, c in enumerate(classes) if str(c) in bc_idx}
print(f"BirdMAE → BC2026 mapping: {len(birdmae_to_bc)} classes")

# Preprocessing
SR = 32000
WIN_SAMP = 5 * SR
N_WIN = 12
FBANK_MEAN = -7.2
FBANK_STD = 4.43
TARGET_LEN = 512

def compute_fbank_batch(wave_batch):
    """Compute (B, 1, 512, 128) fbank tensor — exact training pipeline."""
    fbanks = []
    for y in wave_batch:
        wav = torch.from_numpy(y.astype(np.float32))
        wav = wav - wav.mean()
        wav = wav.unsqueeze(0)
        feat = torchaudio.compliance.kaldi.fbank(
            wav, htk_compat=True, sample_frequency=SR, use_energy=False,
            window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10,
        )
        if feat.shape[0] < TARGET_LEN:
            pad = TARGET_LEN - feat.shape[0]
            feat = F.pad(feat, (0, 0, 0, pad), value=float(feat.min()))
        elif feat.shape[0] > TARGET_LEN:
            feat = feat[:TARGET_LEN]
        feat = (feat - FBANK_MEAN) / (FBANK_STD * 2.0)
        fbanks.append(feat)
    return torch.stack(fbanks, dim=0).unsqueeze(1)

# Process test files
test_files = sorted(TEST_DIR.glob("*.ogg"))
if not test_files:
    print("No test files — emitting all-zero submission")
    out = samp.copy()
    out.iloc[:, 1:] = 0.0
    out.to_csv("submission.csv", index=False)
    sys.exit(0)

print(f"\nProcessing {len(test_files)} files...")
ROW_RE = re.compile(r"_(\d{8})_(\d{6})$")
all_rows = []
all_preds = []
t0 = time.time()

with torch.no_grad():
    for fi, fpath in enumerate(test_files):
        try:
            audio, sr = sf.read(str(fpath))
            if audio.ndim > 1: audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            # 12 × 5s windows
            wins = []
            for w in range(N_WIN):
                s_samp = w * WIN_SAMP
                e_samp = s_samp + WIN_SAMP
                if e_samp <= len(audio):
                    clip = audio[s_samp:e_samp]
                else:
                    clip = np.zeros(WIN_SAMP, dtype=np.float32)
                    avail = max(0, len(audio) - s_samp)
                    if avail > 0: clip[:avail] = audio[s_samp:s_samp+avail]
                wins.append(clip)
            specs = compute_fbank_batch(wins).to(device)
            logits = model(specs)
            probs = torch.sigmoid(logits).cpu().numpy()
            # Map BirdMAE preds → BC2026 columns
            bc_preds = np.zeros((N_WIN, 234), dtype=np.float32)
            for bi, ci in birdmae_to_bc.items():
                bc_preds[:, ci] = probs[:, bi]
            stem = fpath.stem
            for w in range(N_WIN):
                end_sec = (w + 1) * 5
                all_rows.append(f"{stem}_{end_sec}")
                all_preds.append(bc_preds[w])
        except Exception as e:
            print(f"  fail {fpath.name}: {e}")
            continue
        if (fi + 1) % 25 == 0 or fi == len(test_files) - 1:
            el = time.time() - t0
            rate = (fi + 1) / el
            eta = (len(test_files) - fi - 1) / max(rate, 0.01)
            print(f"  [{fi+1}/{len(test_files)}] {el:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")

print(f"\nTotal: {time.time()-t0:.0f}s")
P = np.array(all_preds, dtype=np.float32)
print(f"P shape: {P.shape}")

# Write submission with row_id alignment
out = pd.DataFrame({"row_id": all_rows})
for i, c in enumerate(class_cols):
    out[c] = P[:, i]

# Ensure all sample rows present (Kaggle requires)
expected_rows = set(samp["row_id"].astype(str).tolist())
got_rows = set(out["row_id"].astype(str).tolist())
missing = expected_rows - got_rows
if missing:
    print(f"  WARNING: {len(missing)} expected rows missing — filling with zeros")
    miss_df = samp[samp["row_id"].isin(missing)].copy()
    miss_df.iloc[:, 1:] = 0.0
    out = pd.concat([out, miss_df], ignore_index=True)

# Align order to sample
out = out.set_index("row_id").reindex(samp["row_id"]).reset_index()
out.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv: {out.shape}")
print(f"Range: min={P.min():.4f}, max={P.max():.4f}")
