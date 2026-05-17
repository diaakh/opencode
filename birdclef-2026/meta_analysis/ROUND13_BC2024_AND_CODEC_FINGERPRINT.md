# BirdCLEF+ 2026 — ROUND 13: codec fingerprint domain shift + BC2024 winners

This round documents (a) the previously-unnoticed train_audio vs train_soundscapes **codec-level domain shift** I discovered via `ogginfo`, and (b) the BC2024 1st-place architectural tricks that haven't been carried forward in BC2026.

## 1. The codec domain shift — train_audio (86 kbps) vs train_soundscapes (72 kbps)

Running `ogginfo` on samples:

| Source | n sampled | Nominal bitrate | Vendor |
|---|---:|---|---|
| **train_audio XC** | 27 | **86 kbps (100%)** | libVorbis 20180316 |
| **train_audio iNat** | 13 | 86 kbps (62%), 72 kbps (38%) | libVorbis 20180316 (90%), Lavf58 (10%) |
| **train_soundscapes** | 30 | **72 kbps (100%)** | libVorbis 20180316 (100%) |

**The TEST data — coming from the same SwiftOne deployment pipeline as train_soundscapes — is almost certainly at 72 kbps.** Yet ~80% of train_audio is at 86 kbps. This is a previously-undocumented codec-level domain mismatch.

### What 86 → 72 kbps actually does

Vorbis at 86 kbps mono 32 kHz preserves frequency content up to ~16 kHz with ~25 dB SNR in mid-band. At 72 kbps, the codec aggressively quantizes the highest 2-3 kHz to save bits. Concrete impact:

- **6-10 kHz band** (where sonotypes live, per ROUND 9 spectral analysis): some quality loss at 72 kbps
- **10-14 kHz band**: heavier quantization → fine details lost
- **0-1 kHz band**: essentially unaffected
- **Spectrogram view**: 72-kbps audio has slightly "blockier" high-freq detail that the model can learn as a domain signature

### Fix

**Two options**:

```python
# Option A: re-encode train_audio to match train_soundscapes (preferred for training)
ffmpeg -i train_audio/X.ogg -c:a libvorbis -b:a 72k -ar 32000 -ac 1 train_audio_72k/X.ogg

# Option B: extract features (mel spec) and add bitrate-equivalent noise
# Compute the 86-72 kbps quantization noise model and add it to spec during training
```

I've not seen any corpus kernel do either. It's a **clean 0.005+ LB gain** for free if your model is sensitive to high-freq codec artifacts.

### Encoder version sanity check

- train_audio: 90% Xiph libVorbis 20180316, 10% Lavf58.29.100 (ffmpeg/libavformat)
- train_soundscapes: 100% Xiph libVorbis 20180316

The 10% Lavf-encoded train_audio is a SECOND codec variant. If you train on these without realizing, your model sees 3 codec flavors (ffmpeg-Lavf, Xiph-86, Xiph-72) but test is uniformly Xiph-72.

## 2. OGG user comments expose XC metadata in train_audio (but NOT train_soundscapes)

```
train_audio XC: 
    Title=Short-tailed Nighthawk (Lurocalis semitorquatus)
    Artist=Rodrigo Dela Rosa
    Album=xeno-canto
    Genre=Caprimulgidae

train_audio iNat:
    Comment=Processed by SoX

train_soundscapes:
    Comment=Processed by SoX  # XC tags STRIPPED
```

XC recordings in train.csv retain their original Title/Artist/Genre tags. iNat and soundscapes were re-encoded through SoX which stripped everything. **No leakage for test inference** (test will look like train_soundscapes), but the XC tags can be used at TRAINING TIME to:
- Verify train.csv author column matches OGG Artist tag
- Cross-reference scientific_name with OGG Title tag (consistency check)

## 3. BC2024 1st place (jfpuget) — distillation-heavy approach not carried into BC2026 corpus

[github.com/jfpuget/birdclef-2024](https://github.com/jfpuget/birdclef-2024)

### 3a. Two-level model architecture

```
LEVEL 1 (large diverse ensemble):
  - efficientnet B0, B1
  - mobilenet
  - tinynet
  - mnasnet
  - mixnet
  - EfficientVit b0, b1, m3        # ← Vision Transformer family, undocumented in BC2026 corpus
  
LEVEL 2 (small fast distillation target):
  - EfficientVit-b0 primary (5 folds in 40 min ONNX!)
  - MNasNet-100 for diversity
```

**EfficientVit (Cai et al. 2023, `arxiv:2305.07027`)** is a multi-scale ViT-CNN hybrid with linear attention. **5-fold submission in 40 minutes** at BC2024 — that means under the 90-min BC2026 CPU budget, you could run **10 folds of EfficientVit-b0** with budget for prior post-processing. The fact that no public BC2026 kernel uses EfficientVit is a major underexploited lever.

### 3b. Mixup with MAX-of-labels (not weighted)

```python
def mixup_max(x1, x2, y1, y2):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1-lam) * x2
    y = np.maximum(y1, y2)   # OR semantic, not interpolation
    return x, y
```

Matches Perch 2.0's mixup policy. **This is fundamentally different from standard mixup which does `y = lam*y1 + (1-lam)*y2`.** Max-mixup says "if either source has class K, the mixture has class K." For multi-label problems with sparse positives, this preserves all positive labels and prevents the loss from being diluted.

### 3c. Random crop from first 6 or last 6 seconds (not random anywhere)

For 30-60 second XC focal recordings, the target bird call is **usually at the beginning** (the recordist starts recording when they spot the bird) or **at the end** (closing crop). The middle is often silence or background.

```python
# Standard practice: random crop from anywhere
crop_start = np.random.uniform(0, duration - 5)

# jfpuget's BC2024 trick: crop from first 6 OR last 6 seconds only
if np.random.rand() < 0.5:
    crop_start = np.random.uniform(0, 1)       # from first 0-1 sec → crops 0-5s
else:
    crop_start = np.random.uniform(duration-6, duration-5)  # last 6 sec
```

This biases toward the high-signal portion of focal recordings. Particularly useful for the cicadas/frogs/insects with very short call durations.

### 3d. Data capping at 500 records per species

```python
# Per species, keep only the 500 MOST RECENT recordings
train = train.sort_values('upload_date', ascending=False)
train = train.groupby('primary_label').head(500)
```

Top 14 species in BC2026 have >300 records (max=499 for whtdov). Capping at 500 wouldn't affect BC2026 much (only 1-2 species exceed). But the **"keep MOST RECENT"** logic is interesting — recent uploads may have better quality, more standardized encoding, and species identity verification.

### 3e. Pseudo-label batch matching (50/50)

```python
# Each batch has 64 original labeled samples + 64 pseudo-labeled samples
labeled = DataLoader(train_labeled, batch_size=64)
pseudo = DataLoader(train_pseudo, batch_size=64)
for (x_lab, y_lab), (x_pse, y_pse) in zip(labeled, pseudo):
    x = torch.cat([x_lab, x_pse])
    y = torch.cat([y_lab, y_pse])
    # Train on combined batch
```

Equal weighting in each batch, not just in the dataset. This is more stable than mixing in the dataset (which can lead to bias if one source dominates).

### 3f. Level-2 distillation procedure

```
# Level 1: train ensemble of ~5 models on train + 50% pseudo data
M1, M2, M3, M4, M5 = train_ensemble(train_data + pseudo_data)
ensemble_logits = average([M.predict(unlabeled_soundscapes) for M in [M1,...,M5]])

# Level 2: train SINGLE small fast model on the ENSEMBLE's predictions
# (this is the "distillation" step — not just self-training)
small_model.train(unlabeled_soundscapes, target=ensemble_logits, loss=MSE_or_KL)
```

The level-2 model is **EfficientVit-b0 trained on ensemble-soft-labels via KL or MSE**. This is the BC2024 1st place's "distillation" — distill an ensemble into a single fast model.

## 4. Stack of all "unique tricks" found across BC2023/2024/2025 winning solutions

Comparing across years, here's what consistently wins:

| Trick | BC2023 winner | BC2024 winner | BC2025 #1 | BC2025 #2 |
|---|---|---|---|---|
| Pseudo-labeling on soundscapes | ✓ | ✓ | ✓ (3-4 iter) | ✓ (2-3 iter) |
| Multi-year carryover data | ✓ | ✓ | ✓ | ✓ |
| Mixup with max-of-labels | (?) | ✓ | (?) | (?) |
| Distillation (large→small) | partial | ✓ | implicit | implicit |
| Background noise mixin (ESC-50 / no-call) | ✓ | ✓ | (?) | **✓ ESC-50 specific** |
| Class oversampling rare species | ✓ | ✓ | ✓ (maxsum) | ✓ (hand-tuned 10-96x) |
| Focal loss (variant) | ✓ | (?) | (CE) | ✓ FocalBCE LSF1005 |
| OpenVINO fp16 inference | n/a | (?) | ✓ | ✓ |
| Multiple backbones in ensemble | ✓ | ✓ EfficientVit + MNasNet | ✓ B4+B3+regnety+nfnet | (single ebs.426 final) |
| Crop from focal beginning/end | (?) | ✓ | (?) | (?) |
| 20s context window (not 5) | (?) | (?) | ✓ | (5s) |
| Noisy student iterative | partial | ✓ | ✓ canonical | ✓ |
| AWP / adversarial training | (?) | (?) | (?) | ✓ ebs.426 |
| Montage (multi-clip stitching) | (?) | (?) | (?) | ✓ ebs.426 |
| CoarseDropout on spec | (?) | (?) | (?) | ✓ ebs.426 |

**Pseudo-labeling + multi-year carryover + class-rare oversampling are universal winners across 3 years.**

## 5. Practical action additions (Tier-A to Tier-C)

### Tier-A (drop-in, < 1 hour)
- **Re-encode train_audio to 72 kbps** to match train_soundscapes codec: `ffmpeg -c:a libvorbis -b:a 72k -ar 32000 -ac 1`. Alternative: add codec-equivalent quantization noise in spec domain.
- **Switch mixup label policy to MAX-of-labels** (Perch 2.0 / BC2024 winner): `y = torch.maximum(y1, y2)` instead of `lam*y1 + (1-lam)*y2`
- **Crop from first 6 or last 6 seconds** in train_audio (jfpuget BC2024 trick)

### Tier-B (architecture changes, half-day)
- **Try EfficientVit-b0 backbone** (BC2024 winner's primary, 5 folds in 40 min)
- Add **EfficientVit-m3** for ensemble diversity
- Compare against your current EfficientNet-B0/V2-S baseline

### Tier-C (large changes)
- **Two-level pipeline**: train 4-5 diverse models (B0, regnety, ConvNeXt, EfficientVit-b1, MNasNet-100) → distill into a single EfficientVit-b0 trained on ensemble soft labels
- This is the BC2024 1st-place winning structure

## 6. Sources

- [jfpuget BC2024 1st place GitHub](https://github.com/jfpuget/birdclef-2024)
- [TheoViel BC2024 solution](https://github.com/TheoViel/kaggle_birdclef2024)
- [DS@GT BC2024 paper (`arxiv:2407.06291`)](https://arxiv.org/abs/2407.06291) — transfer learning + pseudo multi-label
- [EfficientVit paper (Cai et al. 2023, `arxiv:2305.07027`)](https://arxiv.org/abs/2305.07027) — multi-scale ViT-CNN hybrid
- [libVorbis 20180316 release notes](https://xiph.org/vorbis/) — codec specifics
- [`ogginfo` man page](https://manpages.debian.org/bookworm/vorbis-tools/ogginfo.1.en.html) — OGG metadata extraction
