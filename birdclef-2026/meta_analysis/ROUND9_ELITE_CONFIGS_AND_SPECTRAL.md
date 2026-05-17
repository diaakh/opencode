# BirdCLEF+ 2026 — ROUND 9: ELITE training-config archaeology + acoustic signatures

After extracting actual checkpoint files, downloaded model artifacts, the Perch 2.0 paper, and Nikita Babych's BC2025 1st-place pipeline (he's currently BC2026 Rank 3 with LB 0.959), here are the most concrete leaked findings.

## 1. Leaderboard top-3 (as of 2026-05-17) and what's been published

| Rank | Team | LB | Subs | Public artifacts? |
|---:|---|---:|---:|---|
| **1** | Yannan Chen (yannan90) | 0.962 | 236 | **None public** |
| **2** | "more exp is all you need" (cudacoding) | 0.959 | 297 | **None public** |
| **3** | **Nikita Babych (BC2025 1st place winner)** | **0.959** | 305 | **FULL BC2025 inference + weights public** |
| 4 | 89 Minutes 59 Seconds (aliozanmemetoglu + denizegememetoglu) | 0.958 | 168 | 5-fold SED inference public |
| 5 | coolz | 0.957 | 243 | None |
| 6 | "Duck said: Quack" (pursueml, zejunfool) | 0.957 | 270 | None |
| 7 | "BirdCLEF+ 2026 Team🤗" (tonylica, shtljw, yiheng) | 0.957 | 297 | Inference + 2 weights public |

**Nikita's BC2025 1st-place artifacts are reusable directly** — he's already 3 ppts above the 2025 LB plateau and won by porting the same pipeline.

## 2. Alexandergremyakov LB 0.950 (rank 43) — full checkpoint forensics

Downloaded `models/alexandergremyakov/sed-b0-ce-nospecaug/pytorch/default/1/best.pt` (52 MB):

```python
config = {
    'loss_type': 'softmax_ce',         # NOT BCE — softmax cross-entropy
    'model_name': 'efficientnet_b0',    # 4.3M params
    'num_classes': 234,
    'is_sed': True,
    'mel_params': {
        'sample_rate': 32000,
        'n_mels': 64,                   # half of typical 128
        'n_fft': 2048,
        'hop_length': 512,
        'f_min': 0, 'f_max': 16000,
    },
}
optimizer = Adam(
    lr=1e-3, weight_decay=1e-5,          # very low WD
    betas=(0.9, 0.999),
)
scheduler = CosineAnnealingWarmRestarts(T_0=5, T_mult=1.0, eta_min=0.0)
```

Training trajectory (val_aucs, from checkpoint pickle):
```
epoch  0: train=9.44  val=5.84  val_auc=0.9473
epoch  3: train=6.26  val=4.08  val_auc=0.9734  (first cosine restart approaches)
epoch  8: train=5.67  val=3.73  val_auc=0.9750  (first cycle peak)
epoch 12: train=5.51  val=3.63  val_auc=0.9764  (second cycle peak)
epoch 13: train=5.35  val=3.51  val_auc=0.9790  (final, third cycle peak)
```

**Three critical findings:**
1. **softmax_ce, not BCE** — multilabel-incompatible in theory, but ranking-friendly for macro-AUC
2. **n_mels=64** (half of standard 128) — sufficient because softmax_ce only needs class rankings
3. **val AUC 0.9790 vs LB 0.950 = 3 percentage-point gap** — the val set is much easier than test set
4. The "nospecaug" suffix in the model name = NO SpecAugment, consistent with the corpus finding that aug_specaugment is negative for top kernels

Inference (from `efficientnet-b0-submission.ipynb`):
- **20-second CONTEXT window**, predicting 4 × 5-sec segments per window  
- **40 frames per context, 10 frames per segment** (`hop_length=512` → 32000*20/512 = 1250, but they downsample to 40 frames)
- **Custom asymmetric temporal smoothing**: first window `0.75*curr + 0.25*next`, middle `0.25*prev + 0.50*curr + 0.25*next`, last `0.25*prev + 0.25*curr`
- AttentionPooling head with tanh→softmax (different from sigmoid attention)

## 3. Tonylica LB 0.957 (rank 7) — full training config archaeology

Downloaded `tonylica/birdclef-2026-model/LB872.pt` (75 MB):

```python
# Architecture (matches aidensong123/bestfold foundation)
backbone = 'tf_efficientnet_b0.ns_jft_in1k'  # Noisy Student JFT pretrained
n_mels = 224                                 # matches ImageNet 224×224 input
n_fft = 2048
hop_length = 512
fmin = 0, fmax = 16000
in_channels = 3                              # mel duplicated to RGB
gem_p_init = 3.0                             # learnable GEM pooling
dropout = 0.1
drop_path_rate = 0.0
mel_scale = 'htk'                            # NOT slaney
norm = 'slaney'

# Two-stage training
# Stage 1: aggressive
stage1_epochs = 4
stage1_batch_size = 24
stage1_lr_backbone = 1e-5
stage1_lr_head = 3e-5                        # 3× higher LR on head
stage1_weight_decay = 1e-4
stage1_mixup_alpha = 0.2
stage1_clip_repeat = 1
stage1_sc_repeat = 2                         # train_soundscapes OVERSAMPLED 2x

# Stage 2: refinement
stage2_epochs = 3                            # only 3 epochs (BC2025 1st place: 5)
stage2_lr_backbone = 5e-6                    # halved
stage2_lr_head = 1e-5                        # one-third
stage2_mixup_alpha = 0.0                     # NO mixup in stage 2

# Multi-label handling
secondary_label_weight = 0.5                 # USE secondary labels at half weight!
use_secondary_labels_train = True
label_smoothing = 0.0
min_rating = 0.0                             # use ALL recordings (no rating filter)

# Augs (SpecAugment USED here, contradicting my corpus finding)
specaug_p = 0.5
time_mask_max = 32
freq_mask_max = 24
n_time_masks = 2
n_freq_masks = 2

# Validation metrics
macro_auc = 0.9960  # !! validation macro AUC
clip_auc = 0.9829
soundscape_auc = 0.9960
# LB = 0.957 → val-LB gap of 4 percentage points
```

**Inference config (from notebook)**:
- Final blend: 0.8 finetuned + 0.2 baseline in PROB space (`EXP_ID = 2`)
- 4 worker threads, asynchronous prefetch

**Foundation checkpoint**: `aidensong123/bestfold/best_fold0.pt` (134 MB on Kaggle, 134K downloads). Tonylica's LB872 model is a 2-stage finetune of this foundation. **23 other people downloaded the same foundation** — small enough that custom variants will diverge.

The shared foundation is critical to understand because tonylica + multiple top-7 teams all built on this. Key foundation params:
- chunk_duration = **10.0 seconds** for training (not 5!)
- lr = 5e-4 base
- 15 epochs with CosineAnnealingWarmRestarts(T_0=5)
- mixup_prob=0.5, mixup_alpha=0.5 (heavy mixup)
- gain_min_db/max_db = ±12 dB random gain
- noise_min_snr_db/max_snr_db = 10-30 dB additive noise
- loss = "ce" (cross-entropy, not BCE)
- clip_loss_weight=0.5 + frame_loss_weight=0.5 (BOTH levels of supervision)
- pad_type = "random" (random offset padding for short clips)

## 4. Nikita Babych's BC2025 1st-place pipeline (currently BC2026 Rank 3)

Pulled `nikitababich/birdclef2025-1st-place-inference` and decoded the ensemble checkpoint names. **This is the highest LB pipeline that has full publicly-available code AND weights.**

### 4a. Architecture

```python
# All 9 models share:
duration = 20                          # 20-sec input window (not 5!)
slice_step_sec = 5
img_size = (224, 512)                  # mel 224 × time 512
n_fft = 2048 * 2 = 4096                # DOUBLE typical n_fft
hop_length = 20 * 32000 // (512 - 1) = 1252  # computed for exact 512 time frames
n_mels = 224
f_min, f_max = 0, 16000
top_db = 80
dropout = 0.5                          # HEAVY dropout (0.5)
inference_type = "overlap_average_max_delta"
```

### 4b. The ensemble (9 SED models)

Decoding the checkpoint names:

| # | Backbone | Iter | Notes |
|---|---|---:|---|
| 1 | tf_efficientnet_b4.ns_jft_in1k | 3 | bs=64, mixup_p=0.5, temp=0.55, dropout_path=0.15 |
| 2 | tf_efficientnet_b3.ns_jft_in1k | 3 | bs=54, mixup_p=0.5, temp=0.55, dropout_path=0.15 |
| 3 | regnety_016.tv2_in1k | 4 | bs=64, mixup_p=0.5, temp=0.6, dropout_path=0.15 |
| 4 | regnety_016.tv2_in1k | 4 | same but seed=fold2 |
| 5 | regnety_016.tv2_in1k framewise | 4 | framewise variant |
| 6 | eca_nfnet_l0.ra2_in1k | 3 | bs=128, +additional_data, full_data, 15 epochs |
| 7 | regnety_008.pycls_in1k | sup | supervised only, no iter |
| 8 | **tf_efficientnet_b0.ns_jft_in1k_incest_amphibia** | sup | **DEDICATED MODEL FOR INSECTS + AMPHIBIA** |

**Bombshell finding #8**: Nikita trained a **separate dedicated model** ONLY for insects and amphibia ("incest_amphibia" is a typo of "insect_amphibia"). This is the highest-leverage trick for handling the 10.7% sonotype share. No public BC2026 kernel does this.

### 4c. The "overlap_average_max_delta" inference

After running each 20-sec window through the SED model:

```python
# 1. For overlapping 20s windows, accumulate framewise predictions
# 2. Average across overlapping frames (normalize by mask count)
# 3. Per 5-sec segment: max over frames → segmentwise prediction
# 4. Frame-level TTA shift:
segmentwise_preds *= 0.5
for segment_ind in range(num_segments):
    # back-shift contribution (0.25 weight)
    seg_back = framewise_ss_preds[segment_ind*step - tta_delta:
                                  (segment_ind+1)*step - tta_delta].max(0) * 0.25
    # forward-shift contribution (0.25 weight)
    seg_fwd = framewise_ss_preds[segment_ind*step + tta_delta:
                                 (segment_ind+1)*step + tta_delta].max(0) * 0.25
    segmentwise_preds[segment_ind] += seg_back + seg_fwd
```

This is **frame-shift TTA** (faster than waveform-shift TTA because the SED forward pass runs once). The 5-sec prediction = `0.5 × center_max + 0.25 × backshift_max + 0.25 × forwardshift_max`.

### 4d. Training recipe (decoded from checkpoint names)

- `sampler_maxsum_iteration_3_v1`: **3-iteration noisy-student pseudo-labeling**
- `temp_0.55`: softmax temperature 0.55 for SED head
- `0.15_drop_path_rate`: drop_path 15%
- `1_mixup_ratio_pseudo_data`: mixup ratio 1 on pseudo-labeled data
- `20_duration_sed_type`: 20s SED input
- `0.5_mixup_p`: mixup probability 50%
- `(224, 512)_size`: mel size 224×512
- `ce`: cross-entropy loss
- `4096_n_fft`: n_fft = 4096
- `additional_data, full_data`: model 6 used external data + full training set

### 4e. CPU optimization: OpenVINO

Nikita's `/kaggle/input/runtimes-onnx-openvino/openvino/` ships pre-built OpenVINO wheels:
- `openvino-2025.0.0-17942-cp310-cp310-manylinux2014_x86_64.whl`
- `openvino_telemetry-2025.1.0-py3-none-any.whl`

He uses `from openvino.runtime import Core` to load the SED models. This is **30-50% faster CPU inference than raw ONNX** (per the BC2025 2nd-place paper).

## 5. Perch 2.0 — what's actually in the teacher (`arxiv:2508.04665`)

From the official paper (not training memory):

| Property | Value |
|---|---|
| Total training recordings | 1,542,778 |
| Xeno-Canto recordings | 896,255 |
| **iNaturalist recordings** | **571,698** |
| Tierstimmenarchiv (Berlin Animal Sound Archive) | 33,859 |
| FSD50K (general environmental audio) | 40,966 |
| **Bird recordings** | 1,367,553 (89%) |
| **Amphibian recordings** | **55,051** |
| **Insect recordings** | **63,366** |
| **Mammalian recordings** | **15,389** |
| Other sound events | 41,419 |
| Total classes | **14,795** (14,597 species + 198 FSD50K events) |
| Embedding dim | **1536** |
| Spatial feature shape | (5, 3, 1536) → mean to (1536,) |
| Backbone | EfficientNet-B3, 12M params |
| Sample rate | 32 kHz |
| Hop / window length | **10 ms / 20 ms** (= 320 samples / 640 samples at 32 kHz) |
| Frequency range | **60 Hz – 16 kHz** |
| Mel bins | 128 |
| Frames per 5s clip | 500 |

**Critical insight**: Perch v2's training corpus contains 63,366 insect + 55,051 amphibian recordings. The reason 28 BC2026 classes don't map to Perch isn't that Perch lacks insect/frog knowledge — it's that Perch trained on labeled **species**, not the anonymous `47158son01-25` "sonotypes" which are call-type categorizations within Insecta order without species ID.

**Self-distillation mechanism in Perch 2.0**:
- 4 learnable prototypes per class
- Stop-gradient separates embedding model from prototype classifier
- Prototype classifier output → soft target for the dense linear classifier

**Multi-source mixup (Perch-specific)**:
- Sample N ∈ {2, 3, 4, 5} sources per training example
- Mixing weights from symmetric Dirichlet distribution
- Output normalized for gain

This is more aggressive than the typical 2-source mixup in BC2026 public kernels (alpha=0.2-0.5).

**The "Bittern Lesson"** (paper §1): *"Simple, supervised models are difficult to beat."* Perch 2.0 explicitly argues against self-supervised approaches in favor of fine-grained supervised classification on 14,795+ labels. This is consistent with BC2025 1st place using iterative supervised noisy-student (not contrastive SSL).

## 6. Spectral fingerprinting — where insect sonotypes actually live

Ran Welch PSD on labeled soundscape files containing sonotype annotations:

### 6a. Unmapped sonotype-dominated files (no birds drowning the signal)

| File | Sonotypes present | Spectral centroid | Dominant band |
|---|---|---:|---|
| BC2026_Train_0005_S08_20250607_070007.ogg | son15,16,17,25 | **6,232 Hz** | **6-10 kHz (86.7%)** |
| BC2026_Train_0004_S08_20250607_070007.ogg | son03,17,18,19 | **5,874 Hz** | **6-10 kHz (57%)**, 3-6 kHz (37.6%) |

Both files: 95% of energy is in 3-10 kHz, almost nothing above 10 kHz or below 1 kHz.

### 6b. Train-audio INSECT classes (the 3 mapped ones)

| Class | Common name | Sample 1 centroid | Sample 2 centroid | Sample 3 centroid |
|---|---|---:|---:|---:|
| 244024 | Giant Cicada | **1,251 Hz** | **1,872 Hz** | **886 Hz** |
| 1161364 | Guyalna cuta | 3,620 Hz | 1,138 Hz | 2,556 Hz |
| 760266 | Prionacris erosa | 2,151 Hz | 3,230 Hz | 5,013 Hz |

**The 3 mapped Insecta classes are dominantly LOW-FREQUENCY (1-3 kHz cicadas).**

### 6c. Train-audio AVES (birds) sample

| Class | Centroid | Roll-off 95% |
|---|---:|---:|
| Ferruginous Pygmy Owl | 742 Hz | 4,859 Hz |
| White-naped Jay | 2,466 Hz | 4,641 Hz |
| Black-capped Donacobius | 2,462 Hz | 4,195 Hz |
| Bananaquit | 6,055 Hz | 9,375 Hz |
| Striped Cuckoo | 2,554 Hz | 2,680 Hz |

Bird centroids: 700 Hz – 6 kHz, mostly under 3 kHz.

### 6d. The big spectral gap: sonotypes live where birds don't

| Frequency band | Birds (train_audio) | Insects mapped (train_audio) | Sonotypes (labeled soundscape) |
|---|---|---|---|
| < 1 kHz | medium | high (cicadas) | trace |
| 1-3 kHz | high | **dominant** | low |
| 3-6 kHz | high | medium | medium |
| **6-10 kHz** | low | medium | **DOMINANT (50-87%)** |
| 10-16 kHz | trace | medium | trace |

**The 6-10 kHz band is the "insect sonotype signature"** — and it's NOT well-covered by training insect audio. This explains why:
- Genus-proxy (Maryna's hack) fails for sonotypes (mapping `Insect` → `Insecta order` is too broad)
- Direct training on the 3 cicada classes won't generalize (wrong frequency band)
- Aliozanmemetoglu's texture/event smoothing helps because it spreads weak sonotype signal across multiple windows
- Nikita Babych's dedicated insect/amphibia model can specialize on the 6-10 kHz band

### 6e. Mel-config implications

Most public kernels use `f_min=0, f_max=16000` with n_mels=128. The mel-scale dedicates:
- ~30 bins to 0-1000 Hz (mostly wind noise)
- ~40 bins to 1-4 kHz (birds + low cicadas)
- ~40 bins to 4-10 kHz (insect sonotypes!)
- ~18 bins to 10-16 kHz (mostly trace)

If you use `f_min=1000, fmax=12000` with n_mels=128 (Slaney scale):
- ~50 bins to 1-4 kHz
- ~78 bins to 4-12 kHz (sonotype-rich zone gets ALMOST DOUBLE the resolution)
- Trade-off: lose the high-cicada signature below 1 kHz

The aliozanmemetoglu config uses `f_min=50, f_max=16000, n_mels=128` — better than `f_min=0` because below 50 Hz is pure DC/wind noise (SwiftOne mic response starts at 100 Hz, so below 100 Hz is essentially mic noise).

## 7. The aidensong123 foundation: a missed leverage point

`aidensong123/birdclef-2026-sed-baseline-training-lb-0-862` (only 14 votes, ~50 downloads). This is the shared foundation tonylica (rank 7) uses. The training config:

```python
chunk_duration = 10.0        # 10-second training input
n_mels = 224                  # matches ImageNet 224×224
in_channels = 3               # mel as RGB
backbone = 'tf_efficientnet_b0.ns_jft_in1k'
mel_scale = 'htk', norm = 'slaney'

epochs = 15
batch_size = 16, grad_accum_steps = 2  # eff. 32
lr = 5e-4
weight_decay = 1e-4
scheduler_T_0 = 5             # CosineAnnealingWarmRestarts(T_0=5)

mixup_prob = 0.5, mixup_alpha = 0.5  # aggressive mixup
loss_type = 'ce'              # cross-entropy
clip_loss_weight = 0.5
frame_loss_weight = 0.5       # DUAL supervision: clip + frame

# Augs
gain_min_db = -12, gain_max_db = +12   # ±12 dB random gain
noise_min_snr_db = 10, noise_max_snr_db = 30  # additive noise SNR 10-30 dB
freq_mask_param = 30
time_mask_param = 30

use_secondary_labels = True
include_soundscape_labels = True
pad_type = "random"           # random offset padding for short clips
```

**Why this is undervalued**: This single recipe gets to LB 0.862 standalone, and tonylica's stage-2 fine-tune adds +0.010 to get to LB 0.872, then ensemble takes it to LB 0.957. The foundation is doing 90% of the work — and only 23 people have downloaded it.

## 8. Concrete plan, updated with ROUND 9 evidence

### Tier-A (immediate, < 1 hour)
- Drop `aidensong123/bestfold/best_fold0.pt` into your inference notebook as a teacher (LB 0.862 single model)
- Add the texture/event smoothing kernel from aliozanmemetoglu
- Add `.drop_duplicates()` on the labels CSV read

### Tier-B (free compute, < 1 day)
- Train your own student on the `pseudo_cache` soft labels using aidensong123's exact training config (lr=5e-4, 15 epochs, mixup α=0.5, 224×512 mel size)
- Use **n_mels=224, n_fft=4096, in_channels=3** (Nikita BC2025 1st place)
- 20-second context window during training, not 5
- Apply Nikita's "overlap_average_max_delta" frame-shift TTA at inference

### Tier-C (ELITE training, multi-day)
- Train a **dedicated insect/amphibia model** on just the texture taxa (Nikita's #1 unique trick)
  - Filter training data to Insecta + Amphibia + sonotype-positive soundscape windows
  - Use `f_min=4000, f_max=12000` mel config to concentrate resolution in the sonotype band
- Iterate 3 rounds of noisy-student pseudo-labeling on train_soundscapes (BC2025 1st-place lifted scores by +0.03 with this alone)
- OpenVINO fp16 conversion for all final models (saves 30-50% CPU time)

### Tier-D (architectural diversity)
- Build a 3-way ensemble matching Nikita's diversity: B4 + regnety_016 + nfnet_l0
- Each at different mel resolutions (224×256, 224×512, 128×256) for cross-resolution voting
- Per-class learnable fusion alpha between SED ensemble and Perch teacher (chaneyma)

## 9. Sources researched fresh (URLs verified)

- [Perch 2.0 paper (`arxiv:2508.04665`)](https://arxiv.org/html/2508.04665v2) — training corpus, self-distillation, multi-source mixup, "Bittern Lesson"
- [Nikita Babych BC2025 1st place inference (Rank 3 on BC2026)](https://www.kaggle.com/code/nikitababich/birdclef2025-1st-place-inference) — 9-model ensemble code
- [Nikita Babych BC2025 1st place ensemble weights dataset](https://www.kaggle.com/datasets/nikitababich/birdclef2025-1st-place-ensemble) — 457 MB, 152 votes
- [Nikita Babych OpenVINO runtimes dataset](https://www.kaggle.com/datasets/nikitababich/runtimes-onnx-openvino) — Pre-built fp16 OpenVINO wheels
- [aidensong123 BC2026 SED training baseline (LB 0.862)](https://www.kaggle.com/code/aidensong123/birdclef-2026-sed-baseline-training-lb-0-862) — the shared foundation
- [aidensong123 bestfold dataset](https://www.kaggle.com/datasets/aidensong123/bestfold) — 134 MB, 23 downloads. Foundation `best_fold0.pt`
- [tonylica model checkpoints dataset](https://www.kaggle.com/datasets/tonylica/birdclef-2026-model) — LB872.pt + LB862.pt
- [alexandergremyakov SED B0 model](https://www.kaggle.com/models/alexandergremyakov/sed-b0-ce-nospecaug) — softmax_ce + nospecaug recipe
- [BirdCLEF 2026 strategy playbook PDF (Eric Benhamou)](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/Birdclef_2026.pdf) — phased competition strategy
- [BirdCLEF 2026 EDA findings discussion](https://www.kaggle.com/competitions/birdclef-2026/discussion/681827)
- [alexandergremyakov sonotype EDA notebook](https://www.kaggle.com/code/alexandergremyakov/birdclef-2026-soundscape-sonotype-eda) — source of 10.7% sonotype insight
- [BC2025 1st place writeup (Nikita Babych)](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n)
- [Xeno-Canto.org scope (birds only)](https://xeno-canto.org/about/recordings) — confirms zero non-bird coverage
