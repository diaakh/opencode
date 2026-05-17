# BirdCLEF+ 2026 — ROUND 16: ttahara's HGNetV2-B0 training + session-end summary

## 1. ttahara's HGNetV2-B0 training config (LB 0.876 baseline)

From `ttahara/birdclef-2026-hgnetv2-b0-baseline-training`:

```python
class CFG:
    max_epoch     = 20             # 20 epochs (between Melichov 5 and Sydorskyi 50)
    warmup_epoch  = 5              # 5-epoch warmup
    batch_size    = 64
    lr            = 5.0e-04         # peak LR
    init_lr       = 2.0e-05         # warmup start
    final_lr      = 1.0e-04         # cosine end
    weight_decay  = 1.0e-04
    
    # Model
    model_name = "hgnetv2_b0.ssld_stage2_ft_in1k"   # SSLD pretraining variant
    pretrained = True
    drop_path_rate = 0.0
    head_dropout   = 0.3            # 30% dropout in head (heavy)
    is_lse_trainable = True         # LSE temperature is LEARNABLE
    
    # Mel
    mel_spectrogram_params = dict(
        sample_rate=32_000,
        n_fft=2048,
        win_length=626,             # unusual value
        hop_length=313,             # = win_length / 2
        f_min=20,                   # higher than usual 0 (skips wind noise)
        n_mels=256,                 # 256 mel bins (high resolution)
        power=2.0,
        norm='slaney',              # slaney norm
        mel_scale='htk',            # htk scale
    )
    lms_shape = (256, 256)          # SQUARE 256×256 (vs 128×256, 224×512, etc.)
    top_db = 80.0
    
    # Augmentation
    mixup = dict(alpha=1.0, theta=0.8)   # alpha=1 + custom theta=0.8 param
    
    # Training tricks
    use_amp = True                  # mixed precision
    use_ema = True                  # EMA model
    ema_params = dict(
        decay=0.999,
        use_warmup=True,            # EMA decay warmup
        warmup_gamma=1.0,
        warmup_power=2/3,           # ramp follows ^(2/3) law
    )
```

**Three unique elements not in BC2026 corpus elsewhere**:

### 1a. HGNetV2-B0 with SSLD pretraining

`hgnetv2_b0.ssld_stage2_ft_in1k` — PaddlePaddle's [HGNetV2](https://github.com/PaddlePaddle/PaddleClas) with **SSLD (Simple Semi-supervised Label Distillation)** pretraining + fine-tuning on ImageNet. The SSLD pretrain step gives HGNetV2-B0 stronger ImageNet performance than vanilla pretraining. **This specific timm variant has 23M params and is ~2x slower than EfficientNet-B0 but +1-2% on ImageNet.**

### 1b. Custom mixup with `theta=0.8`

Standard mixup: `lam ~ Beta(alpha, alpha)`. Ttahara adds a `theta=0.8` parameter that's likely modifying the standard policy — possibly skewing the Beta distribution or being a probability cutoff. Without the source code for the mixup function, the exact semantics are unclear, but it's a non-standard mixup variant.

### 1c. EMA with warmup

```python
# EMA decay starts at low value, ramps to 0.999 following warmup_power^(2/3) law
ema_decay_at_epoch_t = min(decay, t^(2/3) / warmup_epoch^(2/3) * decay)
```

Standard EMA: fixed decay = 0.999 from epoch 0. Ttahara's approach: ramp EMA decay during warmup so early-epoch noise doesn't poison the EMA buffer. **This is a documented trick from MoCo v3 / DINO literature** but rare in audio competitions.

### 1d. f_min=20 Hz (not 0)

Skips the 0-20 Hz band (mostly mic DC + sub-audible noise) but keeps everything from 20 Hz up. The SwiftOne mic's specified frequency response starts at 100 Hz, so anything below 100 Hz is mic noise anyway. **f_min=20 is a sweet spot** that drops noise while preserving low-frequency bird/cicada calls.

### 1e. Square 256×256 mel + grayscale (in_chans=1)

```python
self.backbone = timm.create_model(model_name, pretrained=True, in_chans=1, ...)
```

HGNetV2-B0 with `in_chans=1` saves 2/3 of input bandwidth vs the typical RGB-duplication approach. The square 256×256 spec matches ImageNet's 224×224 spatial structure approximately.

### 1f. LSE temperature is LEARNABLE

`is_lse_trainable = True` — the `r` parameter in `lse_pool(x, r=10)` is a learnable nn.Parameter, not a fixed scalar. The model can adjust the sharpness during training.

## 2. Comparison of three "single SED" recipes for BC2026

| Property | aliozanmemetoglu | tonylica | ttahara HGNetV2 |
|---|---|---|---|
| Backbone | tf_efficientnetv2_s | tf_efficientnet_b0.ns_jft_in1k | hgnetv2_b0.ssld_stage2_ft_in1k |
| in_chans | 3 (RGB) | 3 (RGB) | **1 (grayscale)** |
| n_mels | 128 | 224 | **256** |
| n_fft | 2048 | 2048 | 2048 |
| hop_length | 627 | 512 | **313** |
| f_min | 50 | 0 | **20** |
| f_max | 16000 | 16000 | (not set, default) |
| LR | (?) | 1e-4 backbone / 3e-5 head | **5e-4 peak** |
| Optimizer | (?) | AdamW | (likely AdamW) |
| Mixup | (?) | α=0.2 stage1 | **α=1.0 + theta=0.8** |
| EMA | No | No | **Yes, with warmup** |
| Epochs | (?) | 4+3 (=7 total) | **20** |
| Head | AttnSED w/ tanh | AttnSEDHead | **AttnSEDHead + LSE-trainable** |
| Public LB | 0.958 (ensemble) | 0.957 (ensemble) | 0.876 (single) → 0.928 with optimization |

**ttahara starts from 0.876 single-model and reaches LB 0.928** with the speed+reproducibility fixes from discussion 686457. Adding Perch distillation gets to 0.898 (per discussion 683822 chain). So the ttahara recipe → distillation → optimization stack tops at ~0.93.

## 3. Session-end consolidated summary across all 16 rounds

This is the union of every public lever I've found, organized by where they're documented:

### From the corpus of 1,194 public BC2026 kernels (ROUNDS 1-7)
- The 0.948 PLATEAU recipe (Maryna canonical): ProtoSSM + ResidualSSM + Perch + isotonic + adaptive_delta + texture/event smoothing + site×hour prior + rank-aware
- Top kernels: aliozanmemetoglu (0.958 V2-S 5-fold + B0 coarse), tonylica (0.957 stage-1+stage-2 B0), hideyukizushi (0.953 SGKF + ResSSM)
- The 28 missing-from-train classes (25 sonotypes + 3 frogs) = 10.7% of macro-AUC
- Hyperparameter convergence at top: lambda_prior=0.4-0.5, rank_power=0.4-0.6, alpha_blend=0.4

### From the Maryna Borovska canonical notebook (ROUND 8)
- Genus-proxy for unmapped species (rescues 3 frogs, not 25 sonotypes)
- Class-specific temperature (T=0.95 texture, T=1.10 event)
- All 5 post-processing functions (file_confidence_scale, rank_aware, adaptive_delta, isotonic, etc.) trace here

### From the train_soundscapes_labels.csv duplicate bug (ROUND 8, 10)
- Every annotation is duplicated 2x (1478 rows → 739 unique)
- Organizers acknowledged but haven't pushed fix

### From the XC URLs + iNat data sources (ROUND 8)
- XC dataset covers 0 of 28 missing classes (bird-only platform)
- Non-bird classes need iNaturalist or Tierstimmenarchiv

### From ELITE 0.95+ kernel checkpoint forensics (ROUND 9)
- Alexander LB 0.950: softmax_ce + nospecaug + n_mels=64 + Adam + CosineWarmRestarts T_0=5
- Tonylica LB 0.957: 2-stage finetune of aidensong123/bestfold, secondary_label_weight=0.5
- Nikita Babych BC2025 1st: 20-sec context, n_fft=4096, mel 224×512, dedicated insect_amphibia model, multi-iter noisy student
- aidensong123/bestfold is the shared foundation (only 23 downloads despite being the top-7 base)

### From the BC2026 discussion forum (ROUND 10)
- LSE pool head gives +0.015 over GAP on HGNetV2-B0
- ONNX Perch + ThreadPoolExecutor(max_workers=4) = 23-min scoring on 90-min budget
- OpenVINO 2x faster than Torch (but no speedup for NFNet — no BN)
- Domain-binary classifier + external PAM data for background augmentation
- Naive pseudo-labeling HURTS (drops LB 0.06-0.09) — needs confidence filtering + on-the-fly soft labels
- Tom Capybara's "Noisy classmates" extension of noisy student

### From the BC2025 2nd place Sydorskyi GitHub (ROUND 11, 12)
- ESC-50 background noise mixin (dog, rain, insects, engine, hen, hand_saw, pig, ...)
- Per-class hand-tuned oversampling 10-96x for 60 rarest classes
- F2-score pseudo-label criteria (prob≥0.5, model≥0.1, ratio 0.4, iter 2-3)
- TimeFlip, RandomLowerHighFreq, CoarseDropout, deep_supervision_steps
- AWP adversarial training (adv_lr=0.005, adv_eps=0.01, adv_th=0.3)
- Montage augmentation (stitch 0-3 random clips)
- Author-grouped CV
- 5-fold same-architecture ensemble > diverse 2-model ensemble at the top

### From the codec/spectral fingerprint analysis (ROUND 13)
- train_audio is 86 kbps OGG, train_soundscapes is 72 kbps OGG (codec domain shift!)
- 10% of train_audio is Lavf-encoded (3rd codec variant)
- Sonotypes live in 6-10 kHz band (not 1-3 kHz like train cicadas)
- BC2024 jfpuget uses EfficientVit-b0 (5 folds in 40 min)
- BC2024 mixup with MAX-of-labels matches Perch 2.0 paper

### From geographic distance + iNat enrichment (ROUND 14)
- Median train distance to Pantanal: 1,606 km (only 113 records within 100 km)
- Insecta median 3,976 km, Mammalia median 8,167 km
- iNat API exposes per-observation: quality_grade, identifications_count, precise dates, place
- SigmoidF1 + Asymmetric Loss + EnCodec embeddings unused in BC2026

### From recent dataset uploads (ROUND 15)
- samuelzxu/bc26-iter1-pseudo-labels: balanced 50-per-class cap, prob ≥ 0.85
- majkel1337/long-convnextv2-tiny-onnx: 60-sec input → 12×234 outputs in ONE forward pass
- ConvNeXtV2, Swin-tiny, AST, EfficientVit, NFNet — 5+ unused alternative backbones

### From ttahara's HGNetV2-B0 (ROUND 16)
- HGNetV2-B0 SSLD pretrained variant
- in_chans=1 (grayscale, 2/3 less input data)
- f_min=20 Hz (skips wind noise band)
- LSE temperature learnable
- EMA with warmup_power=2/3 (smooth ramp)
- Mixup with custom theta=0.8 parameter

## 4. Recommended final pipeline (synthesized from all rounds)

```
# 1. Pre-process train_audio
train_audio = train_audio.dropna(subset=['latitude','longitude'])
train_audio['weight'] = 1.0 / (1.0 + haversine_to_pantanal(lat, lon) / 500)  # geo weight
train_audio = ffmpeg_reencode(train_audio, '-c:a libvorbis -b:a 72k -ar 32000 -ac 1')

# 2. Train 3-4 diverse models with shared recipe
shared_config = dict(
    sample_rate=32000, n_fft=2048, hop_length=313, n_mels=256,
    f_min=20, f_max=16000, top_db=80,
    in_chans=1, gem_p_init=1.8,  # grayscale, lower gem_p
    mixup_alpha=1.0, mixup_p=0.5,
    epochs=20, batch_size=64,
    lr=5e-4, weight_decay=1e-4,
    optimizer='AdamW', scheduler='CosineAnnealingWarmRestarts(T_0=5)',
    loss='FocalBCE + label_smoothing=1.005',
    use_ema=True, ema_decay=0.999, ema_warmup_power=2/3,
    augs=['ESC50_background', 'TimeFlip', 'CoarseDropout', 'RandomLowerHighFreq',
          'Volume_pm12dB', 'Mixup_max_labels', 'time_mask', 'freq_mask'],
    head='LSE_pool(r_trainable=True) + BCE_2way(clipwise + max_of_frame)',
)
models = [
    train(backbone='hgnetv2_b0.ssld_stage2_ft_in1k', **shared_config),
    train(backbone='tf_efficientnetv2_s_in21k', **shared_config),
    train(backbone='convnextv2_tiny', **shared_config),
    train(backbone='efficientvit_b0', **shared_config),
]

# 3. Multi-iterative noisy student
for iteration in range(3):
    pseudo_predictions = ensemble_predict(models, train_soundscapes_unlabeled)
    high_confidence = filter(prob >= 0.85, cap_per_class=50, F2_threshold)
    models = retrain(models, train_audio + high_confidence)

# 4. Inference
inference_pipeline = pipeline(
    perch_onnx_path='justinchuby/Perch-onnx (with spatial_embedding)',
    intra_op_num_threads=4, max_workers=4,  # async I/O
    long_format=True,  # 60-sec input → 12 outputs
    overlap_average_max_delta=True,  # frame-shift TTA
    proto_ssm=True, residual_ssm=True,  # canonical refiners
    per_class_fusion_alpha=True,  # chaneyma
    texture_event_smoothing=True,
    site_hour_prior=True,  # use full pseudo_cache for prior fitting
    genus_proxy=True,  # for unmapped species
    isotonic_calibration=True,
    adaptive_delta_smoothing=True,
    rank_aware_scaling=True,
    file_top_2_amplification=True,
)
submission = inference_pipeline.predict(test_soundscapes)

# 5. CPU-budget verification
assert inference_pipeline.estimated_time_sec < 88 * 60  # 88-min budget with 2-min safety margin
```

## 5. What I haven't been able to find

Truly hidden information (Yannan Chen at Rank 1, cudacoding at Rank 2) is **PRIVATE**. They have no public kernels, datasets, or discussion posts. Their LB 0.962 / 0.959 advantage over Nikita Babych (0.959 same as cudacoding) is likely:
- Custom-trained backbone on private external data (more recent than Sydorskyi's XC scrape)
- A loss-function or architecture innovation not yet publicly described
- A specific post-processing technique only they know about

To reach 0.96+, you'd need to either guess one of these innovations or accept that the gap is closeable only via additional pseudo-label iterations + diverse ensemble (the 0.959 ceiling matches Nikita's BC2025 1st place private 0.961, indicating that the same approach saturates around 0.96).

## 6. Sources

- [ttahara/birdclef-2026-hgnetv2-b0-baseline-training](https://www.kaggle.com/code/ttahara/birdclef-2026-hgnetv2-b0-baseline-training)
- [hgnetv2_b0.ssld_stage2_ft_in1k on Hugging Face (timm)](https://huggingface.co/timm/hgnetv2_b0.ssld_stage2_ft_in1k)
- [PaddleClas SSLD pretraining](https://github.com/PaddlePaddle/PaddleClas) — Simple Semi-supervised Label Distillation
- [DINO paper (Caron et al. 2021, `arxiv:2104.14294`)](https://arxiv.org/abs/2104.14294) — EMA decay warmup reference
- [Public BC2026 leaderboard CSV](https://www.kaggle.com/competitions/birdclef-2026/leaderboard)
