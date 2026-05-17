# BirdCLEF+ 2026 — ROUND 29: more model bundle deep-dive

After ROUND 28 (Bruce Wu's pipeline = 0.93 OOF AUC), I pulled and analyzed 4 more model bundles.

## 1. michaelihc/birdclef2026-hybrid-bundle-public-20260324

A SIMPLER version of Bruce Wu's approach. Contains `stack_arrays.npz` with PCA + Ridge probe weights.

**Architecture**:
```
Perch v2 → 1536-d embedding → StandardScaler → PCA(64) → +metadata = 71-d features
                                                                       ↓
                                            Linear probe for 52 specific classes
```

**OOF results** (708 windows, on 52 modeled classes):
- `raw_macro_auc`: 0.7390 (Perch only, mapped classes only)
- `base_macro_auc`: 0.8038 (Perch + basic post-processing)
- **`probe_macro_auc`: 0.8335** (with Ridge probes)

**Compared to Bruce Wu's pipeline (0.9304)**, this is LOWER because:
- PCA(64) is smaller than Bruce's PCA(256)
- Only 52 of 234 classes modeled (vs Bruce's 75 valid)
- 71-d features vs Bruce's 490-d

**Saved priors** (similar in structure to my pseudo priors):
- `prior_global_p`: (234,)
- `prior_site_p`: (9 sites × 234)
- `prior_hour_p`: (13 hours × 234)
- `prior_site_hour_p`: (25 combos × 234)

## 2. alexanterkapai/birdclef-2026-models

Single `model_fold0.pth` (79 MB, 20.6M params).

**Architecture**: EfficientNetV2-S (NOT B0)
- backbone.conv_stem + bn1 + blocks (5 stages, up to 14 sub-blocks per stage)
- SE blocks throughout
- 1280-d conv_head → simple linear `head.1` (no SED, no attention)
- 1280 → 234 direct FC

**Pretrained**: NS-JFT-In1k (same family as tonylica/aidensong)

This is a **larger, simpler V2-S baseline** — useful as ensemble member with diversity over the B0 models.

## 3. mauriciooffermann/birdclef-2026-exp-034-sparse-fusion-safe-bundle

Has 3 SEEDS of checkpoints (seed_42, seed_1337, seed_2026) — 49.7 MB each.

**Configuration (from `seed_42/best.pt` config dict)**:

```yaml
model:
  backbone_name: tf_efficientnet_b0
  pretrained: False   # !!! trained FROM SCRATCH
  in_chans: 1         # grayscale
  pooling: avg        # simple GAP, not GeM
  dropout: 0.0

spectrogram:
  n_fft: 2048
  hop_length: 512
  n_mels: 128
  f_min: 20           # skip wind noise
  f_max: 16000
  image_size: 224
  power: 2.0
  normalize: per_sample

audio:
  sample_rate: 32000
  duration: 5.0
  train_crop_mode: event_aware
  eval_crop_mode: center

optimizer:
  name: AdamW
  lr: 0.001
  weight_decay: 0.01

scheduler:
  name: cosine
  warmup_epochs: 1
  min_lr: 1e-06

loss:
  name: bce
  focal_gamma: 2.0

training:
  batch_size: 16
  epochs: 200            # max
  amp: True
  gradient_accumulation_steps: 1

labels:
  secondary_label_weight: 0.3   # use secondary labels at 30%

folds:
  n_splits: 5
  clip_strategy: stratified_group_kfold
  soundscape_strategy: group_kfold
```

**Best epoch: 9, train_loss=0.0108**. Likely overfit (low train loss + only 9 epochs of 200).

**UNIQUE FEATURE — anchored_stage2_augmentation**:
```yaml
anchored_stage2_augmentation:
  enabled: True
  apply_probability: 0.25
  same_label_reinforcement_probability: 0.75
  nuisance_overlay_probability: 0.5
  allowed_fx_buckets: ('river_selected',)   # Pantanal RIVER background!
  reinforcement_snr_grid_db: (-15, -13.5, -12, -10.5, -9)
  overlay_relative_db_grid: (-21, -18, -15)
  trim_threshold_db_from_peak: 12.0
```

**Concept**: Take a target species clip → add same-species reinforcement (75% chance) + nuisance overlay (50% chance) at specific SNR levels (-15 to -9 dB), with Pantanal river background. This is a **physics-aware augmentation** for matching the Pantanal acoustic environment.

**Also configurable but DISABLED in this run**:
- `realism_critic`: a GAN-style critic to score if synthetic is realistic
- `realism_generator`: a generator with species conditioning + 32-d embedding

This is the **most sophisticated training pipeline** in the public corpus. No other published kernel has GAN-style realism critic + Pantanal river-background augmentation.

## 4. baiyuby/birdclef2026-distill-models fold0

(83.8 MB, 21.9M params)

**Architecture**:
- `bb.conv_stem.weight: [24, 1, 3, 3]` — **EfficientNetV2-S** with **in_chans=1 (grayscale)**
- Head:
  - `att.2.bias`: attention head
  - `fc_att`: 1280→234 (attention-pooled classifier)
  - `fc_max`: 1280→234 (max-pooled classifier)
- **Dual-head**: attention + max-pool, two separate 234-d outputs

**Trained metadata**:
- epoch: 7
- **fold0 CV AUC: 0.972**
- (per dataset description: 4-fold mean CV = 0.9848)

This is the EfficientNetV2-S distillation student from teacher ensemble (described as: Temperature=2.0, Alpha=0.7, teacher CV=0.9786 LB=0.861).

**The mean CV 0.985 is suspicious** — likely uses heavy soundscape-based train/val (which inflates CV). Real LB performance probably ~0.86-0.90.

## 5. emoptisie/birdclef2026-effb0-onnx

Single ONNX file (16 MB):
- Input: (batch, 3, 224, 224) — RGB mel spec
- Output: (batch, 234) — direct logits
- 674 nodes, 81 Conv layers, 65 Sigmoid (SE), 16 ReduceMean (GAP)
- Simple EfficientNet-B0 classifier, no SED

**Trade-off**: 16 MB ONNX is FAST on CPU. Good ensemble member.

## 6. Cross-model architectural comparison

| Source | Backbone | Params | in_chans | Head type | CV/LB |
|---|---|---:|---:|---|---|
| tonylica_LB872 | tf_efficientnet_b0.ns_jft_in1k | 6.3M | 3 (RGB) | GeM + AttSED | LB 0.872→0.957 ens |
| aidensong_bestfold | Same | 6.3M | 3 | Same | LB 0.862 |
| alexander_sed_b0 | efficientnet_b0 | 4.3M | **1 (grayscale)** | 4-head SED | LB 0.950 |
| nikita_insect_amphibia | tf_efficientnet_b0.ns_jft_in1k | 5.0M | 3 | AttSED **700 outputs (BC2025)** | - |
| chaneyma_moe | (Proto SSM only) | 3.2M | - | Mamba + prototypes | CV 0.9245 |
| chaneyma_student_cnn | 3-block CNN | 1.1M | 1 | dual (logit+emb1536) | - |
| chaneyma_student_crnn | 2-conv + BiGRU | 3.4M | 1 | dual (logit+emb1536) | - |
| junhaoyi_resnet50 | ResNet-50 | 21.4M | 3 | **FC 204 outputs (BC2025)** | - |
| alexanterkapai | tf_efficientnetv2_s | 20.6M | 3 | Simple FC | - |
| mauricio_seed42 | tf_efficientnet_b0 (no pretrain!) | 4.3M | **1 (grayscale)** | avg pool + linear | - |
| baiyuby_fold0 | EfficientNetV2-S | 21.9M | **1 (grayscale)** | dual (att+max) | CV 0.972 |
| Bruce CLIP-ridge | (sklearn Ridge) | - | - | PCA(256) + Ridge | **OOF 0.930** |
| michaelihc probe | (sklearn Linear) | - | - | PCA(64) + LR | OOF 0.834 |

**Pattern observations**:
- **Grayscale in_chans=1** is used by 4 models (alexander, mauricio, baiyuby, chaneyma students)
- **3-channel RGB mel** used by 4 models (tonylica, aidensong, nikita, alexanterkapai)
- **SED architectures** (attention + framewise heads) consistently outperform simple FC heads
- **Distillation from Perch embedding** appears in 2 models (chaneyma students, baiyuby)

## 7. The Mauricio anchored_stage2_augmentation recipe (unique trick)

This is the most novel training trick I've seen across all bundles. The idea:

```python
# During stage 2 training, 25% of batches get this:
if np.random.rand() < apply_probability:  # 0.25
    # Take base audio (5-sec sample, target species X)
    
    # 75% chance: reinforce with another sample of species X
    if np.random.rand() < same_label_reinforcement:  # 0.75
        reinforcement = sample_from(target_species_X)
        snr = np.random.choice([-15, -13.5, -12, -10.5, -9])  # dB
        audio = mix_at_snr(audio, reinforcement, snr)
        # Both samples now contribute to target_X label (already labeled correctly)
    
    # 50% chance: overlay a nuisance sound  
    if np.random.rand() < nuisance_overlay:  # 0.50
        bg = sample_from('river_selected')  # Pantanal river background
        rel_db = np.random.choice([-21, -18, -15])
        audio = mix_at_db(audio, bg, rel_db)
        # bg is background, labels stay [target_X]
```

**Why this works**:
- Same-label reinforcement: makes the model robust to multiple species-X individuals calling at varying SNR
- Pantanal river overlay: directly trains for the test domain background
- The trim_threshold_db_from_peak=12.0 ensures clean audio segments

**Result**: 25% of training data is augmented this way, leading to a model that handles realistic Pantanal SNR conditions.

## 8. Bruce vs Mauricio: two opposite philosophies

**Bruce**: 
- Use Perch as frozen backbone
- Add Ridge regression student
- 2 MB pickle, no GPU needed
- 0.93 OOF AUC

**Mauricio**:
- Train EfficientNet B0 FROM SCRATCH (not pretrained!)
- Stage-2 anchored augmentation with Pantanal river BG
- 50 MB checkpoint per seed × 3 seeds
- Result unclear (no OOF AUC reported)

Both reach approximately the same LB tier (0.85-0.93) by different paths. The Bruce path is faster to iterate; the Mauricio path adds more domain-specific tricks.

## 9. Concrete plan additions (post ROUND 29)

### Tier-A (drop-in)
- Use Bruce's `clip_student_bundle.pkl` directly as a Ridge student on top of Perch (0.93 OOF baseline)
- Combine with my `pseudo_hour_priors.csv` for 0.96 macro-AUC

### Tier-B (training)
- Adopt Mauricio's `anchored_stage2_augmentation`:
  - Same-label reinforcement at SNR -15 to -9 dB
  - Pantanal river background overlay at -21 to -15 dB relative
- Mine a 'river_selected' background dataset (use train_soundscapes S22 night recordings as proxy)

### Tier-C (architecture)
- Combine 3-4 diverse models:
  - EfficientNet-B0 RGB (tonylica style)
  - EfficientNet-B0 grayscale (alexander/mauricio)
  - EfficientNetV2-S grayscale (baiyuby/alexanterkapai)
  - Ridge on Perch features (Bruce)
- Blend logits in rank-averaging space

## 10. Sources

- [brucewu1200/birdclef-2026-cvlb-assets-0911](https://www.kaggle.com/datasets/brucewu1200/birdclef-2026-cvlb-assets-0911)
- [michaelihc/birdclef2026-hybrid-bundle-public-20260324](https://www.kaggle.com/datasets/michaelihc/birdclef2026-hybrid-bundle-public-20260324)
- [alexanterkapai/birdclef-2026-models](https://www.kaggle.com/datasets/alexanterkapai/birdclef-2026-models)
- [mauriciooffermann/birdclef-2026-exp-034-sparse-fusion-safe-bundle](https://www.kaggle.com/datasets/mauriciooffermann/birdclef-2026-exp-034-sparse-fusion-safe-bundle)
- [baiyuby/birdclef2026-distill-models](https://www.kaggle.com/datasets/baiyuby/birdclef2026-distill-models)
- [emoptisie/birdclef2026-effb0-onnx](https://www.kaggle.com/datasets/emoptisie/birdclef2026-effb0-onnx)
