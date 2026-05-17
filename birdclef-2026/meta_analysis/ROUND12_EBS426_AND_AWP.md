# BirdCLEF+ 2026 — ROUND 12: The third BC2025 2nd-place model (ebs.426) + adversarial training

After mining Sydorskyi's GitHub I found that the actual Kaggle inference notebook (`vladimirsydor/bird-clef-2025-ensemble-v2-final-final`) reveals a **third teammate's model** (`ebs.426` by vialactea/Fernando Goncalves) with a completely different training recipe.

## 1. The full ebs.426 config (decoded from the inference notebook)

```python
ebs.426 = dict(
    # Architecture
    model_name='SpecNetImg',
    encoder='tf_efficientnetv2_s',
    img_dim=(128, 256),       # mel 128 × time 256 (HALF resolution vs aidensong's 224)
    img_duration=5,
    in_chans=1,               # grayscale, NOT 3-channel RGB
    gem_p=1.8,                # lower than tonylica's 3.0 / aliozanmemetoglu's 3.0
    head_dropout=0.0,
    drop_path_rate=None,
    out_indices=2,            # 2nd-to-last feature map (multi-scale)
    
    # Training
    epochs=50,
    train_batch_size=64,
    val_batch_size=64,
    train_size=28_000,        # 28k samples per epoch
    n_folds=5,
    seed=10,
    
    # Optimizer
    optimizer='AdamW',
    lr=[(1e-05, 0.001), (3, 0.00025)],  # multi_lr: warmup 1e-5→1e-3, then at epoch 3 drop to 2.5e-4
    end_lr=1e-06,
    eps=1e-08,
    betas=(0.9, 0.999),
    weight_decay=1e-06,       # VERY LOW (vs alexander's 1e-5)
    max_grad_norm=10,
    
    # Scheduler
    schedule_type='multi_lr',
    step_scheduler_after='step',
    warmup_share=0.02,
    
    # Loss
    loss_schedule={0: 'focal_volodymyr'},   # custom focal at epoch 0
    label_smoothing=0.0,
    mask_non_labels=True,                   # mask out non-target class logits
    
    # CV
    fold_col='author',                      # !!! GROUP BY AUTHOR (not file or species)
    n_preload_species=15,
    
    # Augmentation
    aug={
        'CoarseDropout': (0.375, 0.375, 1, 0.7),   # block-dropout 37.5% × 37.5%
        'Flip': 0.5,                                # spec axis flip
        'audio': False,
        'mixup': 1,                                 # mixup always on
        'volume': (0.3333, 3),                      # ±3x volume gain
    },
    background_noise_prob=0.5,
    background_noise_cache_size=1000,
    background_noise_max_usage=6,
    background_noise_reference=True,
    
    # Montage augmentation (audio stitching)
    montage=(0, 3),                          # 0-3 montage clips per training sample
    montage_cache_size=10,
    montage_bird_prob=False,
    
    # Adversarial training (AWP-style)
    adv_lr=0.005,
    adv_eps=0.01,
    adv_th=0.3,
    awp=False,                              # AWP framework off, but uses adv_lr/eps anyway
    
    # Pseudo-labeling
    pseudo_label_zero_th=0.1,
    sample_unlabeled_prob=0.0,              # they don't sample pseudo, mix during training
    train_unlabeled_dataset='h5-unlabeled-2',
    unlabeled_primary_th=0.5,
    unlabeled_weight=1,
    unlabeled_pseudo_preds=['b5-data-pseudo-pred-v3/unlabeled pseudo pred v2'],
    
    # Data
    dataset='h5-6',
    dataset_val='val-128-256-m1-4',
    files_edges='/kaggle/input/b5-cache/files_edges speech 5',
    previous_dataset='add-h5-6',
    curation_mode='speech',                 # SPEECH CURATION (remove human voice)
    train_mode='random',
    train_primary_th=0.5,
    ws_power=0.5,                           # weighting power (sqrt class freq)
    extension='.ogg',
    n_preload_species=15,
    
    # Pretrained init
    pretrained_encoder='/kaggle/input/b5-pretrained-weights/tf_efficientnetv2_s_in21k_Pretrainversion1.pth',
)
```

## 2. Final ensemble weights (the actual production submission)

```python
ebs_426 = {
    'ebs.426_f0': 1/5,
    'ebs.426_f1': 1/5,
    'ebs.426_f2': 1/5,
    'ebs.426_f3': 1/5,
    'ebs.426_f4': 1/5,
}

class ARGS:
    engine = 'onnx'        # NOT openvino (they switched back for final)
    lot_size = 90          # files per batch lot
    num_workers = 3        # async workers
    batch_size = 4         # per-window batch
    ws = ebs_426
    ensemble_logit = False # average probs, not logits
    norm_pred = False
```

The final BC2025 2nd-place submission **is just 5-fold ebs.426 averaged uniformly** in PROBABILITY space (not logit space). No multi-architecture ensemble in the production inference. Just one model, 5 folds, 1/5 weights.

## 3. New unique tricks in ebs.426 not in BC2026 corpus

### 3a. Adversarial Weight Perturbation (AWP)
- `adv_lr=0.005, adv_eps=0.01, adv_th=0.3`
- Even with `awp=False` flag, the adversarial gradient is computed and applied
- During each forward pass, perturb the weights by `+adv_lr × normalized_gradient` if loss > `adv_th`
- This is a form of **Sharpness-Aware Minimization (SAM)** lite (Foret et al. 2020, `arxiv:2010.01412`)

### 3b. Montage Augmentation
- `montage=(0, 3)` with `montage_cache_size=10`
- During training, stitch 0-3 random additional clips into the current 5-sec window
- Forces the model to handle multi-event windows with abrupt boundaries
- Similar to mixup but TIME-DOMAIN concatenation, not amplitude mixing

### 3c. CoarseDropout (block-mask augmentation on spectrogram)
- `(0.375, 0.375, 1, 0.7)` = (max_height_fraction, max_width_fraction, min_holes, max_holes_prob)
- Drops 37.5% × 37.5% blocks of the spectrogram at probability 0.7
- More aggressive than SpecAugment's `time_mask` + `freq_mask` (which mask whole rows/columns)
- Forces the model to use partial information from non-blocked regions

### 3d. Author-grouped CV
- `fold_col='author'` (not filename or species)
- Each author's recordings go entirely into one fold
- Catches "recorder fingerprint" leakage that file-grouping misses
- More aggressive than `GroupKFold(filename)` which still allows same-author across folds

### 3e. Curation mode 'speech'
- `curation_mode='speech'`
- Cleans human voice from training audio (the BC2025 kdmitrie approach generalized)
- Even though BC2026 has less voice contamination (1.6% CSA-style vs BC2025 majority), this is still valuable

### 3f. Multi-step LR schedule
- `lr=[(1e-05, 0.001), (3, 0.00025)]`
- Step 1: warmup from 1e-5 to 1e-3 over a few epochs
- Step 2: at epoch 3, drop to 2.5e-4 (1/4 of peak)
- This is unusual — most use cosine decay. Multi-step gives **abrupt LR drop** that breaks out of local minima
- Combined with `end_lr=1e-6` for the final decay

### 3g. Single channel mel input
- `in_chans=1` (grayscale)
- Tonylica uses `in_chans=3` (mel tripled to RGB)
- Saving 2/3 of input bandwidth — but also losing the ImageNet-pretrained color filters' benefit
- Why this works: they use a custom `pretrained_encoder` (not raw ImageNet), so the in_chans=1 patch isn't a problem

### 3h. mask_non_labels=True
- Mask out non-target-class logits during loss computation
- Forces the model to ONLY rank the 234 target classes against each other
- Standard BCE with 234-d output already does this implicitly; explicit masking may help when the model has additional output dims

### 3i. ws_power=0.5 (sqrt class weighting)
- Class weights = (1/class_count)^0.5
- Sqrt-balancing: more aggressive than 1/freq (true balancing) but less than uniform
- Combined with random sampling, this approximates the SqrtBalancing from the train config

## 4. Loss functions verified in Sydorskyi's repo

From `code_base/losses/focal_loss.py` and `combined_losses.py`:

```python
# Standard FocalLoss (alpha=0.25, gamma=2.0)
class FocalLoss(nn.Module):
    def forward(self, inputs, targets):
        return torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs, targets, alpha=0.25, gamma=2, reduction='mean'
        )

# Combined BCE + Focal (with separate weights)
class FocalLossBCE(nn.Module):
    def forward(self, inputs, targets):
        focal = sigmoid_focal_loss(inputs, targets, alpha, gamma)
        bce = BCEWithLogitsLoss()(inputs, targets)
        return bce_weight*bce + focal_weight*focal

# BCE focalized on positives only (NOT negatives)
class BCEFocalLossPaper(nn.Module):
    def forward(self, preds, targets):
        bce = BCEWithLogitsLoss(reduction='none')(preds, targets)
        proba = sigmoid(preds)
        loss = (
            targets * alpha * (1-proba)**gamma * bce        # focal on positives
          + (1-targets) * (1-alpha) * proba**gamma * bce    # focal on negatives  
        )

# TWO-WAY loss (clip + max-of-frame)
class BCEFocal2WayLoss(nn.Module):
    def forward(self, input, target):
        # Auxiliary loss from MAX over frame logits
        framewise = input['framewise_logits_long']
        clipwise_via_max = framewise.max(dim=1)  
        loss_main = focal(input['clipwise_logits_long'], target)
        loss_aux  = focal(clipwise_via_max, target)
        return self.weights[0]*loss_main + self.weights[1]*loss_aux  # default [1, 1]
```

The **2-way loss** (clip + max-of-frame) is what `aidensong123`'s `clip_loss_weight=0.5 + frame_loss_weight=0.5` is implementing. Sydorskyi uses equal `[1, 1]` weights.

## 5. Comparison of BC2025 top-3 actual ensemble configs

| Property | BC2025 #1 Nikita | BC2025 #2 Sydorskyi | BC2025 #2 Sydorskyi/ebs.426 (Fernando) |
|---|---|---|---|
| # models | 9 | 2 | 1 (5-fold) |
| Mel input | 224×512 | 128×256 (mel A) / 224×512 (mel B) | **128×256** |
| n_fft | **4096** | 2048 | 2048 |
| Window | **20s** input | 5s | **5s** |
| in_chans | 3 (RGB) | 3 | **1 (grayscale)** |
| Backbones | B4, B3, regnety_016/008, nfnet_l0, B0+texture | nfnet_l0 + V2-S | tf_efficientnetv2_s only |
| Optimizer | (decoded) | RAdam (A) / AdamW (B) | AdamW |
| Epochs | n/a per ckpt | 50 | 50 |
| LR | (decoded) | (decoded) | multi-step 1e-3→2.5e-4 |
| WD | 0.15 drop_path | 1e-4 | **1e-6 (very low!)** |
| Loss | CE | FocalBCE + LSF=1.005 | focal_volodymyr (custom) |
| Adversarial | No | No | **AWP-like (adv_lr=0.005)** |
| Mixup | p=0.5, α≈0.4 | p=0.5, α=None | p=1 (always) |
| Specaug | No (?) | RandomLowerHighFreq + time/freq mask | CoarseDropout |
| Bkg noise | (?) | ESC-50 + no-call | yes, custom |
| Montage | No | No | **Yes (0-3 clips)** |
| Pseudo-label iter | 3-4 | 2-3 (F2 criteria) | 1 (on-the-fly) |
| Final inference | OpenVINO | OpenVINO fp16 + AsyncInferQueue | ONNX, 5-fold avg in prob space |
| Public LB | (BC2025) 0.937 | 0.925 | 0.925 (same as A+B) |

## 6. Why ebs.426 stands alone in production

Sydorskyi tested A+B (NFNet + V2-S) but final submission used just ebs.426 (5 folds). The 2 models A+B from the training pipeline were apparently superseded by the single ebs.426 model. This is **a major lesson** about ensembling: just because you trained 10 models doesn't mean ensemble > best single. The 2nd place submitted a **5-fold same-model average** (not a diverse ensemble). Ensemble diversity loses to a strong single model + 5 folds.

This contradicts the common wisdom that "diverse ensemble beats single model" — at the very top of the leaderboard, **5 strong same-architecture folds > 2 mediocre diverse models**.

## 7. Concrete plan additions for BC2026

### Tier-A (drop-in <1 hour)
- Add **CoarseDropout (0.375, 0.375, 1, 0.7)** to spectrogram augs
- Increase volume gain range to **(0.333, 3)** = ±3x
- Add **author-grouped CV split** (group by train.csv `author` column)

### Tier-B (config changes)
- Adopt the **multi-step LR schedule**: warmup to 1e-3, drop to 2.5e-4 at epoch 3
- Lower **weight decay to 1e-6** (alexander 1e-5; tonylica 1e-4; Sydorskyi 1e-6)
- Implement **2-way BCEFocal loss** (clipwise + max-of-frame, equal weights)
- Add **Montage augmentation** (concat 0-3 random clips per training sample)

### Tier-C (advanced)
- Add **AWP-style adversarial training** (adv_lr=0.005, adv_eps=0.01, threshold=0.3)
- Build a **speech curation pipeline** to clean human voice from train_audio (1.6% of BC2026 data)
- Try **single-channel input** (in_chans=1) with custom-pretrained encoder

### Tier-D (validation discipline)
- **Trust 5-fold same-model ensemble over diverse 2-model ensemble** — the BC2025 2nd place lesson
- Final submission should be 5 folds of your strongest model, averaged in PROBABILITY space (not logit)

## 8. Sources

- [Sydorskyi BC2025 ensemble inference notebook (final final)](https://www.kaggle.com/code/vladimirsydor/bird-clef-2025-ensemble-v2-final-final)
- [vialactea (Fernando) ebs.426 training notebook (still in Sydorskyi's pipeline)](https://www.kaggle.com/code/vialactea/b5-train-ebs-426)
- [Sydorskyi `focal_loss.py`](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place/blob/main/code_base/losses/focal_loss.py)
- [Sydorskyi `combined_losses.py`](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place/blob/main/code_base/losses/combined_losses.py)
- [Foret et al. 2020 — Sharpness-Aware Minimization (SAM, `arxiv:2010.01412`)](https://arxiv.org/abs/2010.01412)
- [BC2025 1st/2nd CEUR proceedings (Sydorskyi & Goncalves, "Tackling Domain Shift")](https://ceur-ws.org/Vol-4038/paper_256.pdf)
