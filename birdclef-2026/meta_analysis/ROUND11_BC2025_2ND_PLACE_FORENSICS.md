# BirdCLEF+ 2026 — ROUND 11: BC2025 2nd place full pipeline mining (Sydorskyi)

Nikita Babych (BC2025 1st = BC2026 Rank 3) is the most visible. But the **2nd place BC2025 from Sydorskyi & Goncalves** open-sourced their entire pipeline on GitHub (Public LB 0.925, Private 0.928). The config files reveal a LOT of unique training tricks not in any public BC2026 kernel.

## 1. The two final ensemble models — verbatim names

```
Model A (the eca_nfnet one):
eca_nfnet_l0_Exp_noamp_64bs_5sec_BasicAug_SqrtBalancing_Radamlr1e3_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005_FromXCV2Best_PseudoF2PT05MT01P04I3_MinorOverSampleV1

Model B (the EfficientNetV2-S one):
tf_efficientnetv2_s_in21k_Exp_noamp_64bs_5sec_BasicAug_EqualBalancing_AdamW1e4_CosBatchLR1e6_Epoch50_FocalBCELoss_LSF1005_FromPrebs1_PseudoF2PT05MT01P04I2_AddRareBirdsNoLeak
```

Decoded:

| Field | Model A | Model B |
|---|---|---|
| Backbone | `eca_nfnet_l0` | `tf_efficientnetv2_s_in21k` |
| Mixed precision | **no AMP** | no AMP |
| Batch size | 64 | 64 |
| Window | 5 sec | 5 sec |
| Augmentation | BasicAug | BasicAug |
| Sampling balance | **SqrtBalancing** | **EqualBalancing** |
| Optimizer | **RAdam lr=1e-3** | AdamW lr=1e-4 |
| Scheduler | CosBatchLR → 1e-6 | CosBatchLR → 1e-6 |
| Epochs | **50** | **50** |
| Loss | FocalBCELoss | FocalBCELoss |
| Label smoothing | **1.005** (LSF1005) | 1.005 |
| Init | FromXCV2Best (XC pretrain) | FromPrebs1 (prev best run) |
| Pseudo-label criteria | F2≥0.5, model≥0.1, prob≥0.4, iter=3 | iter=2 |
| Rare-class handling | MinorOverSampleV1 | AddRareBirdsNoLeak |

The two models are intentionally diverse: NFNet+RAdam+SqrtBalance vs EfficientNetV2-S+AdamW+EqualBalance. **Same loss, same scheduler, same epochs — only the backbone+optimizer+sampling differ**.

## 2. Per-class oversampling multipliers (from `selected_eca.py`)

Sydorskyi explicitly hand-tunes oversampling for 60 rare classes:

```python
OVERSAMPLE_CONFIG = {
    "turvul":  96,   "piwtyr1": 90,  "bubcur1": 86,  "plctan1": 83,
    "sahpar1": 83,   "shghum1": 81,  "woosto":  81,  "ampkin1": 79,
    "bafibi1": 79,   "blctit1": 77,  "whmtyr1": 74,  "rosspo1": 73,
    "plukit1": 71,   "olipic1": 69,  "cocher1": 68,  "rutpuf1": 68,
    ...
    "thlsch3": 10,   "rufmot1": 10,
}
```

**60 rare classes get explicit oversampling multipliers 10-96x.** This is the "MinorOverSampleV1" trick. Many of these class codes (`piwtyr1`, `bubcur1`, etc.) are also in BC2026's 234-class set (the codes are shared between BC2025 and BC2026 for overlapping species), so this list is partially TRANSFERABLE.

## 3. ESC-50 background noise augmentation (the secret augmentation)

From `selected_eca.py` and `best_ensem_ebs1.py`:

```python
late_aug = OneOf([
    BackgroundNoise(
        p=0.5,
        esc50_root="data/soundscapes_nocall/train_audio",  # custom no-call dataset
        esc50_df_path="data/v1_no_call_meta.csv",
        normalize=True, precompute=False,
    ),
    BackgroundNoise(
        p=0.5,
        esc50_root="data/esc50/audio",
        esc50_df_path="data/esc50_background.csv",
        esc50_cats_to_include=[
            "dog", "rain", "insects", "hen", "engine",
            "hand_saw", "pig", "rooster", "sea_waves",
            "cat", "crackling_fire", ...
        ],
    ),
])
```

**This is the realization of the BC2026 discussion 690887 idea** — but Sydorskyi already had this working in BC2025. They use:
1. **ESC-50** (Piczak 2015, environmental sound classification with 50 classes) — but ONLY specific categories that match Pantanal-like sounds (dog, rain, insects, engine, etc.)
2. **A custom "no-call" soundscape collection** — soundscape windows with no bird vocalization, used as clean background

Mixed in at p=0.5 during training. This teaches the model to ignore farm/weather/insect background noise.

## 4. Auxiliary external data sources

```python
"filename_change_mapping": {
    "base": "train_audio",
    "train_audio": "train_audio",
    "add_train_audio_from_prev_comps": "add_train_audio_from_prev_comps",  # BC2021-2024 carryovers
    "add_train_audio_from_xeno_canto_28032025": "add_train_audio_from_xeno_canto_28032025",  # custom XC scrape
    "soundscape_0": "train_features_soundscapes",
    "soundscape_1": "train_features_soundscapes",
}
```

External data includes:
- **`add_train_audio_from_prev_comps`** — recordings of overlapping species from PREVIOUS BirdCLEF competitions (2021-2024). For BC2026 the analogue would be including BC2024 + BC2025 species that overlap with BC2026's 162 Aves classes.
- **`add_train_audio_from_xeno_canto_28032025`** — a fresh XC scrape from March 28, 2025. Sydorskyi did a custom XC scrape beyond what's in train.csv. **The yasunorim XC-URLs dataset in BC2026 is the analogue but covers only birds, not the texture taxa.**

## 5. CV split design

```python
"split_path": "data/cv_split_base_and_prev_comps_XCsnipet28032025_group_allbirds_hdf5.npy",
```

The split is GROUPED by `allbirds` — meaning groups are formed by ALL birds in the recording (multi-species recordings stay in the same fold to avoid co-occurrence leakage). For rare-birds variant:
- `cv_split_base_and_prev_comps_XCsnipet28032025_group_allrarebirds_hdf5_noleak.npy`

The "no_leak" suffix is critical — the split explicitly prevents rare-bird leakage across folds.

## 6. The model architecture: WaveCNNAttenClassifier

Modular `nn.Module` with these levers (`code_base/models/wave_clasifier.py`):

```python
WaveCNNAttenClasifier(
    backbone='eca_nfnet_l0' or 'tf_efficientnetv2_s_in21k',
    spec_extractor='Melspec' or 'CQT' or 'LEAF',   # 3 frontends supported
    head_type='AttHead' or 'AttHeadSimplified',
    use_sigmoid=False,                              # they use BCE-with-logits
    transformer_backbone=False,                     # CNN backbones
    central_crop_input=None,                        # optional crop
    spec_augment_config={
        "power_aug": ...,                          # RandomSpecPower
        "lower_high_freq": ...,                    # RandomLowerHighFreq (spectral cutoff aug)
        "freq_mask": ...,                          # SpecAugment freq mask
        "time_mask": ...,                          # SpecAugment time mask
        "white_noise": ...,                        # white noise injection
        "bandpass_noise": ...,                     # bandpass-filtered noise
    },
    atten_smoothing_config=...,                    # attention smoothing
    deep_supervision_steps=...,                    # multi-scale loss
    spec_resize=...,                               # resize spec input
)
```

Six different spec-augmentation flavors stack together — `RandomSpecPower`, `RandomLowerHighFreq`, `freq_mask`, `time_mask`, `white_noise`, `bandpass_noise`. No public BC2026 kernel uses this rich spec-aug stack.

**`RandomLowerHighFreq` is particularly interesting** — random spectral cutoff (mask out either lower band or higher band). This forces the model to be robust to band-limited test audio (which matches the SwiftOne 100Hz–20kHz response).

## 7. Audio-level augmentation transforms (`code_base/augmentations/transforms.py`)

Sydorskyi's `OneOf` augmentation stack:
- `NoiseInjection` (uniform random noise)
- `TimeFlip` (reverse the audio!) — controversial but they use it
- `GaussianNoise` (SNR 5-20 dB)
- `BackgroundNoise` (ESC-50 + no-call soundscapes)

**TimeFlip — playing the audio BACKWARDS** is an aggressive augmentation that breaks frequency-modulated calls (most bird songs sweep frequencies in time). But it preserves spectral content. Their use of TimeFlip suggests they found it helps for spectral-only features.

## 8. Pseudo-label criteria explained (the "F2PT05MT01P04I3" decoder)

```
F2  = F2-score-based selection (favors recall over precision)
PT05 = Pseudo Threshold prob > 0.5
MT01 = Model Threshold > 0.1
P04 = Pseudo ratio = 0.4 (40% of training data is pseudo, 60% is labeled)
I3  = 3 iterations of teacher-student loop
```

For model A (NFNet) they use 3 iterations; for model B (EfficientNetV2-S) only 2 iterations. **The exact pseudo-label hyperparameters that work**:
- Use F2 score (β=2, favoring recall) to pick pseudo-positives
- Threshold prob > 0.5 OR model threshold > 0.1 (combo)
- Pseudo ratio 0.4 (don't go higher — this is the "naive pseudo hurts LB" point from BC2026 disc 694815)
- 2-3 iterations is the sweet spot

## 9. CPU inference pipeline (`bird-clef-2025-models` dataset)

```
.pth (PyTorch fp32) 
  → torch.onnx.export (ONNX fp32)
  → ov.convert_model() (OpenVINO IR fp32)  
  → quantize to fp16 (`onnx_ensem_5first_folds_openvino_fp16`)
  → core.compile_model("CPU")
  → AsyncInferQueue (asynchronous batched inference)
```

`AsyncInferQueue` is the OpenVINO equivalent of `ThreadPoolExecutor` — it overlaps multiple inference requests on the same model on CPU, achieving wall-clock speedup without extra cores.

## 10. The `deep_supervision_steps` mechanism

This is a unique trick in Sydorskyi's model — adding auxiliary classifier heads at MULTIPLE backbone stages, not just the final layer. Loss = main_loss + Σᵢ aux_loss_i (each with smaller weight). Forces the backbone to learn discriminative features at every layer, not just the final stage. **No public BC2026 kernel uses deep supervision.**

## 11. What this round adds — the unique BC2025-2nd-place tricks not in BC2026 corpus

| Trick | Currently in BC2026 corpus? | Source |
|---|---|---|
| **ESC-50 background mixin (dog, rain, insects, engine, etc.)** | No | Sydorskyi `late_aug` |
| **TimeFlip audio reversal** (p=0.5) | No | Sydorskyi `TimeFlip` |
| **RandomLowerHighFreq** spectral cutoff aug | No | Sydorskyi spec_aug |
| **Per-class hand-tuned oversampling (10-96x)** | No | Sydorskyi `OVERSAMPLE_CONFIG` |
| **SqrtBalancing vs EqualBalancing ensemble** | No | Sydorskyi 2 models |
| **F2-score pseudo-label criteria** (not F1) | No | `PseudoF2PT05MT01P04I3` |
| **Pseudo ratio 0.4** (40% pseudo, 60% labeled) | No | `P04` config |
| **2-3 pseudo iterations** (not 1) | Partial (Nikita's BC2025 uses 3-4) | `I2`/`I3` |
| **3 spec extractors: Melspec / CQT / LEAF** | No | Sydorskyi `spec_extractor` |
| **deep_supervision_steps** (auxiliary heads at multi-scale) | No | Sydorskyi model |
| **AsyncInferQueue with OpenVINO** | Partial (others use ThreadPoolExecutor) | Sydorskyi inference |
| **FocalBCELoss + LSF1005** (label smoothing 1.005) | Partial (most use BCE only) | `FocalBCELoss_LSF1005` |
| **Group split by ALL species in clip** (not just primary) | No | `group_allbirds_hdf5` |
| **External XC scrape with custom date** (28032025) | Partial (yasunorim's list) | Sydorskyi data pipeline |
| **From-XC-V2-Best pre-init** (XC-only pretrain warm start) | No | `FromXCV2Best` |

## 12. Concrete plan, expanded

### Tier-A (drop-in <1 hour)
- Add ESC-50 background noise mixup (download esc50 dataset from Kaggle, p=0.5 mix)
- Add `TimeFlip` (audio reversal) augmentation at p=0.2-0.5
- Switch BCE → FocalBCELoss with label smoothing 1.005

### Tier-B (custom training, multi-day)
- Build pseudo-label pipeline with `F2 threshold ≥ 0.5, model threshold ≥ 0.1, ratio 0.4, iterations 2-3`
- Train 2 ensembled models: eca_nfnet_l0 (RAdam) + tf_efficientnetv2_s_in21k (AdamW) at 50 epochs each
- Use SqrtBalancing for one, EqualBalancing for the other
- Per-class hand-oversample the 60 rarest classes (10-96x multipliers)
- Group CV split by ALL species in clip (not just primary_label)

### Tier-C (architectural)
- Try CQT spectrogram (better for harmonic-rich bird/insect calls than mel)
- Try LEAF learnable frontend (jointly trained spec extraction)
- Add deep_supervision_steps with auxiliary heads at intermediate backbone stages

## 13. Sources researched fresh

- [Sydorskyi BC2025 2nd place GitHub](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)
- [Sydorskyi `selected_eca.py`](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place/blob/main/train_configs/selected_eca.py)
- [Sydorskyi `selected_ebs.py`](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place/blob/main/train_configs/selected_ebs.py)
- [Sydorskyi `code_base/models/wave_clasifier.py`](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place/blob/main/code_base/models/wave_clasifier.py)
- [Sydorskyi `code_base/augmentations/transforms.py`](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place/blob/main/code_base/augmentations/transforms.py)
- [Sydorskyi `code_base/models/__init__.py`](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place/blob/main/code_base/models/__init__.py)
- [vladimirsydor/bird-clef-2025-models Kaggle dataset (final OpenVINO models)](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-models)
- [ESC-50 dataset (Piczak 2015) — environmental sounds 50 classes](https://github.com/karolpiczak/ESC-50)
- [LEAF: Learnable Audio Frontend (Google Research)](https://github.com/google-research/leaf-audio)
- [nnAudio CQT1992v2 (Constant-Q Transform on GPU)](https://github.com/KinWaiCheuk/nnAudio)
