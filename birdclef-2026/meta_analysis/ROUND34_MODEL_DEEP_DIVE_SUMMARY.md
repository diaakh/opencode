# BirdCLEF+ 2026 — ROUND 34: Final model deep-dive summary (Rounds 27-33)

## 1. Complete model inventory analyzed

I downloaded, inspected, and (where possible) ran inference on 15+ public model bundles:

| Owner | Bundle | Type | Macro-AUC on labeled OOF |
|---|---|---|---:|
| **Bruce Wu** | clip_student_bundle.pkl + Perch ONNX | Ridge regression on PCA(Perch) | **0.9304** ⭐ |
| Tonylica | LB872.pt + LB862.pt | EfficientNet-B0 + GeM + AttSED | (not run, has LB 0.957) |
| aidensong123 | bestfold/best_fold0.pt | Same as Tonylica (foundation) | (foundation) |
| **Alexander** | sed-b0-ce-nospecaug | EfficientNet-B0 + 4-head SED, gray | 0.6344 |
| Nikita Babych | BC2025 ensemble | 9 SED models, multi-iter pseudo | (BC2025 LB 0.937) |
| Mauricio | exp-034 sparse fusion safe | EffNet-B0 from scratch + Pantanal river aug | 0.6232 |
| baiyuby | distill-models fold0 | EfficientNetV2-S gray + dual head | (CV 0.972, requires their pipeline) |
| chaneyma | MoE artifacts | ProtoSSM (Mamba) + student CNN + student CRNN | (CV 0.9245) |
| junhaoyi | best_auc_r2_fp16 | ResNet-50, 204 classes (BC2025) | (incompatible) |
| alexanterkapai | model_fold0 | EfficientNetV2-S + simple FC | (not benchmarked) |
| pulkitsahu89 | BirdTransform | CNN + 4-layer LLaMA transformer (RoPE+SwiGLU+RMSNorm) | (not run, no preprocessing) |
| majkel1337 | long-convnextv2-tiny | ConvNeXtV2-tiny, 60s→12×234 | 0.6303 |
| **tsubasatech** | snowflake-sed | ConvNeXt-tiny + EffNetV2-M, 5s SED | 0.5988 / 0.6010 |
| habedi | CLAP int8 bundle | 5-fold linear probes on CLAP int8 | (not benchmarked) |
| **emoptisie** | effb0.onnx | EfficientNet-B0 RGB 224×224 | (not run, mel preproc unknown) |
| michaelihc | hybrid bundle | PCA(64)+linear probes for 52 classes | (OOF 0.834 self-reported) |
| **yuyajk** | effnet-b0-soft-pseudo-exp0273 | 5-fold ResNet-EffB0 + 50% pseudo + mixup | (CV 0.616) |
| Pseudo cache hour prior | (my derivation from `pseudo_cache`) | Lookup table per hour | 0.9175 |

## 2. The 3-tier model performance hierarchy

**Tier 1 (0.92+ OOF macro-AUC):**
- Bruce Wu Ridge on Perch (0.9304)
- Pseudo hour prior alone (0.9175)
- Bruce + 3·log(pseudo prior) ⭐ best: **0.9586**

**Tier 2 (0.60-0.65 OOF):**
- All single-fold CNN SED models (Alexander, Mauricio, long_convnext, Snowflake variants, yuyajk)
- ALL converge to ~0.60-0.65 because they:
  - Trained on similar data (train_audio + train_soundscapes labels)
  - Use similar augmentation
  - Use similar architectures (EfficientNet-B0 / V2-S / ConvNeXt-tiny)
  - 1 fold doesn't generalize well

**Tier 3 (~0.50 OOF):**
- Raw Perch v2 (frozen)
- Bare model output without any post-processing

## 3. The CRITICAL insight: per-class optimal routing

Oracle analysis (pick best model per class on labeled OOF):

| Best model | # classes won | Class types won |
|---|---:|---|
| **Bruce** | 34 | Common Pantanal frogs + birds (well-trained classes) |
| **Hour prior** | 32 | Missing classes (sonotypes, rare frogs — no audio info needed) |
| **Alex SED** | 5 | son15, son16, son18, son20 + thlwre1 (perfect 1.0 AUC!) |
| **Mauricio** | 2 | son17, son19 |
| **long_convnext** | 1 | son21 |
| **snowflake_cn** | 1 | plcjay1 (1 positive) |

**Oracle macro-AUC: 0.9570** (slightly worse than global blend 0.9586).

**Practical heuristic routing**: 0.9541 — slightly worse than global blend.

**Conclusion**: A SIMPLE WEIGHTED BLEND beats per-class routing because:
1. The per-class winners overfit to the small labeled set (only 12-24 positives per sonotype)
2. The global blend captures cross-class regularities

## 4. Common training config patterns I extracted

From inspecting all model configs:

```python
# Most popular config (5+ models use this)
config = {
    'backbone': 'tf_efficientnet_b0' or 'tf_efficientnet_b0.ns_jft_in1k',
    'in_chans': 1 (4 models) or 3 (4 models),
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 128 or 224 or 256,
    'f_min': 20 (3 models) or 50 (1) or 0 (rest),
    'f_max': 16000,
    'sample_rate': 32000,
    'duration': 5 sec (most) or 10 sec (Mauricio, yuyajk) or 20 sec (Nikita, Alexander),
    'lr': 5e-4 (aidensong) or 1e-3 (Mauricio) or 2e-4 (yuyajk),
    'optimizer': 'AdamW',
    'weight_decay': 1e-4 to 1e-2,
    'epochs': 3-15,
    'batch_size': 16 to 64,
    'mixup_alpha': 0.2-1.0,
    'loss': 'BCE' or 'BCE + Focal' or 'softmax_ce' (Alexander!),
    'use_amp': True,
}
```

## 5. Unique training tricks per model

| Owner | Unique trick |
|---|---|
| **Mauricio** | Pantanal RIVER background overlay at -21 to -15 dB SNR, anchored stage2 same-label reinforcement |
| **Sydorskyi (BC2025 #2)** | ESC-50 (dog/rain/insect/engine) background augmentation |
| **Nikita Babych (BC2025 #1)** | 20-sec context input, dedicated insect_amphibia model, multi-iter pseudo (3-4 rounds) |
| **Bruce Wu** | Sklearn Ridge on PCA(Perch features) — fastest 0.93 baseline |
| **chaneyma** | ProtoSSM (Mamba-style SelectiveSSM) + per-class fusion_alpha + top-2 amplification |
| **aliozanmemetoglu** | Texture/event smoothing [0.35,0.30,0.35] vs [0.20,0.60,0.20] |
| **alexander** | softmax_ce loss (NOT BCE!), 20-sec context, custom asymmetric temporal smoothing |
| **yuyajk** | pseudo_soundscape_ratio=0.5, pseudo_weight_scale=0.55, train_audio_max=2000 per class |

## 6. The optimal inference recipe (validated on labeled OOF)

```python
import numpy as np, pandas as pd, pickle
import onnxruntime as ort

# Load Bruce's trained Ridge bundle
with open('clip_student_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)
scaler = bundle['clip_bundle']['emb_scaler']
pca = bundle['clip_bundle']['pca']
fscaler = bundle['clip_bundle']['feature_scaler']
ridge = bundle['clip_bundle']['model']

# Load Perch ONNX
perch = ort.InferenceSession('perch_v2_no_dft.onnx', providers=['CPUExecutionProvider'])

# Load my pseudo_hour_priors (derived from backtracking/pseudo-cache-v1)
hour_priors = pd.read_csv('pseudo_hour_priors.csv').set_index('hour')

def predict_test_window(audio_5sec, hour):
    # 1. Perch features
    out = perch.run(None, {'inputs': audio_5sec[np.newaxis]})
    emb = out[0]  # (1, 1536)
    logits = out[1]  # (1, 234)
    
    # 2. Bruce's Ridge pipeline
    emb_scaled = scaler.transform(emb)
    emb_pca = pca.transform(emb_scaled)
    features = np.concatenate([emb_pca, logits], axis=1)  # (1, 490)
    features = fscaler.transform(features)
    bruce_logits = ridge.predict(features)  # (1, 234)
    
    # 3. Apply hour prior
    prior = hour_priors.loc[hour].values  # (234,)
    combined = bruce_logits[0] + 3.0 * np.log(prior + 1e-7)
    
    return 1 / (1 + np.exp(-combined))
```

**Expected performance**:
- Labeled OOF: **0.9586 macro-AUC**
- Private LB: likely **0.94-0.95** (typical OOF→test gap)

## 7. To push beyond 0.95 LB

Would require:
1. **Multi-iterative noisy student** (Nikita BC2025): +0.03 to +0.05
2. **5-fold ensemble** of diverse backbones: +0.005 to +0.015
3. **Per-class isotonic calibration** (hideyukizushi): +0.005
4. **Mauricio's Pantanal river augmentation**: +0.005 estimated
5. **Custom-trained student on pseudo_cache embeddings**: +0.01 to +0.02

Combined gain: +0.05 to +0.10 (i.e., 0.94 → 0.99 — but with diminishing returns).

**The Yannan Chen private 0.962 ceiling** likely combines all of these.

## 8. Final actionable summary

For a competitor wanting to maximize LB:

### Tier-A (immediate, no training, < 1 hour)
1. Download `brucewu1200/birdclef-2026-cvlb-assets-0911` (Bruce's clip_student_bundle.pkl)
2. Download `rishikeshjani/perch-onnx-for-birdclef-2026` (Perch ONNX)
3. Use my `pseudo_hour_priors.csv` (saved in this repo)
4. Inference: Bruce + 3*log(prior) → 0.95 LB target

### Tier-B (custom training, multi-day)
1. Replicate Mauricio's anchored_stage2_augmentation pipeline
2. Train 5-fold ensemble: B0 grayscale + V2-S grayscale + ConvNeXt-tiny + EffVit-b0
3. Add Nikita's multi-iter pseudo (3 rounds)
4. Apply per-class calibration from labeled OOF

### Tier-C (research)
1. BirdTransform-style LLaMA transformer for temporal modeling
2. Custom Perch fine-tune (using hengck23's PyTorch port)
3. Multi-source mixup (Perch 2.0 paper)

## 9. Sources

All findings derived from local model analysis + Kaggle dataset downloads. 12+ public bundles inspected. 6 models actually run on the labeled 739-row OOF benchmark.
