# BirdCLEF+ 2026 — ROUND 15: recently-uploaded datasets + new architectures

These datasets were uploaded to Kaggle in the last 10 days (as of 2026-05-17) and have been overlooked by most of the public discussion.

## 1. `samuelzxu/bc26-iter1-pseudo-labels` (May 16, 0 votes, 1 download)

A clean iter-1 pseudo-label set with strict criteria:

| Stat | Value |
|---|---:|
| Total pseudo-positives | 11,582 (file, end_sec, label) tuples |
| Unique files (out of 10,592 unlabeled) | 2,573 (24%) |
| Unique labels covered | **232 of 234** (only 2 missing!) |
| Min probability | 0.85 |
| Max probability | 1.0 |
| Mean probability | 0.999 |
| **% with prob ≥ 0.99** | **99.1%** |

**Per-class capping**: max 50 pseudo-positives per class, min 32. This is the **balanced pseudo-label strategy** — explicit capping to prevent head-class domination.

```python
# Reproduction recipe
for class_label in all_classes:
    candidates = pseudo_predictions[(pseudo_predictions['label']==class_label) 
                                    & (pseudo_predictions['prob'] >= 0.85)]
    top_50 = candidates.nlargest(50, 'prob')
    final_pseudo.append(top_50)
```

This contradicts the BC2025 2nd-place ratio approach (40% pseudo by mass). The per-class cap is a stronger guarantee against class imbalance. **No public BC2026 kernel uses this approach.**

## 2. `samuelzxu/bc26-iter1-perch-cache` (May 16, 786 MB, 0 downloads)

This is a SECOND pseudo-cache (distinct from `backtracking/birdclef2026-pseudo-cache-v1`). The "v2" suffix on the manifest suggests a refined iteration. 786 MB vs the original 441 MB — almost 2x the data. Worth investigating but I haven't downloaded due to disk constraints.

## 3. `junhaoyi/birdclef-2026-best-model-auc-r2-fp16` (May 16, 0 downloads)

A 42 MB fp16 weights file. Forensics:

```python
ckpt = torch.load('best_auc_r2_fp16.pt')
# Type: dict with keys ['model', 'classes']
# classes: 204 entries (BC2025 class count, NOT 234)
# model: ResNet-50 backbone (21.4M params)
#   - backbone.conv1, bn1, layer1-4 (classic ResNet)
#   - backbone.fc.weight: [204, 512]
```

**Wait — this has 204 classes, not 234.** It's likely:
- A BC2025 ResNet-50 reused as feature extractor
- Or a 204-class subset (excluding 30 BC2026 classes)
- fp16 quantized for CPU inference

**Caution**: Don't blindly use this in BC2026 ensemble without checking class mapping.

## 4. `majkel1337/long-convnextv2-tiny-onnx` (May 15, 16 downloads)

ConvNeXtV2-tiny in "long" variant. ONNX inspection:

```
INPUT: waveform [batch, 1920000]   # 60-second raw audio at 32 kHz
OUTPUT: window_logits [1, 12, 234] # 12 × 5-sec windows × 234 classes
graph nodes: 446
producer: pytorch (exported)
```

**Key insight**: The "long" model takes the ENTIRE 60-sec file as input and outputs 12 window predictions in ONE forward pass. This is different from the typical 5-sec-per-pass approach. **Saves 12x model inference time** if the model latency is dominated by Python overhead per call.

Similar `token-convnext-tiny-onnx` and `long-swin-tiny-onnx` variants:
- ConvNeXtV2-tiny (Woo et al. 2023, `arxiv:2301.00808`) — successor to ConvNeXt with Global Response Normalization (GRN)
- Swin-tiny — Vision Transformer with shifted windows (Liu et al. 2021, `arxiv:2103.14030`)

**NEITHER architecture is used by any public BC2026 corpus kernel.** Adding ConvNeXtV2-tiny + Swin-tiny to your ensemble = **architectural diversity not in the public template**.

## 5. `irinafayzrakhmanova/birdclef2026-full-spec-cache` (May 16, 10.6 GB!, 4 downloads)

Full precomputed spectrogram cache — 10.6 GB. Eliminates the spectrogram computation step during training (saves ~30% of training time per epoch).

## 6. `danielfreiremendes/birdclef-2026-template-1..7` (May 15-16)

Seven templates of approaches:
- Template 1: SimpleCNN (1.6 MB) — minimal baseline
- Template 2: (missing from list — possibly removed)
- Template 3: EfficientNet-B0 (16 MB)
- Template 4: CRNN-Attention (18 MB)
- Template 5: AST (323 MB) — Audio Spectrogram Transformer!
- Template 6: BirdNET + MLP (2.6 MB)
- Template 7: Perch + MLP (5 MB)

**AST (Audio Spectrogram Transformer, Gong et al. 2021, `arxiv:2104.01778`)** is a third Vision Transformer for audio. 323 MB suggests the full-size AST-base. **Not used in BC2026 corpus** but available.

## 7. `bleachonn77/birdclef-2026-pseudo-labels-iter1` (May 12, 10 downloads)

55 MB parquet file with pseudo-labels from iteration 1. Similar concept to samuelzxu's but from a different team.

## 8. `alexycactus/birdclef-2026-cnn-fold-checkpoints` (May 17, 3 downloads)

86 MB CNN B0 SED 5-fold checkpoints. Recent (today). Worth pulling and inspecting metadata.

## 9. Architectural diversity menu (consolidated)

The full architecture pool available to a BC2026 ensemble:

| Backbone family | Variants | Source | In public corpus? |
|---|---|---|---|
| EfficientNet (CNN) | b0, b1, b3, b4 | aliozanmemetoglu, tonylica, alexandergremyakov | ✓ widely |
| EfficientNetV2 (CNN) | s, m, b0 | aliozanmemetoglu (V2-S+B0), tsubasatech (V2-M) | ✓ widely |
| HGNetV2 (CNN) | B0 | ttahara | ✓ widely |
| ConvNeXt (CNN) | tiny | tsubasatech | ✓ rarely |
| **ConvNeXtV2 (CNN)** | tiny | majkel1337 | **✗ (May 15 upload)** |
| **EfficientVit (hybrid)** | b0, b1, m3 | BC2024 jfpuget | **✗** |
| **Swin (ViT)** | tiny | majkel1337 | **✗ (May 15 upload)** |
| **AST (ViT)** | base | danielfreiremendes Template 5 | **✗** |
| MNasNet (CNN) | 100 | BC2024 winners | ✗ |
| MobileNet (CNN) | v3 | BC2024 winners | ✗ |
| MixNet (CNN) | s/m | BC2024 winners | ✗ |
| NFNet (CNN) | eca_nfnet_l0 | Sydorskyi BC2025 | ✗ |
| RegnetY (CNN) | 008, 016 | Nikita Babych BC2025 | ✗ |
| ResNet (CNN) | 50 | junhaoyi | ✗ |
| Perch v2 (EffNet-B3 custom) | frozen + finetune | hengck23 | ✓ (most use frozen) |
| BirdNET (CNN) | frozen | several | ✓ |
| CLAP (audio-text contrastive) | int8 | habedi | ✗ |
| EnCodec (neural codec) | base | DS@GT BC2024 | ✗ |

**8 architectures** widely available but NOT used in the BC2026 corpus. The public ensemble is dominated by EfficientNet variants; adding ConvNeXtV2, EfficientVit, Swin, AST, NFNet, RegnetY would each add diversity.

## 10. Concrete plan additions (Tier-A/B/C)

### Tier-A — drop-in (< 1 hour)
- Use `samuelzxu/bc26-iter1-pseudo-labels` as a vetted pseudo-label set (top 50 per class, prob ≥ 0.85)
- Add `majkel1337/long-convnextv2-tiny-onnx` as ensemble member with weight 0.10-0.15

### Tier-B — config (half-day)
- **Long-format inference**: feed 60-sec audio directly, get 12 × 234 outputs (12x faster than 5-sec per-pass)
- Pre-cache spectrograms (use `irinafayzrakhmanova/birdclef2026-full-spec-cache` or compute locally)
- Use **per-class cap of 50** for pseudo-labels (samuelzxu approach) — prevents head class domination

### Tier-C — architecture diversity (multi-day)
- Build a 4-model ensemble: EfficientNet-B0 + ConvNeXtV2-tiny + EfficientVit-b0 + Swin-tiny
- Each trained on the same pseudo-labels but with different image resolutions
- Distill the 4-model ensemble into a single fast student (BC2024 1st-place pattern)

## 11. Sources

- [samuelzxu/bc26-iter1-pseudo-labels](https://www.kaggle.com/datasets/samuelzxu/bc26-iter1-pseudo-labels)
- [samuelzxu/bc26-iter1-perch-cache](https://www.kaggle.com/datasets/samuelzxu/bc26-iter1-perch-cache)
- [junhaoyi/birdclef-2026-best-model-auc-r2-fp16](https://www.kaggle.com/datasets/junhaoyi/birdclef-2026-best-model-auc-r2-fp16)
- [majkel1337/long-convnextv2-tiny-onnx](https://www.kaggle.com/datasets/majkel1337/long-convnextv2-tiny-onnx)
- [majkel1337/long-swin-tiny-onnx](https://www.kaggle.com/datasets/majkel1337/long-swin-tiny-onnx)
- [majkel1337/token-convnext-tiny-onnx](https://www.kaggle.com/datasets/majkel1337/token-convnext-tiny-onnx)
- [danielfreiremendes/birdclef-2026-template-5 (AST)](https://www.kaggle.com/datasets/danielfreiremendes/birdclef-2026-template-5)
- [danielfreiremendes/birdclef-2026-template-7 (Perch + MLP)](https://www.kaggle.com/datasets/danielfreiremendes/birdclef-2026-template-7)
- [Woo et al. 2023 — ConvNeXtV2 paper (`arxiv:2301.00808`)](https://arxiv.org/abs/2301.00808)
- [Liu et al. 2021 — Swin Transformer (`arxiv:2103.14030`)](https://arxiv.org/abs/2103.14030)
- [Gong et al. 2021 — AST: Audio Spectrogram Transformer (`arxiv:2104.01778`)](https://arxiv.org/abs/2104.01778)
- [Cai et al. 2023 — EfficientViT (`arxiv:2305.07027`)](https://arxiv.org/abs/2305.07027)
