# BirdCLEF+ 2026 — ROUND 32: BirdTransform — modern LLaMA-style transformer

## 1. The architecture (pulkitsahu89/birdtransform-birdclef-2026-transformer-model)

The first **fully transformer-based** BC2026 model I've found (17.9M params, 68 MB):

```
Audio (1, T)
  ↓
3-layer CNN front-end (1→64→128→512 channels, 3x3 convs)
  ↓
Reshape into sequence of 512-dim tokens
  ↓
4 × Transformer blocks (each with):
  • RMSNorm (norm1, norm2 — only .scale, no .bias)
  • Multi-head attention (Q,K,V projection 512→1536 = 3×512)
  • Output projection 512→512
  • SwiGLU FFN: x = (W1(x) * SiLU(W2(x))).proj(2048→512)
  • RoPE rotary position encoding (inv_freq: 256)
  ↓
Head: Linear(512→512) → activation → Linear(512→234)
```

**This is LLaMA-style architecture applied to audio**:
- RMSNorm instead of LayerNorm
- SwiGLU instead of GeLU MLP
- RoPE instead of learned/sinusoidal positional embedding

**Modern Transformer choices** are uncommon in audio competitions but increasingly used in language modeling.

## 2. Why this matters

- **4 transformer layers × 8-head attention** can model long-range temporal dependencies in audio
- **SwiGLU FFN** has 2x more parameters per FFN (2 weight matrices to 2048) but better activations
- **RoPE** generalizes to longer sequences than training (test soundscape may have longer effective context)
- **17.9M params** is similar to EfficientNet-B0 SED (4-6M) + transformer head — more capacity in the head

## 3. Training metadata

- `species_list.npy`: 234 species (standard BC2026)
- `thresholds.npy`: all set to 0.5 (no per-class calibration)
- Model trained but thresholds not optimized

## 4. Could this be the missing piece?

The single-fold SED models I tested all use:
- EfficientNet-B0 or V2-S backbone (CNN)
- Simple FC head or AttentionSED

None use **transformer attention** as the main processing. BirdTransform is the only model with a pure transformer body.

**Hypothesis**: Long-range temporal dependencies (across the 5-sec window) may be better captured by transformers than CNNs. This could explain why aliozanmemetoglu's "20-sec context for 5-sec prediction" approach works — it leverages long-range patterns the model can capture.

A transformer over the whole 60s audio (12 windows × 5s) could:
- Identify call onset patterns
- Detect call sequences (e.g., bird→silence→bird pattern)
- Use the structural information of when species call in sequence

## 5. Why I couldn't easily benchmark

Pulkit's checkpoint doesn't include:
- The exact mel spectrogram parameters used
- The audio preprocessing pipeline
- The input format (raw waveform or precomputed spec?)

Without the inference code, I can only inspect the architecture, not run it directly.

## 6. Sources

- [pulkitsahu89/birdtransform-birdclef-2026-transformer-model](https://www.kaggle.com/datasets/pulkitsahu89/birdtransform-birdclef-2026-transformer-model)
- Local file: `meta_corpus/datasets/bird_model.pth` (17.9M params)
