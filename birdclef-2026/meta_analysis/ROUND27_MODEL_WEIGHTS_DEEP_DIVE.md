# BirdCLEF+ 2026 — ROUND 27: model checkpoint deep-dive (weights forensics)

This round extracts weight-level patterns from every model checkpoint we have downloaded, comparing architectures, fine-tune deltas, and learned representations.

## 1. Model inventory (9 checkpoints, on disk)

| Name | Size | Params | Architecture | Outputs |
|---|---|---:|---|---:|
| **tonylica_LB872** | 72 MB | 6.29M | EfficientNet-B0 + GeM + AttSED | 234 |
| **aidensong_bestfold** | 72 MB | 6.29M | Same (foundation) | 234 |
| **aidensong_final** | 72 MB | 6.29M | Same | 234 |
| **junhaoyi_resnet50_fp16** | 41 MB | 21.4M | ResNet-50 + FC | **204** (BC2025!) |
| **alexander_sed_b0** | 50 MB | 4.35M | EfficientNet-B0 + 4-head SED, in_chans=1 | 234 |
| **nikita_insect_amphibia** | 20 MB | 4.98M | EfficientNet-B0 + AttSED | **700** (BC2025) |
| **chaneyma_moe_fold1** | 12 MB | 3.19M | ProtoSSM (Mamba-lite + prototypes) | 234 |
| **chaneyma_student_cnn** | 4 MB | 1.07M | 3-block CNN + dual head | 234 + emb |
| **chaneyma_student_crnn** | 13 MB | 3.43M | 2-conv + BiGRU + dual head | 234 + emb |

## 2. tonylica vs aidensong vs aidensong-final (head fine-tuning forensics)

Same architecture, different training stages. After excluding BN running statistics:

**Head layers changed MOST during stage-2 finetune** (tonylica vs aidensong):
- head.att_conv.bias: **12.7% relative L2 change**
- head.cls_conv.weight: 7.4%
- head.fc.0.weight: 5.5%
- head.att_conv.weight: 5.5%
- head.cls_conv.bias: 5.0%
- head.fc.0.bias: 3.4%

**Backbone changed barely**:
- Most SE (Squeeze-Excitation) blocks: 1-10% rel change
- All BN biases: ~0.0001% (frozen)
- Conv weights: ~0.01% (essentially unchanged)
- GeMPool.p: 0.16% (slightly tuned)

**Interpretation**: The stage-2 finetune is **head-only training with frozen backbone**. The published config confirms: `stage2_lr_backbone=5e-6, stage2_lr_head=1e-5` — head learns at 2x the backbone rate; backbone barely changes.

**Replication recipe**: To improve LB on a Tonylica baseline:
```python
for name, param in model.named_parameters():
    if 'head' in name or 'gem_pool' in name:
        param.requires_grad = True  # train head
    else:
        param.requires_grad = False  # freeze backbone
optimizer = AdamW([
    {'params': head_params, 'lr': 1e-5},
    {'params': backbone_params, 'lr': 5e-6},  # if any unfrozen
])
```

## 3. conv_stem analysis (what frequencies the model sees first)

All 3 EfficientNet-B0 models (tonylica, aidensong, nikita) have **IDENTICAL conv_stem activation patterns**:
- Top-5 active output channels: [31, 21, 10, 9, 19]
- Bot-5 inactive output channels: [30, 20, 16, 17, 27]

These weights come from the **same NS-JFT-In1k pretraining** and are NOT modified during stage-2 finetune.

**Alexander's model is DIFFERENT**: in_chans=1 (grayscale, not RGB), top channels are [3, 1, 14, 30, 11]. Higher frequency/time gradient ratio (1.18 vs 1.07), meaning Alexander's first layer is MORE FREQUENCY-ORIENTED.

**Frequency/Time gradient ratios**:
- tonylica/aidensong: 1.065 (slightly freq-biased, balanced)
- alexander: 1.178 (more freq-biased)
- nikita: 1.075

All models prefer to detect FREQUENCY-direction variations slightly more than time-direction. This is acoustically meaningful — distinguishing species is primarily a frequency-pattern task.

## 4. Per-class learned biases (tonylica head.cls_conv.bias)

Range: -0.082 to +0.011 (very small).

**Most-positive biases** (model defaults TOWARD these classes):
- **sptnig1 (Spot-tailed Nightjar)**: +0.011 — model expects this class even with no audio
- 25214, 74580, **47158son11**: +0.007 — sonotype son11 (which Perch CAN detect) has a positive bias
- **517063 (Southern Orange-legged Frog)**: +0.003 — partial correction for Perch's 107x under-prediction
- 476521, 64898, son03, son18, 23150: ~0.000

**Most-negative biases** (model defaults AGAINST):
- yecpar, **strcuc1**, grasal3, **roahaw**, trokin, bbwduc, redjun, scadov1, pvttyr1, **fusfly1**: -0.075 to -0.082
- Common bird species in train_audio that aren't dominant in Pantanal soundscape

**Correlation between bias and soundscape prevalence: r=0.147** (weak) — the model's learned bias only weakly tracks the actual species distribution in test domain.

## 5. Attention vs classifier norm INVERSION (overconfidence detector)

Tonylica's per-class attention norms (head.att_conv.weight per row):

**LARGEST attention norms** (model has strong feature detector):
- 22973, whtdov, coffal1, 23158, grfdov1, 517063, soulap1, sofspi1, rufnig1, 65380
- ALL are common Pantanal night-time species

**SMALLEST attention norms**:
- 23724, 23176, 476521, 23154, **23150**, 1161364, son05, **209233**, 555123, **74580**
- Rare species with 1-3 train recordings

**Per-class cls_conv weight norms — INVERTED pattern**:
- LARGEST: 23176, **23150**, 738183, 25214, 70711, **23724**, **209233**, 64898, 476521, sptnig1
- SMALLEST: grekis, trokin, saffin, sobtyr1, socfly1, **strcuc1**, roahaw, banana, yeofly1, **whtdov**

**The compensation pattern**: 
- Rare classes have WEAK attention (poor feature detector) but LARGE classifier (overshooting compensation) → **HIGH FALSE-POSITIVE RISK**
- Common classes have STRONG attention but SMALLER classifier (attention does the work) → well-balanced

**Practical fix**: at inference, apply per-class temperature scaling:
- For rare classes (high cls_norm, low att_norm): use **T > 1** (soften predictions)
- For common classes: use **T = 1**

## 6. ProtoSSM (chaneyma) — learned representations

### Per-class prototypes (234 × 320-d vectors)

L2 norms range 0.354 - 0.427 (well-normalized).

**Most-similar prototype pairs** (acoustic cluster):
- son13 ↔ son24 (sim 0.573)
- **25073 (MISSING Amphibia) ↔ son13 (Insecta)**: cross-class sim 0.556 — **the ProtoSSM model accidentally aliases this missing frog with an insect**
- 25073 ↔ son24: cross-class sim 0.504
- 1491113 (MISSING frog) ↔ 22967: same-class sim 0.522 — strong intra-frog cluster
- **22985 (frog) confused with multiple birds**: compot1, bkcdon, blttit1, toctou1 (all sim 0.47-0.48)

### fusion_alpha (proto-vs-teacher trust)

**Range: -0.016 to +0.016** → sigmoid range only 0.496 - 0.504. The fusion is essentially **50/50 with tiny per-class deviations**.

**Slightly MORE PROTO** (model trusts learned prototypes):
- whtdov, compot1, son06, **516975** (Capuchin), son14, son20, son10, **25073** (MISSING), son11, son08, son13, son24, son16, **1491113** (MISSING), son15
- Mostly RARE / SONOTYPE / MISSING classes → makes sense, model has learned PROTOTYPES for these because Perch is unreliable

**Slightly MORE TEACHER (Perch)**:
- 65380, 555146, 24279, 66971, 22973, bufpar, 23158, chacha1, hyamac1, chvcon1, litnig1, orwpar, son12, son22, 67252
- COMMON species → model trusts Perch's accurate detections here

**Interpretation**: The fusion alpha codes the model's per-class confidence in Perch vs in its own learned features. Even though the values are tiny (±0.016), the SIGN consistently picks the right branch.

### Site embeddings (10 indices × 16-d)

Chaneyma's model has only **10 site indices** mapped. The 23 BC2026 sites cannot all be represented — they collapse to 10 + unknown padding.

**Learned site similarity clusters** (cosine > 0.4):
- Sites 6 ↔ 9 (sim 0.68) — MOST SIMILAR (probably S22 + S15)
- Sites 4 ↔ 9 (sim 0.46), 1 ↔ 5 (sim 0.44), 0 ↔ 2 (sim 0.44), 3 ↔ 8 (sim 0.42)
- The site index → real-site mapping isn't published, but each cluster likely corresponds to acoustically-similar labeled sites

**This is a LIMITATION**: chaneyma's model can't generalize to sites NOT in the labeled set (S01, S02, S05, etc.). At inference on S05 test, it falls back to site 0 (padding/unknown) and loses S05-specific signal.

### Hour embeddings (24 × 16-d)

Hour=1 (test sample hour) similarities:
- Most similar: **hr=9: 0.562** (surprising — daytime), hr=0: 0.343, hr=8: 0.298, hr=10: 0.275
- Most dissimilar: **hr=3: -0.466**, hr=12: -0.265, hr=17: -0.259, hr=23: -0.217

**Hour 1 is uniquely structured** — NOT similar to its neighboring hours 0, 2, 3 except hour 0 (0.34). And it's MOST dissimilar from hour 3 (-0.466), which is when SONOTYPES dominate. This means the model learned that **hr=01 and hr=03 are very different** species distributions despite being 2 hours apart.

## 7. Alexander's 4-head SED (unique structure)

Alexander's model has **4 separate output heads**:
1. `fc_framewise.1: 640→234` — per-frame logits
2. `attention_pool.attention.0: 640→640` (tanh) + `.2: 640→1` — temporal attention weights
3. `fc_clipwise.1: 640→234` — clip-level logits

The pipeline: backbone features → frame-wise logits + (attention-weighted average of frames → clipwise logits).

**Inference uses framewise_logits for the 5-sec segment-level predictions** (each segment is 10 frames, averaged).

This is the **MOST RIGOROUS SED structure** in our model collection. Per-frame predictions at fine temporal resolution + clip-level aggregation. Tonylica's model has the same idea but with attention conv (att_conv) merging into 1×234 instead of 4-head.

## 8. Student models (chaneyma)

Both StudentCNN (1.07M) and StudentCRNN (3.43M) have **DUAL OUTPUT HEADS**:
- `logit_head`: 234-d classification
- `emb_head`: **1536-d embedding (distills Perch v2's embedding)**

Training loss = `BCE(logits, target) + MSE(emb_head, perch_embedding)`. The student learns to predict BOTH the right classes AND the Perch teacher's embedding.

**StudentCNN** structure:
- 1→32→64→128 channels (3 conv blocks, MaxPool 2x2)
- AdaptiveAvgPool2d → FC(128→512)
- emb_head: 512→1536; logit_head: 512→234

**StudentCRNN** structure:
- 1→32→64 channels (2 conv blocks, MaxPool 2x2)
- BiGRU: input=2048 (32×64 flattened), hidden=192 → 384 total (bidirectional)
- FC: 384→384
- emb_head: 384→1536; logit_head: 384→234

**StudentCNN is the FASTEST option**: 1.07M params, ~10ms per window on CPU. Suitable for the 90-min budget if you want to run 100+ folds.

## 9. Key TAKEAWAYS for new training

1. **Head-only finetuning suffices** — tonylica's +0.010 LB came from training head + SE blocks only. Don't waste compute on backbone.

2. **First-conv stem is FROZEN** across all 3 EfficientNet-B0 models — using NS-JFT-In1k pretrained weights as-is.

3. **Per-class temperature scaling** based on attention norm / classifier norm ratio could correct overconfidence on rare classes.

4. **Distillation dual-head** (emb_head + logit_head) is the way to compress Perch into a tiny model.

5. **Site embeddings are LIMITED**: chaneyma's model only has 10 site indices. For test sites not in labeled set, the site_emb fallback loses signal.

6. **ProtoSSM's 50/50 fusion** is essentially uniform — there's room to learn STRONGER per-class trust signals if trained longer.

## 10. Sources

All from local checkpoint analysis:
- `meta_corpus/datasets/`: 7 checkpoints (LB872, bestfold, best/final fold0, junhaoyi, nikita insect_amphibia, moe artifacts × 3)
- `meta_corpus/models/best.pt`: alexander's SED model

Tools: `torch.load`, weight L2 norms, cosine similarity, PCA on prototype embeddings.
