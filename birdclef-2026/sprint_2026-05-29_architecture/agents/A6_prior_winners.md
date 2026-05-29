# A6 — Prior BirdCLEF Winners: Actionable Recipe Distillation (2025 primary, 2024 secondary)

**Agent A6** | Sprint 2026-05-29 | Metric: macro ROC-AUC | Submission: CPU ≤90min, no GPU/internet
**Track key:** T1 = doable now, GPU-free (inference / post-proc / blend / code). T2 = needs GPU training, spec'd to run the instant GPU returns.

Sources read at code level: VSydorskyy/BirdCLEF_2025_2nd_place (configs + transforms, via ROUND11), **myso1987/BirdCLEF-2025-5th-place (actual YAML configs + pseudo-label loop pulled this session)**, detkov/BirdCLEFplus_2025 README/journal, Nikita Babych 1st-place writeup (search-snippet extraction — Kaggle writeup is JS-rendered, body not directly fetchable), tekkix top-5 overview, arXiv 2507.08236 (token-distillation paper).

---

## RANKED TECHNIQUE TABLE

| # | Technique | Track | Expected LB lever | Feasibility | Recipe / config detail | Source |
|---|-----------|-------|-------------------|-------------|------------------------|--------|
| 1 | **Multi-iterative noisy-student / self-distillation** on the 10,592 unlabeled soundscapes | **T2** (highest value) | **+0.03–0.06** (Nikita: 0.898→0.930, the single biggest jump) | High value, moderate effort. Need GPU. Pseudocode below is verified from 5th-place loop. | Teacher (5-fold ens) soft-labels unlabeled audio → student trains on labeled+pseudo → student becomes next teacher. **5th place: 6 rounds** (base→re1→re2→re3→re4→re5→tss). **Nikita: 4 rounds power-scaling**. 5th-place exact params (constant every round): `pseudo_alpha=0.7, pseudo_th=0.3, pseudo_power=2`. Mixing rule: `label = 0.7·pseudo + 0.3·hard`. PowerTransform: `p = p·(p>0.3) + p^2`. | 5th-place `train_pesudo_label.py` + YAML; Nikita writeup |
| 2 | **SED head (AttBlockV2 attention pooling)** on a CNN backbone | T2 (train) / T1 (use public 0.958 SED ckpts inference-only) | +0.01–0.02 vs plain pooling; SED corr +0.609 in our meta | T1 inference now via public ckpts; T2 to train fresh | `clipwise = Σ softmax(tanh(att(h)))·sigmoid(cla(h))` over frames. Backbone = timm CNN minus last 2 layers, then `meanpool(freq)→maxpool+avgpool(time)→fc→AttBlockV2`. Input 1ch mel→tiled to 3ch. **logit = Σ norm_att·cla(x)** used for loss (not clipwise prob). | 5th-place `TimmSED`/`AttBlockV2`; MASTER_BOARD (SED corr) |
| 3 | **OpenVINO FP16 + AsyncInferQueue** CPU inference | **T1** | 0 LB directly, but **2–4× speedup → fit 2–4× bigger ensemble in 90min → +0.005–0.015** | Drop-in now, no training | `.pth → torch.onnx.export → ov.convert_model → compress_to_fp16 → core.compile_model("CPU") → AsyncInferQueue`. Async overlaps requests on one model = wall-clock win without extra cores. Both 2nd & 5th place ship OpenVINO models as final submission. | 2nd-place ROUND11; 5th-place convert/inference notebooks |
| 4 | **ESC-50 + no-call soundscape background mixup** (p=0.5) | T2 | +0.005–0.01 (domain robustness to Pantanal farm/weather/insect noise) | Easy add to training; need ESC-50 download | `OneOf([BackgroundNoise(esc50, cats=[dog,rain,insects,engine,rooster,sea_waves,...], p=0.5), BackgroundNoise(custom_nocall_soundscapes, p=0.5)])`. Teaches model to ignore non-bird texture. Mauricio's analogue: Pantanal river overlay −21 to −15 dB SNR. | 2nd-place `late_aug` (ROUND11) |
| 5 | **Separate insect/amphibia model** | T2 | +0.003 (Nikita: 0.930→0.933) | Cheap once main pipeline exists | Dedicated EfficientNet-B0 trained only on "other" (non-bird) classes with +17,197 extra insect/amphibia recordings; predictions slotted into the relevant class columns. BC2026 has the texture/sonotype taxa (son15-son21) — direct analogue. | Nikita writeup; tekkix |
| 6 | **External data: prev-comp audio + custom XC scrape** | T2 (train) — **legality caveat below** | +0.005–0.01 (more data for rare classes) | Need to assemble dataset; reproducibility/no-leak required | 2nd place: `add_train_audio_from_prev_comps` (BC2021-24 overlapping species) + `add_train_audio_from_xeno_canto_28032025`. Nikita: +5,489 XC bird + 17,197 insect/amphibia. **First 30s of recording (60s for rare), upsample classes with <20 samples.** | 2nd-place data pipeline; Nikita |
| 7 | **FocalBCELoss + label smoothing + sumix/mixup** | T2 | +0.003–0.008 | Easy training change | 5th: `Focal_MultiLabel_Loss(γ=2)` = `(1−e^{−bce})^2·bce`. 2nd: FocalBCELoss + LSF1005 (smoothing 1.005). **sumix_freq** (5th): blend two clips with coeffs∈[0.3,1.0], label = soft-weighted mix, clipped [0,1]. | 5th-place loss/`sumix_freq`; 2nd-place ROUND11 |
| 8 | **SoftAUCLoss (direct AUC optimization, soft-label compatible)** | T2 | +0.003–0.005 (metric is AUC) | Custom loss, easy to code | Pairwise positive−negative score differences passed through log-loss; resistant to overfit, accepts soft pseudo-labels. Optimizes the ranking metric directly instead of BCE proxy. | Nikita writeup; tekkix |
| 9 | **Rich SED spec-aug stack** (FilterAug, time/freq mask, gain, resample) | T2 | +0.003–0.005 | Easy training add | 5th-place exact: time-mask×2 (param 10), freq-mask×1 (param 10), random gain ±9 dB, resample ±5%, **FilterAugment** (DCASE2021, random per-band dB gain, 3-6 bands). 2nd adds RandomLowerHighFreq (spectral cutoff) + TimeFlip (reverse audio, p=0.5). | 5th-place `filt_aug`/`time_freq_mask`; 2nd ROUND11 |
| 10 | **Per-class hand-tuned oversampling (rare classes 10–96×)** | T2 | +0.002–0.005 on rare-class macro-AUC | Cheap; partially transferable code list | 2nd place `OVERSAMPLE_CONFIG` hand-tunes 60 rare classes (turvul 96×, piwtyr1 90×…). 5th place: oversample any species with <20 samples. Macro-AUC weights every class equally → rare-class recall matters disproportionately. | 2nd ROUND11; 5th `train_meta` |
| 11 | **Group CV split by ALL species in clip (no-leak)** | T2 (correct CV) | Indirect — prevents OOF over-optimism / bad model selection | Free, just split design | Group folds so multi-species recordings stay together; `_noleak` variant for rare birds. Avoids co-occurrence leakage inflating CV. | 2nd ROUND11 |
| 12 | **Diverse-backbone ensemble** | T1 (public ckpts) / T2 | +0.005–0.015 | T1 now: blend public checkpoints | Nikita final: EffNet-l0,B4,B3 + RegNetY-016, RegNetY-008 + B0(insect). 2nd: eca_nfnet_l0(RAdam,SqrtBalance) + effv2_s(AdamW,EqualBalance). 5th: B0+B3+effv2-b3+effv2-s. **Diversity from backbone×optimizer×sampling**, same loss/scheduler. | Nikita; 2nd; 5th |
| 13 | **20-sec (long) context input** | T2 | +0.003 (Nikita uses 20s; Alexander too) | Training change | Nikita/Alexander 20s window; 5th uses 10s window from 30s crop; 2nd uses 5s. Longer context helps sparse-call detection but costs inference. | Nikita; ROUND34 |
| 14 | **Silero-VAD voice/no-call separation** | T2 | +0.002 (cleaner pseudo-label targets) | Need VAD preprocess | detkov used `kdmitrie/bc25-separation-voice-from-data-by-silero-vad` to strip human-voice/no-call regions before training. | detkov README |
| 15 | **Token-distillation of mel into compact transformer** | T2 (research) | speed play, not accuracy | High effort, low priority for 5-day window | arXiv 2507.08236: distill spectrogram→token sequence for fast lightweight inference. Interesting but unlikely to land in budget. | arXiv 2507.08236 |

---

## THE NOISY-STUDENT PROTOCOL — runnable pseudocode (verified from 5th-place `train_pesudo_label.py`)

This is the #1 T2 play. We have 10,592 unlabeled `train_soundscapes` — exactly the regime this exploits.
5th place did **6 self-distillation rounds**; Nikita did **4 power-scaling rounds**. Sweet spot from both: **3–4 rounds**, soft labels, low threshold.

```python
# ---- HYPERPARAMETERS (5th-place values, verified) ----
PSEUDO_ALPHA = 0.7    # weight on pseudo vs hard label:  label = 0.7*pseudo + 0.3*hard
PSEUDO_TH    = 0.3    # below this, the linear pseudo term is zeroed (only power term survives)
PSEUDO_POWER = 2      # PowerTransform exponent — sharpens confident preds, suppresses noise
N_ROUNDS     = 4      # 3-4 is the sweet spot (Nikita 4, 5th 6 incl. final "tss" stage)
# 2nd-place analogue ("F2PT05MT01P04I3"): F2-score selection, prob>0.5, model>0.1,
#   pseudo-ratio 0.4 (40% pseudo / 60% labeled), 2-3 iterations.  DO NOT exceed ratio ~0.4.

def power_transform(p):                       # the core soft-label transform
    # zero out weak signal, keep sharpened confident signal
    return p * (p > PSEUDO_TH) + p**PSEUDO_POWER          # then clamp [0,1]

def make_soft_targets(teacher_models, mel_batch, hard_labels):
    # 1. average the 5-fold teacher ensemble (soft probs, NOT argmax)
    p = mean(sigmoid(m(mel_batch)['logit']) for m in teacher_models)   # (B, n_classes)
    # 2. PowerTransform
    p = clamp(power_transform(p), 0.0, 1.0)
    # 3. blend with hard labels (labeled rows have real hard_labels;
    #    pure-unlabeled soundscape rows pass hard_labels=0 so target = 0.7*pseudo)
    return PSEUDO_ALPHA * p + (1 - PSEUDO_ALPHA) * hard_labels

# ---- ITERATIVE LOOP ----
teacher = train_supervised(labeled_data)            # round 0: 5-fold on train_audio (Focal, Adam,
                                                    #          cosine+warmup, 10 epochs, sumix+specaug)
for r in range(N_ROUNDS):
    student = init_model(pretrained=True)           # fresh init each round (noisy-student: student
                                                    #   bigger/equal + MORE aug than teacher)
    data = labeled_data + unlabeled_soundscapes     # 5th: all data; 2nd: cap pseudo-ratio ≈0.4
    for epoch in range(EPOCHS):
        for mel, hard in dataloader(data):
            with torch.no_grad():
                soft = make_soft_targets(teacher_ens, mel, hard)
            mel, soft = sumix_freq(mel, soft)        # noise injection on the STUDENT (key!)
            mel = filt_aug(time_freq_mask(mel))      # heavier aug than teacher saw
            loss = focal_bce(sigmoid(student(mel)['logit']), soft)
            loss.backward(); opt.step()
    teacher_ens = kfold_checkpoints(student)         # student -> next round's teacher
    # optional final "tss" stage: re-label with a STRICTER threshold (5th: pseudo_tss_th=0.7,
    #   power=2) to clean targets one last time before the final model.

# Expected lever: +0.03-0.06 macro-AUC (Nikita 0.898->0.930). Diminishing returns after ~4 rounds.
# T1 note: we can PRE-COMPUTE pseudo-labels for train_soundscapes NOW (CPU, inference-only) using
#   public 0.958 SED checkpoints, so the dataset is ready the instant GPU returns.
```

**Why soft (not hard) pseudo-labels:** macro ROC-AUC is a ranking metric. Hard thresholding destroys the score ordering the metric rewards and injects label noise on the long tail. Both Nikita and the 5th place keep probabilities soft and merely *sharpen* them via PowerTransform. 2nd place's caution (`naive pseudo hurts LB`, BC2026 disc 694815) is resolved by (a) soft labels, (b) capped pseudo-ratio ≤0.4, (c) F2/recall-biased selection.

---

## EXTERNAL DATA / PRETRAINING — legality for 2026

| Resource | Used by | Usable in BC2026? | Notes |
|----------|---------|-------------------|-------|
| **Google Perch v2** (frozen embeddings) | everyone in 2026 | **Yes** — host-endorsed starting point | BC2026 explicitly frames Perch v2 as the canonical transfer base. Our 0.948 plateau is the Perch ceiling. |
| **Xeno-Canto recordings (CC-licensed)** | Nikita +5,489, 2nd custom scrape | **Yes IF** CC-licensed, referenced, publicly accessible, and excludes test obs | **Crawling xeno-canto.org directly is forbidden** per CLEF rules — must use an already-published CC dataset (e.g. yasunorim XC-URL list) and respect per-recording license. |
| **Prev-competition audio (BC2021-2025)** | 2nd `add_train_audio_from_prev_comps`, Nikita | **Yes for overlapping species** that are CC and not in BC2026 test | BC2025 species codes are shared with BC2026 for overlapping species → checkpoints/audio partially transferable. BC2025 used a different species set (204) so direct ckpt reuse needs head remap. |
| **ESC-50** (environmental sounds) | 2nd-place background aug | **Yes** — CC BY-NC, public, not test data | Use as background-noise augmentation only. |
| **iNaturalist audio** | not seen in top-5 | Conditionally — must verify CC license per clip | Not a proven lever; low priority. |
| **Public BC2026 SED 0.958 5-fold ckpts** (aliozanmemetoglu) | 0 of 1,194 kernels | **Yes** — public Kaggle dataset, inference-only | Flagged in MASTER_BOARD as free T1 lever. |

**Rule of thumb for 2026:** any external resource must be (1) publicly accessible to all, (2) clearly referenced for reproducibility, (3) free of test observations. CC-licensed published datasets are fine; live crawling of XC is not.

---

## INFERENCE TRICKS TO FIT BIG ENSEMBLES IN 90-min CPU BUDGET

1. **OpenVINO IR + FP16 compression** — `ov.convert_model` then `compress_to_fp16`; ~2× faster than ONNX-CPU, near-zero AUC loss (both 2nd & 5th ship this as final).
2. **AsyncInferQueue** — overlap inference requests on one compiled model; wall-clock speedup without extra cores (2nd place). Superior to ThreadPoolExecutor others use.
3. **5-sec / 10-sec windowing with batched mel precompute** — compute all mels once, batch the CNN forward.
4. **Frozen-embedding shortcut** (Bruce Wu) — Perch ONNX embeddings → tiny sklearn Ridge head; the 0.93 baseline costs almost nothing, leaves the whole budget for the heavy SED ensemble.
5. **Single shared mel frontend** across ensemble members where params match (n_fft 2048, hop 512/768, 32 kHz) — avoid recomputing spectrograms per model.

---

## SOURCES
- Nikita Babych 1st-place writeup: https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
- 5th-place repo (configs + pseudo loop read this session): https://github.com/myso1987/BirdCLEF-2025-5th-place-solution
- 5th-place inference/convert (OpenVINO): https://www.kaggle.com/code/zuoliao11/birdclef2025-inference-openvino , https://www.kaggle.com/code/zuoliao11/birdclef2025-convert-models-openvino
- 2nd-place repo: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place (full forensics in meta_analysis/ROUND11_BC2025_2ND_PLACE_FORENSICS.md)
- detkov solution (Silero-VAD): https://github.com/detkov/BirdCLEFplus_2025
- Top-5 overview: https://tekkix.com/articles/ai/2025/07/birdclef-2025-overview-of-the-competition-a
- Token distillation paper: https://arxiv.org/pdf/2507.08236
- Noisy Student (Xie et al.): https://arxiv.org/pdf/1911.04252
- FilterAugment (DCASE2021): https://github.com/frednam93/FilterAugSED
