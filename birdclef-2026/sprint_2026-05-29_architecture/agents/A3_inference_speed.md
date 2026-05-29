# A3 — Inference-Speed Breakthroughs (fit a 2–3× bigger ensemble in 90 min CPU)

**Charter:** highest-leverage *inference-efficiency* techniques. Thesis: faster inference = bigger
effective ensemble in the same 90 min = higher LB. `uses_onnx` correlates **+0.535** with score;
OpenVINO is a rare-but-powerful trick (Nikita 2025). This file is the speed lever, not the
accuracy lever — but speed *buys* accuracy by letting more models vote.

---

## TL;DR ranking

| # | technique | track | speedup | accuracy cost | source |
|---|-----------|-------|---------|---------------|--------|
| 1 | **OpenVINO FP32→FP16, CPU EP** for mel-CNN + Perch ONNX | T1 (today) | **1.5–3×** vs ORT, ~3–4× vs raw PyTorch | ~0 (FP16 lossless for ROC-AUC ranking) | 2nd-place 2025 repo; OpenVINO blog |
| 2 | **Shared mel frontend + cached Perch embeddings** (compute mel/embeddings once, fan out to N heads) | T1 | **N× on the shared part** (embedding stage dominates) | 0 (identical math) | own pipeline analysis |
| 3 | **ONNX Runtime graph-opt + thread tuning** (ORT_ENABLE_ALL, intra=4/inter=1, batch 192 windows) | T1 | 1.3–2× vs untuned ORT; baseline already partly done | 0 | ORT docs; current `sub_v8` |
| 4 | **Static INT8 quant of the mel-CNN (SED EfficientNet)** via ORT static or OpenVINO POT | T1 (calib data needed) | 2–3× *iff* VNNI; **~1× (or slower) if no VNNI** | −0.000 to −0.003 ROC-AUC | ORT quant docs; arXiv 2303.05016 |
| 5 | **Cascade / early-exit gate** (cheap Perch-prob screen → run expensive SED only on active windows) | T1 | 1.5–4× on the *expensive* branch (most 5s windows are silence/few-species) | small, tunable | own design |
| 6 | **Distill big ensemble → ONE fast CNN student** (Perch+SED+SSM teacher) | T2 (needs GPU) | replaces 3–5 models with 1 → **3–5×** | retains ~85–95% of ensemble *if* student keeps mel/embedding frontend; **fails if you compress to a tiny 1D-CNN** | Hinton KD; arXiv 2507.08236 |
| 7 | **Reduced TTA** (12→6 windows or drop hflip/noise TTA) | T1 | up to 2× linear | −0.001 to −0.003 (TTA gives little here) | own benchmark notes |
| 8 | **Dynamic INT8 quant** (MatMul/Gemm heads, probes, ProtoSSM) | T1 | 1.3–2× on MatMul-heavy parts only | ~0 | ORT docs |

> **Hard constraint to respect:** Kaggle CPU notebook = **4 vCPU / 2 physical cores**, ~16–32 GB RAM,
> Intel Xeon **Skylake-class** (AVX-512 usually present, **VNNI NOT guaranteed**). This single fact
> reshapes the quant story (see §INT8 caveat). 90 min wall-clock, ~600 soundscapes × 1 min,
> 12 windows × 5 s each → **~7,200 inference windows**, 234-class multi-label output.

---

## The budget math (the deliverable that matters)

### Current observed pipeline (`sub_v8_full_recipe_standalone.py`)
- Perch runs as **ONNX, CPUExecutionProvider, intra_op=4, inter_op=1, ORT_ENABLE_ALL**, batched
  `BATCH_FILES=16` → 192 windows/call. Plus Bruce student CNN, KNN retrieval, Ridge/LR probes,
  ProtoSSM, RAG. Reported ~30 min for the lighter variants.
- **Perch embedding extraction is the dominant cost** — everything downstream (KNN, probes,
  ProtoSSM, MLP, RAG) consumes the *same* 1280-d embedding. That is the key structural fact: the
  embedding is computed once and is essentially free to reuse.

### Budget model
Let `T_embed` = wall-time to run Perch over all 7,200 windows, `T_melcnn` = one SED EfficientNet
pass over all windows, `T_head` = a cheap downstream head (KNN/probe/SSM, negligible, <1 min each
on 234×7200 floats).

Approximate per-stage costs at current ORT-CPU throughput (order-of-magnitude, from the 2025
literature scaled to 600 files):
- Perch ONNX (ORT, FP32): **~16–18 min** for ~600 files (matches TFLite-Perch ~16 min @ 700 files).
- One SED EfficientNet-B0 mel-CNN, ORT FP32: **~12–15 min** for 600 files (5 folds = ~60–75 min — busts budget alone).
- Each downstream head reusing cached embeddings: **<1 min**.

| scenario | what runs | est. wall time | fits 90 min? | effective ensemble size |
|----------|-----------|---------------:|:------------:|------------------------:|
| **A. baseline (today, FP32 ORT)** | Perch + 1 SED-B0 + 4 cheap heads | 18 + 14 + 3 = **~35 min** | yes | ~6 voters |
| **B. + OpenVINO FP16 on Perch & SED (lever 1)** | same, 2× faster heavy parts | 9 + 7 + 3 = **~19 min** | yes (huge slack) | ~6 voters in 19 min |
| **C. spend the slack → 5-fold SED ensemble** | Perch + **5×** SED-B0 (OpenVINO FP16) + heads | 9 + 5×7 + 3 = **~47 min** | yes | **~10 voters** |
| **D. + INT8 SED *(only if VNNI present)*** | Perch FP16 + 5× SED INT8 + heads | 9 + 5×3 + 3 = **~27 min** | yes | room for **15+ voters** |
| **E. cascade gate (lever 5) on D** | run SED only on ~40% active windows | 9 + 5×1.5 + 3 = **~20 min** | yes | room for a 2nd backbone family |
| **F. distilled single student (T2)** | 1 student CNN OpenVINO FP16 replacing Perch+SED+SSM | **~9–11 min** | yes | frees budget for diverse 2nd model |

**Headline:** lever 1 alone (OpenVINO FP16) roughly **halves** the heavy-stage cost, taking the
baseline from ~35 min to ~19 min. That ~16 min of freed budget is enough to add the **public
aliozanmemetoglu 5-fold SED (LB 0.958)** ensemble that the meta-analysis flags as the single
biggest unexploited lever — i.e. **speed work directly unlocks the accuracy lever**. Going from
~6 voters to ~10–15 voters in the same wall-clock is a realistic 2–3× effective-ensemble gain.

---

## 1. Knowledge distillation (T2 — spec now, train when GPU returns)

**Goal:** collapse the slow teacher ensemble (Perch + SED + ProtoSSM/residual-SSM) into ONE fast
CPU student so 3–5 forward passes become 1.

**Math (standard Hinton KD, multi-label variant):**
- Teacher produces soft per-class probabilities `p_t = σ(z_t / T)` per 5 s window (sigmoid, not
  softmax, because multi-label). Use **T = 2–3** (sweet spot from the literature; T≥10 over-smooths).
- Student loss: `L = α · BCE(p_s, hard_labels) + (1−α) · T² · KL/BCE(p_s^{(T)}, p_t^{(T)})`,
  with α ≈ 0.3 (70% weight toward the teacher distribution — the config that beat the hard-label
  student in the ensemble-KD literature). The `T²` factor rescales gradient magnitude.
- Teacher target = the **rank-blended ensemble output** you already produce in `sub_v8`
  (0.30 Bruce + 0.40 KNN + 0.20 Probe + 0.10 Perch …). Distill the *blend*, not individual models.

**Architecture choice — the load-bearing caveat:**
- **DO** distill into a mel-CNN that keeps a real spectrogram frontend (EfficientNet-B0 / NFNet-l0
  student). This retains **~85–95%** of ensemble accuracy and is what the 2025 top solutions did.
- **DO NOT** distill into a tiny 1D-CNN over precomputed embeddings. The arXiv "Distilling
  Spectrograms into Tokens" paper tried exactly this (STSG 1D-CNN student on Perch embeddings) and
  the student **collapsed from 0.80 → 0.47 macro-F1** (0.520 private ROC-AUC). The embedding
  bottleneck destroys the information the heads need. Distill the *spectrogram model*, keep the
  frontend.

**Expected payoff:** one student CNN ≈ 9–11 min for all 600 files (OpenVINO FP16). If it retains
~90% of the ~0.95 ensemble it lands ~0.945–0.948 *alone*, and frees enough budget to ensemble it
with a *diverse* 2nd family (e.g. Perch logits) for the structural diversity that actually moves LB.

**Recipe (ready-to-run when GPU is back):**
1. Run current best ensemble over the 10,592 unlabeled train_soundscapes → save soft targets (234-d
   per 5 s window) as the distillation set (this doubles as noisy-student data — see A6).
2. Train EfficientNet-B0 / eca_nfnet_l0 student on mel-specs with the KD loss above, T=3, α=0.3,
   time-shift + mixup aug (both +corr with score in meta-analysis).
3. Export student → ONNX → OpenVINO FP16. Validate retention on OOF before trusting it.

---

## 2. Quantization (INT8 / dynamic / static / OpenVINO)

**The VNNI caveat governs everything here.** INT8 only delivers its 2–4× on CPUs with VNNI
(AVX-512-VNNI, Cascade Lake+). Kaggle CPU nodes are commonly **Skylake-class without VNNI**, where
INT8 can be **break-even or slower** (documented ORT issue #6695: "dynamic quant slow on non-VNNI
CPUs"; issue #12854: uint8 model *slower*). **Verify on the actual Kaggle CPU first** (`lscpu | grep
vnni`) before committing INT8.

| variant | best for | speedup (VNNI) | speedup (no VNNI) | accuracy cost | notes |
|---------|----------|---------------:|------------------:|---------------|-------|
| **Static INT8 (QDQ, S8S8)** | the **mel-CNN / SED EfficientNet** (Conv-heavy) | 2–3× | ~1× | −0.000…−0.003 | ORT docs say **static** for CNNs; needs ~100–500 calib windows. **S8S8 is the recommended default** (U8S8 saturates on AVX2/512 w/o VNNI). |
| **Dynamic INT8** | **MatMul/Gemm-heavy** heads: probes, ProtoSSM, MLP, transformer-ish | 1.3–2× | ~1× | ~0 | ORT docs: dynamic for RNN/transformer/MatMul; **little help for Conv**. |
| **OpenVINO INT8 (POT/NNCF)** | mel-CNN, full graph | 2.95–4× (their bench) | modest | small | best tooling for Intel CPU; pairs with lever 1. |
| **FP16 (OpenVINO)** | everything | 1.5–3× | 1.5–3× | ~0 (ranking metric) | **the safe win — no VNNI dependency, no calibration, no accuracy risk.** Do this first. |

**Which ops quantize cleanly:** `Conv`, `MatMul`, `Gemm` (with Conv+BatchNorm fusion happening at
graph-opt time). Avoid quantizing softmax/sigmoid/normalization tails and the final classifier if
you see AUC wobble — quantize the backbone, keep the head FP. Mel/STFT frontend stays FP (it's an
FFT, not a quantizable matmul; in the `perch_v2_no_dft` ONNX the DFT is already stripped out and mel
is done outside the graph — good, leave it).

**Recommendation:** ship **FP16 (lever 1)** everywhere as the no-risk default. Treat INT8 as a
*conditional* lever — turn it on only if `lscpu` shows `avx512_vnni` on the Kaggle CPU, and only
*static* INT8 on the SED CNN. Expected real gain there is the difference between scenario C and D
above (47 → 27 min).

---

## 3. ONNX Runtime vs OpenVINO vs raw PyTorch on Kaggle CPU

**Hardware:** 4 vCPU / 2 physical cores, Skylake-class Xeon, AVX-512 (VNNI uncertain).

| runtime | relative CPU speed | setup cost | notes |
|---------|-------------------:|-----------|-------|
| raw PyTorch (eager) | 1.0× (baseline) | none | slowest; no graph fusion |
| **ONNX Runtime** (current) | ~2–3× | export to ONNX (already have Perch ONNX) | graph-opt + operator fusion; `uses_onnx` +0.535 corr |
| **OpenVINO** | ~1.5–3× *on top of* ORT for Intel CPU | ONNX→OpenVINO IR + FP16 compress | best Intel-CPU backend; the rare Nikita-2025 trick |

**Threading (critical and partly already tuned in `sub_v8`):**
- `intra_op_num_threads = 4` (= vCPU count), `inter_op_num_threads = 1` — correct for a single
  sequential model. With only 2 physical cores, **do not over-subscribe**: setting intra>4 hurts.
- `graph_optimization_level = ORT_ENABLE_ALL` — already set. Good (fuses Conv+BN, etc.).
- **Batch the mel windows**: current code does 192 windows/call (BATCH_FILES=16 × 12). Keep batches
  large to amortize per-call overhead and saturate SIMD — this is one of the biggest free wins and
  is already in place.
- For OpenVINO, set `PERFORMANCE_HINT=THROUGHPUT` (not LATENCY) since we process a big static batch,
  and let it pick `NUM_STREAMS` for the 2-core box.

**Conversion path (today, T1):** PyTorch → `torch.onnx.export` → `openvino.convert_model(onnx,
compress_to_fp16=True)` → `core.compile_model(..., "CPU")`. The Perch ONNX is already in the bundle;
just wrap it through OpenVINO and benchmark vs the existing ORT session before switching.

---

## 4. Smart compute reduction

| technique | track | speedup | cost | how |
|-----------|-------|--------:|------|-----|
| **Shared mel frontend** | T1 | removes duplicate STFT/mel across models | 0 | compute mel once per window, feed all CNNs that share input resolution; skip recompute |
| **Cache Perch embeddings** | T1 | **N× on downstream** | 0 | already implicit — KNN/probe/SSM/MLP/RAG all reuse the one 1280-d embedding. Make this explicit and ensure no model recomputes Perch. |
| **Cascade / early-exit gate** | T1 | 1.5–4× on expensive branch | small, tunable | cheap Perch sigmoid screens each 5 s window; only windows with max-prob > τ (or non-silent energy) get the expensive SED + SSM pass. Most Pantanal windows are sparse → big skip rate. Tune τ on OOF so recall ≈ 1.0 for ranking. |
| **Reduced TTA** | T1 | up to 2× | −0.001…−0.003 | drop low-value TTA (hflip/noise); meta-analysis shows n_windows=12 is the convention but TTA augments give little. Benchmark 12→6 windows. |
| **fp16 on CPU** | — | **avoid raw torch fp16 on CPU** | n/a | PyTorch CPU fp16 is generally *slower* (no native kernels). Use FP16 **only via OpenVINO IR**, where it's a real speedup. Don't `model.half()` on CPU. |

**Cascade is the sleeper lever:** because soundscapes are mostly quiet/single-species, gating the
expensive SED+SSM behind a cheap Perch-prob screen can cut the heavy branch by 2–4× with negligible
ranking-AUC loss if the gate is tuned for high recall. Pairs multiplicatively with levers 1 and 4.

---

## Concrete next actions (priority order)

1. **(T1, do first)** Wrap the existing Perch ONNX and any SED CNN through **OpenVINO FP16**;
   benchmark vs current ORT on a few train_soundscapes. Expect ~2× on the heavy stages, zero AUC
   cost. This is the rare +0.535-correlated trick taken one step further.
2. **(T1)** Spend the freed ~16 min to **add the public aliozanmemetoglu 5-fold SED (LB 0.958)** to
   the ensemble (the meta-analysis's #1 unexploited lever) — speed work *is* the accuracy unlock.
3. **(T1)** `lscpu | grep vnni` on the Kaggle CPU. **If VNNI present**, add **static INT8** on the
   SED backbone (S8S8 QDQ, ~200 calib windows) for another ~2×. If absent, skip INT8 entirely.
4. **(T1)** Make embedding-caching explicit + add a **cascade gate** (Perch-prob screen → SED/SSM)
   tuned for recall on OOF.
5. **(T2, GPU-ready code now)** Spec the **distillation** of the rank-blend ensemble into one
   EfficientNet-B0/NFNet student (T=3, α=0.3), trained on the 10,592 unlabeled soundscapes' soft
   labels — but **keep the spectrogram frontend** (the 1D-CNN-on-embeddings student fails, 0.80→0.47).

## Sources
- ONNX Runtime quantization docs — https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ORT dynamic-quant slow on non-VNNI CPUs — https://github.com/microsoft/onnxruntime/issues/6695 ; uint8 slower — https://github.com/microsoft/onnxruntime/issues/12854
- OpenVINO INT8 (2.95–4× FP32) — https://medium.com/openvino-toolkit/easily-optimize-deep-learning-with-8-bit-quantization-1f9021926bd3 ; https://www.edge-ai-vision.com/2019/02/introducing-int8-quantization-for-fast-cpu-inference-using-openvino/
- Quantization-on-edge benchmark (3.08–4× INT8) — https://arxiv.org/pdf/2303.05016
- BirdCLEF 2025 2nd place (ONNX→OpenVINO FP16, eca_nfnet_l0 + effv2_s ensemble) — https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- "Distilling Spectrograms into Tokens" (Perch→TFLite ~10× = 16 min/700 files; 1D-CNN student collapse 0.80→0.47) — https://arxiv.org/html/2507.08236
- Ensemble KD accuracy retention / T & α — https://huggingface.co/blog/Kseniase/kd ; https://arxiv.org/pdf/2204.00548
- Kaggle CPU notebook spec (4 vCPU / 2 core, Xeon) — https://www.kaggle.com/code/teeyee314/cpu-kernel-specs ; https://www.kaggle.com/code/bconsolvo/hardware-available-on-kaggle
- Local pipeline analyzed: `inference_notebooks/sub_v8_full_recipe_standalone.py`
