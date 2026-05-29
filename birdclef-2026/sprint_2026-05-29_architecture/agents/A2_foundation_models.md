# A2 — Bioacoustic & General-Audio Foundation Models Survey

Agent A2 charter: survey SOTA bioacoustic / general-audio foundation models, find the
highest-leverage ones legally usable at **CPU inference** (≤90 min, no GPU, no internet,
pretrained-OK), and tackle the **28-unmapped-class** problem. Metric = macro ROC-AUC over
234 species (162 Aves, 35 Amphibia, 28 Insecta sonotypes, 8 Mammalia, 1 Reptilia).

---

## TL;DR ranked table

| # | model / technique | track | expected LB lever | CPU-feasible? | how to use | license | source |
|---|---|---|---|---|---|---|---|
| 1 | **Perch v2 (ONNX)** — incumbent | T1 | baseline (already the 0.948 ceiling); *not* a new lever but the anchor every ensemble member must beat/diversify | **YES** — ONNX export already on Kaggle (`rishikeshjani/perch-onnx`, 401 kernels); 12M-param EffNet-B3, 1536-D embed, ~5s/clip CPU | embed 5s @32kHz → 1536-D; linear/MLP probe to 234 logits | Apache-2.0 | [arXiv 2508.04665](https://arxiv.org/html/2508.04665v1), [Kaggle model](https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2) |
| 2 | **BirdNET v2.4 branch (TFLite/ONNX)** for the 28 unmapped classes | T1 | **+0.004–0.010 macro** (28/234 = 12% of macro avg; Perch scores these near-random ~0.5 → BirdNET can lift each toward 0.7–0.85). Meta confirms `uses_birdnet` HI 0.22 vs LO 0.02 | **YES** — TFLite is CPU-only by design, ~6k species + Geomodel v3 covers insects/amphibia; 1024-D embed, EffNetB0-like, tiny | run BirdNET logits, weight HEAVY on unmapped classes (mapped 50/30/20 Proto/SED/BirdNET; unmapped 20/40/40 + 1.8× spike) | CC-BY-NC-SA 4.0 (⚠ non-commercial — fine for Kaggle research use) | [Zenodo 15050749](https://zenodo.org/records/15050749), [BirdNET-Lite](https://github.com/birdnet-team/BirdNET-Lite) |
| 3 | **Sonotype max-pooling / mirroring** (technique, not a model) | T1 | **+0.003–0.008 macro** — meta: `sonotype_mirror` HI 0.41 vs LO 0.00 (+0.408 corr). Highest-corr handling of zero-train Insecta sonotypes | **YES** — pure post-proc, ~0 cost | for each of 28 zero-audio classes, max-pool/borrow scores from acoustically-similar mapped sonotypes (kNN in Perch embed space of labeled soundscape clips) | n/a | [meta FINAL_META_FINDINGS](../../meta_analysis/FINAL_META_FINDINGS.md) |
| 4 | **kNN retrieval in Perch/BirdNET embedding space** (for 28 classes) | T1 | **+0.003–0.006 macro** — the labeled train_soundscapes DO contain the 28 classes; build a per-class prototype/kNN index from those clips, score test clips by cosine sim | **YES** — embed once, FAISS/numpy kNN at inference | embed the labeled-soundscape exemplars of each unmapped class → prototype; score test = cosine sim. Complements/replaces sonotype-mirror | n/a (uses comp data) | technique; see ProtoCLR prototypical probing |
| 5 | **BEATs (general-audio SSL)** | T2 (probe-train needs GPU); T1 if pre-probed | **+0.002–0.005 macro ensemble diversity** — strongest *non-bird* embedding; beats bird models on BEANS (97.98 AUROC w/ attentive probe). Adds Insecta/Amphibia diversity Perch lacks | **YES at inference** (90M ViT-B, 768-D), PyTorch→ONNX exportable; probe training is GPU | freeze BEATs, attentive/MLP probe to 234, ensemble with Perch | MIT (Microsoft `unilm/beats`) | [arXiv 2508.01277 review](https://arxiv.org/html/2508.01277v1) |
| 6 | **NatureLM-audio (zero-shot text)** for the 28 classes | T2 (heavy) / research | **+0.002–0.005 macro** on unmapped IF it runs — zero-shot SOTA on unseen species via text prompts ("call of a {species}"). BEATs encoder + 8B LLaMA = 665M trained | **NO for full LLM** in 90 min CPU; **YES if you extract just the BEATs encoder** (`esp-aves2-naturelm-audio-v1-beats` on HF) and use it as embedder | full model = offline zero-shot label generation for the 28 classes, or use extracted BEATs encoder as embedder #5 | CC-BY-NC-SA (check card) | [ESP HF](https://huggingface.co/EarthSpeciesProject/NatureLM-audio), [arXiv 2411.07186](https://arxiv.org/abs/2411.07186) |
| 7 | **BirdAVES-bioxn (ONNX available!)** | T1/T2 | **+0.002–0.004 macro diversity** — HuBERT-style waveform SSL, 316M, 768-D; +20% over AVES on bird tasks; **official ONNX + torchaudio weights** | **YES** — ONNX shipped by Earth Species; waveform input (no spectrogram) = orthogonal feature path | freeze, probe to 234, ensemble | CC-BY-NC-SA 4.0 | [ESP BirdAVES blog](https://www.earthspecies.org/blog/introducing-birdaves-self-supervised-audio-foundation-model-for-birds) |
| 8 | **BirdMAE (ViT-L/16)** | T2 | **+0.002–0.004 macro** — current BirdSet SOTA (44.0 cmAP, beats Perch by up to +16% on POW), 1024-D. BUT needs **attentive probing** (transformer) to shine = GPU train | **borderline** — 300M ViT-L is slow on CPU; ViT-B variant more feasible | freeze, attentive-probe to 234 (GPU), ONNX-export for CPU inference | check DBD-research-group card | [arXiv 2504.12880](https://arxiv.org/abs/2504.12880), [HF Bird-MAE](https://huggingface.co/DBD-research-group/Bird-MAE-Large) |
| 9 | **ProtoCLR (CvT-13)** | T1/T2 | **+0.001–0.003 macro** — smallest (20M, 384-D), domain-invariant, built for **few-shot/prototypical probing** → ideal cheap branch for the 28 low-data classes | **YES** — tiny, fast CPU; CvT-13 | prototypical probe; especially for unmapped-class prototypes (item 4) | CC-BY-4.0 (permissive!) | [HF ProtoCLR](https://huggingface.co/ilyassmoummad/ProtoCLR), [arXiv 2409.08589](https://arxiv.org/html/2409.08589v4) |
| 10 | **BioLingual (HTS-AT + RoBERTa)** | T2 | **+0.001–0.003 macro** — audio-text contrastive, multi-taxa incl mammals/marine, 1024-D; zero-shot via text | **YES at inference** (190M) | zero-shot text scoring of unmapped classes; or embed branch | check card | [arXiv 2508.01277 review](https://arxiv.org/html/2508.01277v1) |
| 11 | **AudioMAE / SSAST / AST (general)** | T2 | **low, +0.001–0.002** — generic AudioSet SSL, weaker than BEATs here; only if cheap diversity needed | YES (86M ViT-B) | probe + ensemble | varies | [arXiv 2508.01277 review](https://arxiv.org/html/2508.01277v1) |
| 12 | **SurfPerch** | — | **~0** for this comp — marine/reef extension of Perch, no terrestrial gain | YES | skip | Apache-2.0 | [arXiv 2512.03219](https://arxiv.org/html/2512.03219) |

---

## 1. Perch 2.0 — exact facts

- **Architecture:** EfficientNet-B3 embedding net, **~12M params**, depthwise convs. Frontend = log-mel
  spectrogram, **5s @ 32 kHz → 500 frames × 128 mel bins (60 Hz–16 kHz)**.
- **Embedding:** spatial (5,3,1536) → mean-pooled to **1536-D**. (Note: Perch *v1* was EffNet-B1, 1280-D —
  the 0.948-cluster's `perch-meta` may be v1; v2 = 1536-D.)
- **Output head:** **14,795 classes** (14,597 species). Trained on **1.54M recordings**: Aves 1.37M,
  **Insecta 63k, Amphibia 55k, Mammalia 15k** (XC + iNaturalist + Tierstimmenarchiv + FSD50K).
- **Training criteria (the structural novelty):** (1) species cross-entropy, (2) **self-distillation** via a
  prototype-learning classifier with stop-gradient, (3) **source-prediction** self-supervised objective
  (each example = its own class). This is what gives the strong transfer embeddings.
- **What it knows:** the 162 Aves + most Mammalia/Amphibia/Insecta that overlap iNat/XC taxa →
  this is exactly why the public crowd plateaus at **0.948 = Perch's transfer ceiling on the 206 mapped classes.**
- **What it does NOT know:** the **28 zero-train-audio sonotypes** (custom Insecta sonotype labels that don't
  map to any Perch species). Perch scores these ~random → drags macro AUC.
- **CPU / format:** HF card says "**requires TF 2.20.rc0 + GPU, CPU variant coming soon**" and ships only TF
  SavedModel/Keras. **BUT in-competition CPU is already a solved problem** — the community ONNX export
  (`rishikeshjani/perch-onnx-for-birdclef-2026`, used by 401/1194 kernels; `tuckerarrants/perch-v2-no-dft-onnx`)
  runs Perch v2 on CPU within budget. Use ONNX, not the HF/TF path.
- **License:** Apache-2.0 (commercially clean — unlike most alternatives below).
- **Newer from DeepMind?** As of May 2026 the only post-Perch-2.0 release is the **marine/whale transfer
  paper (arXiv 2512.03219, "Perch 2.0 transfers whale")** — same weights, no new terrestrial model. Tom Denton
  (comp organizer) co-authors Perch; **no Perch 3 exists yet.** No new embedding to chase here.

## 2. Diversity members for a Perch ensemble (ranked by lever × CPU-feasibility)

Best diversity-per-CPU-dollar: **BEATs (#5)** and **BirdAVES (#7)** — both are *non-Perch lineages*
(general-audio SSL and waveform HuBERT-SSL respectively), so their errors decorrelate from Perch's
spectrogram-supervised errors. BirdAVES is the only strong alt with **official ONNX** = drop-in CPU.
**ProtoCLR (#9)** is the cheapest (20M/384-D, CC-BY-4.0) and is purpose-built for prototypical few-shot,
which doubles as the engine for the 28-class fix. **BirdMAE (#8)** is the highest-accuracy alt but its
gains need attentive probing (GPU train) and the ViT-L is heavy on CPU — defer to T2.

Caveat from the comparative review (arXiv 2508.01277): **always train a linear/attentive probe** on top of
any frozen embedder — raw zero-shot use of these models underperforms. Transformer models (BEATs, BirdMAE)
need **attentive probing**, not plain linear, to realize their headline numbers (+5–10 AUROC).

## 3. The 28-unmapped-class problem (12% of the macro average — the real bottleneck)

These 28 Insecta/Amphibia sonotypes have **zero train_audio**; they appear only in labeled
train_soundscapes. Perch can't score them. Macro-AUC averages all 234 classes, so 28 near-random columns
cost roughly **0.5×(28/234) ≈ 0.06 of headroom** if left unaddressed — this is why the 0.948→0.96 gap is
dominated by these classes. Ranked fixes (combine them, they stack):

1. **BirdNET branch with heavy unmapped weighting** (#2) — Geomodel v3 covers insects/amphibia; mapped
   blend 50/30/20, unmapped 20/40/40 with ~1.8× spike pull. **Highest single lever, +0.004–0.010.**
2. **Sonotype max-pooling / mirroring** (#3, corr +0.408 in meta) — borrow scores from acoustically-similar
   *mapped* sonotypes. **+0.003–0.008.**
3. **kNN / prototype retrieval in embedding space** (#4) — the labeled soundscapes DO contain exemplars of
   all 28; build per-class prototypes (Perch or ProtoCLR embeds) and score test by cosine sim. More
   principled than mirroring; **+0.003–0.006.** ProtoCLR (#9) is the natural backbone here.
4. **NatureLM-audio / BioLingual zero-shot text** (#6, #10) — prompt "the sound of {species}" to score the
   28 unseen classes. Real but offline (LLM too heavy for 90-min CPU) — precompute zero-shot priors, fold in
   as a fixed branch. **+0.002–0.005.**

These four are **independent signal sources** for the same 28 columns → ensemble/rank-blend all four.
Expected combined headroom on the macro metric from the 28 classes alone: **+0.008 to +0.015** — the
single biggest structural lever available, far above any param tweak (<0.002).

## 4. Insect / Amphibia domain-shift (the bottleneck taxa)

- **No dedicated public Insecta foundation model** exists that beats the above on CPU. Best coverage:
  **BirdNET Geomodel v3.0** (12,012-species, explicitly birds+mammals+insects+amphibians+reptiles) and
  **Perch v2's Insecta(63k)/Amphibia(55k) training shards** — but Perch's *output head* doesn't expose the
  28 sonotypes, so you must probe its *embeddings*, not its logits, for these taxa.
- **BEATs / general-audio SSL is the diversity unlock for insects** — the review found general AudioSet SSL
  *outperforms* bird-specific models on the multi-taxa BEANS benchmark. Insect stridulation is closer to
  generic broadband texture than birdsong, so BEATs/AudioMAE features transfer better there.
- **Texture-aware time smoothing** (from aliozanmemetoglu, per meta) — apply *heavier* temporal smoothing to
  Insecta/Amphibia predictions (continuous stridulation/chorus) vs standard for Aves (discrete calls).
  Cheap post-proc, real gain on exactly the bottleneck taxa.

---

## Recommendations to orchestrator

**T1 (do today, CPU-only):** (a) keep Perch-v2-ONNX as anchor; (b) add the **BirdNET unmapped branch +
sonotype max-pool + embedding-kNN stack** for the 28 classes — this is the highest combined lever
(+0.008–0.015 macro) and is pure post-proc + a tiny TFLite model; (c) add **BirdAVES-ONNX** as a free
diversity member (only strong alt with shipped ONNX); (d) **ProtoCLR** as the cheap prototypical engine for
the 28-class index.

**T2 (queue for GPU return):** attentive-probe **BEATs** and **BirdMAE-ViT-L** on train_audio, ONNX-export,
fold into the ensemble; precompute **NatureLM-audio/BioLingual** zero-shot text priors for the 28 classes
offline.

**Do NOT chase:** a "Perch 3" (doesn't exist), SurfPerch (marine), or plain AST/SSAST (dominated by BEATs).

### Licensing flag
Perch (Apache-2.0) and ProtoCLR (CC-BY-4.0) are permissive. **BirdNET, BirdAVES, NatureLM-audio are
CC-BY-NC-SA (non-commercial)** — fine for a Kaggle research competition, but note it if any winning code is
later commercialized.
