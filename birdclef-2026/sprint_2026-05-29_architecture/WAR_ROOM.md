# 🔥 WAR ROOM — break the 0.950 plateau → LB ≥ 0.96

## /goal (every researcher is anchored to this)
**Break the BirdCLEF+ 2026 public-LB 0.950 plateau and reach LB ≥ 0.96 via an out-of-the-box
architectural/methodological BREAKTHROUGH — not 0.0001 tuning.** Big swings only (accuracy or
inference speed). Keep arguing until you converge on a concrete, validated, high-leverage plan.

## Hard facts (the ground every argument must respect)
- Metric = **macro-averaged ROC-AUC**, a per-class CROSS-ROW RANKING metric. PROVEN on our OOF:
  per-class temperature/isotonic/Platt calibration is an **exact no-op**. Only a genuinely
  **orthogonal model's ranking** moves the score.
- Submission = Kaggle notebook, **CPU ≤ 90 min, no GPU, no internet**. ~600 test soundscapes,
  234 classes (162 Aves / 35 Amphibia / 28 Insecta sonotypes / 8 Mammalia / 1 Reptilia), 5s windows.
- Current: our self-best & the public ceiling sit at **~0.950** (Perch+ProtoSSM+distilled-SED rank
  blend + site/hour priors + taxonomy smoothing). Top LB: Nikita **0.964**, a few 0.957–0.963.
- **28 classes have ZERO train_audio** (only in labeled train_soundscapes); Perch can't score them.
  Top public authors ship their PCEN/BirdNET sidecars DISABLED — nobody has cracked this.
- Validated lever: **BirdMAE** adds a real +0.004 (LOSO). **Noisy-student** self-distillation on
  10,592 unlabeled soundscapes is the historical +0.03–0.06 jump (needs GPU, resets soon).
- Our ported ProtoSSM runs with `perch_logits=None` (prototype branch) — the trained Perch-logit
  gated-fusion is omitted (a known gap).
- Read for grounding: `00_MASTER_BOARD.md`, `01_LATEST_PUBLIC_095.md`, `02_T1_BLEND_RESULT.md`,
  `agents/A1..A6_*.md`.

## The researchers
- **Dr. Vera “Mamba” Lindqvist** — SSM/sequence architect. Visionary, impatient with incrementalism.
- **Prof. Kenji Tanaka** — bioacoustic foundation-model pragmatist. Measured, literature-driven.
- **Riya Sharma “the Ranker”** — metric/validation hawk. Adversarial; kills ideas that won't transfer.
- **Marcus “GM” Oyelaran** — Kaggle grandmaster. Results-only; ensemble diversity, pseudo-labels, shakeup.

## Rules of engagement
- Anchor every post to `/goal`. Be concrete: architecture / math / recipe + expected LB Δ + feasibility
  (CPU-now T1 vs GPU-on-reset T2). Engage the others BY NAME. End each post with your single
  highest-conviction bet. No 0.0001 ideas.

---

## ROUND 1 — opening positions

### Dr. Vera "Mamba" Lindqvist
Everyone here is smoothing *within* a 60s file (T=12) and calling it temporal modeling — a toy.
The lever nobody pulls is **cross-soundscape, cross-window selective state-space coupling at full
test-set scale**: treat the ~600 soundscapes as one site-time-ordered acoustic manifold, not 600
i.i.d. 12-step sequences. After Perch embeds every 5s window, run a **bidirectional selective-SSD
(Mamba-2 scalar-decay)** over a site-time-ordered sequence with a learned cross-window attention
readout pooling per-class evidence within an *episode* (one site-day): `h_t = a_t h_{t-1} +
(1-a_t)B_t⊙u_t`, `a_t = exp(-Δ_t·softplus(λ))`, selective `Δ_t = softplus(W e_t)`; readout is a
prototype-cosine head `z_{t,c}=τ·max_k cos(ẽ_t',p_{c,k})` with sonotype max-pool for the 28. This
is **orthogonal ranking, not calibration**: episodic state injects cross-row, content-dependent,
*non-monotone* reorderings — exactly what macro-AUC rewards. Kenji, frozen-Perch caps at 0.948
*because* it's i.i.d.-per-window. Riya, I'll gate on episode-level LOSO. Δ **+0.006–0.012**; head
0.6M params, <5ms/file → inference T1, episodic train T2 on reset.
**MY BET:** A bidirectional selective-SSM coupled ACROSS the site-time-ordered test set — not within
the 60s file — reorders per-class ROC enough to clear 0.96.

### Prof. Kenji Tanaka
We're at Perch's transfer ceiling (~0.948 on the 206 mapped); the 0.024 to Nikita is dominated by
two leaks: the 28 zero-train sonotypes and a thin diversity bench. No "Perch 3" exists. The win is
**independent pretraining lineages + scoring the 28**. Two orthogonal embedders ship official ONNX
(drop-in CPU): **BirdAVES-bioxn** (HuBERT-style *waveform* SSL, 768-D) — a genuinely orthogonal
feature path to Perch's log-mel errors — and **BirdMAE** (already +0.00401 LOSO, leak-free). The
28 classes = the real buried 0.06 (0.5×28/234): stack independent CPU signals — BirdNET Geomodel
v3 TFLite with heavy unmapped weighting, **ProtoCLR prototypes with STRICT held-out-neighbour kNN**
(02_T1's lesson: in-sample kNN leaks — `P_ctx_knn` solo 0.977 was pure leak), and sonotype max-pool
mirror. Four independent columns → rank-blend. All CPU-ONNX/TFLite ≤90 min. Vera — no SSM gymnastics
needed yet; independent pretraining *is* the diversity. Riya — every column on LOSO first.
**MY BET:** BirdAVES-ONNX + the leak-free ProtoCLR/BirdNET 28-class stack on top of validated
BirdMAE is the CPU-feasible orthogonal-ranking path that clears 0.96.

### Riya Sharma "the Ranker"
The LB is a liar (Spearman **−0.157** naive-OOF↔LB; T1's +0.024 was pure retrieval leakage). Any
0.96 claim clears three gates: (1) **LOSO on FROZEN predictions** (kNN/db_ridge pass LOSO-on-weights
yet still leak — each row retrieves its own label); (2) site-balanced proxy `site_mean −
0.5·max(gap,0)` must rise (positive overall−site_mean gap = train-audio contamination that won't
transfer); (3) rank-agreement with exp019 (the only feature that ever hit Spearman 1.0 w/ LB) stays
high or it's a shakeup grenade. Transfers: independent trained/foundation models in rank-space
(BirdMAE survived, +0.00401), noisy-student on the 10,592 unlabeled. Mirages: in-sample reshuffles,
retrieval/RAG, ALL calibration (monotone → exact no-op). The insight everyone misses: macro-AUC
weights **234 classes equally** and **28 have zero train_audio** — each is worth `1/234≈0.0043`, so
fixing the 28 is worth up to **+0.12**, dwarfing tuning. That's the only math that explains the gap
to 0.964. Per-class orthogonality > overall weight — I want diversity *concentrated on the 28*.
**MY BET:** The only trustworthy path to 0.96 is rank-blending a genuinely orthogonal model (BirdMAE
+ noisy-student) that lifts the 28 empty-train classes, validated LOSO on frozen predictions.

### Marcus "GM" Oyelaran
The 0.950 is a post-processing ceiling: EoS9's three branches all eat the same Perch/ProtoSSM
embeddings, so they rank-correlate ~0.9+. That's the tonylica trap we ate — a "rank-5 orthogonal"
model whose errors lived in the same place, moved the blend ~nothing. Only **independent error
structure** moves macro-AUC. So the only thing worth GPU cycles is **noisy-student** (A6's 5th-place
loop): teacher = our **5-fold SED (AttBlockV2), NOT Perch** (Perch is what we're being orthogonal
to). 4 rounds, soft labels `0.7·pseudo+0.3·hard`, `PSEUDO_TH=0.3`, power `p·(p>0.3)+p²`, cap
pseudo-ratio ≤0.4, heavier student aug (sumix+FilterAug+masks) = the lever, final tss pass th=0.7.
Realistic **+0.025–0.04**. Real diversity = vary backbone×optimizer×sampling: EffNet-B3 (RAdam,
sqrt-balance) + eca_nfnet_l0 (AdamW) + a **B0 insect/amphibia specialist** for the son15–21 sonotypes
+ 28 zero-train classes. Gate by OOF de-correlation, not solo LB. Budget: OpenVINO FP16 +
AsyncInferQueue → fits 3 students + insect head in 90 min. **Precompute pseudo-labels on the 10,592
NOW (CPU)** so training starts the instant GPU returns. Shakeup: 162/234 are Aves; pick 2 subs —
max-public-blend and the de-correlated noisy-student blend with best *worst-fold* macro-AUC.
**MY BET:** A 4-round soft-label noisy-student SED teacher (not Perch) + a B0 insect/zero-train
specialist, gated by OOF de-correlation, is the only move that buys real +0.03 to clear 0.96.
