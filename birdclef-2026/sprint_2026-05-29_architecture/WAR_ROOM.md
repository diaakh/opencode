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

---

## ROUND 2

### Dr. Vera "Mamba" Lindqvist
Riya, your 0.12 math is the only honest arithmetic in this room, and it guts my opening. Be told
plainly: a Perch-embedded SSM does **nothing** for the 28 zero-train classes — Perch never sees
them, so `u_t` is null and my prototype head has no `p_{c,k}` to cosine against. My episode coupling
only re-ranks the 206 mapped Aves. Conceding that. So I will NOT defend a Perch-only branch for the
28 — that's dead.

Marcus, you're also right that a Perch-fed SSM is a correlated branch — same log-mel error manifold,
ρ~0.9, the tonylica trap again. I won't pretend a state-space readout launders that away. So I adapt,
hard: **mount the episodic SSM on YOUR noisy-student SED logits, not Perch.** The SED teacher gives a
genuinely orthogonal per-window emission; my Mamba-2 episode layer then does what SED *can't* —
cross-soundscape, site-time-ordered, non-monotone re-ordering of those emissions. That's orthogonal
error structure AND a temporal lever stacked on it, not a redundant column. The 28 get scored through
the SED/insect-B0 head with sonotype max-pool over the episode — your specialist, my pooling.

Riya, gate 3 (exp019 rank-agreement): cross-episode coupling IS a shakeup risk — state leaks across
windows and can swing worst-fold. So constrain it: **scalar decay floored** (`λ` lower-bounded so
`a_t≥0.6`, memory ≤ one site-day), readout as a **residual** `z = z_SED + γ·Δ_episode`, `γ` tuned to
hold Spearman-vs-exp019 ≥ 0.95 on LOSO-frozen. If γ→0 helps the worst fold, it ships at γ→0. No grenade.

This collapses four egos into one column stack: SED-teacher noisy-student (Marcus) → BirdMAE +
BirdAVES independent lineages (Kenji) → leak-free ProtoCLR for the 28 (Kenji/Riya) → my episodic-SSM
residual *on the SED logits only*, gated to exp019. T1 inference, T2 training on reset.

**MY BET:** The SSM earns its seat ONLY as a γ-gated episodic residual on the noisy-student SED head —
orthogonal, 28-safe via the specialist, shakeup-bounded — not as a Perch branch.
**CONSENSUS PLAN I'D SIGN:** Noisy-student SED teacher + insect/28 B0 specialist + BirdMAE/BirdAVES
lineages + leak-free ProtoCLR, rank-blended, with my episodic SSM as a γ-gated SED-only residual,
all on LOSO-frozen with exp019 rank-agreement ≥ 0.95.

### Prof. Kenji Tanaka
Riya, your +0.12 math is the lodestar, and I concede my R1 "four columns" was hand-wavy on the
leakage gate. The leak-free recipe, concrete: the 28 have NO train_audio — only positives inside
labeled train_soundscapes, so the only class anchors are test-adjacent. The gate is **held-out-
neighbour retrieval by site-day**: build ProtoCLR/BirdMAE prototypes per sonotype; at scoring a query
window may retrieve ONLY anchors from a *different* site-day episode (strict group-disjoint kNN, k=5,
cosine). That kills the `P_ctx_knn`-0.977 self-retrieval leak. Validate as you demand: LOSO on frozen
predictions, column dead if overall−site_mean gap goes positive. BirdNET Geomodel-v3 maps to sonotypes
via species→family logits collapsed onto the 28 as an *unmapped-weighted prior* (a retriever-free,
leak-free column). Realistic fraction: the 0.12 ceiling assumes 0.5→1.0 on all 28; held-out few-shot
realistically lands ~0.65–0.72 per class → **+0.03 to +0.045**, still the biggest lever. Marcus —
your B0 noisy-student specialist has independent error structure mine lacks and likely beats cold
retrieval on son15–21; they COMBINE — ship retrieval NOW (T1, CPU-ONNX), swap/blend the B0 on reset.
Vera — episodic SSM is a distraction for the 28 (no anchors to couple), a modest complement elsewhere.
**MY BET:** Strict held-out-neighbour ProtoCLR/BirdMAE retrieval + BirdNET sonotype prior lifts the 28
by ~+0.035 on CPU now; Marcus's B0 specialist stacks on top at reset.
**CONSENSUS PLAN I'D SIGN:** T1 now — group-disjoint held-out retrieval (BirdMAE+ProtoCLR) + BirdNET
prior on the 28, LOSO-gated; T2 on reset — Marcus's noisy-student B0 specialist blended by OOF
de-correlation; Vera's SSM deferred.

### Riya Sharma "the Ranker"
Three of you are about to hand me a leak wearing a lab coat. The gates: **Marcus** — your SED teacher
trains on the SAME labeled soundscapes the test sites neighbour; pseudo-labels on the 10,592 encode
dead-hour/site identity. Gate: **leave-one-recording-group-out**, every pseudo clip from a held-out
site purged from teacher AND student; pass only if `site_mean − 0.5·max(gap,0)` rises AND worst-fold
rises. **Kenji** — prototypes built ONLY from train_audio, never the query site's soundscapes;
strict held-out-neighbour exclusion by recording group; any class with <K real neighbours reports
NaN, not a borrowed score; measure per-class AUC on the 28 specifically. **Vera** — episodic cross-
test coupling violates i.i.d. scoring. Concrete test: **shuffle episode assignment at inference and
re-score; if macro-AUC moves >0.002 it's order-dependent → shakeup grenade, out.** In only if intra-
site-day AND shuffle-invariant. My +0.12 is the ceiling; realistic in 5 days: recover ~8–12 of the
28 to half-decent AUC ≈ **+0.008–0.015**, plus noisy-student **+0.01–0.02** net of contamination →
honest **0.958–0.962**, not 0.964.
**MY BET:** Noisy-student (site-grouped LOSO, purged teacher) as the orthogonal engine + the 28 via
leak-free ProtoCLR scored per-class on held-out sites — everything frozen-prediction LOSO before a sub.
**CONSENSUS PLAN I'D SIGN:** Purged noisy-student SED + strict held-out-neighbour 28-class stack on
BirdMAE, gated by site-grouped LOSO (worst-fold + site_mean must rise), Vera's SSM only if shuffle-
invariant; ship two subs — max-public-blend and the de-correlated best-worst-fold blend.

### Marcus "GM" Oyelaran
Convergence is obvious; I'll sign it. **Kenji — the 28:** noisy-student B0 specialist BEATS zero-shot
retrieval but we blend both. Recipe: cut labeled-soundscape positive segments into 5s clips → *seed*
hard labels for the B0 son-specialist; noisy-student over the 10,592 UNLABELED generates pseudo-
positives, soft `0.7·pseudo+0.3·hard`, cap ≤0.4. Real out-of-sample signal, not retrieval. Blend B0
(primary ranker for son15–21) + Kenji's leak-free ProtoCLR/BirdNET (orthogonal 2nd column) = two
independent error structures on the 28. **Riya — LOSO:** on the 206 mapped, noisy-student survives
site-held-out (teacher is OUR SED, no row sees its own label); on the 28 I gate on the site-balanced
proxy, zero the column if `site_mean−0.5·max(gap,0)` doesn't rise. **Vera** — episodic SSD is cheap
and orthogonal but rides the embeddings; train it on noisy-student logits, not raw Perch. **5-DAY
SEQUENCE:** NOW(CPU) precompute SED pseudo-labels on 10,592 + cut 28-class clips → D1–2(GPU) noisy-
student rounds 1–2 (B3-RAdam + nfnet_l0-AdamW) → D2–3 B0 son/zero-train specialist → D3–4 rounds 3–4
+ BirdAVES/BirdMAE ONNX export (CPU-parallel) → D4 episodic-SSD on student logits → D5 rank-blend,
LOSO-gate every column, OpenVINO FP16 ≤90min, freeze. **2 SUBS:** (1) max-public-blend (defensive),
(2) de-correlated noisy-student + B0/ProtoCLR-28 + episodic-SSD, selected on **best worst-fold**.
**MY BET:** Noisy-student SED (rounds 1–2 first) feeding both a B0 zero-train specialist AND Vera's
episodic SSD readout — orthogonal at embedding AND sequence level — is the +0.03 that clears 0.96.
**CONSENSUS PLAN I'D SIGN:** Pseudo-label now → noisy-student SED first → B0/ProtoCLR 28-class dual-
column → BirdMAE/AVES + episodic-SSD on student logits → LOSO+site-proxy gate every column → ship
worst-fold-best blend vs public-max.

---

## 🏁 BREAKTHROUGH — the converged plan (all four signed)
The room broke through to ONE plan. Unanimous structure:

1. **Engine = noisy-student self-distillation**, teacher = our **SED (AttBlockV2), NOT Perch** (Perch
   is what we're being orthogonal to). 4 rounds, soft labels `0.7·pseudo+0.3·hard`, TH=0.3, power
   `p·(p>0.3)+p²`, pseudo-ratio ≤0.4, heavier student aug. → the +0.01–0.02 net orthogonal lift.
2. **The 28 zero-train classes = the prize** (Riya: up to +0.12 ceiling; realistic +0.008–0.015),
   attacked by a **DUAL independent column**: (a) a **B0 insect/amphibia specialist** seeded from
   labeled-soundscape 5s clips + noisy-student pseudo-positives; (b) **strict held-out-neighbour
   (group-disjoint by site-day) ProtoCLR/BirdMAE retrieval** + a leak-free **BirdNET sonotype prior**.
3. **Independent lineages for diversity**: **BirdMAE** (validated +0.004) + **BirdAVES** (waveform SSL,
   orthogonal feature path), CPU-ONNX, rank-blended.
4. **Vera's episodic SSM survives — demoted**: a **γ-gated residual on the noisy-student SED logits**
   (not Perch), floored decay (memory ≤ one site-day), **only if shuffle-invariant** (Riya's test:
   shuffle episode assignment, macro-AUC must move <0.002) and rank-agreement-with-exp019 ≥ 0.95.
5. **Validation gates (non-negotiable, Riya)**: **leave-one-recording-group-out** (not random fold);
   pseudo clips from held-out sites purged from teacher AND student; a column ships only if
   `site_mean − 0.5·max(gap,0)` rises AND **worst-fold** macro-AUC rises; the 28 measured **per-class**
   on held-out sites; NaN (not borrowed) for classes with <K real neighbours.
6. **Submission selection (shakeup hedge)**: 2 subs — (1) max-public-blend (defensive, 162/234 Aves),
   (2) the de-correlated noisy-student+28-dual+SSD blend chosen on **best worst-fold**, not mean.
7. **5-day sequence**: NOW(CPU) precompute pseudo-labels on 10,592 + cut 28-class clips → D1–2 NS
   rounds 1–2 → D2–3 B0 specialist → D3–4 NS rounds 3–4 + BirdAVES/BirdMAE ONNX export → D4 episodic
   SSD on student logits → D5 blend + LOSO-gate + OpenVINO FP16 ≤90min + freeze.

**Honest converged forecast: 0.958–0.962** (Riya's number, the others didn't dispute it). Clears the
0.950 plateau; 0.96 is reachable but the top (0.964) likely needs more than 5 days. The single most
time-critical action — **start the CPU pseudo-label precompute now** (B1's `pseudo_label_precompute/`
already built) so noisy-student fires the instant GPU returns.
