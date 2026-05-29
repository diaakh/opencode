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
_(orchestrator appends each researcher's turn below)_
