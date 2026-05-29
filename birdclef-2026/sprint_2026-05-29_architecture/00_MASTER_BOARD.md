# 🛰️ Architecture Sprint — 2026-05-29 (GPU-free window)

**Window:** ~7h GPU-free (GPU usage resets after). Comp deadline **2026-06-03** (5 days).
**Goal:** out-of-the-box *structural* breakthroughs → push toward LB **0.96** (accuracy AND/OR
inference speed). NOT 0.0001 param tweaks.
**Metric:** macro-averaged ROC-AUC (ranking, skips empty classes). Submission = CPU ≤90 min,
no internet, no GPU.

## Where we stand (from FINAL_META_FINDINGS.md + AGENT_REPORTS.md)
- Best **self**-submission ≈ 0.947. Public exp019 baseline 0.949. The 0.948 plateau = **Perch
  transfer ceiling**. Top LB: Yannan 0.962 / Team🤗 0.960 / Nikita 0.959.
- Winning structural features (corr w/ score): **ProtoSSM +0.664, residual-SSM +0.656,
  rank-aware +0.651, SED +0.609**, Perch-v2, MLP probes, site-hour priors, BirdNET (28 unmapped
  classes). **Param tuning is saturated** — only architecture moves the needle.
- Free lever already identified: ensemble the **publicly-available 0.958 SED 5-fold checkpoints**
  (aliozanmemetoglu) — used by 0 of 1,194 kernels.

## Two tracks
- **T1 (move LB *today*, no GPU):** post-proc / calibration / rank-blend search on existing OOF;
  ensembling public pre-trained checkpoints (inference-only); OpenVINO/ONNX speed so a bigger
  ensemble fits in 90 min.
- **T2 (research + ready-to-run code):** SSM-SED, noisy-student pseudo-labels, distillation —
  spec'd + coded now, executed the instant GPU returns.

## Coordination protocol
- Each agent writes ONE file: `agents/<ID>_<topic>.md`. No shared-file write races.
- Required output format per agent: ranked findings table with columns
  **[technique | track T1/T2 | expected LB lever | feasibility | impl sketch/math | source]**.
- Headline rule: prioritize BIG levers (inference-speed multipliers, +0.005 LB class of ideas),
  flag anything < +0.002 as low-priority.
- This board (`00_MASTER_BOARD.md`) is the live index — orchestrator updates status + the
  consolidated shortlist as agents report.

## Fleet status
| ID | Charter | Model | Status |
|----|---------|-------|--------|
| A1 | SSM / linear-attention SOTA for audio SED (Mamba lineage, DeepSeek/Qwen/Kimi transferable math) | opus | ⏳ launching |
| A2 | Bioacoustic foundation models (Perch2, BirdMAE, NatureLM-audio, BEATs/AVES) + 28-unmapped-class problem | opus | ⏳ launching |
| A3 | Inference-speed breakthroughs (distillation, INT8/quant, ONNX/OpenVINO, pruning) — fit bigger ensemble in 90min | opus | ⏳ launching |
| A4 | SED head + post-proc math directly optimizing macro ROC-AUC (rank loss, calibration, TTA) — T1 prototypeable now | opus | ⏳ launching |
| A5 | BirdCLEF 2026 LIVE forum + public-kernel scan (fresh, last ~10 days) | opus | ⏳ launching |
| A6 | BirdCLEF 2025/2024 winning-solution deep dive (Nikita, 2nd-place repo, noisy-student recipe) | opus | ⏳ launching |

## Consolidated shortlist (orchestrator fills in as reports land)
_TBD — updated after wave 1._
