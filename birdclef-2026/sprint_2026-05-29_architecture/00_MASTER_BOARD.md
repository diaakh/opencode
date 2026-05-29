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
| A1 | SSM / linear-attention SOTA for audio SED | opus | ✅ done → `agents/A1_ssm_linear_attention.md` |
| A2 | Bioacoustic foundation models + 28-unmapped-class problem | opus | ✅ done → `agents/A2_foundation_models.md` |
| A3 | Inference-speed breakthroughs | opus | ✅ done → `agents/A3_inference_speed.md` |
| A4 | macro-ROC-AUC post-proc math + tested rank-blend code | opus | ✅ done → `agents/A4_rocauc_postproc.md` + `a4_rankblend.py` |
| A5 | LIVE 2026 forum/LB/kernel intel | opus | ✅ done → `agents/A5_live_intel_2026.md` |
| A6 | 2025/2024 winning-solution deep dive | opus | ✅ done → `agents/A6_prior_winners.md` |

---

# 🎯 CONSOLIDATED BREAKTHROUGH PLAN (wave 1 synthesis, 2026-05-29)

## Live LB (A5, today): Nikita #1 **0.964**, aliozanmemetoglu-team **0.963**, several 0.957–0.961.
**No public kernel above 0.952.** Public plateau moved 0.948 → ~0.950/0.952 via a new base
stack (not a new architecture). Our self-best ≈ 0.947.

## The one thing all six agents agree on
macro ROC-AUC is a **per-class cross-row RANKING** metric. **A4 proved on the real OOF** that
per-class temperature / isotonic / Platt calibration are *exact no-ops* (raw vs rank vs T=2 all
give identical 0.96429). Therefore **the ONLY thing that moves the score is adding a genuinely
orthogonal model whose ranking differs** — post-proc reshuffling of the same public branches is
saturated. The 0.96 jump is an *ensemble-diversity + new-model* problem, not a tuning problem.

## Verified live, unexploited levers (HTTP 200 today, importable)
| asset | what | importers | status |
|---|---|---|---|
| `tonylica/birdclef-2026-model` | rank-5 trained weights (orthogonal model) | 0 | ✅ LIVE |
| `aliozanmemetoglu/raw-pseudos` + `pseudo-text-init-iter-0` | the **0.963 author's own pseudo-labels** | ~0 | ✅ LIVE |
| `hideyukizushi/sgkfk-202604041716` | rank-17 trained models | few | ✅ LIVE |
| `tuckerarrants/bc2026-distilled-sed-public` + `perch-v2-no-dft-onnx` | the new base stack | many | ✅ LIVE |
| `needless090/sed-v5-trio` | SED ensemble | — | ❌ GONE (403) |
| `aliozanmemetoglu` *Models* (SED folds) | 0.958→0.963 SED | 0 | ⚠️ verify in Kaggle UI (API scope blocked) |

## TRACK T1 — do NOW, GPU-free (CPU), can move LB or prep the GPU run
| # | play | source | expected | notes |
|---|---|---|---|---:|---|
| T1-1 | **Rank-blend live orthogonal trained models** (tonylica rank-5 + hideyukizushi sgkfk; aliozanmemetoglu folds if UI-confirmed) into our pipeline via `a4_rankblend.py` + site-balanced local proxy | A4,A5 | **+0.004–0.009** | biggest *immediate* lever; 0 competitors use these |
| T1-2 | **Catch up to the new public base stack** (tuckerarrants distilled-SED + perch-v2-no-dft + sgkfk + proto-residualssm) | A5 | to ~0.950/0.952 | cheapest catch-up if we're not already on it |
| T1-3 | **28-unmapped-class stack** on just those columns: BirdNET-TfLite + sonotype max-pool + embedding-kNN (off labeled soundscapes) + NatureLM/BioLingual zero-shot, rank-blended | A2,A5 | **+0.008–0.015** | ~12% of macro avg, currently near-random; top authors *can't* crack it = our edge |
| T1-4 | **OpenVINO FP16 + AsyncInferQueue** on Perch+SED (~2–4×, zero acc cost) | A3,A6 | unlocks bigger ensemble in 90min | FP16 safe (no VNNI on Kaggle CPU → INT8 only conditional) |
| T1-5 | **Pre-compute pseudo-labels** on 10,592 unlabeled soundscapes (CPU, inference-only) | A6 | enables T2-A instantly | dataset ready the moment GPU returns |
| T1-6 | **Robust local macro-AUC proxy** (site-balanced − gap penalty) driving greedy blend search | A4 | meta-lever | LB is near-useless now (A5) → CV is our compass |
| T1-7 | Free post-proc: temporal-flip TTA, per-taxon time-smoothing, hierarchical taxonomy smoothing, symmetric smoother + max/mean logit adjust | A1,A4,A5 | +0.002–0.005 | low risk, gate on CV |

## TRACK T2 — fire the instant GPU resets (spec'd + coded now)
| # | play | source | expected | notes |
|---|---|---|---|---:|---|
| T2-A | **Multi-iter noisy-student self-distillation** on the soundscapes, 3–4 rounds, SOFT labels α=0.7 thresh=0.3 power=2 | A6 | **+0.03–0.06** | THE historical jump (Nikita 0.898→0.930); g124 scaffold exists |
| T2-B | **Bidirectional Mamba-2/SSD selective-SSM head + prototype/cosine classifier** over the 12×5s Perch sequence | A1 | +0.003–0.008 | structurally past Perch ceiling; helps 28 classes; ~0.15M params, <1ms/file CPU |
| T2-C | **SED AttBlockV2 attention-pooling head** on diverse timm backbones (B0/effv2s/NFNet) | A6,A4 | the 0.948→0.955 step | diversity = backbone×optimizer×sampling |
| T2-D | **Distill ensemble → one fast spectrogram-CNN student** (keep mel frontend, NOT embedding-only) | A3 | speed + retain ~85–95% | embedding-only student collapses (arXiv) |
| T2-E | Separate insect/amphibia model + **SoftAUC / pairwise-rank surrogate loss** (replace BCE/focal) | A6,A4 | +0.003 + metric-aligned | directly optimizes the metric |

## Realistic projection (honest)
0.947→0.96 is current rank-2/3 (Nikita 0.964). Compounding the T1 plays (catch-up to 0.950 base
+ orthogonal-model blend + 28-class stack) is a realistic **~0.953–0.957**. Crossing into
**0.958–0.96 requires T2-A (noisy-student) landing as a real diverse member** — that's the
moonshot, and it's exactly why pre-computing pseudo-labels now (T1-5) matters. **0.96 = stretch;
0.954–0.957 = strong realistic outcome** in 5 days.

## Immediate next actions (recommended order)
1. T1-6 + T1-1: stand up the local proxy and rank-blend tonylica/sgkfk into current OOF **in this
   container today** (OOF npz are on disk — runnable now).
2. T1-5: kick the pseudo-label pre-compute (CPU notebook on Kaggle — many can run without GPU).
3. T1-3: prototype the 28-class stack on the labeled-soundscape OOF.
4. Code T2-A/T2-B so they're commit-ready for the GPU reset.

---

# ✅ SPRINT DELIVERABLES & HANDOFF (built + verified this window)

| deliverable | path | status |
|---|---|---|
| Local macro-AUC proxy + rank-blend search | `agents/a4_rankblend.py`, `t1_blend_search.py` | ✅ ran; calibration-invariance + leakage proven |
| **LOSO validation** | `agents/t1_loso_validate.py`, `02_T1_BLEND_RESULT.md` | ✅ **BirdMAE +0.004 confirmed real**; leaky helpers rejected |
| **Pseudo-label pre-compute** (B1) | `pseudo_label_precompute/` | ✅ end-to-end tested; CPU Kaggle kernel; schema documented |
| **Noisy-student + Bi-SSD/proto head** (B2) | `g124_2025pre_pseudo/{ssm_head,noisy_student,run_ssm_smoke}.py` | ✅ smoke 1.7s, **21/21 tests**, 422k params, 10.1 MFLOP/file |
| **Orthoblend inference notebook** (B3) | `inference_notebooks/sprint_orthoblend/` | ✅ compiles; live mounts verified; **~26–32 min / 90** |
| B1→B2 schema reconciliation | `train_g124.py load_pseudo_parquet` | ✅ integration-tested (win_confidence carried) |

## Execution sequence
**Now, GPU-free (CPU):**
1. Run B1 `pseudo_label_precompute/` as a Kaggle CPU utility kernel → produces the soft-label
   dataset (~2–3.7 h, inside the 12 h limit).
2. Submit B3 `sprint_orthoblend/` as the inference notebook → catches up to the 0.950 base AND
   adds the unexploited orthogonal `tonylica`/`sgkfk` members (LB-measure the orthogonal lift).

**The instant GPU resets:**
3. Train B2: `python train_g124.py --ssm-head --perch-embeddings perch.npz
   --pseudo-parquet <B1 output> --noisy-student-rounds 4 --ns-final-tss` → `g124_ssm_fold1_fp16.pt`.
4. Package the trained g124 as a Kaggle dataset; add it to B3's `ENSEMBLE` config as a real
   orthogonal member; wire the validated **BirdMAE** branch (TODO marked in B3).

## Known issue (pre-existing, not from this sprint)
`g124_2025pre_pseudo/tests/test_g124_assets.py` is uncollectable — it imports
`evidence_*`/`run_kaggle_evidence_traces` modules that the earlier "work" commit never
committed (`git ls-files` confirms none tracked). Either commit those modules or trim the test.

## Honest projection (unchanged, now evidence-backed)
T1 plays (base catch-up + orthogonal blend + 28-class stack) → realistic **0.953–0.957**.
Crossing **0.958–0.96** needs B2's noisy-student g124 landing as a real diverse member
(why B1's pseudo-labels are pre-computed now). 0.96 = stretch; mid-0.95 = strong realistic.
