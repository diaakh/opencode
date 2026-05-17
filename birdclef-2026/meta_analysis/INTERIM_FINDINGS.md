# Meta-analysis interim findings (490/1,250 kernels pulled)

## Data

- **1,250 unique BirdCLEF 2026 public kernels** identified via Kaggle API (across 3 sort orders, deduped).
- **3,602 teams on the public LB**; top score 0.962.
- **4,275 unique LB users** (single-user + multi-user teams expanded).
- **1,172 of 1,250 (94%)** kernel authors matched to LB → know their best LB score.
- **490 kernels pulled so far** (~40% complete).

## Public LB score distribution (3,602 teams)

| score bucket | n teams |
|---|---:|
| 0.96+ | 1 |
| 0.955–0.96 | 10 |
| 0.95–0.955 | 25 |
| 0.945–0.95 | **868 ← massive pile-up** |
| 0.94–0.945 | 409 |
| 0.93–0.94 | 146 |
| 0.92–0.93 | 498 |
| 0.9–0.92 | 300 |
| 0.85–0.9 | 300 |
| <0.85 | ~1,045 |

The pile-up at 0.945–0.95 is the "copy-the-public-0.948-kernel" crowd. Above 0.95 there's a long tail of 36 teams with real innovations.

## Top-50 LB authors with public kernels (only 8!)

| user | LB | rank | public kernels |
|---|---:|---:|---|
| aliozanmemetoglu | 0.958 | 4 | enb0-coarse-ensemble + 5-fold-ensemble (SED EfficientNet) |
| tonylica | 0.957 | 7 | birdclef-lb-0.872-0.862-16mins-runtime |
| kdmitrie | 0.954 | 15 | birdclef26-google-perch-starter (founding Perch starter, 208 votes) |
| hideyukizushi | 0.953 | 17 | bird26-reproduce-perch-protossm-resssm (241 votes) + train kernel |
| hyh273279 | 0.950 | 47 | birdclef-2026-simple-script (not yet pulled) |
| yash9439 | 0.951 | 34 | pantanal-distill + timeoptimized + keepimproving |
| mattiaangeli | 0.951 | 36 | better-blend + protossm-efficientnet-sed-all-public |
| alexandergremyakov | 0.950 | 43 | efficientnet-b0-submission + eda |

**The top-50 keep their training kernels PRIVATE.** Only inference + light starters are public.

## Most-referenced baseline kernels (corpus-wide fork lineage)

| baseline kernel | referenced by N kernels |
|---|---:|
| tuckerarrants/bc2026-distilled-sed | 8 |
| **nikitababich/birdclef2025-1st-place-inference** | **7** |
| nina2025/birdclef-2026-eos-4 | 6 |
| ravi20076/birdclef2026-openvino-starter-v1 | 6 |
| ravi20076/birdclef2026-preprocessing-v1 | 6 |
| hideyukizushi/bird26-reproduce-perch-protossm-resssm-inf-train | 5 |
| mtoshidesu/tedbirdclef-2026-improved | 5 |
| ravi20076/birdclef2026-public-blend-v1 | 5 |
| jaejohn/perch-v2-starter-train-infer | 4 |
| kdmitrie/birdclef26-google-perch-starter | 4 |
| marynaborovska/birdclef-26-two-pass-ssm-advanced-pp | 4 |
| ravi20076/birdclef2026-supplements-v1 | 4 |
| waterjoe/birdclef2026-submit-baseline | 4 |
| ravi20076/birdclef2026-public-blend-v2 | 4 |

**Big surprise: Nikita Babych's 2025 1st-place inference notebook is referenced 7 times in 2026 submissions.** He's also rank 3 in 2026 LB. He's reusing his own 2025 approach (SED EfficientNet via OpenVINO) for 2026.

## Numeric parameter convergence (most kernels use IDENTICAL values)

| param | n with value | range | median | corr with score |
|---|---:|---|---:|---:|
| lambda_prior | 93 | 0.3–0.5 | **0.4** | -0.01 (zero) |
| ensemble_w_mapped | 20 | 0.6 only | 0.6 | 0 |
| file_conf_power | 87 | 0.4 only | 0.4 | 0 |

**The 0.948+ cluster has converged on lambda_prior=0.4, ensemble_w_mapped=0.6, file_conf_power=0.4.** No correlation with score → these are saturated. exp019's tweak (lambda_prior 0.4→0.5) was a probe, not a proven gain.

## Architecture used by top-LB authors

From inspecting the inference kernels:

| author | LB | architecture |
|---|---:|---|
| aliozanmemetoglu (rank 4) | 0.958 | **5-fold SED EfficientNet** ensemble, texture-aware time smoothing (Insecta/Amphibia continuous), site/hour Bayesian priors |
| alexandergremyakov (rank 43) | 0.950 | EfficientNet-B0 SED, 20s context window for 5s prediction, custom checkpoint |
| hideyukizushi (rank 17) | 0.953 | Perch + ProtoSSM + ResidualSSM, OOF with StratifiedGroupKFold |
| mattiaangeli (rank 36) | 0.951 | Perch + EfficientNet + SED (all-public template, document references) |
| nikitababich (rank 3, his 2025 kernel) | 0.959 (in 2026) | **SED with GeMFreq + AttHead, OpenVINO acceleration, multi-model ensemble** |

**Common theme**: custom-trained SED EfficientNet checkpoints + Perch + texture-aware postprocessing.

## What separates 0.94-0.948 from 0.95+ (qualitative based on top kernels read)

The public 0.948 cluster (Nina EoS-4, Karnakbayev, Pilkwang, Safar1, Sunderekkiz exp019) uses:
- Perch ONNX (frozen) + ProtoSSM (trained on labeled soundscapes) + Distilled SED (frozen) + BirdNET (frozen)
- Site/hour priors with lambda_prior=0.4
- Per-class blend (mapped 50/30/20, unmapped 20/40/40)
- Sonotype mirroring (9 of 25 covered)
- TTA shifts ±2.5s

The 0.95+ tier adds:
- **CUSTOM-TRAINED SED EfficientNet on train_audio** (multiple folds → ensemble)
- **OpenVINO acceleration** for more inference budget
- **Pseudo-labels** on unlabeled soundscapes (iterative)
- **Larger / longer context windows** (20s context for 5s prediction)
- **Texture-aware postprocessing** that splits Insecta/Amphibia vs Aves/Mammalia

## Implications for the test data

From the convergence at 0.948 and the gap to 0.962:

1. **Train_audio (35k clips, 344 hours) is essential** for 0.95+, and unused by the entire public 0.948 cluster.
2. **The label space is partly trivial** (the 0.948 baseline gets there without train_audio because Perch already knows the 206 mapped species from its pretraining).
3. **The 28 missing classes are the hardest** (insect sonotypes + 3 frogs) — top kernels use BirdNET + custom SED to cover them.
4. **OpenVINO matters** for CPU budget — top kernels can run bigger ensembles within 90 min.
5. **5-fold ensemble of SED EfficientNets** is the dominant 0.95+ recipe.

## Next steps

- Complete the bulk pull (760 more kernels, ~16 min).
- Run final extraction + analysis on full corpus.
- Look at the LOWER end (kernels at 0.7-0.85) to see what approaches FAIL.
- Identify the EXACT param/feature combos that differentiate close score bands (0.945 → 0.948 → 0.949).
- Cluster the 8 top-LB authors' approaches to find common-but-private patterns.
