# BirdCLEF 2026 — Meta-analysis of 1,250 public kernels

## 1. Corpus and LB coverage

- **1250** unique public BirdCLEF 2026 kernels enumerated via Kaggle API
- **1172** kernels matched to a public LB user
- **93.8%** coverage
- Author LB best: range 0.455–0.958, median 0.9450

## 2. Public LB distribution (3,602 teams)

| score range | n teams |
|---|---:|

## 3. Score buckets in our corpus (best LB by kernel author)

| bucket | count |
|---|---:|
| 0.no_signal | 10 |
| 0.weak_0.500-0.900 | 215 |
| 1.low_0.900-0.925 | 56 |
| 2.mid_0.925-0.940 | 236 |
| 3.high_0.940-0.948 | 331 |
| 4.elite_0.948-0.955 | 321 |
| 5.top_0.955+ | 3 |

## 4. Top-50 LB authors WITH public kernels

| author | LB | rank | kernels |
|---|---:|---:|---|
| alexandergremyakov | 0.95 | 43 | alexandergremyakov/efficientnet-b0-submission<br>alexandergremyakov/birdclef-2026-soundscape-sonotype-eda |
| aliozanmemetoglu | 0.958 | 4 | aliozanmemetoglu/birdclef-enb0-coarse-ensemble-submission<br>aliozanmemetoglu/birdclef-5-fold-ensemble-submission |
| hideyukizushi | 0.953 | 17 | hideyukizushi/bird26-reproduce-perch-protossm-resssm-inf-train<br>hideyukizushi/bird26-reprod-perch-proto-residualssm-train-s7177 |
| hyh273279 | 0.95 | 47 | hyh273279/birdclef-2026-simple-script |
| kdmitrie | 0.954 | 15 | kdmitrie/birdclef26-google-perch-starter |
| mattiaangeli | 0.951 | 36 | mattiaangeli/birdclef-2026-0-943-better-blend<br>mattiaangeli/birdclef-protossm-efficientnet-sed-all-public |
| tonylica | 0.957 | 7 | tonylica/birdclef-lb-0-872-0-862-16mins-runtime |
| yash9439 | 0.951 | 34 | yash9439/birdclef2026-keepimproving<br>yash9439/birdclef2026-timeoptimized<br>yash9439/pantanal-distill-birdclef2026-improvement |

## 5. Architecture features by score bucket


Top 25 features by HI (≥0.948) vs LO (<0.925) delta

| feature | hi share | lo share | delta |
|---|---:|---:|---:|
| uses_protossm | 0.72 | 0.06 | +0.664 |
| uses_residual_ssm | 0.70 | 0.05 | +0.656 |
| rank_aware | 0.69 | 0.04 | +0.651 |
| aug_time_shift | 0.75 | 0.11 | +0.634 |
| uses_mlp_probes | 0.71 | 0.08 | +0.626 |
| adaptive_delta | 0.66 | 0.04 | +0.626 |
| uses_sed | 0.67 | 0.06 | +0.609 |
| site_hour_prior | 0.58 | 0.04 | +0.539 |
| file_confidence | 0.55 | 0.02 | +0.537 |
| uses_onnx | 0.63 | 0.09 | +0.535 |
| val_groupkfold | 0.63 | 0.12 | +0.507 |
| uses_perch_v2 | 0.77 | 0.27 | +0.503 |
| uses_perch | 0.78 | 0.32 | +0.460 |
| aug_mixup | 0.61 | 0.20 | +0.417 |
| sonotype_mirror | 0.41 | 0.00 | +0.408 |
| uses_tucker | 0.40 | 0.00 | +0.399 |
| uses_tf | 0.70 | 0.31 | +0.395 |
| uses_torch | 0.88 | 0.61 | +0.270 |
| uses_train_audio | 0.27 | 0.51 | -0.236 |
| uses_birdnet | 0.22 | 0.02 | +0.191 |
| loss_bce | 0.02 | 0.17 | -0.149 |
| aug_cutmix | 0.17 | 0.04 | +0.134 |
| loss_focal | 0.24 | 0.11 | +0.125 |
| tweak_C_residual | 0.09 | 0.00 | +0.093 |
| uses_efficientnet | 0.44 | 0.35 | +0.084 |

## 6. Most-imported public dependencies (kernels using them)


### Top notebook outputs imported

| notebook | used by N kernels |
|---|---:|
| ashok205/tf-wheels | 337 |
| vyankteshdwivedi/notebook1b25083f0d | 321 |
| kdmitrie/bc26-tensorflow-2-20-0 | 58 |
| antoinemasq/birdclef-2026-pytorch-baseline-training | 13 |
| ttahara/birdclef-2026-download-wheels | 9 |
| udaysonawane/cnn-finetune | 8 |
| ttahara/birdclef-2026-hgnetv2-b0-baseline-training | 6 |
| raunakdey07/offline-training-efficientnet-b0-focal-recording | 3 |
| emanuellcs/birdclef-2026-i-o-preprocessing | 3 |
| skidive/birdclef-2026-onnx-perch-sequence-model-302597 | 3 |
| waterjoe/birdclef2026-train-baseline | 3 |
| aliozanmemetoglu/birdclef-sed-fold-1 | 2 |
| aliozanmemetoglu/birdclef-sed-fold-2 | 2 |
| denizegememetoglu/birdclef-sed | 2 |
| blamerx/birdclef-2026-training | 2 |
| emanuellcs/birdclef-2026-training | 2 |
| atahalam/perch-meta-0-943 | 2 |
| troyxxf/bc26-build-final-proto-bank-fixed302-v1 | 2 |
| troyxxf/birdclef-2026-build-final-proto-bank | 2 |
| troyxxf/birdclef-2026-build-soundscape-pseudo-bank | 2 |

### Top datasets imported

| dataset | used by N kernels |
|---|---:|
| jaejohn/perch-meta | 554 |
| rishikeshjani/perch-onnx-for-birdclef-2026 | 401 |
| tuckerarrants/bc2026-distilled-sed-public | 114 |
| tuckerarrants/perch-v2-no-dft-onnx | 93 |
| lixin73/birdclef2026-v27-onnx-perch-meta-forum-v1-lb872 | 49 |
| yuriygreben/perch-meta | 47 |
| chaneyma/birdclef2026-edits-protossm-sed-onnx-infer-artifacts | 42 |
| chaneyma/bc26-edits-protossm-sed-v7-all66-40x20 | 40 |
| chaneyma/bc26-edits-protossm-sed-v8-all66-synth-p010-40x20 | 40 |
| chaneyma/bc26-gate-fake008-head0015-baseline-onnx | 38 |
| chaneyma/bc26-probe-middle-pca128-raw085-logreg015 | 30 |
| hideyukizushi/sgkfk-202604041716 | 28 |
| tuckerarrants/birdclef-2026-waveform-cache | 28 |
| tonylica/birdclef-2026-model | 24 |
| needless090/birdclef2026-sed-v5-trio | 24 |
| needless090/birdclef2026-perch-tflite | 17 |
| needless090/birdclef2026-sed-ensemble | 16 |
| mlnjsh/birdclef2026-effnet-5fold | 13 |
| mlnjsh/perch-onnx | 13 |
| yananaaaaa/birdclef2026-models | 11 |

### Top models imported

| model | used by N kernels |
|---|---:|
| google/bird-vocalization-classifier | 774 |
| shadiakiki1/birdnet-analyzer | 6 |
| timm/tf-efficientnet | 6 |
| google/yamnet | 5 |
| imronrsya/efficientnetv2-s | 3 |
| kospintr/birdclef | 3 |
| aliozanmemetoglu/birdclef-sed-fold-3 | 2 |
| aliozanmemetoglu/birdclef-sed-fold-4 | 2 |
| haradibots/bird-efficientnet-model | 2 |
| marcusewang/gem-efficient-net | 2 |
| marcusewang/test-model | 2 |
| dheyeong/tf-efficientnet-b0-ns-jft-in1k | 2 |
| aadigupta1601/birdclef-models | 1 |
| aiaiaioooo/tf-efficientnetv2-b0 | 1 |
| alexandergremyakov/sed-b0-ce-nospecaug | 1 |
| chesteryuan/perch-v2 | 1 |
| ashishkubade/timm-convnextv2-tiny-fcmae-ft-in22k-in1k-384 | 1 |
| farshidamira/birdclef-2026-sed-enb0-stage1-5fold | 1 |
| ikkimasuta/20260511-baseline | 1 |
| jek1wantaufik/buddy | 1 |