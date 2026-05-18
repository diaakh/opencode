# Session summary — BirdMAE breakthrough (0.9706 → 0.9813 honest OOF)

## The journey

| Stage | Honest OOF | What changed |
|---|---:|---|
| Session start | 0.9706 | Bruce + KNN + Probe + Perch + LR-stackers + MLP + prior |
| + clean prototype | 0.9710 | Per-class pure-call Perch centroids |
| + Multi-K external RAG | 0.9717 | Rishikesh 49k AnuraSet+Amazon+Coffee Farms |
| + external Ridge classifier | 0.9725 | Ridge trained on Rishikesh DB |
| + ConvNeXt-RAG | 0.9748 | 12k unlabeled (Perch_emb, ConvNeXt_pred) pairs |
| **+ BirdMAE (alpha=0.5)** | **0.9813** | SOTA bird-pretrained ViT-B/16 MAE |

**Total session gain: +0.0107 OOF**

## What broke the chorus frog ceiling

The 4-frog S22-night chorus problem was unsolvable until **Bird-MAE**. Per-class
improvements (final stack vs session start):

| Frog | Start | After BirdMAE | Δ |
|---|---:|---:|---:|
| 326272 Weeping | 0.687 | **0.959** | **+0.272** |
| 1491113 Adenomera | 0.792 | **0.954** | **+0.162** |
| 22961 Pointedbelly | 0.853 | **0.970** | +0.117 |
| 24279 Lesser Snouted | 0.911 | 0.954 | +0.043 |
| 555146 Chaco TreeFrog | 0.933 | 0.962 | +0.029 |
| 65380 Dwarf TreeFrog | 0.976 | 0.988 | +0.012 |

## Why BirdMAE worked when everything else stalled

Bird-MAE (arxiv 2504.12880) is fundamentally different:
- **Architecture**: ViT-B/16 Masked Autoencoder (vs Bruce/Perch's CNN-Ridge stack)
- **Pre-training**: SSL on bird vocalizations specifically (75% masking ratio)
- **Pre-processing**: Kaldi-fbank (htk_compat, 128 mels, frame_shift=10ms)
- **Normalization**: (feat - mean=-7.2) / (std=4.43 * 2.0) — critical for inference

All prior signals (KNN, prototypes, ConvNeXt-RAG, Rishikesh-RAG) were Perch-derived
or ConvNeXt-derived — sharing acoustic representations. BirdMAE is the first
truly orthogonal model class.

## Deployment artifacts

**Production submission kernels** (tomorrow's priority order):

1. **`slot6_birdmae_exp019`** — exp019 (LB 0.949) + BirdMAE 70/30 blend
   - Expected LB: **0.95-0.97**
   - Runtime: ~67 min (under 90)
   - Files: `/inference_notebooks/slot6_birdmae_exp019/slot6_birdmae_exp019.py`

2. **`birdclef-2026-birdmae-submit`** — BirdMAE standalone
   - Honest OOF: 0.9721
   - Expected LB: **0.94-0.97**
   - Runtime: ~47 min
   - Files: `/inference_notebooks/birdmae_submit/birdmae_submit.py`

3. **`slot5_combined`** — exp019 + sub_v8 (0.9748 OOF) 70/30 blend
   - Expected LB: 0.95-0.96 (sub_v8 has Bruce-derived diversity)
   - Runtime: ~50 min

4. **exp019 vanilla** — LB 0.949 baseline anchor

5. **`sub_v8_full_recipe_standalone`** — Bruce/KNN/RAG diversity slot

## Key dataset additions this session

- `adkasd/birdmae-finetune-234` (945MB) — fine-tuned Bird-MAE (was sitting unused)
- `rishikeshjani/birdclef-external-data` (158MB) — 49k labeled AnuraSet+Amazon+Coffee
- `convnext_rag_bundle.pkl` (43MB) — 12k Perch+ConvNeXt-pred pairs
- `external_rag_bundle.pkl` (164MB) — 49k Perch+labels at float16
- `prototype_bundle.pkl` (1.4MB) — 106 clean per-class prototypes

## Critical fixes uncovered

1. **sklearn 1.6 compat**: sub_v8's pickled LR objects lost `multi_class` attr in
   sklearn 1.8→1.6 round-trip. Without `_sklearn_compat_fix()`, sub_v8 crashes on LB.

2. **BirdMAE preprocessing**: Must use `torchaudio.compliance.kaldi.fbank` (not librosa
   melspec), shape `(B, 1, 512, 128)`, normalize (mean=-7.2, std=4.43*2). Wrong preprocessing
   gives 0.51 AUC; correct gives 0.97.

3. **KNN DB leakage**: All 66 labeled OOF files were in the KNN DB → prototype-sim
   inflated to 0.92 (was actually 0.68 honest). Fixed by removing OOF files from DB
   for prototype construction.

## SOTA research findings (web search)

- **Bird-MAE** (DBD-research-group, arxiv 2504.12880): SOTA on BirdSet ← we used this
- **Spectrogram Token Skip-Gram** (arxiv 2507.08236): Faiss K-means + Word2Vec
- **Wav2Vec2/HuBERT/WavLM**: 92.75% bioacoustic test acc (vs RF 78.62%)
- **BirdSetEfficientNetB1**: public LB 0.81, private 0.78
- **Domain shift** is the core BirdCLEF challenge — handled via in-domain SSL distillation

