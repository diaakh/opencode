# SOTA Audio Classification Research — BirdCLEF 2026

## Models researched + deployment status

| Model | Type | Status | Notes |
|---|---|---|---|
| **Bird-MAE** | ViT-B/16 MAE pretrained on bird vocalizations | ✅ **DEPLOYED** | SOTA on BirdSet, +0.0065 OOF, breakthrough for chorus frogs |
| **BirdAVES** (BEATs-bio) | BEATs encoder + SSL bioacoustic | ✅ DEPLOYED (RAG) | +0.0010 OOF, embedding-only via cached samuelzxu/birdaves-train-cache |
| **Perch 2.0** | EfficientNet-B3 multi-taxa SSL+distillation | ❌ BLOCKED | XLA v10 mismatch + ABI issues on Kaggle TF 2.18; needs TF 2.20.rc0+ |
| **ConvNeXt-Tiny** | Mel-CNN supervised on labeled BC2026 | ✅ DEPLOYED (RAG) | +0.0023 OOF (long_convnextv2_tiny majkel) |
| **BirdNET v2.4** | Cornell Lab bird ID | ⚠ Not tried | TFLite available (mansianilkadam/birdnet-v2-4-tflite, 77MB) |
| **YERMES Perch v5** | PyTorch BC2026 Perch fine-tune (50MB) | ⚠ Not tried | Complete inference package, downloaded for inspection |
| **BirdTransform** | Custom CNN+Transformer (4-layer) | ⚠ Inspected only | 71MB pth, BC2026-specific, custom mel pipeline |
| **F11** (DBD-research EfficientNet) | EfficientNet-B1 on AnuraSet (42 frogs) | ✅ Tested | Marginal (mapping mismatches), 16/234 mapped |
| **SurfPerch** (Google) | Coral-tuned Perch | ❌ Skipped | Domain mismatch (underwater) |
| **CLAP** (LAION/Microsoft) | Contrastive Audio-Text | ❌ Skipped | Better for zero-shot, not closed-class |
| **BEATs** (Microsoft) | Audio acoustic tokenizers | ❌ Skipped | AudioSet-tuned not bird-specific |
| **AST** (MIT) | Audio Spectrogram Transformer | ❌ Skipped | AudioSet-tuned |
| **PANNs** | Pretrained Audio Neural Networks | ❌ Skipped | AudioSet, 2019 era |

## Datasets found

| Dataset | Size | Content |
|---|---|---|
| `adkasd/birdmae-finetune-234` | 945MB | **Fine-tuned Bird-MAE for 234 BC2026** (us) ⭐ |
| `samuelzxu/birdaves-train-cache` | 243MB | **BirdAVES embeddings on ALL train_soundscapes** ⭐ |
| `samuelzxu/birdaves-biox-large` | 763MB | BirdAVES model weights |
| `rishikeshjani/birdclef-external-data` | 158MB | **49k Perch embeddings: AnuraSet+Amazon+Coffee Farms** ⭐ |
| `nikitababich/birdclef2025-1st-place-extra-data` | 7.8GB | 17k recordings (mostly European frogs) |
| `denden12/anuraset-bc26-32k-mono-ogg` | 715MB | AnuraSet preprocessed audio |
| `bengtlueers/anuraset-v2-raw` | 7.2GB | AnuraSet raw |
| `lvweibin/birdclef-2026-f11-ep10-pytorch` | 24MB | F11 AnuraSet EfficientNet-B1 |
| `hengck23/pytorch-perchv2` | 425MB | PyTorch port of Perch v2 |
| `mansianilkadam/birdnet-v2-4-tflite` | 77MB | BirdNET 2.4 TFLite |

## Key Kaggle model IDs

```
google/bird-vocalization-classifier/TensorFlow2/perch_v2/2          (Perch 2.0 CUDA)
google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1      (Perch 2.0 CPU)
google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8  (Perch v1)
yermes/birdclef-2026-perch-yermes-v5/PyTorch/default/1              (YERMES Perch finetune)
```

## Research papers / repos

- [Can Masked Autoencoders Also Listen to Birds? (arxiv 2504.12880)](https://arxiv.org/abs/2504.12880)
- [Perch 2.0: The Bittern Lesson for Bioacoustics (arxiv 2508.04665)](https://arxiv.org/abs/2508.04665)
- [AVES: Animal Vocalization Encoder (arxiv 2210.14493)](https://arxiv.org/abs/2210.14493)
- [Distilling Spectrograms into Tokens (BirdCLEF+ 2025, arxiv 2507.08236)](https://arxiv.org/abs/2507.08236)
- [BEATs: audio acoustic tokenizers (NeurIPS 2023)](https://openreview.net/forum?id=Fj0PRtd4e6)
- [Comparing SSL for Bioacoustics (arxiv 2501.05987)](https://arxiv.org/pdf/2501.05987)
- [Bird-MAE GitHub](https://github.com/DBD-research-group/Bird-MAE)
- [Bird-MAE Base/Large HuggingFace](https://huggingface.co/collections/DBD-research-group/bird-mae)
- [EarthSpeciesProject AVES2](https://huggingface.co/EarthSpeciesProject)

## Honest OOF ladder (final)

```
0.9706  baseline (Bruce + KNN + Probe + Perch v1 + LR/MLP stackers + prior)
0.9710  + clean prototype (no-leak rebuild)
0.9717  + Multi-K external RAG (Rishikesh 49k)
0.9725  + external Ridge classifier
0.9748  + ConvNeXt-RAG (12k Perch+ConvNeXt-Tiny pred pairs)
0.9813  + BirdMAE @ alpha=0.5 (THE breakthrough)
0.9823  + BirdAVES-RAG with ConvNeXt soft labels @ alpha=0.3
```

**Net session gain: +0.0117 OOF (0.9706 → 0.9823)**

## Production deployment status

**Validated kernels for LB submission:**

1. **`slot6_birdmae_exp019`** — exp019 (LB 0.949) + BirdMAE (OOF 0.9721) 70/30 blend
   - Expected LB: **0.95-0.97** ⭐
   - Runtime: ~67 min (under 90)
2. **`birdclef-2026-birdmae-submit`** — BirdMAE standalone
   - Honest OOF: 0.9721
   - Runtime: ~47 min
3. **`slot5_combined`** — exp019 + sub_v8 (BirdMAE+RAGs) 70/30
   - sub_v8 has all RAG stacks baked in (ConvNeXt-RAG + External-RAG + Prototype)
4. **exp019 vanilla** — LB 0.949 baseline anchor
5. **`sub_v8_full_recipe_standalone`** — Bruce + RAG diversity slot

**Current LB top:** 0.962 (Yannan Chen)

## Future improvement paths (not deployed)

1. **Perch 2.0 deployment**: needs TF 2.20.rc0+ in Kaggle kernel. Could use custom Docker image override or precompute predictions locally (need test audio access).

2. **BirdNET v2.4 TFLite**: Cornell's bird ID model — completely different lineage. 77MB TFLite, would add another orthogonal signal.

3. **YERMES Perch v5**: PyTorch fine-tune of Perch v2 for 234 BC2026. 50MB model — could ensemble with our existing Perch predictions for variance reduction.

4. **BirdTransform**: Custom 4-layer CNN+Transformer (71MB), BC2026 fine-tune. Different architecture than everything we use.

5. **Iterative noisy-student** (BirdCLEF 2025 1st place recipe): pseudo-label unlabeled audio with current ensemble, retrain BirdMAE on labeled+pseudo, repeat. Requires GPU training.


## Perch 2.0 deployment attempts log (Phase 2 final)

Attempted 15+ approaches to deploy Perch 2.0 on Kaggle. All blocked by the same
fundamental issue: Perch 2.0's SavedModel contains XLA-compiled computations
(XlaCallModule v10) that require a TF runtime Kaggle doesn't provide.

| Attempt | Approach | Error |
|---|---|---|
| v1 | Default TF 2.18 | XlaCallModule version 10 not supported |
| v2 | Set jit=False env | No effect (XLA baked in SavedModel) |
| v3 | perch_v2_cpu variant | Same XLA error |
| v4 | Install TF 2.21 wheel offline | keras dep conflict |
| v5 | + --no-deps | libtfkernel_sobol_op undefined absl symbol |
| v6 | enable_internet + pip install --upgrade | Same ABI mismatch |
| v7 | venv with --system-site-packages | venv create failed |
| v8 | pip --target=/tmp/tfnew | protobuf 5 vs gencode 6 mismatch |
| v9 | + protobuf>=6 | System protobuf shadows |
| v10 | --force-reinstall direct | libtf absl symbol still broken |
| v11 | docker_image: gcr.io/kaggle-images/python:v168 | Cannot deserialize XLA |
| v12 | perch-hoplite library install | TF 2.19 in v168, same XLA issue |
| v13 | + ModelConfigName.PERCH_V2_CPU enum loading | Same Cannot deserialize |

**Root cause:** Perch 2.0's XLA modules require **TF 2.20+ with matching XLA backend
build**. Kaggle's containers ship TF 2.18-2.19 with older XLA backend. Even
the v168 docker image release (March 2026) doesn't fully resolve this.

**Locally we have TF 2.21 working** with Perch 2.0 CPU (verified). But cannot
deploy this to Kaggle without test_soundscapes access locally.

**Future workarounds:**
1. Use `tf2onnx` to convert Perch 2.0 to ONNX format (requires GPU local
   inference of one test sample to establish architecture, then weight export)
2. Use PyTorch reimplementation if Google releases one (current is TF-only)
3. Wait for Kaggle to upgrade default docker image to TF 2.20+

**Concession:** Perch 2.0 stays out of our LB stack. BirdMAE remains our
strongest deployed signal. Production stack: **0.9823 honest OOF**, no Perch 2.0.
