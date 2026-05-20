# G124 Research-Backed Training Plan

## Recommendation

Build G124 as a BirdCLEF 2025-style EfficientNetV2-S sidecar, not as a generic ImageNet model.

The strongest public evidence points to:

1. `tf_efficientnetv2_s_in21k` or close EfficientNetV2-S backbone.
2. 5-second log-mel training windows.
3. Focal BCE / label smoothing style regularization.
4. BirdCLEF 2025 or previous-competition pretraining before 2026 fine-tuning.
5. Pseudo-labeled soundscapes, weighted below clean labels.
6. Fast fp16/ONNX/OpenVINO-style inference for Kaggle runtime.

## Evidence

- BirdCLEF+ 2025 overview reports that top systems commonly used SED-style modeling, EfficientNet backbones, pretraining on external/large audio datasets, pseudo-labeling/self-training, TTA, and optimized inference.
- The BirdCLEF 2025 second-place public repository names a selected `tf_efficientnetv2_s_in21k` model trained with 5-second windows, focal BCE, label smoothing, pseudo labels, and additional rare-bird/no-leak handling. It also describes previous-competition pretraining and fp16/OpenVINO conversion.
- EfficientNetV2 is designed for faster training and parameter efficiency, which matches Kaggle runtime constraints.
- Perch 2.0 and PANNs support the broader conclusion that large bioacoustic/audio pretraining transfers strongly to downstream acoustic tasks.
- BirdCLEF 2024/2025 solution summaries repeatedly show pseudo-labeling and 5-second soundscape segmentation as high-impact for leaderboard performance.

## Concrete Pipeline

### Stage 0: Acquire A Strong 2025pre Backbone

Preferred:

- Use the public BirdCLEF 2025 second-place pretrained EfficientNetV2-S backbone if accessible.
- Convert only the backbone into our `g124_2025pre.pt` format.

Fallback:

- Train EfficientNetV2-S on BirdCLEF 2025 + compatible previous-competition/additional species audio.
- Use 5-second crops, log-mel spectrograms, focal BCE, balanced sampling, and label smoothing.
- Extract backbone only, then attach a 234-class BirdCLEF 2026 head.

Avoid:

- Starting from random weights for the real run.
- Treating plain ImageNet/timm initialization as true `2025pre`.

### Stage 1: 2026 Fold-1 Fine-Tune

Train fold 1 because the S124 notebook expects:

```text
g124_fold1_fp16.pt
```

Training data:

- BirdCLEF 2026 `train_audio`
- labeled `train_soundscapes`
- pseudo-labeled `train_soundscapes` / unlabeled windows from our strongest ensemble

Important settings:

- 5-second random crops for focal recordings.
- fixed 5-second windows for soundscapes.
- pseudo labels as soft labels with lower sample weight, initially `0.25-0.35`.
- focal BCE or BCE-with-logits plus class/frequency balancing.
- mixup/cutmix/spec masking only after the baseline is stable.

### Stage 2: Candidate Validation Before Submission

Before packaging, run G124 on `train_soundscapes` and feed the predictions to:

```text
birdclef-2026/analysis/model_zoo_transfer/
```

Accept the sidecar only if:

- it is not a duplicate of existing Perch/public cache behavior,
- it has useful diversity versus `exp019`/S114,
- site gap is not extreme,
- the LB predictor does not mark it as a private-risk trap,
- S124 rank-blend simulation moves features in the same direction as high-LB public analogs.

### Stage 3: Package For S124

Package exactly:

```text
infer.py
g124_fold1_fp16.pt
dataset-metadata.json
```

Attach it to the S124 notebook under:

```text
birdclef2026-g124-effv2s-2025pre-pseudo-assets
```

## Changes Needed In Our Current Scaffold

The current scaffold is enough for smoke tests and packaging, but the production training script should be upgraded before a full run:

- Add a true SED/attention head instead of plain image-classification pooling.
- Add focal BCE and optional label smoothing.
- Add labeled `train_soundscapes` rows, not only focal `train_audio`.
- Add balanced sampling by class/source.
- Add optional export to ONNX or OpenVINO after checkpoint selection.
- Add a train-soundscape prediction export for the LB correlator.

## Source Links

- BirdCLEF 2025 second-place repository: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- BirdCLEF+ 2025 overview paper: https://ceur-ws.org/Vol-4038/paper_232.pdf
- EfficientNetV2 paper: https://arxiv.org/abs/2104.00298
- Perch 2.0 paper page: https://research.google/pubs/perch-20-the-bittern-lesson-for-bioacoustics/
- PANNs paper: https://arxiv.org/abs/1912.10211
- PaSST paper: https://arxiv.org/abs/2110.05069
- BirdCLEF 2024 solution summary: https://zenn.dev/yuto_mo/articles/ad43c630729073

