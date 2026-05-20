# G124 2025pre Pseudo Sidecar

This folder builds a compatible replacement for the private S124 sidecar dataset:

```text
birdclef2026-g124-effv2s-2025pre-pseudo-assets/
  infer.py
  g124_fold1_fp16.pt
```

The public S124 notebook looks for exactly this contract and then rank-blends the sidecar submission into the S114 anchor with weight `0.115`.

## What `2025pre` Means

Use BirdCLEF 2025 acoustic pretraining before BirdCLEF 2026 fine-tuning:

```text
EfficientNetV2-S
  -> train/pretrain on BirdCLEF 2025
  -> initialize BirdCLEF 2026 234-class model
  -> fine-tune fold 1 on BirdCLEF 2026
  -> add pseudo-labeled train_soundscapes rows from our best ensemble
```

If a 2025 checkpoint is not available yet, `train_g124.py` can still train from the timm EfficientNetV2-S pretrained weights. That is a G124-like fallback, not true `2025pre`.

## Train On Kaggle

Create a GPU Kaggle notebook/kernel using this folder as code. Attach:

- `birdclef-2026`
- optionally `birdclef-2025`
- a pseudo-label CSV dataset with `row_id` plus 234 class columns
- optionally a previous `g124_2025pre.pt`

Fold-1 2026 fine-tune:

```bash
python train_g124.py \
  --competition-dir /kaggle/input/competitions/birdclef-2026 \
  --output-dir /kaggle/working/g124_assets \
  --stage finetune2026 \
  --fold 1 \
  --pretrained-checkpoint /kaggle/input/g124-2025pre/g124_2025pre.pt \
  --pseudo-csv /kaggle/input/our-pseudo/train_soundscapes_pseudo.csv \
  --epochs 8 \
  --batch-size 48 \
  --num-workers 4
```

Smoke test with a small file cap:

```bash
python train_g124.py \
  --competition-dir /kaggle/input/competitions/birdclef-2026 \
  --output-dir /kaggle/working/g124_smoke \
  --stage finetune2026 \
  --fold 1 \
  --epochs 1 \
  --max-train-files 2 \
  --batch-size 8 \
  --num-workers 2
```

## Package Dataset

After training:

```bash
python package_assets.py \
  --checkpoint /kaggle/working/g124_assets/g124_fold1_fp16.pt \
  --output-dir /kaggle/working/birdclef2026-g124-effv2s-2025pre-pseudo-assets \
  --dataset-id diaakh/birdclef2026-g124-effv2s-2025pre-pseudo-assets
```

Then create or version the Kaggle dataset:

```bash
kaggle datasets create -p /kaggle/working/birdclef2026-g124-effv2s-2025pre-pseudo-assets
```

or:

```bash
kaggle datasets version -p /kaggle/working/birdclef2026-g124-effv2s-2025pre-pseudo-assets -m "g124 fold1 sidecar"
```

## Inference Contract

`infer.py` accepts the S124 sidecar arguments:

```bash
python infer.py \
  --data-dir /kaggle/input/competitions/birdclef-2026 \
  --input-dir /kaggle/input/competitions/birdclef-2026/test_soundscapes \
  --output submission_g124_effv2s_fold1_s124.csv \
  --device cpu \
  --batch-size 64 \
  --num-workers 0 \
  --window-seconds 5.0 \
  --tta-shifts 0 \
  --prior-weight 0.0 \
  --smooth-weight 0.12 \
  --disable-context-postprocess \
  --fast-fixed-60s \
  --assume-sr 32000 \
  --assume-duration 60 \
  --checkpoint g124_fold1_fp16.pt
```

## Efficiency Choices

- Inference reads each soundscape file once and slices all 5-second windows from memory.
- Mel extraction is batched with `torchaudio` on the selected device.
- Training uses AMP, `channels_last`, `AdamW`, cosine LR, pinned memory, and persistent dataloader workers.
- Pseudo labels are weighted soft-label examples, so they help without overwhelming focal audio.

