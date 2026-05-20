# Model Zoo Transfer Analyzer

This folder builds a phase-1 model-zoo table for BirdCLEF 2026. It compares our models and public notebooks only when actual `train_soundscapes` predictions are available.

## Run

```bash
python3 birdclef-2026/analysis/model_zoo_transfer/analyze.py
```

To include downloaded public notebook outputs without committing large caches:

```bash
python3 birdclef-2026/analysis/model_zoo_transfer/analyze.py \
  --public-output-root /tmp/birdclef_public_outputs
```

Some public `full_oof_meta_features.npz` outputs omit row metadata. You can opt into a known 708-row train-soundscape metadata template when you have verified the cache family uses the same row order:

```bash
python3 birdclef-2026/analysis/model_zoo_transfer/analyze.py \
  --public-output-root /tmp/birdclef_public_outputs \
  --public-meta-template /tmp/birdclef_public_outputs/yaroslavkholmirzayev__v6-0949-replay/perch_cache/full_perch_meta.parquet
```

Outputs:

- `model_zoo_features.csv`
- `model_zoo_feature_correlations.csv`
- `model_zoo_leave_one_out.csv`
- `model_zoo_report.md`

## Scope

The analyzer intentionally avoids notebooks that only have LB scores or code structure. Phase 1 requires extractable prediction/cache outputs on labeled or unlabeled `train_soundscapes`.

Supported public cache formats:

- Kaggle submission-style CSVs with `row_id` plus class columns.
- `perch_arrays.npz`/`perch_meta.parquet`.
- `full_perch_arrays.npz`/`full_perch_meta.parquet`.
- `full_oof_meta_features.npz` when paired row metadata is present.

## Interpretation

Use the report for risk tiering and candidate ranking. Do not treat exact LB regression as reliable to 0.001 precision.
