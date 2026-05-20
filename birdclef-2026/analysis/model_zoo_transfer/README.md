# Model Zoo Transfer Analyzer

This folder builds a phase-1 model-zoo table for BirdCLEF 2026. It compares our models and public notebooks only when actual `train_soundscapes` predictions are available.

## Run

```bash
python3 birdclef-2026/analysis/model_zoo_transfer/analyze.py
```

Outputs:

- `model_zoo_features.csv`
- `model_zoo_feature_correlations.csv`
- `model_zoo_leave_one_out.csv`
- `model_zoo_report.md`

## Scope

The analyzer intentionally avoids notebooks that only have LB scores or code structure. Phase 1 requires extractable prediction/cache outputs on labeled or unlabeled `train_soundscapes`.

## Interpretation

Use the report for risk tiering and candidate ranking. Do not treat exact LB regression as reliable to 0.001 precision.
