# BirdCLEF Model Zoo Transfer Analyzer Design

## Goal

Build a phase-1 research pipeline that compares our BirdCLEF models and public notebooks using actual prediction/cache outputs on `train_soundscapes`. The purpose is to discover which prediction behaviors transfer to public leaderboard score, especially cases where labeled OOF looks strong but LB underperforms.

Phase 1 only includes notebooks/models where we can extract real predictions for labeled or unlabeled `train_soundscapes` rows. Models with only code structure or only known LB scores are out of scope until a later phase.

## Inputs

The pipeline ingests prediction artifacts from:

- Our kernels and experiments under `birdclef-2026/inference_notebooks/` and `birdclef-2026/analysis/`.
- Public notebooks downloaded under `birdclef-2026/analysis/entropy_tta/public_kernels/`.
- Public cache/output folders available locally, such as extracted Perch arrays, OOF caches, submission-like CSVs, or parquet metadata.

Each included model needs:

- A stable `model_id`.
- Source metadata: `ours` or `public`.
- Optional public LB score.
- A category label such as `single`, `blend`, `postprocess`, `prior-heavy`, `retrieval`, `sed`, `perch`, `birdmae`, or `other`.
- Prediction coverage: `labeled`, `unlabeled`, or `both`.
- A normalized prediction matrix for rows that can be mapped to `train_soundscapes` windows.

## Normalization

All predictions are normalized to a common representation:

```text
row_id × 234 class columns
```

`row_id` should use the competition-style format:

```text
<soundscape_stem>_<end_time_seconds>
```

The normalizer must preserve row coverage metadata rather than silently filling missing rows. If a model only covers 708 labeled-like rows, it should remain a partial-coverage model with explicit coverage counts.

Class columns must match `sample_submission.csv` / `taxonomy.csv` order. Missing class columns should be represented explicitly and flagged in the model metadata.

## Feature Table

The main output is:

```text
birdclef-2026/analysis/model_zoo_transfer/model_zoo_features.csv
```

Each row represents one model or blend. Columns include:

- `model_id`
- `source`
- `notebook_slug`
- `category`
- `known_lb`
- `coverage_labeled_rows`
- `coverage_unlabeled_rows`
- `coverage_classes`
- labeled metrics where labels are available
- unlabeled/distribution/agreement metrics where labels are unavailable

Core feature groups:

- Labeled performance: macro-AUC, active-class AUC, family-level AUC, site-weighted AUC, site gap.
- Site/hour behavior: per-site variance, per-hour variance, site/hour coverage, sensitivity to biased labeled sites.
- Prediction shape: entropy mean/quantiles, confidence rates, probability quantiles, rank concentration, calibration-like summary statistics.
- Agreement: rank/probability agreement with `exp019`, agreement with known high-LB public models, agreement on unlabeled rows, disagreement with known bad/risky models.
- Class-family behavior: Aves, Amphibia, Insecta, Mammalia, Reptilia, sonotype-only features.
- Risk signals: prior-heavy behavior, excessive site/hour dependence, OOF/LB mismatch patterns, catastrophic-model similarity.

## Analysis Outputs

The pipeline writes:

```text
birdclef-2026/analysis/model_zoo_transfer/model_zoo_features.csv
birdclef-2026/analysis/model_zoo_transfer/model_zoo_report.md
birdclef-2026/analysis/model_zoo_transfer/model_zoo_feature_correlations.csv
```

The report should answer:

- Which features correlate most strongly with known LB?
- Which models look strong on labeled OOF but risky by transfer features?
- Which public notebooks are closest to the high-LB cluster?
- Which of our models are most similar to high-LB public notebooks?
- Which candidate blends are likely safe, risky, or catastrophic?

## Modeling Strategy

Phase 1 uses a two-stage approach:

1. Risk-tier classification:
   - `catastrophic`
   - `risky`
   - `safe`
   - `ceiling`

2. Candidate ranking inside the safe/ceiling tiers.

Exact LB prediction is a secondary output because the anchor count is small and yesterday's metric experiments showed absolute LB regression can overfit. The primary value is avoiding bad submissions and identifying high-transfer behaviors.

Validation uses leave-one-out over models with known LB. Report both:

- rank quality, such as Spearman correlation
- risk-tier mistakes, especially high-risk models incorrectly labeled safe

## Non-Goals

Phase 1 does not:

- Run every public notebook from scratch.
- Include notebooks without extractable `train_soundscapes` predictions.
- Claim exact LB prediction to 0.001 precision.
- Submit anything to Kaggle automatically.
- Modify production submission kernels.

## Success Criteria

Phase 1 is complete when:

- At least our key models and the public notebooks with available caches are ingested.
- Predictions are normalized with explicit coverage metadata.
- `model_zoo_features.csv` is generated reproducibly.
- `model_zoo_report.md` identifies LB-correlated features, OOF traps, and safe/risky model tiers.
- Leave-one-out validation is included for all models with known LB.

## Open Implementation Notes

Use small, inspectable scripts under:

```text
birdclef-2026/analysis/model_zoo_transfer/
```

Suggested modules:

- `registry.py`: declares model artifacts and metadata.
- `normalize_predictions.py`: converts each artifact to normalized row/class format.
- `features.py`: computes labeled, unlabeled, agreement, and distribution features.
- `analyze.py`: builds correlations, risk tiers, and report.

The registry should be explicit at first. Automatic discovery can be added later only after the first pass proves useful.
