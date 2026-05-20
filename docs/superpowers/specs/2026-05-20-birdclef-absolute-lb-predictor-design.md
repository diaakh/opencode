# BirdCLEF Absolute LB Predictor Design

## Goal

Build an absolute LB predictor on top of the existing model-zoo transfer analyzer. The predictor should output both:

- `pred_public_lb`: expected Kaggle public leaderboard score.
- `pred_private_proxy`: expected generalization score proxy for private/final leaderboard behavior.

The purpose is to compare candidate models and blends before submission and estimate whether a candidate is likely to improve on the current best. The system must give numeric predictions, but it must also expose confidence, nearest analogs, and risk flags so we do not over-trust a fragile number.

## Principle

This is a decision-support model, not an oracle. It should never claim guaranteed LB gain. It should answer:

```text
Given the artifacts and historical anchors we have, what public LB and private-proxy score should we expect, and how reliable is that estimate?
```

When evidence is weak, duplicated, assumed-order, or outside the known anchor distribution, the predictor should lower confidence rather than output a falsely precise score.

## Inputs

The predictor consumes the existing generated feature table:

```text
birdclef-2026/analysis/model_zoo_transfer/model_zoo_features.csv
```

Required columns:

- `model_id`
- `source`
- `category`
- `coverage`
- `known_lb`
- coverage counts
- labeled metrics
- site/hour metrics
- entropy/probability summary features
- agreement features
- risk tier

It also consumes optional candidate rows generated from new model/blend predictions. Candidates may or may not have `known_lb`.

## Output Files

The predictor writes:

```text
birdclef-2026/analysis/model_zoo_transfer/lb_predictions.csv
birdclef-2026/analysis/model_zoo_transfer/lb_predictor_validation.csv
birdclef-2026/analysis/model_zoo_transfer/lb_predictor_report.md
```

`lb_predictions.csv` contains one row per model/candidate:

- `model_id`
- `known_lb`
- `pred_public_lb`
- `pred_public_lb_low`
- `pred_public_lb_high`
- `pred_private_proxy`
- `pred_private_proxy_low`
- `pred_private_proxy_high`
- `expected_public_delta_vs_best`
- `expected_private_delta_vs_best`
- `confidence`
- `recommendation`
- `nearest_analogs`
- `risk_flags`

## Architecture

Use a hybrid predictor with three components.

### 1. Absolute Public LB Head

This head predicts visible public LB from known-LB anchors. It should be optimized for public-score matching, even if that includes public-LB quirks.

Initial implementation:

- Robust numeric feature cleaning.
- Median imputation for missing numeric features.
- Robust scaling.
- Ridge or elastic-net regression with leave-one-out validation.
- Optional isotonic or linear calibration when enough anchors exist.

Feature groups:

- labeled AUC metrics
- site gap and site mean AUC
- entropy/confidence/probability quantiles
- agreement with high-LB anchors
- coverage and category one-hot features
- metadata flags such as `public_final_oof`, `public_perch_cache`, and `labeled_assumed_order`

### 2. Private/Generalization Proxy Head

This head predicts generalization risk, not the visible public score. It should penalize features that look like overfit or brittle transfer:

- high site gap
- high site/hour dependence
- assumed-order public artifacts
- duplicate or near-duplicate public caches with inconsistent LB
- extreme entropy or probability concentration
- low diversity from current best model
- similarity to known low-LB final OOF artifacts
- strong labeled AUC but weak public analog behavior

Initial implementation:

- Start from the public LB head prediction.
- Apply deterministic penalties and bonuses from risk features.
- Calibrate the proxy scale to remain near public LB, but more conservative.

The private proxy should be monotonic with obvious risk: a model with the same public prediction but higher site gap and lower confidence should receive a lower private proxy.

### 3. Analog And Pairwise Layer

The predictor should compute nearest analogs in feature space and pairwise ranking signals.

Nearest analogs answer:

```text
Which known models does this candidate most resemble?
```

Pairwise ranking answers:

```text
Does candidate A look more like higher-LB anchors than candidate B?
```

Initial implementation:

- Standardize usable numeric features.
- Compute distance to known-LB anchors.
- Estimate analog prediction as inverse-distance weighted LB.
- Blend regression prediction and analog prediction.
- Report nearest analogs in the output.

This layer matters because anchor count is small. Pairwise and analog signals are often more stable than direct absolute regression.

## Confidence Model

Confidence should be explicit and conservative.

High confidence requires:

- candidate is inside the known feature distribution
- multiple nearby known-LB analogs agree
- coverage is verified, not assumed-order
- no severe duplicate-cache inconsistency
- public and private heads agree closely

Medium confidence allows:

- enough analogs but moderate disagreement
- partial coverage
- mild risk flags

Low confidence is required when:

- candidate is outside the known feature range
- nearest analogs have inconsistent LB
- row order is assumed
- candidate is a duplicate/near-duplicate cache with different known LB history
- too few usable features exist

## Recommendations

Each row receives one of:

- `submit_probe`: predicted public gain is positive and confidence is at least medium.
- `submit_risky_probe`: predicted public gain is positive but private proxy or confidence is weak.
- `blend_only`: model has useful diversity but weak standalone LB prediction.
- `hold`: expected delta is small or negative.
- `unsafe`: predicted risk is materially worse than current best.

These recommendations are advisory. They should not submit automatically.

## Validation

Use leave-one-out validation over known-LB anchors.

Report:

- MAE
- median absolute error
- max absolute error
- Spearman rank correlation
- pairwise win accuracy
- calibration by score band
- performance split by category:
  - ours
  - public final OOF
  - public Perch cache
  - assumed-order artifacts

The report should clearly separate:

- verified-row artifacts
- assumed-order artifacts
- duplicate or near-duplicate caches

If validation is poor on a category, predictions for that category must be marked low confidence.

## Duplicate And Leakage Guardrails

Public notebooks often reuse the same Perch train cache while reporting different LB scores. The predictor must detect exact and near-duplicate prediction-feature rows.

For duplicate groups:

- report all known LB scores in the group
- use the group LB spread as an uncertainty penalty
- avoid treating duplicate rows as independent evidence
- prefer final OOF artifacts over base cache artifacts when both exist

The predictor should never train on a held-out row's duplicate group during leave-one-out validation. Group-aware validation is required once duplicate grouping exists.

## CLI

Add a new script:

```text
birdclef-2026/analysis/model_zoo_transfer/predict_lb.py
```

Example:

```bash
python3 birdclef-2026/analysis/model_zoo_transfer/predict_lb.py \
  --features birdclef-2026/analysis/model_zoo_transfer/model_zoo_features.csv \
  --current-best-lb 0.949
```

The script should be read-only with respect to prediction artifacts. It only reads feature CSVs and writes predictor outputs.

## Non-Goals

This phase does not:

- guarantee LB gain
- submit to Kaggle
- run public notebooks from scratch
- train deep models
- optimize blend weights directly
- claim private LB ground truth exists

Blend-weight discovery can be built later using this predictor as a scoring function.

## Success Criteria

The phase is complete when:

- `predict_lb.py` produces public and private-proxy predictions for all rows in `model_zoo_features.csv`.
- Leave-one-out validation is reported with MAE, median error, max error, Spearman, and pairwise accuracy.
- Duplicate groups are detected and used in validation/confidence.
- The report identifies which candidate rows look like public-LB improvements and which are private-risk traps.
- Tests cover feature cleaning, duplicate grouping, leave-one-out prediction, private proxy penalties, and recommendation labels.

