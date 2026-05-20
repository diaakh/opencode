# BirdCLEF Model Zoo Transfer Report

Models ingested: 12
Models with known LB: 9

## Risk Tiers

| risk_tier | count |
| --- | --- |
| ceiling | 5 |
| risky | 4 |
| safe | 3 |

## Top LB-Correlated Features

| feature | n | pearson | spearman |
| --- | --- | --- | --- |
| agreement_exp019 | 3 | 0.8005984385045469 | 1.0 |
| site_gap | 9 | 0.28186781910841097 | 0.1682008800516714 |
| confidence_rate_gt_0_9 | 9 | -0.1777283468218957 | 0.10623213476947665 |
| entropy_mean | 9 | -0.10414334086887331 | -0.008852677897456388 |
| probability_median | 9 | -0.12414344201121452 | -0.008852677897456388 |
| entropy_p90 | 9 | -0.09039374657878324 | -0.10454545454545455 |
| labeled_macro_auc | 9 | 0.1059320322473173 | -0.19475891374404056 |
| probability_p99 | 9 | 0.08424177992278942 | -0.19475891374404056 |
| coverage_labeled_rows | 9 | 0.05304759868504261 | -0.2383656473113981 |
| n_sites | 9 | 0.05304759868504261 | -0.2383656473113981 |
| site_mean_auc | 9 | -0.12374407646114714 | -0.26558033692369165 |

## Leave-One-Out Validation

MAE: 0.0056; median absolute error: 0.0030; max absolute error: 0.0220.

| model_id | actual_lb | predicted_lb | absolute_error | nearest_model | usable_features |
| --- | --- | --- | --- | --- | --- |
| public__youssefmo942009__edits-birdclef-2026-perch-v2-protossm-0-925__perch_arrays | 0.925 | 0.947 | 0.02199999999999991 | public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 10 |
| v73_rag | 0.941 | 0.949 | 0.008000000000000007 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | 10 |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | 0.949 | 0.941 | 0.008000000000000007 | v73_rag | 10 |
| exp019 | 0.949 | 0.946 | 0.0030000000000000027 | birdmae | 10 |
| birdmae | 0.946 | 0.949 | 0.0030000000000000027 | exp019 | 10 |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | 0.949 | 0.947 | 0.0020000000000000018 | public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 10 |
| public__yaroslavkholmirzayev__v6-0949-replay__full_perch_arrays | 0.949 | 0.947 | 0.0020000000000000018 | public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 10 |
| public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 0.947 | 0.948 | 0.0010000000000000009 | public__mtoshidesu__test-0-948__perch_arrays | 10 |
| public__mtoshidesu__test-0-948__perch_arrays | 0.948 | 0.947 | 0.0010000000000000009 | public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 10 |

## Known-LB Models

| model_id | category | known_lb | labeled_macro_auc | site_gap | risk_tier |
| --- | --- | --- | --- | --- | --- |
| exp019 | blend | 0.949 | 0.9642921634095929 | 0.03392413246500259 | ceiling |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | public_cache | 0.949 | 0.7999212856838273 | 0.3482855676168628 | ceiling |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | public_cache | 0.949 | 0.6027970235038224 | 0.3561650376654051 | ceiling |
| public__yaroslavkholmirzayev__v6-0949-replay__full_perch_arrays | public_cache | 0.949 | 0.739018093921368 | 0.03663901045493212 | ceiling |
| public__mtoshidesu__test-0-948__perch_arrays | public_cache | 0.948 | 0.7390192174151201 | 0.03663778889694591 | ceiling |
| public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | public_cache | 0.947 | 0.7390192174151201 | 0.03663778889694591 | safe |
| birdmae | birdmae | 0.946 | 0.9655316206746958 | 0.08564295624306018 | safe |
| v73_rag | retrieval | 0.941 | 0.7764174891741018 | 0.12528138261703858 | safe |
| public__youssefmo942009__edits-birdclef-2026-perch-v2-protossm-0-925__perch_arrays | public_cache | 0.925 | 0.7390192174151201 | 0.03663778889694591 | risky |
