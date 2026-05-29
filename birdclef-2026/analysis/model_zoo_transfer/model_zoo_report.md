# BirdCLEF Model Zoo Transfer Report

Models ingested: 16
Models with known LB: 13

## Risk Tiers

| risk_tier | count |
| --- | --- |
| ceiling | 8 |
| safe | 4 |
| risky | 4 |

## Top LB-Correlated Features

| feature | n | pearson | spearman |
| --- | --- | --- | --- |
| agreement_exp019 | 3 | 0.8005984385045469 | 1.0 |
| site_gap | 13 | 0.36105222249412905 | 0.36341617810988613 |
| confidence_rate_gt_0_9 | 13 | -0.21755251155055133 | -0.09691098082930297 |
| labeled_macro_auc | 13 | 0.035220687236086956 | -0.1574803438476173 |
| probability_median | 13 | -0.17530008761541105 | -0.1877650253567745 |
| entropy_mean | 13 | -0.18197735147249658 | -0.21199277056410024 |
| coverage_labeled_rows | 13 | 0.01440256047377661 | -0.21218304602782243 |
| n_sites | 13 | 0.014402560473776582 | -0.21218304602782243 |
| entropy_p90 | 13 | -0.15239545119102657 | -0.28359154252722407 |
| probability_p99 | 13 | 0.026703956386085387 | -0.3149606876952346 |
| site_mean_auc | 13 | -0.23171332559202396 | -0.42398554112820047 |

## Leave-One-Out Validation

MAE: 0.0038; median absolute error: 0.0010; max absolute error: 0.0220.

| model_id | actual_lb | predicted_lb | absolute_error | nearest_model | usable_features |
| --- | --- | --- | --- | --- | --- |
| public__youssefmo942009__edits-birdclef-2026-perch-v2-protossm-0-925__perch_arrays | 0.925 | 0.947 | 0.02199999999999991 | public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 10 |
| public__mtoshidesu__0-937-birdclef-2026-true-end-to-end-pipeline__perch_arrays.local | 0.937 | 0.949 | 0.0119999999999999 | public__yaroslavkholmirzayev__v6-0949-replay__full_perch_arrays | 10 |
| v73_rag | 0.941 | 0.949 | 0.008000000000000007 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | 10 |
| exp019 | 0.949 | 0.946 | 0.0030000000000000027 | birdmae | 10 |
| birdmae | 0.946 | 0.949 | 0.0030000000000000027 | exp019 | 10 |
| public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 0.947 | 0.948 | 0.0010000000000000009 | public__mtoshidesu__test-0-948__perch_arrays | 10 |
| public__mtoshidesu__test-0-948__perch_arrays | 0.948 | 0.947 | 0.0010000000000000009 | public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 10 |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | 0.949 | 0.949 | 0.0 | public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_oof_meta_features__oof_base | 10 |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | 0.949 | 0.949 | 0.0 | public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_oof_meta_features__oof_prior | 10 |
| public__yaroslavkholmirzayev__v6-0949-replay__full_perch_arrays | 0.949 | 0.949 | 0.0 | public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_perch_arrays | 10 |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_oof_meta_features__oof_base | 0.949 | 0.949 | 0.0 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | 10 |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_oof_meta_features__oof_prior | 0.949 | 0.949 | 0.0 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | 10 |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_perch_arrays | 0.949 | 0.949 | 0.0 | public__yaroslavkholmirzayev__v6-0949-replay__full_perch_arrays | 10 |

## Known-LB Models

| model_id | category | coverage | known_lb | labeled_macro_auc | site_gap | risk_tier |
| --- | --- | --- | --- | --- | --- | --- |
| exp019 | blend | labeled | 0.949 | 0.9642921634095929 | 0.03392413246500259 | ceiling |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | public_final_oof | labeled | 0.949 | 0.7999212856838273 | 0.3482855676168628 | ceiling |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | public_final_oof | labeled | 0.949 | 0.6027970235038224 | 0.3561650376654051 | ceiling |
| public__yaroslavkholmirzayev__v6-0949-replay__full_perch_arrays | public_perch_cache | labeled | 0.949 | 0.739018093921368 | 0.03663901045493212 | ceiling |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_oof_meta_features__oof_base | public_final_oof | labeled | 0.949 | 0.7999212856838273 | 0.3482855676168628 | ceiling |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_oof_meta_features__oof_prior | public_final_oof | labeled | 0.949 | 0.6027970235038224 | 0.3561650376654051 | ceiling |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_perch_arrays | public_perch_cache | labeled | 0.949 | 0.739018093921368 | 0.03663901045493212 | ceiling |
| public__mtoshidesu__test-0-948__perch_arrays | public_perch_cache | labeled | 0.948 | 0.7390192174151201 | 0.03663778889694591 | ceiling |
| public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | public_perch_cache | labeled | 0.947 | 0.7390192174151201 | 0.03663778889694591 | safe |
| birdmae | birdmae | labeled | 0.946 | 0.9655316206746958 | 0.08564295624306018 | safe |
| v73_rag | retrieval | labeled | 0.941 | 0.7764174891741018 | 0.12528138261703858 | safe |
| public__mtoshidesu__0-937-birdclef-2026-true-end-to-end-pipeline__perch_arrays.local | public_perch_cache | labeled | 0.937 | 0.739018093921368 | 0.03663901045493212 | safe |
| public__youssefmo942009__edits-birdclef-2026-perch-v2-protossm-0-925__perch_arrays | public_perch_cache | labeled | 0.925 | 0.7390192174151201 | 0.03663778889694591 | risky |
