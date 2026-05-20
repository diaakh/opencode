# BirdCLEF Model Zoo Transfer Report

Models ingested: 24
Models with known LB: 21

## Risk Tiers

| risk_tier | count |
| --- | --- |
| risky | 12 |
| ceiling | 8 |
| safe | 4 |

## Top LB-Correlated Features

| feature | n | pearson | spearman |
| --- | --- | --- | --- |
| agreement_exp019 | 3 | 0.8005984385045469 | 1.0 |
| site_mean_auc | 21 | 0.44523239929063263 | 0.35031246268195704 |
| entropy_p90 | 21 | 0.1879522088784683 | 0.2698212702252794 |
| entropy_mean | 21 | 0.24858834543900785 | 0.19788642930125813 |
| coverage_labeled_rows | 21 | 0.27026384223765754 | 0.18389242812245682 |
| n_sites | 21 | 0.2702638422376575 | 0.18389242812245682 |
| probability_median | 21 | 0.14994217465395213 | 0.16391642166185727 |
| confidence_rate_gt_0_9 | 21 | 0.13526045003195017 | 0.1323699763569227 |
| probability_p99 | 21 | 0.09221702437690883 | 0.12033634214265697 |
| labeled_macro_auc | 21 | 0.2354775665667208 | -0.03208969123804186 |
| site_gap | 21 | -0.4774432419179822 | -0.3636831673644744 |

## Leave-One-Out Validation

MAE: 0.0127; median absolute error: 0.0120; max absolute error: 0.0450.

| model_id | actual_lb | predicted_lb | absolute_error | nearest_model | usable_features |
| --- | --- | --- | --- | --- | --- |
| public__baidalinadilzhan__perch-improved-lb-0-904__full_oof_meta_features__oof_base | 0.904 | 0.949 | 0.04499999999999993 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | 10 |
| public__saurabhrajvarma__birdclef-2026-audio-classification-0-922__full_oof_meta_features__oof_base | 0.922 | 0.949 | 0.026999999999999913 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | 10 |
| public__youssefmo942009__edits-birdclef-2026-perch-v2-protossm-0-925__perch_arrays | 0.925 | 0.947 | 0.02199999999999991 | public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 10 |
| public__lingyu07__0-928-perch-yamnet-fast-blend__full_oof_meta_features__oof_base | 0.928 | 0.949 | 0.020999999999999908 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | 10 |
| public__mtoshidesu__0-928-bird26-reproduce-perch-protossm-resssm__full_oof_meta_features__oof_base | 0.928 | 0.949 | 0.020999999999999908 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | 10 |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | 0.949 | 0.928 | 0.020999999999999908 | public__lingyu07__0-928-perch-yamnet-fast-blend__full_oof_meta_features__oof_base | 10 |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | 0.949 | 0.928 | 0.020999999999999908 | public__lingyu07__0-928-perch-yamnet-fast-blend__full_oof_meta_features__oof_prior | 10 |
| public__lingyu07__0-928-perch-yamnet-fast-blend__full_oof_meta_features__oof_prior | 0.928 | 0.949 | 0.020999999999999908 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | 10 |
| public__baidalinadilzhan__perch-improved-lb-0-904__full_oof_meta_features__oof_prior | 0.904 | 0.922 | 0.018000000000000016 | public__saurabhrajvarma__birdclef-2026-audio-classification-0-922__full_oof_meta_features__oof_prior | 10 |
| public__saurabhrajvarma__birdclef-2026-audio-classification-0-922__full_oof_meta_features__oof_prior | 0.922 | 0.904 | 0.018000000000000016 | public__baidalinadilzhan__perch-improved-lb-0-904__full_oof_meta_features__oof_prior | 10 |
| public__mtoshidesu__0-937-birdclef-2026-true-end-to-end-pipeline__perch_arrays.local | 0.937 | 0.949 | 0.0119999999999999 | public__yaroslavkholmirzayev__v6-0949-replay__full_perch_arrays | 10 |
| public__mtoshidesu__0-928-bird26-reproduce-perch-protossm-resssm__full_oof_meta_features__oof_prior | 0.928 | 0.922 | 0.006000000000000005 | public__saurabhrajvarma__birdclef-2026-audio-classification-0-922__full_oof_meta_features__oof_prior | 10 |
| v73_rag | 0.941 | 0.946 | 0.0050000000000000044 | birdmae | 10 |
| birdmae | 0.946 | 0.949 | 0.0030000000000000027 | exp019 | 10 |
| exp019 | 0.949 | 0.946 | 0.0030000000000000027 | birdmae | 10 |
| public__mtoshidesu__test-0-948__perch_arrays | 0.948 | 0.947 | 0.0010000000000000009 | public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 10 |
| public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | 0.947 | 0.948 | 0.0010000000000000009 | public__mtoshidesu__test-0-948__perch_arrays | 10 |
| public__yaroslavkholmirzayev__v6-0949-replay__full_perch_arrays | 0.949 | 0.949 | 0.0 | public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_perch_arrays | 10 |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_oof_meta_features__oof_base | 0.949 | 0.949 | 0.0 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | 10 |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_oof_meta_features__oof_prior | 0.949 | 0.949 | 0.0 | public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | 10 |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_perch_arrays | 0.949 | 0.949 | 0.0 | public__yaroslavkholmirzayev__v6-0949-replay__full_perch_arrays | 10 |

## Known-LB Models

| model_id | category | coverage | known_lb | labeled_macro_auc | site_gap | risk_tier |
| --- | --- | --- | --- | --- | --- | --- |
| exp019 | blend | labeled | 0.949 | 0.9642921634095929 | 0.03392413246500259 | ceiling |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_perch_arrays | public_perch_cache | labeled | 0.949 | 0.739018093921368 | 0.03663901045493212 | ceiling |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_oof_meta_features__oof_prior | public_final_oof | labeled | 0.949 | 0.6027970235038224 | 0.3561650376654051 | ceiling |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_base | public_final_oof | labeled | 0.949 | 0.7999212856838273 | 0.3482855676168628 | ceiling |
| public__yaroslavkholmirzayev__v6-0949-replay__full_oof_meta_features__oof_prior | public_final_oof | labeled | 0.949 | 0.6027970235038224 | 0.3561650376654051 | ceiling |
| public__yaroslavkholmirzayev__v6-0949-replay__full_perch_arrays | public_perch_cache | labeled | 0.949 | 0.739018093921368 | 0.03663901045493212 | ceiling |
| public__itshyao__birdclef-2026-s106-eos5-0949-safealign2__full_oof_meta_features__oof_base | public_final_oof | labeled | 0.949 | 0.7999212856838273 | 0.3482855676168628 | ceiling |
| public__mtoshidesu__test-0-948__perch_arrays | public_perch_cache | labeled | 0.948 | 0.7390192174151201 | 0.03663778889694591 | ceiling |
| public__youssefmo942009__small-tweaks-on-lb-score-0-947__perch_arrays | public_perch_cache | labeled | 0.947 | 0.7390192174151201 | 0.03663778889694591 | safe |
| birdmae | birdmae | labeled | 0.946 | 0.9655316206746958 | 0.08564295624306018 | safe |
| v73_rag | retrieval | labeled | 0.941 | 0.7764174891741018 | 0.12528138261703858 | safe |
| public__mtoshidesu__0-937-birdclef-2026-true-end-to-end-pipeline__perch_arrays.local | public_perch_cache | labeled | 0.937 | 0.739018093921368 | 0.03663901045493212 | safe |
| public__lingyu07__0-928-perch-yamnet-fast-blend__full_oof_meta_features__oof_prior | public_final_oof | labeled_assumed_order | 0.928 | 0.6027970235038224 | 0.3561650376654051 | risky |
| public__lingyu07__0-928-perch-yamnet-fast-blend__full_oof_meta_features__oof_base | public_final_oof | labeled_assumed_order | 0.928 | 0.7999212856838273 | 0.3482855676168628 | risky |
| public__mtoshidesu__0-928-bird26-reproduce-perch-protossm-resssm__full_oof_meta_features__oof_base | public_final_oof | labeled_assumed_order | 0.928 | 0.7938222734797339 | 0.350906075889298 | risky |
| public__mtoshidesu__0-928-bird26-reproduce-perch-protossm-resssm__full_oof_meta_features__oof_prior | public_final_oof | labeled_assumed_order | 0.928 | 0.6103587615863981 | 0.3802807851245934 | risky |
| public__youssefmo942009__edits-birdclef-2026-perch-v2-protossm-0-925__perch_arrays | public_perch_cache | labeled | 0.925 | 0.7390192174151201 | 0.03663778889694591 | risky |
| public__saurabhrajvarma__birdclef-2026-audio-classification-0-922__full_oof_meta_features__oof_prior | public_final_oof | labeled_assumed_order | 0.922 | 0.6103587615863981 | 0.3802807851245934 | risky |
| public__saurabhrajvarma__birdclef-2026-audio-classification-0-922__full_oof_meta_features__oof_base | public_final_oof | labeled_assumed_order | 0.922 | 0.803180582624401 | 0.3549472165718676 | risky |
| public__baidalinadilzhan__perch-improved-lb-0-904__full_oof_meta_features__oof_base | public_final_oof | labeled_assumed_order | 0.904 | 0.8061070123750858 | 0.3480532849762901 | risky |
| public__baidalinadilzhan__perch-improved-lb-0-904__full_oof_meta_features__oof_prior | public_final_oof | labeled_assumed_order | 0.904 | 0.6103587615863981 | 0.3802807851245934 | risky |
