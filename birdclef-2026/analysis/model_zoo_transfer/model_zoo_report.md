# BirdCLEF Model Zoo Transfer Report

Models ingested: 6
Models with known LB: 3

## Risk Tiers

| risk_tier | count |
| --- | --- |
| risky | 3 |
| safe | 2 |
| ceiling | 1 |

## Top LB-Correlated Features

| feature | n | pearson | spearman |
| --- | --- | --- | --- |
| site_mean_auc | 3 | 0.9780717300438782 | 1.0 |
| entropy_p90 | 3 | 0.9028962182368041 | 1.0 |
| confidence_rate_gt_0_9 | 3 | 0.9470976793145094 | 1.0 |
| agreement_exp019 | 3 | 0.8005984385045469 | 1.0 |
| labeled_macro_auc | 3 | 0.926442832518378 | 0.5 |
| entropy_mean | 3 | 0.7199357733883385 | 0.5 |
| probability_median | 3 | 0.7568180410873553 | 0.5 |
| probability_p99 | 3 | 0.9280817498712801 | 0.5 |
| site_gap | 3 | -0.9759968105267718 | -1.0 |

## Known-LB Models

| model_id | category | known_lb | labeled_macro_auc | site_gap | risk_tier |
| --- | --- | --- | --- | --- | --- |
| exp019 | blend | 0.949 | 0.9642921634095929 | 0.03392413246500259 | ceiling |
| birdmae | birdmae | 0.946 | 0.9655316206746958 | 0.08564295624306018 | safe |
| v73_rag | retrieval | 0.941 | 0.7764174891741018 | 0.12528138261703858 | safe |
