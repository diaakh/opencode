# BirdCLEF+ 2026 — submission post-processing variants

5 Kaggle-ready inference variants built on top of the exp019 baseline (LB 0.949).
Each variant differs only in the **final post-processing block** appended to the notebook
after `submission.csv` is written by Model_7.

**Local OOF validation on Bruce Wu's 739 labeled windows** (see `validate_patches.py`):

| Variant | Patches | OOF macro-AUC | Δ vs Bruce baseline |
|---|---|---:|---:|
| Bruce baseline (raw exp019 stand-in) | — | 0.9304 | — |
| **sub1_hour3** | hour_prior w=3.0 | **0.9586** | **+0.028** ⭐ |
| **sub2_bruce_standalone** | Standalone Bruce + hour_prior | 0.9586 | +0.028 |
| sub3_hour2_alias_blind | hour_prior w=2.0 + sonotype_alias + site_blind | 0.9583 | +0.028 |
| sub4_hour3_alias | hour_prior w=3.0 + sonotype_alias | 0.9586 | +0.028 |
| sub5_hour3_calib_alias | hour_prior w=3.0 + perch_calib + sonotype_alias | 0.9586 | +0.028 |

OOF→LB gap is typically 0.01–0.02 on this competition, so we expect each variant to
score **0.94–0.95 on private LB** if the LB gap matches.

## Required Kaggle datasets

Each variant needs these datasets attached to the Kaggle notebook:

1. **`birdclef-2026`** — official competition data (auto-attached)
2. Whatever **exp019 already uses** for Model_7 (Perch ONNX, BirdNET, Distilled SED, etc.)
3. **One new private dataset** containing the contents of `priors_bundle/`:
   - `pseudo_hour_priors.csv` (24×234 — primary baseline)
   - `pseudo_site_hour_priors.csv` (joint priors)
   - `pseudo_site_priors.csv` (per-site)
   - `perch_calibration.csv` (per-class bias correction)
   - `missing_class_strategy.csv` (recovery strategy for 28 missing classes)

   **Upload command** (from this machine, requires Kaggle credentials):
   ```bash
   cd /home/user/opencode/birdclef-2026/inference_notebooks/priors_bundle
   # Create dataset-metadata.json (one-time)
   echo '{"title":"birdclef-2026-priors-research","id":"YOUR_KAGGLE_USERNAME/birdclef-2026-priors-research","licenses":[{"name":"CC0-1.0"}]}' > dataset-metadata.json
   kaggle datasets create -p .
   ```

## Workflow for each submission

1. Fork your `exp019` Kaggle notebook (the one currently at LB 0.949).
2. Attach `<your-username>/birdclef-2026-priors-research` as input.
3. Append the contents of the chosen `subN_*.py` file as a **new code cell at the very end** of the notebook, after Model_7 writes `submission.csv`.
4. Save & submit.

The post-proc cell reads `submission.csv`, applies the patches, and **overwrites** `submission.csv`.

## Variant rationale

- **sub1_hour3** — minimal change, only the single highest-impact patch. Safest pick.
- **sub2_bruce_standalone** — completely independent path (no exp019). If LB shows
  it ≈ sub1, we have a 2-source ensemble for sub5 ensembling.
- **sub3_hour2_alias_blind** — softer hour weight + site-blind boost for the
  14 unlabeled sites. Tests user's priority 2 (site-blind ensemble).
- **sub4_hour3_alias** — sub1 + sonotype alias broadcasting (cheap, neutral OOF).
- **sub5_hour3_calib_alias** — sub4 + per-class Perch calibration. Tests if ROUND 23
  systematic-bias correction helps on actual test data.

## Test the variant locally first

Each `subN_*.py` includes a `--dry-run` mode that:
1. Loads Bruce's pre-computed OOF
2. Applies the same patch sequence
3. Reports macro-AUC delta

```bash
cd inference_notebooks
python3 validate_patches.py   # see all configs at once
```

## Submission ordering recommendation

Today's 5 daily slots, in order of risk:

1. **sub1_hour3** (safest, single patch) — proves the hour-prior path works
2. **sub2_bruce_standalone** (alternative anchor for ensembling)
3. **sub4_hour3_alias** (adds sonotype alias, very low risk)
4. **sub5_hour3_calib_alias** (tests calibration — if it beats sub4, we know calib helps)
5. **sub3_hour2_alias_blind** (more aggressive site-blind blend — last because more change)

If sub1 already shows < expected gain on LB, switch to investigating WHY before
spending more slots.
