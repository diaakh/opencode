# BirdCLEF+ 2026 — How to submit each of the 5 variants

Step-by-step instructions to push each of the 5 daily submissions to Kaggle.

## One-time setup (before sub 1)

### 1. Upload the priors bundle as a Kaggle dataset

```bash
cd /home/user/opencode/birdclef-2026/inference_notebooks/priors_bundle
# Edit dataset-metadata.json and replace REPLACE_WITH_YOUR_USERNAME with your Kaggle username
kaggle datasets create -p .
# wait for upload (~500 KB total)
```

You should now see a dataset at `https://www.kaggle.com/datasets/<your-username>/birdclef-2026-priors-research`.

### 2. Fork your existing 0.949 LB exp019 notebook on Kaggle

1. Open your `birdclef-2026-exp019-eos4-rank-power-06` Kaggle notebook.
2. Click "Copy & Edit" to make a fork.
3. In the right sidebar, click "Add Data" → search for and attach `<your-username>/birdclef-2026-priors-research`.

## Submitting variant N

For each of sub1, sub3, sub4, sub5:

1. Open the variant Python file (e.g. `sub1_hour3.py`).
2. Copy its **entire contents**.
3. In your forked exp019 Kaggle notebook, scroll to the **very last cell** (the one that writes `submission.csv` via `final_submission = write_final_submission(submission, "submission.csv")`).
4. Click "+ Add code cell" **below** that final cell.
5. Paste the entire variant file into the new cell.
6. Save & Run All. Wait for completion.
7. Click "Submit to Competition" → select submission → verify success.

The variant cell will read the `submission.csv` written by exp019, apply patches, and **overwrite** `submission.csv` with the post-processed version. The Kaggle submit button reads `submission.csv` so the patch is what gets scored.

## Submitting sub 2 (standalone Bruce)

Sub 2 is different — it's a **complete standalone notebook** (no exp019 needed).

1. Create a new blank Kaggle notebook.
2. Attach the following datasets:
   - `birdclef-2026` (competition, auto)
   - `brucewu1200/birdclef-2026-cvlb-assets-0911`
   - `<your-username>/birdclef-2026-priors-research`
   - `tuckerarrants/perch-v2-no-dft-onnx` (already in Bruce's bundle, but attach also for redundancy)
3. Set the notebook to **CPU-only**, **internet off**.
4. Paste the entire contents of `sub2_bruce_standalone.py` as the only code cell.
5. Save & Run All. Expect ~30 minutes runtime.
6. Submit to Competition.

## Recommended submission ordering

Today (5 slots):

| Slot | Variant | Risk | Expected delta vs exp019 (LB 0.949) |
|---:|---|---|---:|
| 1 | **sub1_hour3** | low (only adds hour_prior) | +0.005 to +0.015 |
| 2 | **sub2_bruce_standalone** | medium (different model path) | -0.005 to +0.010 |
| 3 | **sub4_hour3_alias** | low (sub1 + free sonotype broadcast) | +0.005 to +0.015 |
| 4 | **sub5_hour3_calib_alias** | medium (calibration may help or hurt on test) | +0.000 to +0.015 |
| 5 | **sub3_hour2_alias_blind** | medium (extra site-blind aggressiveness) | -0.005 to +0.010 |

Important: sub 1, 3, 4, 5 all build on the SAME exp019 base → if exp019 itself crashes on Kaggle today, all 4 of these crash. So:

- **Submit sub1 first** to confirm the patched exp019 path runs end-to-end.
- If sub1 succeeds, batch-submit sub3/sub4/sub5 (each just changes the final cell config).
- **Submit sub2 in parallel** because it's independent and tests an alternative path.

## Watching for failures

Each variant cell prints diagnostics:

```
Priors found at /kaggle/input/...
Loaded N rows × 234 classes; min=X, max=Y
Parsed sites (unique): [...]
Parsed hours (unique): [...]
Applied hour_prior (w=3.0)
Wrote submission.csv: rows=N, cols=235, min=..., max=...
```

If `Parsed hours` is empty or all `-1`, the row_id regex didn't match — the post-processing will be a no-op. Check that the submission row_ids match the format `BC2026_Test_..._SXX_YYYYMMDD_HHMMSS_<endsec>`.

## After submission

LB scores will show on the competition page in ~5 minutes per submission. Record:

- Variant name
- Public LB score
- vs prior best

If a variant DECREASES LB by more than 0.003, that's a STRONG signal the patch hurts on test. Switch off that patch in subsequent submissions.
