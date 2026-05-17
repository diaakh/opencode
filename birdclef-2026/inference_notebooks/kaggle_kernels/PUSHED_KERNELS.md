# Kaggle kernels pushed for BirdCLEF 2026 — May 17 2026

All 5 submission variants pushed as private kernels under user `adkasd`.

| Variant | Kernel URL | Status |
|---|---|---|
| sub1 hour_prior w=3.0 | https://www.kaggle.com/code/adkasd/birdclef-2026-sub1-hour3 | running |
| sub2 standalone Bruce | https://www.kaggle.com/code/adkasd/birdclef-2026-sub2-bruce-standalone | running |
| sub3 hour w=2.0 + alias + blind | https://www.kaggle.com/code/adkasd/birdclef-2026-sub3-hour2-alias-blind | running |
| sub4 hour w=3.0 + alias | https://www.kaggle.com/code/adkasd/birdclef-2026-sub4-hour3-alias | running |
| sub5 hour w=3.0 + calib + alias | https://www.kaggle.com/code/adkasd/birdclef-2026-sub5-hour3-calib-alias | running |

## Datasets attached
- competitions/birdclef-2026 (official)
- adkasd/birdclef-2026-priors-research (uploaded as private dataset)
- + exp019's standard datasets (perch onnx, distilled SED, etc.)

## Monitor
`monitor.sh` polls each kernel every 2 minutes and auto-submits to the competition
upon successful completion. Log at /tmp/kpush/monitor.log.

## Manual submit (fallback if monitor fails)

```bash
kaggle competitions submit birdclef-2026 \
  -k adkasd/birdclef-2026-sub1-hour3 \
  -v 1 \
  -f submission.csv \
  -m "sub1: exp019 + hour_prior w=3.0 (OOF 0.9586)"
```

## Local OOF validation context
| Variant | Patches | OOF macro-AUC on Bruce 739 |
|---|---|---:|
| Bruce baseline | none | 0.9304 |
| sub1 | hour_prior w=3.0 | 0.9586 +0.028 |
| sub3 | hour w=2.0 + alias + blind | 0.9583 |
| sub4 | hour w=3.0 + alias | 0.9586 |
| sub5 | hour w=3.0 + calib + alias | 0.9586 |
