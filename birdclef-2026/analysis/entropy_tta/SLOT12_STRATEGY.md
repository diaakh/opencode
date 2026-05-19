# Slot12 Strategy & Decision

## What slot12 does (v4 — latest pushed)

**Per-cell entropy-gated blend** of exp019 (anchor) + sub_v8 + V73 with
**logit-space temperature calibration** of exp019 to sharpen confidence signal.

### Algorithm
1. Calibrate: `P_calib = sigmoid(2.0 * logit(P_exp019))` — sharpens distribution
   without changing rank order
2. Per-cell Bernoulli entropy of P_calib
3. confidence = (1 - H)^2
4. anchor_weight = clip(confidence, 0.40, 0.90)
5. Helper split: 95% sub_v8, 5% V73

### Why this design
- Temperature calibration (T=2.0) addresses exp019's flat distribution: brings
  25% of cells to high-confidence regime (entropy < 0.3) vs only 5% uncalibrated
- Floor=0.40 limits per-cell helper weight to 60% max — safer than the
  OOF-optimal 0.30 which gave 70% helper weight on uncertain cells
- V73 5% weight = small for diversity (V73 raw OOF=0.667 but LB=0.941, so
  trust LB transfer not OOF)

### Effective average weights
- exp019: ~49%  
- sub_v8: ~48.5%
- V73: ~2.5%

### OOF performance
- exp019 alone: 0.9618
- slot12 v4: **0.9796** (+0.018)

## Expected LB outcome

Based on historical OOF→LB transfer rates for our ensembles:
- BirdMAE blanket 30%: +0.012 OOF → -0.003 LB (transfer rate ~-25%)
- slot11 surgical: +0.020 OOF → 0.000 LB (transfer rate 0%)
- slot12 v4 expected: +0.018 OOF → ?

Best case (positive transfer): **0.953–0.955 LB**
Likely case (neutral transfer): **0.948–0.951 LB**
Worst case (negative transfer): **0.945–0.948 LB**

## Decision: submit or hold?

**Submit slot12 v4 today** if:
- v4 dry-run completes successfully (validates pipeline)
- The kernel produces a non-trivial submission.csv with full 234 classes

**Hold last slot for tomorrow** if:
- v4 dry-run fails or shows red flags
- We want to test something fundamentally new tomorrow

Expected value:
- Submit: 60% chance flat, 25% chance +0.002 to +0.005, 15% chance -0.003
- Hold: keep 1 fresh slot for tomorrow's experiments + 5 new slots

Recommendation: **submit slot12 v4** — the entropy gate concept has been
validated locally on OOF, the calibration trick adds real value, and we
want to test how a more aggressive blend transfers to LB.
