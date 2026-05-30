# Orthoblend submission results + course-correction (2026-05-30)

| ver | config | public LB |
|---|---|---:|
| v5 | ProtoSSM `perch_logits=None` + SED | (no score / dry) |
| v9 | same, 3-fold SED, OOM+speed fixed | **0.809** |
| v12 | **Perch-logit gated fusion RESTORED** (206/234 mapped) + 3-fold SED | **0.880** |
| (existing account replays V237/V242) | public EoS9 0.950 anchor | **0.950** |

## Finding
- The fusion fix gave +0.071 (0.809→0.880) — confirms `perch_logits=None` was the bug.
- **0.880 ≈ the A7 discussion clue's ~0.889 ceiling for a trainable-Perch gated-fusion head.**
  Our custom ProtoSSM reproduction has hit that ceiling; it does NOT reach the public 0.950 anchor.
- Therefore our orthoblend's ProtoSSM members are NOT a good base (they underperform the
  account's existing working 0.950 replays by ~0.07).

## Course-correction (for LB 0.96)
- **BASE = the proven 0.950 public-EoS9 replay** (already working), NOT our 0.880 reproduction.
- **+ orthogonal SED CNN noisy-student** (Nikita recipe; training round-0 now) rank-blended on top.
- **+ EffNetB0 28-class specialist** (the buried headroom).
- Keep the orthoblend INFRASTRUCTURE (rank-blend, OpenVINO, self-test, scale-test) — reusable;
  just swap the base members to the 0.950 config.
- Select final subs by **public-LB + worst-fold** (A7: CV is anti-correlated with LB).

## Orthogonal SED CNN (round-0) trained — 2026-05-30
- Model: `tf_efficientnetv2_s.in21k` (20.48M), 1ch log-mel (128/2048/512), 5s@32k, BCE, AMP.
- P100 fix: `pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1` (cu126, last sm_60 line).
- Val BCE 0.0251→0.0088 (65 min, no OOM). train_audio top-1 = 0.799 (real signal).
- Artifact dataset: `adkasd/bc26-g124-cnn-effv2s` (g124_fold1_fp16.pt + train_audio preds).
- Perch-INDEPENDENT (mel CNN) → the orthogonal member. Orthogonality vs Perch measured at blend time.
- NEXT: rank-blend onto the working 0.950 base + submit (measure lift); then noisy-student rounds.
