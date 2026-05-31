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

## 28-class strategy — CRITICAL clarification (2026-05-30)
- The 28 = **25 Insecta anonymized 2026 sonotypes** (`47158son01..25`, iNat 47158, no species name)
  + 3 Amphibia. **No Xeno-Canto clip maps to the 25 sonotypes** (Nikita's CC0 extra-data is European
  named grasshoppers — irrelevant to the Neotropical son-IDs). XC specialist covers only ~3/28.
- Competition ships **train_audio only** — **NO `train_soundscapes_labels.csv`** (2 agents confirmed).
  So the 28 son-IDs have NO labeled audio anywhere reachable to us.
- ⇒ **The ONLY route to the 28 is PSEUDO-LABELING the unlabeled soundscapes** (teacher = public
  distilled-SED, which can score son-IDs) → noisy-student. This is exactly the running pipeline.
  The XC B0 specialist is a dead end for 25/28 (kept as a diverse ensemble member for the rest).
- Action: when the pseudo-label precompute finishes, VERIFY the teacher produces non-zero preds for
  the 28 son-IDs; if yes, the noisy-student rounds can learn them. If the public SED also can't score
  them, the 28 are stuck at the public baseline for everyone (no public audio exists).

## Blend result: 0.950 base + round-0 CNN (w=0.12) = 0.944  ⬇ (−0.006, HURTS)
| submission | public LB |
|---|---:|
| V237 EoS9 base (proven) | **0.950** |
| base + orthogonal EffNetV2-S CNN (w=0.12) | **0.944** ⬇ |
| our orthoblend v12 (ProtoSSM repro) | 0.880 |

**Lesson:** orthogonality is necessary but NOT sufficient — the member must be GOOD on the TEST
(soundscape) domain. Our CNN is round-0 supervised on FOCAL train_audio only → weak under
focal→soundscape shift → its rank ordering is net-negative even at low weight. **Domain adaptation
(focal→soundscape via noisy-student) is not optional; it's the requirement.** Best submission
remains the proven 0.950 base. Do NOT waste slots tuning the weak member's weight — fix the member
(noisy-student adaptation), then re-blend.

## DEFINITIVE: the 28 son-IDs are unreachable (2026-05-30)
- GPU pipeline confirmed: the public distilled-SED does NOT genuinely score the 28 anonymized
  Insecta/Amphibia son-IDs (raw sigmoid probs are dead/near-constant; the earlier "coverage" was a
  percentile-RANK artifact fabricating uniform 0–1 from a dead column). No public audio exists for
  them anywhere. ⇒ **the 28 sit at ~0.5 AUC for EVERYONE (incl. Nikita)**.
- **Implication: 0.950→0.96 must come ENTIRELY from the 206 mapped classes** — better soundscape-
  domain modeling (noisy-student adaptation + ensemble diversity), NOT the 28.
- Perch ONNX has no native 234 head; its 14795-`label` output needs the labels.csv→234 mapping
  (v12 used it; the pseudo-label teacher is distilled-SED-only, which is already 234-class).

## Noisy-student student (soundscape-adapted) — TRAINING
- tf_efficientnet_b0, soundscape pseudo-labels (SED teacher), Nikita recipe (pure power-transform,
  fixed-0.5 mixup, per-round powers), 2 rounds, 4000 soundscapes + focal anchors. Kernel
  `bc26-noisy-student-gpu`; artifact `bc26-ns-student-b0`. This trains ON the test (soundscape)
  domain — the fix the round-0 focal CNN lacked. Blend onto 0.950 base + MEASURE (don't assume).

## b0-blend TIMED OUT (2026-05-30) — V237 base is at the 90-min cliff
- `bc26-eos9-plus-nsb0` (V237 0.950 base + soundscape-adapted b0 student, w=0.15) → submission
  **exceeded 90-min runtime** (no score). The dry-run (12 min, no real test audio) hid this because
  the member rank-blend is a no-op without resolvable test audio.
- Root cause: the **V237 base pipeline alone runs ~85+ min**; adding any 600-file member is a coin-flip.
  The focal-CNN version barely fit (0.944); the b0 version tipped over. ⇒ member-blending onto V237
  is RUNTIME-FRAGILE.
- FIX for next attempt: make base+member fit 90 min with MARGIN — trim the base (fewer SED folds /
  lighter TTA) OR run the member cheaper (fewer windows), and VALIDATE the real 600-file runtime via
  a scale-test before submitting. The orthoblend harness (3-fold SED + Perch, ~59 min) has headroom
  but its base is weak (0.880) — so the real task is a STRONG base that leaves member headroom.

## DEFINITIVE: noisy-student distillation students do NOT beat 0.950 (2026-05-31)
| submission | LB |
|---|---:|
| EoS9 base (proven) | **0.950** |
| + focal effv2s CNN (w=0.12) | 0.944 |
| + soundscape-adapted b0 NS student, 3-fold-SED base (w=0.15) | **0.945** (runtime-safe now) |
- Both members HURT. The NS students distill the public distilled-SED that's ALREADY in the base →
  correlated/redundant; the b0's overconfident 43%-positive preds add rank noise. Branch A
  (add NS students) is DISCONFIRMED. nfnet/effv2s would be the same — stop adding them.
- The 3-fold-SED trim fixed the timeout (no runtime failure this time).
- **PIVOT:** the ONLY member our own LOSO validation proved helps is **BirdMAE (+0.004, independent
  foundation model, NOT a SED distillation)** — never wired into a submission. That's the next lever.
  Honest ceiling: BirdMAE→~0.954; reaching 0.96 needs Nikita's full from-scratch independent
  ensemble (multi-day), not these single-public-teacher distillations.

## FINAL exhaustive scoreboard (2026-05-31) — we cannot beat 0.950 by adding members
| submission | LB |
|---|---:|
| **EoS9 base (V237, proven)** | **0.950 ← BEST** |
| + BirdMAE (independent FM, w=0.15, 2-fold-SED base, stride-4) | **0.949** |
| + soundscape-adapted b0 NS student (3-fold base) | 0.945 |
| + focal effv2s CNN | 0.944 |
| our ProtoSSM reproduction (v12) | 0.880 |

### Why 0.950 is the ceiling with available resources
1. **The 28 son-IDs are unreachable for everyone** (no public audio; SED/our students can't score
   them) → that headroom is closed for all, including Nikita.
2. **NS-distillation students are correlated** with the public SED already in the base → they hurt.
3. **BirdMAE (the one LOSO-validated independent lever, +0.004) helps marginally, BUT the V237 base
   is at the 90-min runtime cliff**, so fitting BirdMAE requires trimming the base (5→2 SED folds),
   which costs ~what BirdMAE adds → net ~flat (0.949). The runtime budget is the binding constraint.
4. The gap to Nikita's 0.964 is his FROM-SCRATCH 7-backbone ensemble trained on focal+labeled-
   soundscape data, with his own iterative pseudo-labels — a multi-day build we cannot complete AND
   validate in the remaining ~2.5 days under GPU cap=2 and the broken/no-CV situation.

### Recommendation
Finalize the proven **0.950** as the primary submission (+ a decorrelated 2nd for the shakeup).
0.96 is not achievable with available resources/time; it requires replicating Nikita's full
independent ensemble pipeline.
