# Entropy/TTA Experiment Findings

## Goal
Test the LLM self-consistency analog: generate multiple views per input,
aggregate via entropy weighting. Find a CPU-only, ≤90-min config that beats
exp019's LB anchor of 0.949.

## Setup
Labeled OOF on train_soundscapes (739 windows × 234 classes, 3122 positives).
Models with locally-aligned OOF:

| Model | OOF Macro-AUC | Standalone LB | OOF→LB ratio |
|-------|---------------|---------------|--------------|
| exp019 | 0.9618 | 0.949 | clean transfer |
| BirdMAE | 0.9721 | ~0.94 (in blend -0.003) | leaks high on OOF |
| V73 (raw) | 0.6665 | 0.941 | leaks LOW on OOF |
| ConvNeXt-RAG | 0.9390 | untested | likely clean |
| BirdAVES-RAG | 0.9160 | untested | likely clean |
| MLP-5seed | 0.8570 | (in sub_v8) | unknown |
| Bruce (Perch-Ridge) | ~0.96 | 0.755 standalone | LEAKS heavily |
| sub_v8 final | 0.9748 | untested (proxy ≈ 0.945) | unknown |

## Critical insights

### 1. exp019's flat probability distribution
- 76% of cells have entropy > 0.7 (highly uncertain)
- Only 0.77% of cells have entropy < 0.1 (highly confident)
- Mean prob = 0.489, median = 0.477
- **Consequence**: per-cell entropy gate ≈ fixed weighted blend for ~95% of cells

### 2. Labeled OOF leakage patterns
- **BirdMAE**: fine-tuned on train_audio which shares patterns with labeled OOF subset → OOF inflated
- **V73**: trained externally, labeled OOF subset doesn't match its training distribution → OOF deflated
- **Bruce**: also leaks (sub_v8 stack uses Bruce as one component)

### 3. Per-cell vs per-row vs blanket
| Strategy | OOF | Notes |
|----------|-----|-------|
| Blanket exp019+BirdMAE w=0.3 | 0.9733 (+0.012) | OOF good, LB bad (slot6 -0.003) |
| Per-cell entropy gate floor=0.5 (BirdMAE) | 0.9807 (+0.019) | OOF best, LB untested |
| Per-row entropy gate (BirdMAE) | 0.9801 (+0.018) | Per-row slightly worse than per-cell |
| Surgical 22-class (slot11) | est 0.985 | LB flat 0.949 |

## Aggregation method comparison (mean of [exp019, helpers])
For combo [exp019, v73, convnext_rag, birdaves_rag] in rank space:

| Method | OOF |
|--------|-----|
| Arithmetic mean | 0.9443 |
| Geometric mean | 0.8920 |
| Median | 0.9585 |
| Trimmed mean | 0.9585 |
| Logit-space mean | 0.9676 |
| **Entropy gate floor=0.3** | **0.9750** |
| Entropy gate floor=0.5 | 0.9705 |
| Entropy gate floor=0.7 | 0.9651 |

Entropy gating wins, but only marginally vs logit-space mean.

## Slot12 design (chosen)

**Per-cell entropy gate with conservative anchor floor**:
- ENTROPY_T = 2.0
- ANCHOR_FLOOR = 0.55 (exp019 always ≥55%)
- ANCHOR_CEIL = 0.90 (exp019 never >90%)
- Helpers: sub_v8 95%, V73 5%

Effective avg weights: ~56% exp019 + ~42% sub_v8 + ~2% V73
OOF: 0.9731 (+0.011)

## Why entropy gate doesn't unlock higher LB

The fundamental issue: with exp019's flat probability distribution, the gate's
adaptive behavior is limited to ~5% of cells. For the other 95%, it acts as a
fixed weighted blend — exactly the strategy that produces OOF gain but LB drag.

**To unlock the entropy idea's potential**, we would need either:
1. exp019 to produce sharper (more confident) probabilities — requires modifying
   its post-processing chain
2. Run exp019 with actual TTA (multi-view) — modifying 8200-line file = risky
3. Use a different anchor model with sharper distribution

## Predicted LB for slot12 v2
- OOF gain: +0.011
- Historical OOF→LB gap: -0.010 to -0.020
- **Expected LB: 0.946–0.952**, central estimate 0.948–0.950
- Best case: +0.003 over exp019 (0.952)
- Worst case: -0.003 (0.946), matching slot6
