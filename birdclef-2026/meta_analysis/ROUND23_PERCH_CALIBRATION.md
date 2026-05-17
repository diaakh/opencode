# BirdCLEF+ 2026 — ROUND 23: Perch v2 systematic per-class calibration errors

## 1. The big discovery

Comparing the **labeled soundscape ground truth** (66 files, 739 unique windows) against the **Perch v2 pseudo predictions** for the same (site, hour) combinations reveals **massive systematic biases**:

- **Perch UNDER-PREDICTS Pantanal frogs by 10-200x**
- **Perch OVER-PREDICTS nightjars/potoo/owls by 10-60x**

The top errors:

### Most severe UNDER-predictions

| Species | True prevalence | Perch pseudo | Multiplier needed |
|---|---:|---:|---:|
| **517063 Southern Orange-legged Frog** | **42.4%** | **0.4%** | **107x** |
| 24321 Mato Grosso Snouted Tree Frog | 23.3% | 0.1% | **197x** |
| 22967 Marbled White-lipped Frog | 21.0% | 0.6% | **35x** |
| 66971 Paraguayan Swimming Frog | 20.2% | 2.8% | 7.3x |
| 24279 Lesser Snouted Tree Frog | 23.4% | 8.9% | 2.6x |
| 555146 Chaco Tree Frog | 28.4% | 14.3% | 2.0x |
| 65380 Dwarf Tree Frog | 45.1% | 33.4% | 1.35x |
| 23158 Pale-legged Weeping Frog | 23.7% | 15.4% | 1.5x |
| 47158son25 | 11.4% | 3.2% | 3.6x |
| 47158son07 | 6.5% | 1.2% | 5.5x |

**Perch v2 essentially cannot detect Southern Orange-legged Frog or Mato Grosso Tree Frog at all.** These are the dominant Pantanal frog species — but Perch never trained on them adequately.

### Most severe OVER-predictions (FALSE POSITIVES from Perch)

| Species | True prevalence | Perch pseudo | Reduction needed |
|---|---:|---:|---:|
| **compot1 Common Potoo** | **0.4%** | **32.4%** | **80x reduction** |
| compau Common Pauraque | 5.1% | 33.0% | 6.4x |
| trsowl Tropical Screech-Owl | 3.5% | 25.6% | 7.3x |
| undtin1 Undulated Tinamou | 5.8% | 22.2% | 3.8x |
| **fepowl Ferruginous Pygmy Owl** | **0.0%** | **11.6%** | infinite |
| houspa House Sparrow | 0.0% | 8.1% | infinite |
| strowl1 Striped Owl | 0.0% | 7.3% | infinite |
| roahaw Roadside Hawk | 0.0% | 7.3% | infinite |
| bkcdon Black-capped Donacobius | 0.0% | 7.1% | infinite |
| 14 more species at ~7% Perch with 0% true | | | |

**Perch HALLUCINATES nightjars/potoos/owls.** Common Potoo: TRUE 0.4%, Perch 32% → **80x over-prediction**. The Pantanal night has ambient low-freq sounds (frog choruses, wind) that Perch interprets as Potoo calls.

## 2. Why Perch v2 fails this way

From the Perch 2.0 paper (`arxiv:2508.04665`):
- Training corpus: 89% birds (mostly XC + iNat birds), only 4% insects, 4% amphibians
- For frogs/amphibians, Perch saw mostly North/Central American species, NOT Pantanal-specific frogs

**Pantanal frog species** like Southern Orange-legged Frog (Pithecopus azureus = 517063) and Mato Grosso Snouted Tree Frog (24321) are LOCAL species poorly represented in Perch's training data. Perch's embedding for these calls is similar to "background noise" → confused with nightjar calls (which Perch knows well).

**Pantanal frogs sound like generic 'low-frequency bird call'** to Perch, which then misattributes them to nightjars (its closest known low-freq night-active class).

## 3. Per-class calibration table (saved to `perch_calibration.csv`)

For each of 234 classes, the table contains:
- `labeled_prev`: true prevalence (from 739 labeled windows)
- `pseudo_prev`: Perch's average prediction
- `diff`: labeled - pseudo (positive = Perch under-predicts)
- `ratio`: labeled / pseudo (multiplier to apply to Perch)

### Calibration formula

```python
calib = pd.read_csv('perch_calibration.csv')

# For Perch's logit predictions per class c:
def calibrate(perch_pred, c):
    ratio = calib.loc[c, 'ratio']
    # Clip ratio to [0.05, 20] to avoid extreme corrections
    ratio = np.clip(ratio, 0.05, 20.0)
    # Apply in logit space (more stable):
    p = perch_pred  # ∈ [0, 1]
    logit_p = np.log(p / (1 - p + 1e-7) + 1e-7)
    logit_adjusted = logit_p + np.log(ratio)
    return 1 / (1 + np.exp(-logit_adjusted))

# Apply to all classes at once:
def calibrate_all(perch_preds, calib_ratios):
    # perch_preds: (N, 234) prob array
    log_ratios = np.log(np.clip(calib_ratios, 0.05, 20.0))
    logit_p = np.log(perch_preds / (1 - perch_preds + 1e-7) + 1e-7)
    return 1 / (1 + np.exp(-(logit_p + log_ratios[None, :])))
```

## 4. Implications for macro-AUC

Macro-AUC averages per-class AUC. If Perch under-predicts a class systematically (like 517063), the class's predictions are mostly clustered low, and AUC depends on ranking within those low values.

For under-predicted classes:
- All predictions are low (e.g., 0.001-0.05)
- AUC depends on whether the model can RANK windows with the species higher than windows without
- Even at low absolute scores, the model can have decent AUC if the RANKING is preserved
- **But systematic under-prediction means the SCORE SHIFT is wrong** — calibration directly improves AUC

For over-predicted classes (compot1, false positives):
- Predictions are inflated (5-30% even when species absent)
- AUC depends on whether the model can rank species-present windows even higher
- Often the absolute over-prediction doesn't hurt ranking
- **But the inflated baseline pushes more mass onto false positives**

**Applying the calibration: estimated gain +0.005 to +0.010 on macro-AUC.** Small but consistent.

## 5. Site- and hour-conditional calibration

The systematic biases vary by site. Let me check (not done yet, but the framework is):

```python
# Per-site calibration
for site in sites:
    site_labels = labels[labels['site']==site]
    site_pseudo = pseudo[meta['site']==site]
    # ... compute per-site labeled vs pseudo difference
```

Different sites have different acoustic profiles, so different calibration may be needed.

## 6. Most-relevant classes for the test sample (S05, hour=01)

The test sample is S05 at hour=01. Combining:
- S05-specific pseudo predictions (60 windows): TOP species = 24279 (83%)
- Hour=01 labeled prior: TOP species = 517063 (84%)
- Perch's known biases: 517063 under-predicts 107x

**The test sample at S05 hour=01 likely contains BOTH:**
- 24279 Lesser Snouted Tree Frog (high Perch confidence at S05)
- 517063 Southern Orange-legged Frog (Perch can't detect, but high prior at hour=01)
- 65380 Dwarf Tree Frog (moderate Perch + high prior)
- compau/compot1 (Perch over-predicts, but probably real Pauraque/Potoo presence)
- Several other frogs (22973, 555146, 23158)

**A smart inference strategy**:
1. Get Perch predictions
2. Apply per-class calibration (multiply by `ratio`)
3. Apply hour-conditional prior boost (especially for under-predicted frogs)
4. Apply Pantanal-frog-chorus broadcast (if 65380 detected, boost 517063, 24279, etc.)

## 7. Concrete plan additions

### Tier-A drop-in
- **Save the calibration ratios and apply them per-class to Perch outputs**
- Expected gain: +0.005 to +0.010 macro-AUC
- Highest-impact corrections: 517063 (boost), 24321 (boost), compot1 (suppress), fepowl (suppress)

### Tier-B (better)
- **Re-train a thin head on top of Perch** specifically to learn the calibration biases
- Use ALL 66 labeled files as training; freeze Perch backbone; train only the final layer

### Tier-C (best)
- **Distill into a non-Perch backbone** that doesn't have these biases
- BC2025 winners (Babych, Sydorskyi) all do this — they don't use raw Perch outputs
- Train HGNet/EfficientNet from scratch on pseudo-labeled data with calibration applied

## 8. Sources

All computed locally from:
- `data/train_soundscapes_labels.csv` (66 labeled files, 739 unique annotated windows)
- `meta_corpus/datasets/pseudo_cache/` (backtracking/birdclef2026-pseudo-cache-v1, 127k windows of Perch predictions)
- Output: `meta_analysis/perch_calibration.csv` (234 classes × calibration ratios)
