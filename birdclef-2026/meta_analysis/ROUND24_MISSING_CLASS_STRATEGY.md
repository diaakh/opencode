# BirdCLEF+ 2026 — ROUND 24: per-missing-class strategy table

## 1. The complete strategy for all 28 missing-from-train classes

For each of the 28 classes that have no train.csv recordings, I computed:
1. **Perch v2 detection ability** (max prediction in 127k pseudo-cache windows)
2. **Best co-occurrence partner** (Jaccard ≥ X with another class)
3. **Labeled-set hourly prior**

Recommended strategy per class:

### USE_PERCH directly (3 classes — Perch can detect these)

| Class | Perch max | Common name |
|---|---:|---|
| **1491113** | **0.748** | Guaraní leaf-litter frog |
| **47158son07** | **0.535** | Insect sonotype07 |
| **47158son11** | **0.744** | Insect sonotype11 |

These have Perch max prediction > 0.5 in the pseudo cache. Use Perch directly with proper calibration (from ROUND 23).

### USE_HOURLY_PRIOR (5 classes — labeled-set hour pattern works)

| Class | Labeled freq | Best at hour | Strategy |
|---|---:|---|---|
| **517063** | **313 windows** | 84% at hour=01 | Boost to 0.7+ at hour=01 |
| 47158son25 | 84 | hour=03 (67%) | Boost at hour=03-04 |
| 47158son13 | 36 | hour=03 | Boost at hour=03-04 |
| 47158son03 | 33 | hour=03 | Boost at hour=03 |
| 47158son01 | 23 | hour=03 | Boost at hour=03 |

### BROADCAST from co-occurring detected partner (20 classes)

These rely on detecting a co-occurring class (Jaccard ≥ 0.7) then broadcasting:

**Broadcast from 47158son25** (11 classes — son25 is the master alias):
- son15, son16: ALWAYS with son25 (Jaccard 1.0)
- son02, son06, son10, son14, son21, son22, son23, son04: Jaccard 1.0
- son17: Jaccard 0.79

**Broadcast from 47158son11** (4 classes):
- 25073 Chiasmocleis mehelyi (MISSING FROG!): Jaccard 1.0
- son09, son12, son24: Jaccard 1.0

**Broadcast from chacha1** (2 classes — Chaco Chachalaca daytime):
- son08: Jaccard 0.71
- son19: Jaccard 1.0

**Other broadcast** (3 classes):
- son18 from son03 (Jaccard 1.0)
- son20 from son08 (Jaccard 1.0)
- son05 from son13 (Jaccard 1.0)

## 2. The KEY anchor classes

**Anchor 1: 47158son25**
- 84 labeled windows (most common sonotype)
- 11 missing classes broadcast from this
- Perch detection only modest (max 0.33), but reliable in cluster 4 contexts

**Anchor 2: 47158son11**
- Perch max 0.744 (HIGH detection!)
- 4 missing classes broadcast from this (including 25073 missing frog)
- Most valuable single detector for missing classes

**Anchor 3: chacha1 (Chaco Chachalaca)**
- Perch can detect well (daytime bird)
- 2 missing sonotypes broadcast from this

**Anchor 4: 1491113**
- Self-detectable (Perch max 0.75)
- Partner with 22967 (Jaccard 0.66) — boost if 22967 detected

## 3. The inference recipe for missing classes

```python
def predict_missing_classes(perch_probs, hour, site, calibration):
    """
    Apply per-missing-class strategy to recover predictions.
    perch_probs: (12, 234) per-window Perch probabilities (calibrated)
    hour: 0-23
    site: 'S01'..'S23'
    Returns: corrected (12, 234) array
    """
    cls_idx = {c: i for i, c in enumerate(CLASS_LIST)}
    
    # USE_PERCH classes: keep as-is (already calibrated)
    
    # USE_HOURLY_PRIOR: shift up at relevant hour
    if hour in (0, 1, 2):
        # 517063 boost at night
        perch_probs[:, cls_idx['517063']] = np.maximum(
            perch_probs[:, cls_idx['517063']], 0.5
        )
    if hour in (3, 4):
        # Sonotype cluster 4/6 boost
        for sono in ['47158son25', '47158son13', '47158son03']:
            perch_probs[:, cls_idx[sono]] = np.maximum(
                perch_probs[:, cls_idx[sono]], 0.4
            )
    
    # BROADCAST: copy from anchor predictions
    BROADCAST_RULES = {
        '47158son25': ['47158son15', '47158son16', '47158son02', '47158son06', 
                       '47158son10', '47158son14', '47158son21', '47158son22', 
                       '47158son23', '47158son04', '47158son17'],
        '47158son11': ['25073', '47158son09', '47158son12', '47158son24'],
        'chacha1': ['47158son08', '47158son19'],
        '47158son03': ['47158son18'],
        '47158son08': ['47158son20'],
        '47158son13': ['47158son05'],
    }
    for anchor, targets in BROADCAST_RULES.items():
        if anchor not in cls_idx: continue
        anchor_pred = perch_probs[:, cls_idx[anchor]]
        for target in targets:
            if target not in cls_idx: continue
            # Broadcast at 80% of anchor's confidence
            perch_probs[:, cls_idx[target]] = np.maximum(
                perch_probs[:, cls_idx[target]],
                anchor_pred * 0.8
            )
    
    return perch_probs
```

## 4. Expected impact

The 28 missing classes contribute 10.7% of macro-AUC (per alexandergremyakov's analysis). Currently, naive Perch predictions give NEAR-RANDOM AUC for most of them (especially son15-16, son18-20 where Perch is totally blind).

Applying this strategy:
- 3 classes via Perch direct: ~0.65 AUC each
- 5 via hourly prior: ~0.55 AUC each (modest improvement over 0.5)
- 20 via broadcast: ~0.65 AUC each (depending on anchor accuracy)

Net: ~0.55-0.65 AUC for missing classes vs ~0.50 random = **+5-15 points × 10.7% share = +0.5-1.5 macro-AUC points**.

Combined with the per-class calibration from ROUND 23, this strategy could deliver **+1-2 macro-AUC points** purely from leveraging the labeled+pseudo data structure.

## 5. Files saved

- `missing_class_strategy.csv` — per-class recommendation table
- (already saved earlier) `perch_calibration.csv`
- `hourly_species_priors.csv`
- `site_hour_species_priors.csv`

A model can READ these CSVs and apply the corrections at inference time WITHOUT any retraining.

## 6. Sources

All computed from local data:
- `data/train_soundscapes_labels.csv` (ground truth)
- `meta_corpus/datasets/pseudo_cache/` (Perch v2 predictions on 127k windows)
- `data/taxonomy.csv` (class definitions)
- `data/train.csv` (which classes have train recordings)
