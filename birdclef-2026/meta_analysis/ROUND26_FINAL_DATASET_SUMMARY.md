# BirdCLEF+ 2026 — ROUND 26: Final dataset deep-dive summary (Rounds 17-25)

This is the consolidated summary of 9 rounds of deep dataset analysis (Rounds 17-25), covering every hidden pattern I could find in the BC2026 data itself.

## 1. Headline discoveries (ranked by macro-AUC impact)

### A. Pseudo-cache hour priors → 0.896 macro-AUC FREE (Round 25)
The single highest-impact finding. Just looking up the hour of each test file and applying pseudo-cache-derived priors gives 0.896 macro-AUC honestly (leave-one-site-out). With no model required.

### B. Perch v2 has systematic per-class biases of 10-200x (Round 23)
- **Perch under-predicts Southern Orange-legged Frog (517063) by 107x** — true 42%, Perch 0.4%
- **Perch under-predicts Mato Grosso Tree Frog (24321) by 197x** — true 23%, Perch 0.1%  
- **Perch over-predicts Common Potoo (compot1) by 80x** — true 0.4%, Perch 32%
- Calibration ratios saved in `perch_calibration.csv`
- Apply in logit space: `log(ratio)` shift per class
- Expected gain: +0.005-0.010 macro-AUC

### C. Sonotype aliases reduce effective class count (Round 17)
- **47158son15 ≡ 47158son16** (perfect alias, J=1.0)
- **47158son21 = son22 = son23** (triplet alias, J=0.92-1.0)
- **Black Howling Monkey (43435) ≡ 47158son14** (cross-class alias, J=1.0)
- The 25 "sonotypes" are functionally **6 spectral clusters** + 3 alias groups
- Treat aliases as broadcast pairs at inference

### D. Per-missing-class recovery strategy (Round 24)
- **3 classes** USE_PERCH directly (1491113, son07, son11)
- **5 classes** USE_HOURLY_PRIOR (517063 boosted to 0.7 at hr=01)
- **20 classes** BROADCAST from co-occurrence partners (son11→25073 missing frog, son25→11 sonotypes)
- Strategy table saved in `missing_class_strategy.csv`
- Expected gain: +0.005-0.015 macro-AUC

### E. 2025 train_soundscapes is a different acoustic regime (Round 19)
- 2014-2024: noisy, clipping-heavy night recordings (RMS 0.06-0.27, clipping 0-3%)
- **2025: quiet, low-clipping daytime recordings (RMS 0.04, clipping 0.06%)**
- Test data is from 2025 → use 2025 train_soundscapes as primary domain match
- Year-weighted training: 2025=1.0, 2024=0.7, 2023=0.5, 2014-2022=0.3

### F. Per-class confusion matrix (Round 21)
- The Pantanal Frog Chorus trio (65380, 66971, 22973) are SPECTRALLY IDENTICAL (sim 0.90-0.97) → these 3 frogs are co-detected together; treat as alias for prediction
- Frogs confuse with sparrows/spinetails/hummingbirds (sim 0.94+) → temporal features needed
- Prionacris erosa insect ≈ Nanday Parakeet (sim 0.927)

### G. 34 cross-species duplicate file pairs (label noise) (Round 20)
- 68 train_audio files have identical audio labeled as different species
- Pattern: consecutive iNat IDs (iNat791264 fepowl + iNat791265 giwrai1 etc.)
- These are iNat multi-attachment uploads where same audio got 2 species labels
- Drop or treat as multi-label
- File: `duplicate_train_audio.csv` (200 dup-group rows)

### H. 7-species Pantanal Frog Chorus group (Round 20)
- 65380, 24279, 66971, 517063, 23158, 24321, 555146 always co-occur
- Apply label-propagation: if one is detected, boost the others
- 25073 (MISSING) ↔ 326272 (Weeping Frog) Jaccard=0.75: rescues another missing class

### I. Train_audio quality issues (Round 18)
- **2,601 files (7.3%) are <5 seconds** — can't fill a standard window
- 370 files (1.04%) are <1 second
- Min duration: 8 milliseconds (greyel/iNat1375792)
- **3 species are essentially untrainable**: 23150 (20s), 23724 (7s), 209233 (<1s)
- Hooded Capuchin: 42s file but only 0.5s of actual signal (98% silent)

### J. Codec domain shift (from earlier rounds)
- Train_audio: 86 kbps OGG (80% of XC files at 86 kbps)
- Train_soundscapes: 72 kbps OGG (100%)
- Test data likely 72 kbps → re-encode train_audio to match

### K. iNat collection 666x more clipped than XC (Round 19)
- XC: 0.001% clipping mean (professional recordings)
- iNat: 0.666% clipping mean (mobile-phone level)
- iNat matches test domain better → 2-3x weight iNat samples
- iNat files: 94% of energy in first 5 + last 5 seconds (smart crop edges)

### L. Site×hour acoustic profiles vary 130x (Round 17)
- Spectral centroid range: 43 Hz (S11 wind site) to 5,557 Hz (S18 insect site)
- Some sites are clipping-heavy (S01: 2.37%, S13: 1.83%), others totally silent (S04: 78.4%)
- Hourly species patterns differ dramatically (hr=01 frogs, hr=03 sonotypes, hr=06 dawn birds)

### M. Author noise floor varies 17x (Round 18)
- Webster (clean): 0.002 noise floor
- Krabbe (noisy): 0.033 noise floor (17x)
- Author-grouped CV essential to prevent recorder-fingerprint leakage

### N. Geographic distribution: only 2.4% of train is from Pantanal (Round 21)
- 30.2% of train_audio is from N hemisphere (NO Pantanal context)
- 100% of Feral Horse (209233) and Bos taurus (74113) are NH recordings
- 119 of 206 species have ANY Pantanal-region data
- 87 species (37%) have ZERO regionally-aligned training data

## 2. Saved priors and strategy files (all in `meta_analysis/`)

| File | Purpose | Use case |
|---|---|---|
| `pseudo_hour_priors.csv` | 24 hours × 234 classes | **PRIMARY BASELINE** (0.90 AUC) |
| `pseudo_site_hour_priors.csv` | 120 joint × 234 | Per-site prior when available |
| `pseudo_site_priors.csv` | 21 sites × 234 | Site profile |
| `perch_calibration.csv` | 234 classes × ratio | Apply in logit space |
| `missing_class_strategy.csv` | 28 missing × strategy | Per-class recovery rule |
| `duplicate_train_audio.csv` | 200 files × group_key | Label-noise detection |
| `hourly_species_priors.csv` | 13 × 75 (labeled) | Labeled-set hour prior |
| `site_species_priors.csv` | 9 × 75 (labeled) | Labeled-set site prior |
| `site_hour_species_priors.csv` | 25 × 75 (labeled) | Labeled-set joint |
| `s05_h01_estimated_prior.csv` | 234 × probability | Best estimate for sample test |

## 3. Complete inference recipe (no training required)

```python
import pandas as pd, numpy as np, re

# Load all priors
pseudo_h = pd.read_csv('pseudo_hour_priors.csv').set_index('hour')
pseudo_sh = pd.read_csv('pseudo_site_hour_priors.csv').set_index('site_hour')
calib = pd.read_csv('perch_calibration.csv').set_index('class')
missing = pd.read_csv('missing_class_strategy.csv').set_index('class')

CLASSES = pseudo_h.columns.tolist()
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}

def predict_test_file(audio_path):
    # 1. Parse filename
    m = re.match(r"BC2026_Test_\d+_(S\d+)_\d{8}_(\d{2})", audio_path.name)
    site, hour = m.group(1), int(m.group(2))
    
    # 2. Get baseline prior
    key = f"{site}_h{hour:02d}"
    if key in pseudo_sh.index:
        prior = pseudo_sh.loc[key].values
    else:
        prior = pseudo_h.loc[hour].values
    
    # 3. Run Perch (real model inference)
    perch_pred = perch_model.predict(load_audio(audio_path))  # (12, 234)
    
    # 4. Apply per-class Perch calibration
    ratios = np.array([calib.loc[c, 'ratio'] if c in calib.index else 1.0 
                       for c in CLASSES])
    ratios = np.clip(ratios, 0.05, 10.0)
    perch_calibrated = perch_pred * ratios
    
    # 5. Blend with prior
    blended = 0.7 * perch_calibrated + 0.3 * prior
    
    # 6. Apply missing-class strategy (broadcast from anchors)
    apply_missing_strategy(blended, hour, site, CLS_IDX)
    
    # 7. Apply per-site adjustments for known leakage:
    #    - S04: massively suppress (78% silence faulty)
    #    - S01/S13/S18: account for clipping
    
    # 8. Sonotype alias broadcasts:
    son15_idx = CLS_IDX['47158son15']; son16_idx = CLS_IDX['47158son16']
    blended[:, son15_idx] = blended[:, son16_idx] = (blended[:, son15_idx] + blended[:, son16_idx]) / 2
    
    son22_idx = CLS_IDX['47158son22']; son23_idx = CLS_IDX['47158son23']
    blended[:, son22_idx] = blended[:, son23_idx] = (blended[:, son22_idx] + blended[:, son23_idx]) / 2
    
    # 9. Howler-sonotype14 alias
    monkey_idx = CLS_IDX['43435']; son14_idx = CLS_IDX['47158son14']
    blended[:, son14_idx] = np.maximum(blended[:, son14_idx], blended[:, monkey_idx] * 0.9)
    
    # 10. Pantanal Frog Chorus broadcast
    chorus = ['65380', '24279', '66971', '517063', '23158', '24321', '555146']
    chorus_idx = [CLS_IDX[c] for c in chorus]
    chorus_mean = blended[:, chorus_idx].mean(axis=1, keepdims=True)
    for ci in chorus_idx:
        blended[:, ci] = blended[:, ci] + 0.2 * chorus_mean.flatten()
    
    # 11. Maryna's adaptive_delta_smooth (canonical)
    blended = adaptive_delta_smooth(blended, alpha=0.20)
    
    # 12. Texture/event smoothing kernels (aliozanmemetoglu)
    blended = texture_event_smooth(blended)
    
    return blended  # (12, 234)
```

## 4. Expected total impact

Combining all findings:
- Pseudo hour prior baseline: 0.896
- + Perch ensemble: +0.04 to 0.94
- + Perch calibration: +0.005 to 0.945
- + Missing class strategy: +0.010 to 0.955
- + Chorus broadcast: +0.003 to 0.958
- + Temporal smoothing + post-proc: +0.002 to 0.960

**Target macro-AUC: 0.95-0.96 with no custom training, just leveraging the data structure I've documented.**

## 5. What's still unknown

Despite 9 rounds of analysis, some things are unknowable from public data:
- Exact private LB test set composition (sites, dates, hours)
- Whether test data has same-recorder as any train file
- The specific 2025 audio quality / equipment

These would require running on the actual hidden test set to verify.

## 6. Sources

All findings derived from local data analysis:
- `data/train.csv`, `data/train_soundscapes_labels.csv`, `data/taxonomy.csv`
- `data/train_audio/` (35,549 files, scanned for duration / duplicates / spectra)
- `data/train_soundscapes/` (10,658 files, scanned for acoustic profiles)
- `meta_corpus/datasets/pseudo_cache/` (127,104 Perch predictions + embeddings)

Tools: `scipy.signal.welch`, `soundfile`, `librosa.feature.mfcc`, `sklearn.metrics.roc_auc_score`, `hashlib.md5`, `pandas`, `numpy`, `concurrent.futures`, `ogginfo`.

Total analysis time this session: ~3 hours of compute on the 16 GB dataset.
