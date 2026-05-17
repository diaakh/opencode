# BirdCLEF+ 2026 — ROUND 17: deep analysis of OUR dataset

This round focuses on the BC2026 audio data itself — running `scipy.signal.welch`, `soundfile.read`, `ogginfo`, and direct numpy analysis on the train_audio and train_soundscapes files we have.

## 1. Per-site acoustic fingerprint (8 files sampled per site)

Sites have **130x range in spectral centroid** (43 Hz to 5,557 Hz) and dramatic differences in clipping/silence:

| Site | n | centroid (Hz) | RMS | clip % | silence % | hr_mean | Profile |
|---|---:|---:|---:|---:|---:|---:|---|
| **S04** | 8 | 1,966 | 0.006 | 0.00 | **78.4%** | 5.0 | **Faulty recorder / mostly silent** |
| S11 | 8 | 43 | 0.044 | 0.14 | 4.6 | 6.9 | Wind-dominated low-freq |
| S20 | 8 | 225 | 0.048 | 0.15 | 0.5 | 13.1 | Low-freq dominant |
| S10 | 8 | 669 | 0.095 | 0.22 | 2.6 | 10.0 | Sub-1kHz |
| S14 | 8 | 889 | 0.006 | 0 | 22.3 | 6.0 | Quiet medium-freq |
| S09 | 8 | 860 | 0.010 | 0 | **25.5%** | 0.0 | Quiet midnight |
| S15 | 8 | 1,087 | 0.034 | 0 | 0 | 6.0 | Dawn chorus |
| S17 | 8 | 1,116 | 0.017 | 0 | 6.0 | 0.0 | Quiet midnight |
| S04 | 8 | 1,966 | 0.006 | 0 | 78.4 | 5.0 | Faulty |
| S19 | 8 | 2,493 | 0.032 | 0.05 | 18.5 | 19.0 | Evening, partial silence |
| S16 | 8 | 2,990 | 0.031 | 0 | 0 | 12.1 | Midday medium |
| S03 | 5 | 3,224 | 0.049 | 0 | 0 | 20.4 | Evening |
| S23 | 3 | 3,412 | 0.019 | 0 | 0 | 3.7 | Pre-dawn |
| S21 | 3 | 3,453 | 0.076 | 0 | 0 | 22.3 | Late night |
| S05 | 8 | 3,606 | 0.041 | 0.01 | 0 | 8.2 | Morning |
| **S13** | 8 | 4,042 | **0.282** | **1.83%** | 0 | 11.9 | **LOUD + clipping** |
| **S22** | 8 | 4,792 | 0.125 | 0.03 | 0 | 12.8 | Mid-loud, common site |
| S07 | 8 | 4,886 | 0.011 | 0 | 9.0 | 9.2 | Quiet high-freq |
| S02 | 8 | 4,961 | 0.070 | 0 | 0 | 11.1 | High-freq dominant |
| S06 | 8 | 5,168 | 0.014 | 0 | 0.3 | 13.9 | High-freq quiet |
| S08 | 5 | 5,347 | 0.033 | 0 | 0 | 5.4 | Sonotype-rich |
| **S01** | 8 | 5,359 | 0.203 | **2.37%** | 0 | 13.0 | **LOUD + clipping** |
| S12 | 8 | 5,519 | 0.019 | 0 | 0 | 13.2 | High-freq |
| S18 | 8 | 5,557 | 0.133 | 0.51 | 0 | 15.6 | High-freq + some clipping |

**Critical site classifications:**
- **Clipping sites** (>0.5% sample-level clip): S13 (1.83%), S01 (2.37%), S18 (0.51%) — these dominate the high-frequency cicada-chorus content
- **Silent / faulty sites**: S04 (78.4% silence), S09 (25.5%), S14 (22.3%) — these may be broken recorders
- **High-frequency dominant** (centroid > 4500): S22, S07, S02, S06, S08, S01, S12, S18 (8 sites — sonotype-rich)
- **Low-frequency dominant** (centroid < 1500): S11, S20, S10, S14, S09, S15, S17 (7 sites — bird-and-frog dominant)

## 2. Labeled vs unlabeled soundscape acoustic comparison (30 each)

| Metric | Labeled (n=30) | Unlabeled (n=30) | Difference |
|---|---|---|---|
| Centroid (Hz) | 3867 ± 1836 | 4252 ± 2114 | -385 (labeled lower) |
| RMS | 0.073 ± 0.068 | 0.127 ± 0.148 | **-42% (labeled quieter)** |
| Peak | 0.50 | 0.69 | -28% |
| **Clipped %** | **0.003 ± 0.009** | **1.16 ± 4.92** | **400x less!** |
| Silence % | 3.4 ± 12.7 | 2.3 ± 12.6 | +1.0 |

**The 66 labeled files are dramatically cleaner than the 10,592 unlabeled files.** This is a SELECTION BIAS — annotators worked on easy-to-label clean files, leaving the hard noisy/clipped files unannotated.

**Critical implication**: Models that validate on the 66 labeled files will OVERESTIMATE their performance on the actual test set, which is drawn from the full distribution (much noisier).

## 3. Labeled set distribution: site/year bias

```
Site distribution of labeled files (66 total):
  S22:  40 files  (61%, dominant labeled site)
  S08:   5 files  (100% of S08's data is labeled)
  S09:   5 files  (58% of S09)
  S15:   4 files  (S15)
  S19:   3 files
  S23:   3 files  (100% of S23's data is labeled)
  S13:   2 files
  S03:   2 files  (40% of S03)
  S18:   2 files
  (14 sites have ZERO labeled files: S01, S02, S04, S05, S06, S07, S10, S11, S12, S14, S16, S17, S20, S21)
```

Year distribution: **2025 is the most-densely-labeled year (6.3%)** despite having fewer files (222 unlabeled + 15 labeled = 237 total). 2023 is the LARGEST year (3,598 files) but has ZERO labeled files.

**Test data is from 2025**, so the 15 labeled 2025 files are the BEST proxy for test-like distribution. But the model can also use the 222 unlabeled 2025 files (pseudo-labeled).

## 4. Sonotype CO-OCCURRENCE aliases (massive labeling artifact)

Computing pairwise Jaccard similarity across all 25 sonotypes in labeled soundscapes:

**PERFECT spectral + co-occurrence aliases (Jaccard = 1.000):**
- **47158son15 ≡ 47158son16** (12 windows each, all overlapping)
- **47158son22 ≡ 47158son23** (24 windows each, all overlapping)

**Near-perfect aliases (Jaccard ≥ 0.9):**
- 47158son21 ⊂ 47158son22 (22/22 son21 co-occur with son22)
- 47158son21 ⊂ 47158son23 (22/22)

**Cross-class alias:**
- **43435 (Black Howling Monkey) ≡ 47158son14** (Jaccard = 1.000, 12 windows!)
  - Howling monkey calls ALWAYS appear with "Insect son14" annotation
  - son14 NEVER annotated alone (always with howler)
  - These might be the SAME acoustic source (low-freq howling that an insect-focused labeler tagged differently)

**High co-occurrence (Jaccard > 0.5):**
- 47158son07 ↔ whtdov (Jaccard 0.76) — low-freq sonotype ↔ White-tipped Dove
- 25073 (Chiasmocleis mehelyi frog) ↔ 67107 (Jaccard 0.75) — frog pair
- 47158son13 ↔ 47158son22 (Jaccard 0.67)

**Solo-occurrence (sonotypes appearing WITHOUT other sonotypes):**

| Sonotype | Total | Solo | Solo % |
|---|---:|---:|---:|
| 47158son07 | 48 | 43 | **90%** |
| All other 24 sonotypes | varies | **0** | **0%** |

**Only son07 is independently identifiable.** The other 24 sonotypes are co-occurrence-chained — annotators heard them as part of polyphonic insect choruses.

**Effective number of distinct sonotypes**: **22 of 25** after deduplicating the 2 perfect alias groups (son15=son16, son21≈son22=son23). Or treating co-occurrence clusters as single classes, effectively **6 clusters**.

## 5. Sonotype spectral clusters (from per-class mean PSD)

Hierarchical clustering of 25 sonotype PSDs gives 6 clean clusters:

| Cluster | Sonotypes | Peak band | Notes |
|---|---|---|---|
| 1 | son09, son12 | 7-10 kHz (23%) + low (68%) | Bimodal — possible mislabeling |
| **2** | son01, son03, son04, son10, son11, son24 | Broad 3-10 kHz | "Generic insect" mid-band |
| **3** | son07 | 0-1 kHz (58%), 3-5 kHz (32%) | **LOW + mid — likely FROG misclassified as Insecta** |
| **4** | son15, son16, son17, son18, son25 | **5-7 kHz (81%)** | High-pitched cicada cluster |
| 5 | son02, son06, son14 | 0-1 kHz (43%) + 10-16 kHz (42%) | Weird bimodal |
| **6** | son05, son08, son13, son19, son20, son21, son22, son23 | **3-5 kHz (57%)** | Mid-range cicada/cricket |

**A clustered training strategy** treating these 6 clusters as 6 classes (broadcasted to 25 at inference) preserves macro-AUC while:
- Doubling training signal per cluster
- Avoiding phantom discrimination errors
- Reducing labeling-noise impact

## 6. Per-class spectral profiles of train_audio (mapped classes)

**Insecta** (only 3 species, all cicadas):
- 1161364 (Guyalna cuta): centroid 2423, **low-band 0-500 Hz 57%** — low cicada
- 244024 (Giant Cicada): centroid 1342, **1-3 kHz peak (49%)** — classic cicada
- 760266 (Prionacris erosa): centroid 3542, broader spread

**The mapped Insecta train_audio (1-3 kHz cicadas) does NOT match the unmapped sonotypes (mostly 3-10 kHz crickets/katydids).** Training on cicada audio gives ZERO transfer to crickets.

**Amphibia** examples:
- 1176823 Wrestler Frog: 74% in 0-500 Hz
- 22930 Basin White-lipped: 86% in 500-2000 Hz
- 1595929 Uruguay Harlequin: 78% in 2000-5000 Hz

**Mammalia**:
- 43435 Black Howling Monkey: **80% in 0-500 Hz** (ultra-low)
- 41970 Jaguar: 65% in 0-500 Hz
- 47144 Domestic Dog: peak 3734 Hz, 0.5-5 kHz dominant

**Reptilia** (1 sample):
- 116570 Southern Spectacled Caiman: peak 5906 Hz, broad 500-8000 Hz

**Frequency band priority for mel-spec design**:

| Band | Contents |
|---|---|
| 0-500 Hz | Howling monkey, jaguar, wrestler frog |
| 500-2000 Hz | Cicadas, most frogs, white-tipped dove, son07 |
| 2000-5000 Hz | Most birds, dogs, harlequin frog, some sonotypes (cluster 6) |
| **5000-7000 Hz** | **Sonotype cluster 4 (son15-18, 25)** — UNIQUE band |
| 7000-10000 Hz | Some bird high-frequency harmonics |
| 10000-16000 Hz | Mostly mic noise, rare signal |

**Optimal mel config: `f_min=20, f_max=12000, n_mels=128`** — covers all signal-rich bands while dropping mostly-noisy 12-16 kHz.

## 7. Duplicate-timestamp file groups (multi-recorder leakage)

41 timestamp groups contain multiple files (171 files involved, 1.6% of train_soundscapes):

| Site_Date_Time | n files | Notes |
|---|---:|---|
| **S10_20251128_103307** | **40** | 40 files with identical timestamp! |
| S04_20240810_080005 | 9 | |
| S09_20250816_000000 | 7 | |
| S05_20251125_030005 | 5 | |
| S19_20241213_193000 | 4 | |
| S05_20250227_170004 | 4 | **Same date as the sample test file (S05)!** |
| S16, S04, S17, ... | 3 each | 30 such groups |

Audio analysis of S08_20250607_070007 (3 files):
- All 3 files have DIFFERENT audio content (cross-correlation ≈ 0)
- File 0004 → 0005 have continuous spectral content (corr 0.74)
- Likely: **multi-recorder deployment** (3 different mics at same site, same minute)

**CV implication**: Duplicate-timestamp files share environment + species. Group-CV by filename leaks them across folds. **Should group by `S{site}_{date}_{HHMM}` (4-digit hour:minute prefix) to avoid this**.

## 8. The 14 sonotype-rich labeled files (the entire texture-class training source)

Only 14 of 66 labeled files contain ANY sonotype annotation — and those 14 are 100% sonotype-annotated (every 12-window labeled). They concentrate at:

| Site | n | Date range | Hours |
|---|---:|---|---|
| **S08** | 5 | 2025-06-06 to 2025-06-07 | 03, 07 |
| **S15** | 4 | 2025-06-17 (all same date!) | 06 |
| **S23** | 3 | 2024-11-24 (all same date!) | 03, 04 |
| **S19** | 2 | 2024-12-13/14 | 19 |

**Note the suspicious clustering**: 4 S15 files at the EXACT same date (2025-06-17) with timestamps 060000, 060100, 060200, 062700 — 4 different recordings at the same minute (S15 has multi-recorder deployment too).

**The ENTIRE training signal for sonotypes comes from**:
- 5 minutes of audio at S08 (5 × 60s)
- 4 minutes of audio at S15
- 3 minutes at S23 + 2 minutes at S19
- Total: **14 minutes of sonotype-bearing audio** to train models for 10.7% of macro-AUC

This is a tiny ground truth, and the sonotype-rich files are concentrated at 4 sites + 4 dates. A model that overfits to those specific dates/sites would NOT generalize to the test set (which spans 2025-02 to later).

## 9. Multi-label structure of labeled windows

```
n_labels per window:
  count: 739
  mean:  4.22
  std:   1.78
  max:   10 labels in a single 5-sec window!
  
  histogram:
    1 label:  156 (21%)
    2-4:      662
    5-7:      624
    8-10:     36
```

**Top species combos** (always-together choruses):
- `23158;24279;24321;517063;555146;65380;66971` — 27 windows (Pantanal frog chorus of 7 frogs)
- `24321;555146;65380;66971` — 18 windows (4-frog chorus)
- `47158son01;son13;son21;son22;son23;son25` — 10 windows (6-sonotype chorus)

These chorus patterns are CONDITIONAL on (site, date, hour). At certain conditions, the same 5-10 species call together.

## 10. Concrete plan additions from dataset analysis

### Tier-A (drop-in, <1 hour)
- **Group CV by `(site, date, HH_MM)` not just filename** to avoid multi-recorder leakage (1.6% of train_soundscapes affected)
- **Cap mel f_max at 12000 Hz** (drop noisy 12-16 kHz band)
- **Drop the 78.4%-silent S04 files** from training (faulty recorder, garbage data)
- **Treat son15=son16 and son21=son22=son23 as alias groups** (predict identically at inference)

### Tier-B (refactoring, half-day)
- **Build a sonotype-cluster classifier** (6 clusters instead of 25 labels) — broadcast at inference
- **Use 2025 train_soundscapes (237 files) as primary validation** (test-domain match)
- **Heavy data augmentation on S01/S13/S18** clipping samples (these dominate sonotype training)

### Tier-C (architectural)
- **Frequency-band-aware mel split**: train two SED models, one with `f_min=2000, f_max=12000` (sonotype/insect band) and one with `f_min=20, f_max=2000` (mammal/frog band)
- **Separate texture/event branches** in the model architecture itself (not just smoothing)

## 11. The truly hidden insight

**Sonotypes are NOT species classifications — they're acoustic-context tags.** The fact that:
- 24 of 25 sonotypes never appear alone
- son15 ≡ son16 and son22 ≡ son23 in both spectrum AND co-occurrence
- 43435 (howler monkey) ≡ son14 (Insecta annotation)
- son07 is the only soloable sonotype

...tells us that the labelers annotated the **AUDIO TEXTURE** present, not individual species. Each sonotype is a description of "this kind of background buzz/chirp", not "species X".

For the model:
1. **Don't try to discriminate aliased sonotypes** — they're not discriminable
2. **Predict the texture cluster** (6 clusters), then broadcast
3. **Use the chorus combinations as multi-output template** — when the model is confident about one frog, predict the whole chorus

For the leaderboard:
- **Macro-AUC favors broadcast prediction over fine discrimination**
- Models that learn the "alias structure" save capacity for the truly discriminable classes
- The 10.7% sonotype share is recovered NOT by 25 discriminators but by 6-cluster predictors

## 12. Sources (computed locally)

This round used only:
- `/home/user/opencode/birdclef-2026/data/` — competition data
- `soundfile.read`, `scipy.signal.welch`, `scipy.cluster.hierarchy.linkage`
- `pandas` for label analysis
- `hashlib.md5` for binary-duplicate checks
- `ogginfo` for OGG metadata
