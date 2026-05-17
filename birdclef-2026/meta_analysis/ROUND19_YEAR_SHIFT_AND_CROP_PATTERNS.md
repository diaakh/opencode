# BirdCLEF+ 2026 — ROUND 19: year-level acoustic shift + smart-crop patterns

## 1. Year-level acoustic shift in train_soundscapes (2014→2025)

Per-year acoustic profile (10 files per year, full PSD + RMS analysis):

| Year | files | centroid | RMS | clip% | silence% | peak |
|---|---:|---:|---:|---:|---:|---:|
| 2014 | 106 | 5,648 | 0.012 | 0.00% | 0.2% | 0.08 |
| 2021 | 1,646 | 3,776 | 0.063 | 0.00% | 0.0% | 0.44 |
| 2022 | 3,146 | 4,265 | 0.077 | 0.12% | 0.0% | 0.52 |
| 2023 | 3,598 | 4,002 | **0.235** | **2.88%** | 0.0% | 1.02 |
| 2024 | 1,925 | 5,261 | **0.205** | **1.76%** | 0.0% | 1.07 |
| **2025** | **237** | **1,056** | **0.039** | **0.06%** | **14.7%** | 0.42 |

**Year 2025 is a different acoustic regime entirely:**
- Centroid 1,056 Hz vs 4,000-5,648 Hz in other years (4-5x lower)
- 14.7% silence vs 0% in 2021-2024
- 0.06% clipping vs 1.76-2.88% in 2023-2024

**Hour distribution by year** (where 2025 differs dramatically):

| Year | Top hours | Pattern |
|---|---|---|
| 2014 | 21:00, 03:00, 00:00, 22:00, 02:00 | Mostly nighttime |
| 2021 | 18:00, 01:00, 02:00, 21:00, 03:00 | Nighttime |
| 2022 | 01:00, 18:00, 19:00, 23:00, 02:00 | Nighttime |
| 2023 | 01:00, 23:00, 18:00, 22:00, 21:00 | Nighttime |
| 2024 | 01:00, 20:00, 19:00, 22:00, 03:00 | Nighttime |
| **2025** | **06:00, 10:00, 00:00, 17:00, 07:00** | **MORNING + early day (different!)** |

**2025 train_soundscapes are recorded during DAYTIME** (6 AM, 10 AM peaks). The 2021-2024 data was nighttime.

## 2. 2025 NIGHT vs DAY vs 2023 NIGHT (the critical comparison)

Sample test file is from 2025-02-27 01:00 AM. The most acoustically-relevant training subsets:

| Subset | n | centroid | RMS | clip% | silence% |
|---|---:|---:|---:|---:|---:|
| **2025 NIGHT (hr 0-4)** | 39 files | 2,483 | **0.026** | **0.02%** | 3.7% |
| 2025 DAY (hr 6-11) | 157 files | 1,082 | 0.036 | 0.03% | 12.1% |
| 2023 NIGHT (hr 0-4) | many | 3,958 | **0.274** | **6.47%** | 0.0% |

**The test data (1 AM, 2025) should look like 2025 NIGHT**: quieter (RMS 0.026), low clipping (0.02%), some silence (3.7%). 

**2023 nighttime data is THE WRONG ACOUSTIC PROFILE** — it's 10x louder and 300x more clipped than 2025. **A model trained primarily on 2023 night data will see test data as ALIEN.** 

**Critical insight**: The site×hour Bayesian prior — when fitted on 2021-2024 data — is fitted on a DIFFERENT acoustic regime than 2025 test. The prior may even be MISLEADING if it assumes "loud chorus at 01:00 AM" but test is quiet.

**Recommended**: weight 2025 train_soundscapes 5-10x higher than 2014-2024 in any prior fitting or pseudo-label training. The 237 files in 2025 are the highest-domain-match.

## 3. The S05 same-date temporal leak (confirmed)

Test sample: `BC2026_Test_0001_S05_20250227_010002` (S05, **2025-02-27 01:00 AM**)

```
S05 train_soundscapes file list:
  BC2026_Train_0084_S05_20241125_030005.ogg     ← 3 AM, Nov 2024
  BC2026_Train_0085_S05_20241125_030005.ogg     ← 3 AM, Nov 2024
  BC2026_Train_0086_S05_20241125_030005.ogg     ← 3 AM, Nov 2024
  BC2026_Train_0087_S05_20241125_030005.ogg     ← 3 AM, Nov 2024
  BC2026_Train_0088_S05_20241125_030005.ogg     ← 3 AM, Nov 2024
  BC2026_Train_0089_S05_20250227_170004.ogg     ← 5 PM, SAME DATE AS TEST
  BC2026_Train_0090_S05_20250227_170004.ogg     ← 5 PM, SAME DATE AS TEST
  BC2026_Train_0091_S05_20250227_170004.ogg     ← 5 PM, SAME DATE AS TEST
  BC2026_Train_0092_S05_20250227_170004.ogg     ← 5 PM, SAME DATE AS TEST
```

**S05 acoustic profile by hour** (from train data):

| Hour | Centroid | Energy band | Notes |
|---|---:|---|---|
| 03:00 (Nov 2024) | 4186-5591 Hz | 5-7 kHz (54-84%) | **HIGH-FREQ insect chorus** |
| 17:00 (Feb 2025) | 60-1789 Hz | 0-1 kHz (60-99%) | **LOW-FREQ wind/silence** |
| **01:00 (Feb 2025, TEST)** | unknown | likely 5-7 kHz | Should be insect chorus |

The test sample's expected acoustic profile (insect chorus) is most similar to S05 at 03:00 AM (different year), NOT to S05 at 17:00 PM (same date). **Hour-prior is more important than date-prior for S05.**

## 4. Signal onset position in train_audio (random crop is suboptimal)

195 files analyzed:
- **Median signal onset: 3.25 sec** into the file
- **39.5% of files have signal onset >5 sec** (39.5% of files would have a 5-sec random crop hit silence)
- 24% have onset >10 sec
- Max: 123 seconds of leading silence

**The "random 5-sec crop from anywhere" augmentation is silently wasting 39.5% of training data** on near-silent windows.

### Energy concentration by position (5-sec windows)

| Position | XC mean | iNat mean |
|---|---:|---:|
| First 5 sec | 28.3% | **45.7%** |
| Middle 5 sec | (low) | (very low) |
| Last 5 sec | 27.0% | **48.6%** |

**iNat: 94.3% of energy is in the first 5 + last 5 seconds combined.** The middle of iNat recordings is mostly silent. This is the "user opens app → records 14 seconds with bird at start, closes" pattern.

**Implication**: 
- For XC: random crop is OK (relatively uniform energy)
- For iNat: **smart crop from first 6 OR last 6 seconds** (jfpuget BC2024 trick confirmed empirically)
- 52.4% of files have a TARGET-DOMINANT 5-sec window at one of the edges

```python
def smart_crop_5sec(y, sr=32000):
    n = len(y)
    if n < sr * 6:
        return np.pad(y, (0, sr*5 - n))  # too short, pad
    first_5_e = (y[:sr*5]**2).sum()
    last_5_e = (y[-sr*5:]**2).sum()
    if first_5_e > last_5_e:
        start = int(np.random.uniform(0, sr*1))  # crop 0-1 sec offset
    else:
        start = int(n - sr*5 - np.random.uniform(0, sr*1))
    return y[start:start + sr*5]
```

## 5. Per-class energy timeline (100-bin normalized)

Average normalized energy timeline across files for each class:

| Class | Peak position | First 20% | Middle 60% | Last 20% | Pattern |
|---|---|---:|---:|---:|---|
| Aves | 25% | 0.31 | 0.29 | 0.28 | **Slight early peak (XC focal style)** |
| **Amphibia** | **94%** | **0.41** | **0.43** | **0.47** | **INCREASING toward end (frog chorus builds)** |
| Mammalia | 68% | 0.33 | 0.39 | 0.38 | Late-middle peak |
| Insecta | 74% | 0.36 | 0.40 | 0.38 | Late-middle peak (cicada drone) |

**Frogs (Amphibia) call MORE LOUDLY AT THE END of recordings.** A recordist starts when they hear the first call, then more frogs join the chorus → end is loudest.

**Per-class smart crop**:
- Aves: crop from first 0-25% (early-peak)
- **Amphibia: ALWAYS crop from LAST 20%** (end-peak)
- Mammalia: crop from 50-90% range
- Insecta: crop from 50-90% (avoid the silent transitions at edges)

## 6. Rating analysis (12,849 of 35,549 = 36.1% have rating=0.0)

| Rating | n files | centroid | RMS | clip% | silence% |
|---|---:|---:|---:|---:|---:|
| 0.0 | 12,849 | 1,711 | 0.057 | **0.522%** | 24.0% |
| 0.5 | 22 | 3,303 | 0.050 | 0.015% | 25.5% |
| 1.0 | 147 | 1,953 | 0.054 | 0.013% | 29.3% |
| 1.5 | 120 | 3,205 | 0.043 | 0.003% | 28.1% |
| 2.0 | 598 | 1,965 | 0.056 | 0.001% | 12.8% |
| 3.0 | 2,738 | 2,247 | 0.040 | 0.004% | 21.5% |
| 4.0 | 8,018 | 2,849 | 0.029 | 0.000% | 28.4% |
| 5.0 | 6,845 | 2,735 | 0.035 | 0.000% | 28.9% |

**Rating=0.0 files are 500x more clipped than rating 4.0+.** 36.1% of training data has rating=0.0 (likely "unrated" rather than "rated 0"). The clipping pattern suggests these are unrated iNat files with mobile-phone-level quality.

**train_soundscapes for reference**: centroid 3946, RMS 0.156, clip 2.66%, silence 0.0%

**Cosmetically, train_soundscapes most resembles unrated train_audio (rating=0.0)** — both have ~0.5-2.7% clipping. The high-rating XC files are CLEANER than test domain.

**Practical: weight low-rated files HIGHER in training** for domain matching. Or use rating as a feature.

## 7. Secondary labels analysis

| Stat | Value |
|---|---:|
| Files with secondary labels | 4,372 (12.3%) |
| Unique secondary species | 161 |
| All secondaries are in primary label set | ✓ (no OOV) |
| Top file: bkcdon/XC703631.ogg | **15 secondary species!** |

**Per-class secondary frequency**:
- Aves: 21% mean (high)
- Amphibia: 5%
- Insecta: 0% (cicada recordings have no other species labeled)
- Mammalia: 1%

**Use secondary labels at weight 0.3-0.5** (Tonylica BC2026 uses 0.5). Treating secondaries as separate weak supervision adds training signal.

## 8. Concrete plan additions

### Tier-A (drop-in, < 1 hour)
- **Filter to 2025 train_soundscapes for prior fitting** (237 files = most test-aligned)
- **Smart crop iNat files from first 1 sec OR last-1 sec** (jfpuget BC2024 pattern), not random
- **Crop Amphibia files specifically from LAST 20%** (chorus peak)
- **Drop rating=0.0 with extreme clipping** (>3% sample-level clip)

### Tier-B (custom training)
- **Year-weighted training**: 2025 files weight 1.0, 2024 weight 0.7, 2023 weight 0.5, 2014-2022 weight 0.3
- **Per-class smart crop policy**: Aves random/early, Amphibia end, Mammalia/Insecta middle
- **Use secondary labels as soft targets at weight 0.5**

### Tier-C (model)
- **Hour-conditional model** that uses *only 2025-style data* for the hour-prior at night (the 2021-2024 night data is OOD)
- **Recorder-generation embedding** (1 = 2021-2024 loud, 2 = 2025 quiet)

## 9. Sources

All derived from local data analysis:
- `/home/user/opencode/birdclef-2026/data/`
- `scipy.signal.welch`, `soundfile`, `pandas`, `concurrent.futures`
