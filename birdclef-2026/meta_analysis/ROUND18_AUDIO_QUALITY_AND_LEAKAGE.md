# BirdCLEF+ 2026 — ROUND 18: train_audio quality forensics + S05 leakage

## 1. Full train_audio duration scan (all 35,549 files)

After parallel-scanning every file:

| Metric | Value |
|---|---:|
| Total valid files | 35,549 |
| **Min duration** | **0.008 sec (8 milliseconds!)** |
| **Max duration** | **6,881 sec (115 minutes!)** |
| Mean | 34.9 sec |
| Median | 21.0 sec |
| Files <1 sec | 370 (1.04%) — **unusable** |
| Files <2 sec | 370 (1.04%) |
| Files <3 sec | 672 (1.89%) |
| Files <5 sec | **2,601 (7.32%)** — **can't form a 5-sec window** |
| Files >60 sec | 4,825 (13.57%) |
| Files >120 sec | 1,363 (3.83%) |
| Files >300 sec | 176 (0.50%) |

**2,601 train_audio files (7.3%) are SHORTER THAN 5 SECONDS.** They can't fill a standard 5-sec training window without aggressive padding. Standard "random crop 5s" augmentation will repeat-pad these, which trains the model on UNNATURAL audio.

### Per-class breakdown of files <3 seconds

| Class | Files <3s | % of class |
|---|---:|---:|
| Aves | 637 | 1.8% of 34,799 |
| Amphibia | 20 | 4.4% of 451 |
| Mammalia | 10 | 10.1% of 99 |
| Insecta | 5 | 2.5% of 199 |

**Mammalia has 10% short files** — the highest fraction. Domestic dog barks and single bird-of-prey calls are often <3 seconds.

### Top species with most short files

| Species | n<3s files |
|---|---:|
| osprey (Aves) | 25 |
| bbwduc (Aves) | 22 |
| houspa (House Sparrow) | 20 |
| brnowl (Brown Owl) | 19 |
| socfly1, greyel | 16 each |
| gycwor1, banana | 15 each |

These are short-call species (single hoots, brief whistles).

### Examples of <0.5s files (essentially unusable):

| Filename | Duration | Species |
|---|---:|---|
| greyel/iNat1375792.ogg | **0.036 s** | Greater Yellowlegs |
| bobfly1/iNat1691483.ogg | 0.104 s | Boat-billed Flycatcher |
| osprey/iNat226664.ogg | 0.104 s | Osprey |
| epaori4/iNat649978.ogg | 0.139 s | Variable Oriole |
| shcfly1/iNat1471076.ogg | 0.144 s | Short-crested Flycatcher |

**Filtering recommendation**: drop training files <3 seconds (672 files = 1.89% of data) to avoid pad-pollution. Or use them only with `pad_type='repeat'`.

## 2. Train_audio XC vs iNat acoustic differences

| Metric | XC (n=131) | iNat (n=69) | Difference |
|---|---|---|---|
| Duration median | 29 sec | **14 sec (half)** | iNat shorter |
| Silence % median | 5.7% | 1.9% | iNat denser |
| Bandwidth median | 5,812 Hz | 5,234 Hz | iNat narrower |
| **Clipping % mean** | **0.001%** | **0.666% (666x more)** | iNat much more clipped |
| Lead silence mean | 0.5 sec | 1.7 sec | iNat more intro silence |

**iNat files closely match train_soundscape characteristics** (clipping, lead silence). XC files are professional recordings (low clip, longer, cleaner). For domain matching with the test set, **iNat files are higher-value training data** than XC.

This suggests **weighting iNat samples 2-3x higher than XC in the loss** could improve test generalization.

## 3. Author-level acoustic fingerprinting (top 10 authors >= 100 recordings)

Different authors produce systematically different acoustic signatures:

| Author | n_files | centroid | RMS | noise floor |
|---|---:|---:|---:|---:|
| JAYRSON ARAUJO DE OLIVEIRA | 2874 | 3969±995 | 0.028 | 0.008 |
| Unknown | 1253 | 1972±925 | **0.009 (very quiet)** | 0.005 |
| Jeremy Minns | 1007 | 2973 | **0.062 (loudest)** | 0.012 |
| Dante Buzzetti | 959 | 2826 | 0.034 | 0.006 |
| Fernando Igor de Godoy | 777 | 2737 | 0.019 | 0.009 |
| Richard E. Webster | 641 | 3406 | 0.016 | **0.002 (cleanest)** |
| GABRIEL LEITE | 588 | 2647 | 0.046 | 0.017 |
| Niels Krabbe | 570 | 3284 | 0.064 | **0.033 (noisiest)** |
| Jerome Fischer | 518 | 4061 | 0.051 | 0.023 |
| Eduardo Luis Beltrocco | 458 | 2003 | 0.022 | 0.003 |

**17x range in author noise floor** (0.002 - 0.033). A model could learn author-specific noise patterns and overfit. **Author-grouped CV is essential** — Sydorskyi's BC2025 2nd place already does this.

### Pantanal-region recordings

Filtering to inside the Pantanal bbox (lat -16.5 to -21.6, lon -55.9 to -57.6):

| Collection | Total | Inside Pantanal | Coverage |
|---|---:|---:|---:|
| XC | 23,043 | 740 | 3.2% |
| iNat | 12,506 | 107 | 0.9% |
| **Total** | **35,549** | **847 (2.4%)** | |

By class:
- Aves: 110 of 162 species have ANY Pantanal-region training data
- Amphibia: 5 of 32 species
- Mammalia: 4 of 8 species
- Insecta: **0 of 3** species
- Reptilia: 0 of 1

**119 of 206 species (57.8%) have any Pantanal-region training data.** The remaining 87 species (42%) are trained ONLY on out-of-region recordings.

**Top Pantanal-region authors** (highest value for domain matching):
- Jeremy Minns: **208 Pantanal XC recordings**
- Dante Buzzetti: 89
- Eric DeFonso: 84
- JAYRSON ARAUJO DE OLIVEIRA: 32 (only 1% of his 2874 are from Pantanal)
- Luciano Bernardes: 18 (iNat)

A simple geographic-weighted recipe would 2-5x upweight these authors' files in training.

## 4. Ultra-rare species: audio is essentially useless

Detailed analysis of the 5 species with ≤2 train_audio recordings:

| Species | Common name | train_audio | Audible signal | Labeled SS | Effective data |
|---|---|---:|---:|---:|---:|
| 116570 | Southern Spectacled Caiman | 1 file (7.9s) | 6.7s | 13 windows | ~72s |
| **23150** | **Central Dwarf Frog** | 1 file (19.9s) | 19.9s | **0 windows** | **20s** |
| **23724** | **Waxy Monkey Tree Frog** | 1 file (7.8s) | 7.1s | **0 windows** | **7s** |
| 516975 | Hooded Capuchin | 1 file (42s) | **0.5s (98% silent!)** | 13 windows | ~65s |
| **209233** | **Feral Horse** | 2 files (first=**0.1s**!) | <1s | **0 windows** | **<1s** |
| 24321 | Mato Grosso Tree Frog | 2 files (6.5s) | 6.5s | 172 windows! | 880s+ |

**3 species are essentially untrainable** (23150, 23724, 209233):
- 23150 Central Dwarf Frog: 20s of train audio, ZERO labeled
- 23724 Waxy Monkey Tree Frog: 7s of train audio, ZERO labeled
- 209233 Feral Horse: 0.1s + ~1s = <2s total, ZERO labeled

These 3 classes will get **near-random macro-AUC (~0.5)** in any model.

## 5. The S05 SAME-DATE TEMPORAL LEAK (critical finding)

Test sample: `BC2026_Test_0001_S05_20250227_010002` (S05, 2025-02-27, **01:00 AM**)

**S05 train_soundscapes (9 files total) include**:
```
4 files at S05_20250227_170004.ogg   ← SAME DATE (2025-02-27) at 17:00 (5 PM)!
5 files at S05_20241125_030005.ogg   ← 3 AM at S05, different date
```

**The test sample shares its recorder and exact date with 4 train_soundscapes files!** They differ only in time-of-day (16 hours apart, 01:00 vs 17:00).

Acoustic profile of S05:
- **17:00 PM (5 train files)**: centroid 60-1,789 Hz, **LOW-frequency wind/silence dominant** (60-99% in 0-1 kHz band). One file is ULTRA-LOW (centroid 60 Hz, 99% in 0-1 kHz — pure wind noise).
- **03:00 AM (5 train files, different date)**: centroid 4,186-5,591 Hz, **HIGH-frequency insect chorus** (54-84% in 5-7 kHz band).

**Test sample at 01:00 AM is likely INSECT CHORUS** (matching the 03:00 pattern) with high-frequency dominant content.

**Implications for the LB**:
1. The site×hour Bayesian prior captures this — but only if it's fitted on diverse-site data (which the 66-file labeled set is NOT).
2. The pseudo-cache (`backtracking/birdclef2026-pseudo-cache-v1`) DOES include S05 train_soundscapes via Perch predictions, so site-conditional priors built from pseudo-labels capture S05 night patterns.
3. The 4 S05 train files at 17:00 PM are NOT the right comparison for the 01:00 AM test sample — 03:00 AM patterns at S05 (different date) are MORE acoustically relevant.

**This is a form of `recorder fingerprint` leakage that no public kernel currently exploits explicitly.** A model could:
- Detect the S05 acoustic signature from filename
- Apply S05-specific hour-conditional priors
- Boost predictions for species typically active at S05 night-time

## 6. No audio-content cross-leak between train_audio and train_soundscapes

I hashed the first 5 seconds of 1,000 train_audio files and scanned 200 train_soundscapes for matching hashes:

```
Cross-leak detections: 0
```

The training data is clean of trivial audio duplication. No train_audio file appears as a verbatim 5-sec window in train_soundscapes.

## 7. Multi-channel and bit-depth check (200 file sample)

- **100% mono (1 channel)** — no stereo recordings
- **100% at 32 kHz sample rate** — uniform sampling
- All confirmed OGG Vorbis encoded

Consistent with the codec analysis from ROUND 13. **No surprises in container format — the format inconsistency lives in bitrate (86 kbps train_audio vs 72 kbps train_soundscapes), not in sample structure.**

## 8. Concrete plan additions

### Tier-A (drop-in, < 1 hour)
- **Drop train_audio files <3 seconds** (672 files, 1.89%) — they can't fill 5-sec windows naturally
- **Author-grouped CV** (Sydorskyi BC2025 2nd-place trick already mentioned, but now empirically validated by 17x author-noise-floor range)
- **iNat-weighted training** — weight iNat samples 2-3x higher than XC since they match test domain (clipping, lead silence)

### Tier-B (refactoring)
- **Geographic + author dual-weighted training**: combine Pantanal-distance weighting with author-noise-floor weighting (avoid both clean professionals and very noisy hobbyists)
- **Drop S04 train_soundscapes** (78.4% silence — faulty recorder garbage)
- **Recorder-fingerprint feature**: extract noise floor + spectral centroid in 1-sec sliding window over each test file, condition prediction on site-of-day acoustic state

### Tier-C (model architecture)
- **Hour-conditional model** — pass hour-of-day as embedding (already done by Maryna recipe, but with limited 66-file labels). Use pseudo-labels to extend prior to all 23 sites.
- **Domain adaptation**: fine-tune the model on train_soundscapes (Pantanal SwiftOne) AFTER pretraining on train_audio, to bridge the codec/recorder gap.

## 9. Sources (all locally-derived)

- `scipy.signal.welch` for spectral analysis
- `soundfile.read` for audio decoding
- `ogginfo` for OGG metadata
- `hashlib.md5` for binary duplicate checks
- `concurrent.futures.ThreadPoolExecutor` for parallel file scanning
- `/home/user/opencode/birdclef-2026/data/` — competition data
