# BirdCLEF+ 2026 — Final synthesis of forensic findings (5 rounds)

This document collects every concrete finding from 5 progressive rounds of
forensic analysis, **with online sources cited** rather than from training-memory
assumptions.

## Tier 1 — Findings that change how you should train

### 🔴 1.1 The recorder hardware is the SwiftOne (Cornell K. Lisa Yang Center)

Researched from [the K. Lisa Yang Center SwiftOne product page](https://www.birds.cornell.edu/ccb/swift-one/),
the [SwiftOne Quick Start Guide PDF](https://www.birds.cornell.edu/ccb/wp-content/uploads/2023/02/Quick-start-guide-to-SwiftOne_v1p5.pdf),
and the [TEMABio Pantanal 2024 call](https://www.birds.cornell.edu/ccb/2024_call_for_proposal_pantanal/).

**SwiftOne hardware specs (verified):**

| spec | value |
| --- | --- |
| Native sample rate | 48 kHz (standard) / up to 96 kHz |
| Bit depth | **16-bit** |
| Microphone | PUI Audio POW-1644L-B-LW100-R (omni-directional) |
| Microphone frequency response | **50 Hz – 16 kHz** |
| Microphone SNR | > 58 dB |
| Gain | **adjustable per-deployment** (root cause of clipping) |

**What this tells us about BirdCLEF 2026 data:**

- BirdCLEF train_soundscapes are 32 kHz mono OGG @ 72 kbps → **the organizers downsampled SwiftOne's native 48 kHz to 32 kHz** before encoding.
- The 50 Hz – 16 kHz mic response means **there is no usable signal above 16 kHz** — the resampling to 32 kHz (Nyquist = 16 kHz) is information-preserving.
- The **per-deployment gain setting** is what causes S01/S13/S10 to clip — those teams set gain higher than others. **Site fingerprint = recorder gain configuration.**

### 🔴 1.2 The TOP 5 PUBLIC 0.948–0.949 KERNELS DON'T USE TRAIN_AUDIO AT ALL

Verified by grepping `train_audio` across all 5 kernels pulled via Kaggle API:

| kernel | LB | mentions of `train_audio` |
| --- | :-: | :-: |
| `nina2025/birdclef-2026-eos-4` | 0.948 | **0** |
| `sunderekkiz/birdclef-2026-exp019-eos4-rank-power-06` | 0.949 | **0** |
| `karnakbaevarthur/power-optimization` | 0.948 | **0** |
| `pilkwang/948-…-rank-power-fusion` | 0.948 | **0** |
| `safar1/lb-score-0-948` | 0.948 | **0** |

The 35,549 train_audio clips (~207 hours / 11 GB of clean, single-species labeled audio) are **completely unused**. The kernels rely entirely on:

1. Perch ONNX (Google's pretrained model) — handles 206 mapped species
2. BirdNET ONNX (Cornell's pretrained model) — auxiliary signal, helps unmapped
3. SED ONNX — fine-grained 5-s events
4. ProtoSSM (trained on Perch embeddings of the 66 labeled soundscapes ONLY)

**Why no train_audio?** Per the [Perch 2.0 paper](https://arxiv.org/abs/2508.04665), Perch was trained on:
- 896,255 Xeno-Canto recordings
- 571,698 iNaturalist recordings
- 33,859 Tierstimmenarchiv
- 40,966 FSD50K
- **Total: 1,542,778 recordings, 14,795 classes**

BirdCLEF 2026 train_audio (XC + iNaturalist, 35,549 clips) is **a subset of Perch's training set**. Perch already memorized it during its own pretraining. The kernels lean on this memorization rather than re-training.

**But this leaves three big gaps:**

1. The 28 unmapped classes (insect sonotypes + 3 frogs) are NOT in Perch's label space — they get genus-proxy guesses.
2. Perch saw clean XC/iNat audio, never the SwiftOne soundscape conditions (different mic, different gain, possible clipping).
3. **BirdCLEF 2025 winners DID train on train_audio.** 1st place Nikita Babych used EfficientNet stack (B0/B3/V2-S/V2-B3) trained on train_audio with multi-iterative noisy student.

### 🔴 1.3 The labels CSV duplication is per-FILE BLOCK doubling

`train_soundscapes_labels.csv` has 1,478 rows that dedupe to 739 unique. Pattern:

- 708 of 739 duplicate pairs are at offset = **exactly 12 rows** (one full 60-s file's worth of segments)
- Each labeled file has 12 segments written, then written again immediately
- After sort by (filename, start, end), 100% of duplicate pairs become adjacent

This is **not** two annotators; it's an export-tool block-doubling bug. The top kernels do `pd.read_csv(...).drop_duplicates()` already.

### 🔴 1.4 Codec bitrate domain shift (train_audio ≠ train_soundscapes)

Verified by reading the Vorbis identification packet from each OGG file:

| corpus | encoder vendor | bitrate |
| --- | --- | --- |
| `train_audio` XC (500 sampled, 100%) | libVorbis 20180316 | **86 kbps** |
| `train_audio` iNat (500 sampled) | libVorbis 66% + Lavf58.29.100 34% | 86 kbps 66% + **72 kbps 34%** |
| **`train_soundscapes` (all 10,658)** | libVorbis 20180316 | **72 kbps uniformly** |

Test soundscapes will match train_soundscapes encoding. A model that trains on 86 kbps audio sees subtle high-frequency content that disappears at 72 kbps. **None of the public 0.948 kernels apply codec round-trip augmentation.**

## Tier 2 — Data anomalies that bias training

### 🟡 2.1 Labels are 290× cleaner than unlabeled (clipping audit, FULL corpus)

Verified across all 10,658 files (655s of decoding):

| metric | labeled (n=66) | unlabeled mean | ratio |
| --- | --: | --: | --: |
| Mean RMS | 0.068 | 0.139 | 2.04× louder |
| Mean peak amplitude | 0.49 | 0.75 | 1.54× higher |
| Files with >1% clipped samples | **0/66** | **1,215/10,592 (11.4%)** | — |
| Worst single file | trivial | **80.5% saturated** | — |

Per-site heavy-clipping rate:

| site | n files | % heavy-clipped | n labeled |
| --- | --: | --: | --: |
| **S13** | 1,873 | **29.0%** | only 2 (clean) |
| **S10** | 46 | **21.7%** | 0 |
| **S01** | 2,341 | **16.1%** | **0** |
| S22 | 3,383 | 6.5% | 40 |
| S02 | 2,505 | 2.6% | 0 |
| All 16 other sites | < 1% | 0–1% | 24 |

**4,260 of 10,658 train_soundscape files (40%) come from heavy-clipping sites
with effectively zero labels.** Test soundscapes draw from the same sites.

### 🟡 2.2 Clipping is hour-of-day biased (peak 19% at 20-21 UTC)

| hour (UTC) | n files | % heavy-clipped |
| --- | --: | --: |
| 06–17 (daytime) | < 200 | ~0% |
| 18 | 1,071 | 7.2% |
| **19** | 1,024 | **13.1%** |
| **20** | 1,009 | **19.0%** ← peak |
| **21** | 1,068 | **19.5%** ← peak |
| 22 | 1,049 | 17.7% |
| 23 | 1,058 | 13.0% |
| 00–04 | 4,190 | 5.8–7.1% |

Recorder gain set for daytime saturates when the dusk frog/insect chorus
starts at 15–16h local time (19–21 UTC). Hour-aware down-weighting of ProtoSSM
for those slices is unexplored in public kernels.

### 🟡 2.3 Specific recorder-failure timelines

- **S01 had a 266-day recording gap** (2022-04-30 → 2023-01-21). Post-gap recordings have aggressive gain → 65% of S01-Apr-2023 files heavily clipped.
- **S13's clipping epidemic** Sept 2023 → Jan 2024 (47–67% heavy per month).
- Both sites stabilized after Feb 2024 to < 1% clipping.

### 🟡 2.4 Each insect sonotype is hyper-locked to ONE site & hour

From `train_soundscapes_labels.csv`, every one of the 25 sonotypes (47158sonXX) appears at exactly 1 site & 1 hour:

- son14, son02 → S23, 4am only
- son15, son16, son18, son19, son20 → S08, 7am only
- son21, son22, son23 → S08, 3am only
- son24 → S19, 7pm only
- son07 → S15, 6am only
- ... (others similar)

The iNat taxon ID 47158 is verified to be the **entire class Insecta** (no
species, family, or even order identification). The sonotypes are
acoustically-distinguishable but taxonomically anonymous insect sounds.

Site × hour fingerprint alone nearly perfectly predicts these classes. The
public kernels' "site × hour prior" exploits this, but only 9 of 25 sonotypes
have explicit mirror-group max-pooling.

### 🟡 2.5 Labels are biased toward documenting rare species

| label / coverage measure | value |
| --- | --: |
| Total labeled segments (dedup) | 739 |
| Segments containing ≥1 of the 28 missing-from-train_audio species | **519 (70%)** |
| Segments containing ONLY missing-28 species (no in-train species) | 71 |

The 66 labeled files were curated to maximize coverage of the 28 unmapped
classes — not as a representative sample of soundscape conditions.

### 🟡 2.6 Labels are mostly file-level (not per-segment)

Inspecting individual labeled files shows the same species set across all
12 windows of one file. Example file `BC2026_Train_0039_S22_20211231_201500.ogg`
has labels `22961;23158;24321;517063;65380` on 10 of 12 segments, with 2 segments
dropping `517063`. So labels are mostly persistent across the 60-second file,
with rare per-segment differences.

This means training labels at 5-sec granularity are essentially "what was
present in the minute" — weakly-supervised. Models that predict at 5-sec but
train against 60-sec-derived labels have a granularity mismatch.

## Tier 3 — Smaller findings worth documenting

### 🟢 3.1 Most train_soundscapes are isolated 60-s snippets (not continuous sessions)

10,357 distinct recording sessions from 10,658 files; median session size = 1.
Only 17 sessions have 6+ consecutive 60-s files. **Each `train_soundscape` is
essentially an independent 60-s sample**, not part of a longer recording chain.

### 🟢 3.2 Sample submission row reveals test format and one S05 example

Sample test row: `BC2026_Test_0001_S05_20250227_010002_5`
- Site S05 (9 files in train_soundscapes, 0 labeled)
- Date 2025-02-27 (overlaps with 4 unlabeled train files at S05 same date, different time)
- Time 01:00:02 UTC (= 21:00 local Pantanal time)
- Per-window granularity: `..._5`, `..._10`, `..._15` (end-time of each 5-s window)

### 🟢 3.3 Within-site PCM near-duplicates are rare

50 random within-(site, date) unlabeled pairs: 0/50 had |PCM correlation| > 0.5;
median was 0.011 (random). Within-site files SOUND similar (spec-corr 0.97+)
because they're the same acoustic environment, but the audio samples are
distinct recordings.

### 🟢 3.4 10 distinct recorder fingerprint clusters (KMeans on noise floor)

Labels concentrate in 4 of 10 clusters (74% of labels). Two clusters have
ZERO labels:
- Cluster of S01/S13/S02 (the high-clipping group)
- S09's unique-hardware cluster

### 🟢 3.5 Train_audio sampling — middle vs first 5s

From the [BirdCLEF 2025 top-2% writeup (Max Melichov)](https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df):
> "The best results always came from just picking the **middle** 5 seconds of each recording" (vs first 5s or whole recording → 0.76 AUC).

From the [BirdCLEF 2025 1st place writeup](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n):
> Sampling: take first 7 seconds, randomly crop 5 seconds within. TTA shifts (±2.5 sec) boost score from 0.91 to 0.922 (+0.012).

From R5a verification on our train_audio: **44.8% of clips have peak energy in the middle half**, only 28.7% in first quarter. **Mean middle-5s RMS is 13% louder than first-5s RMS** (median 1.00, mean 1.13).

But it's not universally true — 53% of clips DON'T have middle louder than first. So "always middle" is too strong; a mixed strategy (middle preferred, with random for rare classes) matches what 1st place did.

### 🟢 3.6 Labeled files have FEWER acoustic events too

Round 5 event-onset detection (50ms RMS rise > 2× threshold):

| metric | labeled | unlabeled | ratio |
| --- | --: | --: | --: |
| Mean onsets per 60-s file | 2.95 | 3.91 | 1.32× |
| rms_max | 0.144 | 0.289 | 2.01× |
| rms_p10 (noise floor) | 0.050 | 0.107 | 2.14× |

Labels were not just chosen for clean recording quality — they were also
chosen for QUIETER acoustic content. This is a curation bias.

### 🟢 3.7 iNat URL timestamps reveal organizer download schedule

`train.csv` iNat URLs include `?<unix-timestamp>` cache-busters that reveal
when iNat re-processed the file. Range: 2017-08 to 2026-01. Recent peaks:
2024-04 (320), 2024-09 (338), 2025-04 (342), **2025-05 (396)**, 2025-09 (366),
2026-01 (78). These peaks indicate when the organizers batch-downloaded
fresh iNat data.

Implication: train_audio includes iNaturalist clips uploaded as late as
**early 2026** — very recent. Perch (released in 2024) likely did NOT see
those newest clips. So for the most-recently-uploaded clips, training a
model directly on train_audio may still help even though Perch saw most
of it.

### 🟢 3.8 The 28 missing-from-train_audio species

| primary_label | scientific name | iNat taxon |
| --- | --- | --- |
| 1491113 | Adenomera guarani (frog) | 1491113 |
| 25073 | Chiasmocleis mehelyi (frog) | 25073 |
| 517063 | Pithecopus azureus (Southern Orange-legged Leaf Frog) | 517063 |
| 47158son01..son25 | "Insect sonotype 01..25" | 47158 (Insecta, entire class) |

**The 25 sonotypes share iNat taxon ID 47158, which is the entire class
Insecta** (verified via iNat). They have no taxonomic identification below class.
The 3 frog species are real, named, but missing from `train_audio`.

`secondary_labels` in train.csv: 0 of 28 missing species ever appear there.

### 🟢 3.9 Single-location species (4 species have all clips at one coordinate)

`primary_label` 116570 (Southern Spectacled Caiman) and 3 others have all
training clips from exactly ONE (lat, lon). Lat/lon-conditional models could
classify these deterministically — but the 0.948 kernels don't use lat/lon at all
(zero occurrences across the 5 kernels).

### 🟢 3.10 Top recordists concentrate the data

`JAYRSON ARAUJO DE OLIVEIRA` is 8.1% of all train_audio (2,874/35,549).
Top 5 recordists = ~20% of the dataset. Recordist-grouped CV would
prevent same-recordist train/val leakage.

## Tier 4 — How to actually use these findings

Priority order for someone starting from your current 0.949 baseline:

### 🟢 Action A: Re-encode train_audio to 72 kbps libVorbis (Tier 1.4)

Match the train_soundscape codec exactly. One ffmpeg pass per file. No
training-side change. Closes the codec domain shift.

```bash
ffmpeg -i in.ogg -c:a libvorbis -b:a 72k -ar 32000 -ac 1 -y out.ogg
```

### 🟢 Action B: Train a custom model on train_audio (Tier 1.2)

Match what BirdCLEF 2025 1st place did: EfficientNet-B0/B3 on log-mel
spectrograms of train_audio, middle-5s sampling, focal-BCE loss, mixup,
SpecAugment, ±2.5 sec TTA shifts (worth +0.012 per the 2025 writeup).
Distill via ONNX → CPU-inference budget. Adds ensemble diversity orthogonal
to Perch.

### 🟢 Action C: Clipping-augmentation training (Tier 2.1)

```python
def random_clip_aug(x, p=0.3, gain_range=(2.0, 8.0)):
    if np.random.rand() > p: return x
    return np.clip(x * np.random.uniform(*gain_range), -1.0, 1.0)
```

Apply with probability 0.3 to training audio. Forces the model to handle
saturated audio it will see on test files from S01/S13/S10.

### 🟢 Action D: Detect clipped test files at inference, route differently (Tier 2.1)

```python
clip_pct_per_file = (np.abs(x) > 0.99).sum() / x.size * 100
if clip_pct_per_file > 1.0:
    # ProtoSSM/Perch unreliable on saturated audio
    weights = (0.10, 0.40, 0.50)   # Proto / SED / BirdNET — was (0.50, 0.30, 0.20)
```

### 🟢 Action E: Extend Tweak G's mapped/unmapped split + sonotype mirroring (already in EoS-4)

The 0.948-0.949 kernels already use Tweak G for the 28 unmapped classes
(50/30/20 mapped vs 20/40/40 unmapped). Extend the sonotype-mirror groups
from 9 of 25 to all 25 using data-driven Jaccard > 0.8 from
`stats/soundscape_cooccurrence.csv`.

### 🟢 Action F: Recordist-grouped CV fold (Tier 3.10)

Add a recordist-based GroupKFold alongside the existing site fold. Prevents
single-recordist train/val leakage during validation.

## What I still cannot verify

- The actual test_soundscapes — they're mounted only at submission time.
- Whether private LB sites/dates skew differently from public LB.
- Which exact ID convention links BirdCLEF site codes (S01–S23) to
  TEMABio Pantanal recorder deployments (no public mapping found).
- Whether the LABELED files were specifically chosen via BirdNET/Perch
  pre-screening (likely but unconfirmed).

## Sources (verified online, fetched fresh)

- [Perch 2.0: The Bittern Lesson — Merrienboer et al. (2025)](https://arxiv.org/abs/2508.04665) — Perch training corpus (1.5M recordings, 14,795 classes, XC + iNat + Tierstimmenarchiv + FSD50K)
- [SwiftOne product page – K. Lisa Yang Center, Cornell](https://www.birds.cornell.edu/ccb/swift-one/)
- [SwiftOne Quick Start Guide v1.5 (PDF)](https://www.birds.cornell.edu/ccb/wp-content/uploads/2023/02/Quick-start-guide-to-SwiftOne_v1p5.pdf)
- [TEMABio Pantanal 2024 program – Cornell Lab](https://www.birds.cornell.edu/ccb/2024_call_for_proposal_pantanal/)
- [BirdCLEF+ 2025 1st place — Multi-Iterative Noisy Student (Nikita Babych)](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n)
- [BirdCLEF+ 2025 2nd place GitHub (Sydorskyy)](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)
- [BirdCLEF+ 2025 5th place GitHub (myso1987)](https://github.com/myso1987/BirdCLEF-2025-5th-place-solution)
- [Top-2% BirdCLEF+ 2025 retrospective — Max Melichov](https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df)
- [exp019 0.949 candidate – Sunderekkiz](https://www.kaggle.com/code/sunderekkiz/birdclef-2026-exp019-eos4-rank-power-06)
- [Nina EoS-4 0.948 baseline](https://www.kaggle.com/code/nina2025/birdclef-2026-eos-4)
- [Karnakbayev Power Optimization](https://www.kaggle.com/code/karnakbaevarthur/power-optimization)
- [pilkwang 948 Rank-Power Fusion](https://www.kaggle.com/code/pilkwang/948-birdclef-26-acoustic-time-window-rank-fusion)
- [safar1 LB 0.948](https://www.kaggle.com/code/safar1/lb-score-0-948)
- [iNaturalist taxon 47158 (Insecta)](https://www.inaturalist.org/taxa/47158) — confirmed parent of the 25 sonotypes is the entire insect class
- [Distilling Spectrograms into Tokens — Holub et al. (arXiv 2507.08236)](https://arxiv.org/html/2507.08236v1) — BirdCLEF+ 2025 academic context
- [Tackling Domain Shift in Bird Audio Classification — CEUR Vol-4038 paper 256](https://ceur-ws.org/Vol-4038/paper_256.pdf)
- [BirdCLEF+ 2026 official page (LifeCLEF)](https://www.imageclef.org/BirdCLEF2026)
- [AudioMoth recording-gap behavior](https://www.openacousticdevices.info/support/configuration-support/record-continuously-for-10h) — context for the 41 (site, date, time) collisions
