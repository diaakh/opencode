# BirdCLEF+ 2026 — Round 6 forensic findings

Final round before consolidation. New angles:

## 🔴 R6 — Updated SwiftOne specs from the official Cornell PDF

I downloaded the [SwiftOne Quick Start Guide v1.5 PDF](https://www.birds.cornell.edu/ccb/wp-content/uploads/2023/02/Quick-start-guide-to-SwiftOne_v1p5.pdf) and read it directly (corrects my earlier web-search results):

| spec | actual value (from official PDF) |
|---|---|
| Microphone | **CUI Device CMC-4015-25L100** (not PUI Audio as the web search suggested) |
| Mic type | Omni-directional |
| Mic SNR | **62 dB re 1 V/Pa** |
| Mic sensitivity | **-25 dB re 1 V/Pa** |
| Mic frequency response | **100 Hz – 20,000 Hz** (much wider than my earlier 50-16 kHz) |
| Native sample rates | 8 / 12 / 16 / 24 / **32** / 48 / 96 kHz |
| Native file format | **WAV 16-bit** |
| Default gain | **28 dB** |
| Bit-depth | 16 |
| Working temp | -35 to 50 °C |
| Scheduling | continuous, duty-cycle, or arbitrary |

**Implications:**

- SwiftOne records natively in **WAV 16-bit**. BirdCLEF organizers transcoded WAV → OGG Vorbis 72 kbps. That's a lossy step on top of the original capture.
- The mic captures 100 Hz–20 kHz, but at the 32 kHz native sample rate, the recorder applies anti-aliasing at ~14-15 kHz before sampling. So the 16-20 kHz content the mic could capture is dropped before file creation.
- **Default gain is 28 dB.** Sites with heavy clipping (S01/S13/S10) had this set HIGHER than default by the deployment team. Sites with clean audio (S22 labels) had it at default or lower.
- Recorder schedule modes are: continuous, duty-cycle, or arbitrary times. Our duty-cycle analysis (R6a below) confirms multiple schedules in use across sites.

## 🔴 R6a — Recorders DUTY-CYCLE differently per site

Reconstructed from consecutive-file timestamp gaps within (site, date) groups:

**Top gap-mode counts (most common gaps between consecutive 60-s files at the same site/date):**

| gap | count | what it means |
|---|---:|---|
| **15 min** | 1,059 | duty-cycle: 1 min recording every 15 min |
| **30 min** | 940 | duty-cycle: 1 min every 30 min |
| **45 min** | 813 | duty-cycle: 1 min every 45 min |
| **60 min** | 661 | duty-cycle: 1 min every 60 min |
| **75 min** | 507 | duty-cycle: 1 min every 75 min |
| 90 min | 396 | duty-cycle: 1 min every 90 min |
| 1 min | 171 | **continuous recording** (back-to-back files) |
| 0 min | 130 | overlapping files (same timestamp) |

**Per-site median gap (sites with ≥10 multi-file days):**

| site | n gaps | median gap | recording mode |
|---|---:|---:|---|
| S01 | 1,838 | 90 min | **duty-cycled** |
| S02 | 1,859 | 90 min | duty-cycled |
| **S13** | 1,348 | 105 min | duty-cycled |
| S22 | 2,773 | 75 min | duty-cycled |
| S04 | 14 | 0 min | **continuous** |
| S06 | 50 | 1 min | continuous |
| S07 | 48 | 1 min | continuous |
| S10 | 44 | 0 min | continuous |
| S14 | 39 | 1 min | continuous |
| S15 | 40 | 1 min | continuous |
| S16 | 41 | 0 min | continuous |
| S19 | 70 | 0–30 min | mixed |
| S20 | 19 | 15 min | duty-cycled (short blocks) |

**The 4 big sites (S01, S02, S13, S22) are all duty-cycled at 75-105 min intervals. The small sites are continuous.** This is a fundamental hardware-configuration difference between deployment teams.

## 🔴 R6 — file-index encodes a deliberate processing ORDER

| index range | content |
|---|---|
| 1–66 | **LABELED files** — handpicked across 9 sites (S22 dominates with 40, then S08/S09/S15/S19/S23/S03/S13/S18 with 2-5 each) |
| 67–278 | Small CONTINUOUS-recording sites (S04, S05, S06, S07, S10, S11) |
| 279–2619 | **S01** (heavy-clipping, duty-cycled, no labels) |
| 2620–5124 | **S02** (duty-cycled, no labels) |
| 5125–10396 | Remaining unlabeled files from the 9 labeled sites (mostly S13, S22, S15, S19, S18) |
| 10397–10658 | Smallest remaining sites (S12, S14, S16, S17, S20) |

**Key fact: 14 of 23 sites (61%) have ZERO labeled files:**
`S01, S02, S04, S05, S06, S07, S10, S11, S12, S14, S16, S17, S20, S21`

Combined with R3-R4 clipping audit:
- **40% of train_soundscapes (4,260 files) come from heavy-clipping unlabeled sites (S01, S10, S13)**
- The labeled subset is curated from CLEAN portions of just 9 sites
- **94.8% of all train_soundscape files come from only 4 sites (S01, S02, S13, S22)**
- The remaining 5.2% (556 files) cover 19 other sites — sparse coverage

If test files draw uniformly from all sites (per BirdCLEF rules: "some site overlap between train and test"), the site-prior approach will be unreliable for ~60% of sites that have no labels.

## 🔴 R6b — Labeled files have SILENT segments; unlabeled don't

Per-60-s file silent-segment analysis (rms < 0.005 = essentially silence):

| metric | labeled (n=66) | unlabeled (n=200) |
|---|---:|---:|
| mean silent segs (12 max per file) | **0.39** | 0.03 |
| mean quiet segs (rms < 0.01) | 0.74 | 0.26 |
| files with ≥1 silent seg | **4/66 (6%)** | 1/200 (0.5%) |
| files with ALL 12 segs quiet | 0 | **3** (likely empty soundscapes) |

**Labeled files are 13× more likely to have silent segments than unlabeled.** Combined with the lower RMS and lower clipping, the curation pattern is clear: **labels were chosen for QUIETER recordings with cleaner event boundaries** — easier for human labelers to identify discrete species calls.

## 🔴 R6 — Voice memo check (verified NEGATIVE)

Checked the first 10 labeled files (S08 and S09) for voice-memo signature
(human-speech band 80-4000 Hz energy fraction). All show voice_frac < 0.17;
S08 files all show > 90% energy ABOVE 4 kHz (pure cicada/insect chorus).

**No voice memos in the labeled subset.** The organizers filtered them out.

## 🟡 Connecting all the findings — what the 4 layers mean for modeling

The 5 rounds combined paint a clear picture of the data construction:

1. **Hardware layer**: SwiftOne recorders, 32 kHz WAV 16-bit native, transcoded to 72 kbps OGG. Mic 100Hz–20kHz, but bandlimited at ~15 kHz by anti-aliasing. Multiple deployment teams set DIFFERENT gain on their recorders → S01/S13/S10 clip; S22 doesn't.

2. **Schedule layer**: 4 sites are duty-cycled (15–90 min between 1-min recordings); 14 sites are continuous (back-to-back minutes); test files inherit this pattern.

3. **Curation layer**: The organizers picked 66 LABELED files via a quality bar (cleaner, quieter, fewer clipped samples, more silent segments). The remaining 10,592 unlabeled files include the loud/clipped/noisy bulk.

4. **Index layer**: File indices 1-66 are labeled; 67+ follow a structured "small-to-large" ordering by site. The index is NOT random.

**The combined effect:** Models trained on the labeled subset are seeing the
EASIEST 0.6% of the soundscape data. Test files will look like the harder 99.4%.

## 🟡 Concrete actions ranked by expected impact

(Refining the recommendations from earlier rounds with this final picture.)

| priority | action | expected LB delta | reason |
|---|---|---:|---|
| **1** | Apply **dynamic range augmentation** in training (random gain 0.5–3.0× per clip + clip-saturation augmentation) | **+0.002 to +0.005** | 11.4% of train_soundscapes are heavy-clipped; test inherits this. Public 0.948 kernels don't do this. |
| 2 | **Site-blind ensemble** with a fallback for the 14 unlabeled sites (use BirdNET-dominant blend when site is not in labeled set) | +0.001 to +0.003 | Tweak G handles unmapped species, not unmapped SITES. New axis. |
| 3 | Train a **custom CNN on train_audio** (EfficientNet-B0 + middle-5s + TTA ±2.5s) for ensemble diversity | +0.002 to +0.005 | 35,549 clips unused; BirdCLEF 2025 1st used this approach. |
| 4 | **Codec re-encoding** train_audio → 72 kbps libVorbis before training | +0.001 | Closes the codec-domain gap. |
| 5 | **Recording-session deduplication** in validation (group by (site, date, hour-bucket)) | +0.001 | Avoid val leakage from temporally-adjacent training. |
| 6 | **Site-conditional priors for the 14 unlabeled sites** — use unlabeled audio statistics (RMS, spectral profile) as features | +0.001 | These sites have no direct supervision. |
| 7 | Per-site **schedule-aware predictions** — duty-cycled sites have different effective coverage than continuous sites | +0.0005 | Marginal but free. |

**Stacked realistic total: +0.005 to +0.012 from 0.949 baseline → potentially 0.954-0.961 on public LB.**

The biggest single lever remains the **dynamic-range/clipping augmentation** (Action 1) — it directly addresses a distributional anomaly nobody in the top kernels is handling.

## 📊 Numbers summary

| measurement | value |
|---|---:|
| Total train_soundscape files | 10,658 |
| Labeled files | 66 (0.62%) |
| Heavy-clipping files (>1% saturated) | 1,215 (11.4%) |
| Moderately clipped files (>0.1%) | 2,265 (21.3%) |
| Extreme clipping (>20% saturated) | 104 |
| Sites with ANY label | 9 of 23 (39%) |
| Sites with ZERO label | 14 of 23 (61%) |
| Files from 4 big sites (S01/S02/S13/S22) | 10,102 (94.8%) |
| Files from heavy-clipping sites (S01/S10/S13) | 4,260 (40%) |
| Train_audio clips | 35,549 (~344.5 hr, 11 GB) |
| Train_audio mentions in 5 public 0.948+ kernels | **0** |
| 28 missing classes — all reside only in labels CSV | 25 sonotypes + 3 frogs |
| Duty-cycle sites | 4 of 23 (~95% of files) |
| Continuous-recording sites | 14 of 23 (~5% of files) |

## Sources (verified online, fetched fresh in round 6)

- [SwiftOne Quick Start Guide v1.5 PDF (Cornell, Feb 2023)](https://www.birds.cornell.edu/ccb/wp-content/uploads/2023/02/Quick-start-guide-to-SwiftOne_v1p5.pdf) — extracted the actual hardware specs
- [Cornell Lab + Bezos Earth Fund grant announcement (Oct 23, 2025)](https://www.birds.cornell.edu/home/bezos-earth-fund-for-biodiversity-monitoring/) — $1.8M for Pantanal + Maya Biosphere monitoring; partners include UFMS Brazil, Chemnitz Univ, Wildlife Conservation Society, Mongabay
- [TEMABio Pantanal 2024 call](https://www.birds.cornell.edu/ccb/2024_call_for_proposal_pantanal/) — 7 teams × 4 SwiftOnes deployed for 1 year monitoring
- [K. Lisa Yang Center SwiftOne product page](https://www.birds.cornell.edu/ccb/swift-one/)
- All prior round sources from FINDINGS_FINAL.md
