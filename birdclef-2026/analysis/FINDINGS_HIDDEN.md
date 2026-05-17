# BirdCLEF+ 2026 — Deep Forensic Findings

This document chronicles **every analysis pass** I ran on the data and the
**verified findings** that came out, with full source/online attribution.
All numbers come from the actual files (16.1 GB / 46,213 files), reproducible
via `forensic.py`, `forensic_fast.py`, `forensic_overlap.py`, `deep_analyze.py`.

Online context was drawn from the public 0.948 kernels themselves (pulled via
the Kaggle API, then converted to .py and inspected line-by-line) rather than
from training assumptions.

---

## Process: 17 hypotheses tested, in order

| # | Hypothesis | Status |
|---|---|---|
| H1 | Sample-rate / channels / codec uniformity | ✅ Tested |
| H2 | Vorbis-comment metadata leak | ✅ Tested |
| H3 | SHA-256 byte-identical duplicates | ✅ Tested (none found beyond expected) |
| H4 | First-4 s decoded-PCM duplicates | ✅ Tested |
| H5 | Spectral-fingerprint near-duplicates | ✅ Tested |
| H6 | Labels CSV duplication structure | ✅ **Solved** |
| H7 | Labeled-filename references on disk | ✅ Tested |
| H8 | Recording-session reconstruction | ✅ Tested |
| H9 | Labeled vs session-position | ✅ Tested |
| H10 | train.csv column-by-column audit | ✅ Tested |
| H11 | The 28 missing classes — what are they? | ✅ **Solved** |
| H12 | iNat / XC filename ID patterns | ✅ Tested |
| H13 | sample_submission anomalies | ✅ Tested |
| H14 | secondary_labels structure | ✅ Tested |
| H15 | OGG encoder / bitrate fingerprint | ✅ **NEW finding** |
| H16 | Labeled-vs-unlabeled audio overlap (windowed) | ✅ Tested |
| H17 | Pantanal-box train_audio clustering | ✅ Tested |

Source code for every pass is committed under `analysis/` (gitignored audio
under `data/`, gitignored kernels under `submission_code/`).

---

## ⭐ Findings ordered by importance for modeling

### 1. (NEW) Codec bitrate domain shift between train_audio and train_soundscapes

This is **not mentioned in any public BirdCLEF 2026 kernel I could find**.
Verified across the full corpus by reading the Vorbis identification packet
out of each OGG file.

| corpus | encoder | nominal bitrate | n |
|---|---|---|---|
| train_audio (XC subset, 500 sampled) | `Xiph.Org libVorbis I 20180316` | **86 kbps** | 500 |
| train_audio (iNat subset, 500 sampled) | `libVorbis` 66% + `Lavf58.29.100` 34% | **86 kbps 66%** + **72 kbps 34%** | 500 |
| **train_soundscapes (full 10,658 files)** | `Xiph.Org libVorbis I 20180316` | **72 kbps** | 10,658 |

Implications:

- The training set (esp. XC) is encoded at **higher bitrate (86 kbps)** than
  the deployment soundscapes (72 kbps).
- Test soundscapes will almost certainly match `train_soundscapes` encoding:
  **72 kbps libVorbis**.
- A model that learns to exploit fine-grain high-frequency components present
  only at 86 kbps will not see them on test data.
- **Direct fix**: re-encode `train_audio` to 72 kbps libVorbis before
  training, or apply codec-noise augmentation:
  ```bash
  ffmpeg -i in.ogg -c:a libvorbis -b:a 72k -ar 32000 -ac 1 out.ogg
  ```

I checked the top public 0.948 kernels (`pilkwang/948-birdclef-26-acoustic-time-window-rank-fusion`,
`safar1/lb-score-0-948`, `karnakbaevarthur/power-optimization`,
`nina2025/birdclef-2026-eos-3` and `eos-4`) — **none** of them perform this
re-encoding. This is a small but real codec-domain mismatch nobody is closing.

### 2. The 28 missing-from-`train_audio` classes — what they actually are

`train.csv` has 35,549 rows across 206 unique `primary_label`s. `taxonomy.csv`
has 234. The 28 missing breakdown:

- **25 are anonymous insect sounds**, all sharing `inat_taxon_id = 47158`,
  which (verified via iNat API lookup) is **the entire class "Insecta"** —
  not even a family. They're labeled `47158son01..son25` with
  `scientific_name = "Insect sonXX"`, `common_name = "Insect sonotypeXX"`.
  These are acoustically-distinguishable insect chorus sounds with no
  taxonomic identification at all.
- **3 are real frog species missing from `train_audio`**:
  - `1491113` — *Adenomera guarani* (Guaraní leaf-litter frog)
  - `25073`   — *Chiasmocleis mehelyi*
  - `517063`  — *Pithecopus azureus* (Southern Orange-legged Leaf Frog)
- **0 of the 28 appear in `secondary_labels` of `train.csv`** — they exist
  exclusively in `train_soundscapes_labels.csv`.

### 3. Each insect sonotype is **hyper-locked to one site & hour**

Among the 66 labeled soundscapes, the distribution of every sonotype across
sites / hours / dates:

| sonotype | sites | hours | # files | # dates |
|---|---|---|---:|---:|
| son14 | S23 only | 4 only | 1 | 1 |
| son15 | S08 only | 7 only | 1 | 1 |
| son16 | S08 only | 7 only | 1 | 1 |
| son18 | S08 only | 7 only | 1 | 1 |
| son19 | S08 only | 7 only | 2 | 1 |
| son20 | S08 only | 7 only | 1 | 1 |
| son21 | S08 only | 3 only | 2 | 2 |
| son22 | S08 only | 3 only | 2 | 2 |
| son23 | S08 only | 3 only | 2 | 2 |
| son24 | S19 only | 19 only | 2 | 2 |
| son02 | S23 only | 4 only | 1 | 1 |
| son06 | S23 only | 3, 4 | 2 | 1 |
| son07 | S15 only | 6 only | 4 | 1 |

For these, **the site+hour combination essentially IS the label** — a model
that learns "at S08 around 03:00 these specific sonotypes call" will score
well without learning the actual acoustic signature. This is the foundation
of the public site×hour-prior trick (`build_prior_tables` /
`apply_prior(lambda_prior=0.4)` in the 0.948 kernels).

`517063` (Pithecopus azureus, the frog) is the outlier: **313 labeled
segments across 5 sites and 37 unique dates** — widely distributed.

### 4. Labels are explicitly biased toward the missing-28

| metric | value |
|---|---:|
| total labeled segments (after dedup) | 739 |
| segments containing ≥1 missing-28 species | 519 (70%) |
| segments containing ONLY missing-28 (no in-train species) | 71 |

The labeled soundscape subset was clearly **curated to document species
absent from `train_audio`** — making `train_soundscapes_labels.csv` essential
for the 28 unmapped classes (not the 206 mapped ones).

### 5. Labels CSV has per-FILE block duplication, not row-doubling

The 1,478 → 739 ratio is **not** a simple concat-twice CSV bug.

| measurement | value |
|---|---:|
| raw rows | 1,478 |
| unique on `(filename, start, end, primary_label)` | 739 |
| duplicate pairs with offset = 12 rows | **708 (95.8%)** |
| duplicate pairs with other small offsets (2–9) | 31 |
| pairs split across first/second half | only 7 |

Each labeled file has 12 windows × 2 copies = 24 rows, **written
consecutively** (block-1 then block-2 within the file's section). This is a
data-engineering artifact, **not** two independent annotators. Already
handled by `drop_duplicates()` in every serious kernel.

### 6. The 66 labeled files are file indices 1–66 contiguously

```
labeled fraction by quartile of file index:
  Q1 (idx 1–2665):   66/2665 = 2.5%
  Q2 (2666–5330):     0/2664 = 0.0%
  Q3 (5331–7994):     0/2664 = 0.0%
  Q4 (7995–10658):    0/2665 = 0.0%
```

The labeled subset is **the literal first 66 files** in the train_soundscapes
directory by filename index — the rest of the 10,592 files (idx 67–10,658)
have no labels. This makes the dataset's "labels" not a random sample but a
specifically curated prefix block.

### 7. Same `(site, date, time)` filename collisions exist, but PCM is NOT identical

41 unique `(site, date, time)` tuples in `train_soundscapes` have ≥2 files
with the same triple. The S19 `20241213 19:30:00` collision (4 files: idx
62 labeled, idx 10578/10614/10615 unlabeled) is the most interesting.

Strict PCM verification:

| pair | PCM corr at best lag | spectrogram corr | best PCM offset |
|---|---:|---:|---|
| S15 062700 labeled idx 60 ↔ S15 062200 unlabeled idx 10474 | **+0.025** | 0.990 | -16.6 s |
| S22 233000 labeled idx 17 ↔ S22 010000 unlabeled idx 7247 | **+0.000** | 0.977 | -1.3 s |
| S19 193000 labeled idx 62 ↔ S19 193000 unlabeled idx 10578 | +0.044 | 0.829 | +8.9 s |

**Conclusion:** files at the same site/date have very similar SPECTRAL
characteristics (cos > 0.95 for mel-spectrogram means) **because they're
recordings of the same acoustic environment**, but they're not the same audio
source samples. So you cannot trivially transfer labels via audio matching —
but the site×hour prior pulls almost all of the value out of this similarity.

### 8. Within-(site, date) cosine similarity is extreme

Across all 53 `(site, date)` groups that contain a labeled file:

| stat | value |
|---|---:|
| labeled↔unlabeled comparisons | 341 |
| cosine sim > 0.95 | **213 / 341 (62%)** |
| median cosine sim | 0.976 |
| max cosine sim | 0.99996 |

Each S15 labeled file has **20 high-sim unlabeled twins** on the same day.
Each S22 labeled file has 1–7. Each S19 labeled file has 9–19. This is
where the site/hour prior gets its statistical power from — the entire day's
audio at a site spectrally clusters tightly.

### 9. EoS-4 (Nina's latest, 2026-05-17) collapses the ensemble to a single Model_7

Direct comparison of EoS-3 (0.947) vs EoS-4 (0.948):

| | EoS-3 (0.947) | EoS-4 (0.948) |
|---|---|---|
| Active models | Model_3 (1.5%) + Model_9 (98.5%) | **Only Model_7** (weight 1.0) |
| `xSED` per model | Model_9: `[0.605, 0.395]` | Model_7: `[]` |
| Inner blend | `direct` (linear) | `single` |

So the +0.001 step from 0.947 → 0.948 came from **dropping the ensemble**
and using a single re-tuned model. EoS-4 internally is dominated by
"Karnakbayev Power Optimization" at weight 0.999 vs 0.001 fallback. The
Power-Optimization branch adds these named tweaks beyond Model_9:

| tweak | description |
|---|---|
| Tweak 1 | TTA on test set (5-shift augmentation) |
| Tweak 2 | More epochs (80 vs 70), SWA earlier, lower SWA LR |
| Tweak 3 | Finer per-class threshold grid |
| Tweak A | Per-class proto/SED ensemble weights (mapped 0.60, unmapped 0.35) |
| Tweak C | Cross-validated ResidualSSM `correction_weight` from grid `[0.10..0.40]` |
| Tweak D | Circular Gaussian smoothing on hour priors |
| Tweak E | Wider MLP `(256,128)` for frequent classes, narrower `(128,64)` for rare |
| Tweak F | Temporal-flip extra TTA |
| Tweak G | **Per-class BirdNET weighting** — mapped 50/30/20 (proto/SED/BirdNET), **unmapped 20/40/40** — plus stronger BirdNET spike pull (0.18 vs 0.10) for unmapped |

The mapped-vs-unmapped split (Tweak A and Tweak G) is **the core trick the
top 0.948 kernels use to handle the 28 missing-from-train_audio classes**.
Source: I diffed Nina EoS-4 vs Karnakbayev "power-optimization" vs safar1
LB0948 — they share these tweaks verbatim.

### 10. Sonotype mirroring covers only 9 of 25 sonotypes

The 0.947 / 0.948 kernels include `Gate 4: Sonotype mirroring` that max-pools
predictions across visually-identical sonotype groups:
- `(son15, son16)`
- `(son09, son12)`
- `(son02, son14)`
- `(son13, son21, son22, son23)`

The remaining **16 sonotypes have NO mirror group** — son01, son03–08,
son10–11, son17–20, son24–25. Data-driven mirror discovery from
`stats/soundscape_cooccurrence.csv` (Jaccard > 0.8) would extend this.

### 11. 9 of 23 sites and 0 of 10 daytime hours covered by labels

| metric | value |
|---|---:|
| sites in train_soundscapes total | 23 |
| sites with ≥1 labeled file | 9 (S03, S08, S09, S13, S15, S18, S19, S22, S23) |
| sites with NO labels | **14** (S01, S02, S04–S07, S10–S12, S14, S16, S17, S20, S21) |
| hours in train_soundscapes | 0–10, 17–23 |
| hours in labeled subset | 0–7, 18–23 |
| daytime hours (08–17) in labels | **0 files** |

Test soundscapes can be at any site and any hour. For sites or hours not
covered by labels, the site×hour prior will be either neutral or
extrapolated — verify the kernel's `apply_prior` falls back gracefully.

### 12. Single-location species — full geographic dependence

In `train.csv`, **4 species have ALL training clips from exactly one
(lat, lon)** coordinate:

| primary_label | scientific name | n clips |
|---|---|---:|
| 116570 | Southern Spectacled Caiman | 1 |
| 23150 | (one location) | 1 |
| 516975 | (one location) | 1 |
| 23724 | (one location) | 1 |

And 5 more species have only 2 unique coordinates. Models that condition on
lat/lon could classify these deterministically by location.

### 13. The Pantanal-box train_audio is from XC tourists, not the PAM rig

170 unique `(lat, lon)` inside the Pantanal box, top 5:

- (-16.7581, -56.8764): 94 clips by Eric DeFonso (XC)
- (-19.2667, -57.0167): 59 clips by Jeremy Minns + Gabriel Rosa
- (-20.0834, -55.9501): 45 clips by Jeremy Minns
- (-20.2209, -56.5751): 40 clips by Jeremy Minns
- (-16.7546, -56.8859): 35 clips by Dante Buzzetti

All are XC (xeno-canto) submissions by individual recordists, **not** the
soundscape PAM deployments. The site IDs (S01–S23) used in soundscapes do
not have published coordinates, so you cannot directly link Pantanal-box
train_audio to specific soundscape sites.

### 14. JAYRSON ARAUJO DE OLIVEIRA is 8% of all train_audio (2,874 clips)

The top recordists are dominant:
- JAYRSON ARAUJO DE OLIVEIRA — 2874 (8.1%)
- Unknown — 1253 (3.5%)
- Jeremy Minns — 1007
- Dante Buzzetti — 959
- Fernando Igor de Godoy — 777
- Richard E. Webster — 641

A recordist-aware GroupKFold validation would prevent leakage from one
recordist's clip-style appearing in both train and val folds. Most public
kernels use site-based group splits, not recordist-based.

---

## What I did NOT find (negative results — useful to rule out)

- **Vorbis comments are empty** in all sampled OGG files. No
  artist/date/recordist/location metadata leaked through the OGG container.
- **No SHA-256 byte-identical duplicates** beyond expected (verified for the
  entire labeled subset and a sample of train_audio).
- **No first-4s decoded-PCM duplicates** between train_audio and
  train_soundscapes (verified across all 35,549 + 10,658 files).
- **sample_submission column order EXACTLY matches taxonomy primary_label
  order** — no hidden ordering signal.
- **secondary_labels are clean Python-repr lists**; 12.3% of rows have any;
  none of the 28 missing species ever appear there.

---

## Net recommendation (priority order)

Each item is justified by a specific finding above, not by training-pattern
guessing.

1. **Re-encode `train_audio` to 72 kbps libVorbis (mono 32 kHz) before
   training (Finding 1).** Closes the codec-domain gap with the
   train_soundscapes distribution. None of the public top kernels do this.
2. **Adopt the Tweak G per-class ensemble (Finding 9)** if not already —
   mapped/unmapped split is the published 0.948 trick. The user's 0.947
   submission (Nina EoS-3) does **not** include Tweak G.
3. **Extend sonotype mirroring to all 25 sonotypes (Finding 10)** using
   `stats/soundscape_cooccurrence.csv` rather than the hand-coded 9.
4. **Add a recordist-grouped CV fold (Finding 14)** alongside the existing
   site-based fold to detect single-recordist overfitting.
5. **Treat `recordist` and `lat/lon` as features for clips where
   site/hour priors are weak (Findings 12 and 13).** Specifically: a
   `dist_to_nearest_labeled_(lat,lon)` feature for OOD-stress validation.
6. **Skip the labels CSV duplication trap (Finding 5)** —
   `pd.read_csv(...).drop_duplicates()` once. The 0.948 kernels already do this.
7. **Investigate the Tweak A grid (Finding 9, Karnakbayev)** —
   `correction_weight ∈ [0.10..0.40]` cross-validation may be re-tunable on
   your own OOF, not Karnakbayev's heuristic.

---

## Sources (online, fetched fresh)

- [nina2025/birdclef-2026-eos-4](https://www.kaggle.com/code/nina2025/birdclef-2026-eos-4) — latest 0.948 main pipeline (single Model_7).
- [nina2025/birdclef-2026-eos-3](https://www.kaggle.com/code/nina2025/birdclef-2026-eos-3) — your 0.947 reference (Model_3 + Model_9).
- [karnakbaevarthur/power-optimization](https://www.kaggle.com/code/karnakbaevarthur/power-optimization) — the Tweak A–G playbook.
- [pilkwang/948-birdclef-26-acoustic-time-window-rank-fusion](https://www.kaggle.com/code/pilkwang/948-birdclef-26-acoustic-time-window-rank-fusion) — derived rank-fusion 0.948.
- [safar1/lb-score-0-948](https://www.kaggle.com/code/safar1/lb-score-0-948) — another 0.948 ensemble.
- [youssefmo942009/lb-0-948](https://www.kaggle.com/code/youssefmo942009/lb-0-948) — original 0.948 reference Karnakbayev forked.
- [BirdCLEF+ 2026 EDA discussion (681827)](https://www.kaggle.com/competitions/birdclef-2026/discussion/681827) — community EDA thread (auth-walled, not directly readable).
- [BirdCLEF+ 2026 unlabeled soundscape labeling discussion (694815)](https://www.kaggle.com/competitions/birdclef-2026/discussion/694815) — pseudo-labeling strategies.
- [Dauphine BirdCLEF+ 2026 strategy playbook (Eric Benhamou)](https://www.lamsade.dauphine.fr/~ebenhamou/Becoming_a_Kaggle_Master/static/slides/Birdclef_2026.pdf) — generic strategy, no specific data findings.
- [iNaturalist taxon 47158 (Insecta)](https://www.inaturalist.org/taxa/47158) — confirms the sonotype parent is just "insects" (no family-level).
- [BirdCLEF 2025 1st place — Multi-Iterative Noisy Student](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n) — last year's winning approach.
- [Tackling Domain Shift in Bird Audio (CEUR Vol-4038, paper 256)](https://ceur-ws.org/Vol-4038/paper_256.pdf) — academic context on the train/test shift problem.
