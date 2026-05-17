# BirdCLEF+ 2026 — ROUND 20: train_audio label noise + chorus groups

## 1. Audio-content duplicates in train_audio (LABEL NOISE)

Computed via thread-pooled `sf.info()` to extract (frames, samplerate, filesize) for all 35,549 files, then deduplicated:

- **98 duplicate-tuple groups** (same audio metadata)
- **200 files (0.56%) are duplicates** of another file
- **MD5-of-decoded-audio confirmed 79 of 98 are true binary duplicates** (others differ slightly due to Vorbis encoder noise)
- **34 cross-species duplicate groups → 68 files with WRONG label supervision**

### Sample cross-species duplicates

| File 1 | Label 1 | File 2 | Label 2 | Duration | iNat IDs |
|---|---|---|---|---:|---|
| fepowl/iNat791264 | Ferruginous Pygmy Owl | giwrai1/iNat791265 | Giant Wood Rail | 37.78s | **791264, 791265** |
| grasal3/iNat693450 | Grayish Saltator | gretho2/iNat693449 | Great Thorntail | 46.49s | **693449, 693450** |
| giwrai1/iNat693433 | Wood Rail | sobcac1/iNat693429 | Solitary Cacique | 54.17s | **693429, 693433** |
| tattin1/iNat714747 | Tataupa Tinamou | whtdov/iNat714748 | White-tipped Dove | 42.42s | **714747, 714748** |
| plcjay1/iNat1734624 | Plush-crested Jay | purjay1/iNat1734625 | Purplish Jay | 11.05s | **1734624, 1734625** |
| 47144/iNat1346365 | Domestic Dog | strcuc1/iNat1670795 | Striped Cuckoo | 0.14s | (no consecutive) |

**14 of 34 cross-species duplicates have CONSECUTIVE iNat IDs** (within 1-4 of each other) — strong evidence these are **iNaturalist observations with multiple sound attachments**. The observer recorded once but tagged the observation with multiple species; iNat split them into separate files with consecutive IDs.

### Species most affected by cross-species duplicates

| Species | files affected |
|---|---:|
| gretho2 (Great Thorntail) | 3 |
| sobcac1 (Solitary Cacique) | 3 |
| strcuc1 (Striped Cuckoo) | 3 |
| soulap1 | 3 |
| grasal3, grekis, gycwor1, sofspi1, saffin, compau, fepowl, giwrai1 | 2 each |

### Same-species duplicate groups (within a single species)

64 groups have multiple files with same species (just plain duplicates within one folder):
- chobla1: 3 identical files (iNat1116775, iNat1119727, iNat1124462)
- pluibi1: 3 identical (iNat520491, iNat520495, iNat520496)
- sobtyr1: 3 identical (iNat532177, iNat532178, iNat532174)

These are less harmful (only inflate sample count by 1-2) but still bias the per-species data distribution.

### Action items

```python
# Detect duplicates and mark for special handling
duplicates_df = pd.read_csv('/home/user/opencode/birdclef-2026/meta_analysis/duplicate_train_audio.csv')

# For training:
# 1. DROP cross-species duplicates (or treat as multi-label)
cross_species = duplicates_df[duplicates_df['is_cross_species']]
# 2. DEDUPLICATE same-species groups (keep 1 file per group)
```

**File saved**: `/home/user/opencode/birdclef-2026/meta_analysis/duplicate_train_audio.csv`

## 2. Pantanal acoustic chorus groups (transitive Jaccard ≥ 0.4)

Hierarchical clustering of labeled-soundscape co-occurrence:

### Group 1: THE PANTANAL FROG CHORUS (7 species, J >= 0.4)
- 65380 — Dwarf Tree Frog
- 24279 — Lesser Snouted Tree Frog
- 66971 — Paraguayan Swimming Frog
- 517063 — Southern Orange-legged Leaf Frog
- 23158 — Pale-legged Weeping Frog
- 24321 — Mato Grosso Snouted Tree Frog
- 555146 — Chaco Tree Frog

When ONE of these 7 frogs is detected, the others are very likely co-present. **A label-propagation rule** can boost predictions:

```python
PANTANAL_FROG_CHORUS = {'65380', '24279', '66971', '517063', '23158', '24321', '555146'}

# After prediction:
chorus_score = mean([preds[s] for s in PANTANAL_FROG_CHORUS])
for s in PANTANAL_FROG_CHORUS:
    preds[s] = (preds[s] + 0.3 * chorus_score)  # boost by chorus signal
```

### Group 2: Daytime Pantanal bird trio (J=0.47-0.56)
- chvcon1 — Chestnut-vented Conebill
- whtdov — White-tipped Dove
- chacha1 — Chaco Chachalaca

### Group 3: Parrot pair (J=0.71)
- bufpar — Turquoise-fronted Amazon
- hyamac1 — Hyacinth Macaw

### Group 4: Marsh frog pair (J=0.52)
- 22967 — Marbled White-lipped Frog
- 22973 — Whistling Grass Frog

### Group 5: Cryptic frog duo (J=0.75) — **RESCUES A MISSING CLASS**
- **25073 — Chiasmocleis mehelyi** (one of the 28 missing-from-train classes!)
- 326272 — Weeping Frog

If model predicts P(326272)=0.9, you can broadcast P(25073)≥0.7. This **rescues a missing class via behavioral co-occurrence**, more powerful than genus-proxy.

### Cross-class alias (J=1.0)
- 43435 — Black Howling Monkey **≡** 47158son14 — "Insect sonotype14"

**Howling Monkey calls are ALWAYS labeled as having "son14" in their windows.** Either:
- The annotator heard the howler and also heard a specific insect texture present at the same time
- OR "son14" is actually a low-frequency texture that overlaps with howler frequencies in the labeler's perception

Either way: **for prediction, P(43435) and P(son14) should be tightly coupled.**

## 3. Within-file species persistence patterns

For 25 top species, computed mean consecutive-run length within each labeled file:

### TEXTURE species (mean_run = 12 windows = entire file)
- 47158son25, 47158son07, 47158son11, 47158son13, 22961, 47158son03 — all 12.00 mean run
- 47158son17: 10.00
- chvcon1: 11.00

**When these species appear, they appear ALL 12 WINDOWS of the file.**

### EVENT species (mean_run < 4 windows)
- undtin1: 2.00 (isolated calls)
- trsowl: 1.80 (single hoots)
- compau: 4.57

### Window distribution (1-12) for all top species
Almost all species have UNIFORM distribution across windows 1-12 (mean ≈ 6.5, no temporal preference within file).

**Implication for the model**: 
- For texture species, predict consistently across all 12 windows (smoothing window = uniform)
- For event species, allow per-window discrimination
- This validates the texture/event smoothing kernel split (aliozanmemetoglu's [0.35,0.30,0.35] vs [0.20,0.60,0.20])

## 4. Within-file label density is uniform

Mean species count per window position:

| Window | Mean species | Max species |
|---|---:|---:|
| 1 | 4.08 | 8 |
| 2 | 4.13 | 8 |
| 3 | 4.21 | 8 |
| 5 | 4.31 | 8 |
| **7** | **4.42** | **9** |
| 10 | 4.28 | **10** |
| 12 | 4.19 | 9 |

**Species count is essentially flat across windows** (4.08-4.42, range only 0.34). The middle (windows 5-9) has slightly more species. The maximum-ever-seen is **10 species in a single 5-sec window** at window 10.

This means: **within a file, species composition is STABLE**. A model that predicts wildly different species across windows of the same file is wrong. Temporal smoothing should be the default.

## 5. Labeled soundscape files are exactly 60.000 seconds

ALL 66 labeled files have duration EXACTLY 60.0 seconds (no jitter). All 500 unlabeled samples also exactly 60.0s. The 12-window 5-sec structure is hard-coded by the data pipeline.

## 6. Concrete plan additions

### Tier-A drop-in (<1 hour)
- **Drop cross-species duplicate files** (or treat as multi-label) — saved to `duplicate_train_audio.csv`
- **Apply the 7-species Pantanal frog chorus label propagation** at inference
- **Use 326272↔25073 (Weeping Frog↔Chiasmocleis) co-occurrence** to rescue one missing class

### Tier-B (refactoring)
- **Multi-label fix for cross-species duplicates**: instead of dropping iNat791264 (fepowl) and iNat791265 (giwrai1), label BOTH with multi-label [fepowl, giwrai1]
- **Texture-vs-event detection per species**: use the run-length statistic to auto-classify each species as texture (run≥6) or event (run<4)
- **Window-uniform prediction for texture species**: predict once, broadcast to all 12 windows

## 7. Sources

All derived from local data analysis:
- `/home/user/opencode/birdclef-2026/data/`
- `soundfile.info`, `concurrent.futures.ThreadPoolExecutor`
- `hashlib.md5` for content-level dedup verification
- `pandas` co-occurrence analysis
