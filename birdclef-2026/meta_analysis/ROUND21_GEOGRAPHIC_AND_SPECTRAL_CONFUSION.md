# BirdCLEF+ 2026 — ROUND 21: geographic distribution + spectral confusion matrix

## 1. Regional distribution of train_audio (35,549 files)

Using rough geographic regions:

| Region | Recordings | % of total |
|---|---:|---:|
| **N_hemisphere** (lat > 0) | **10,726** | **30.2%** |
| Brazil central/east (lat -10 to -23, lon > -60) | 9,579 | 26.9% |
| S_temperate (lat < -23.5) | 8,500 | 23.9% |
| Tropical_other | 4,735 | 13.3% |
| **Pantanal_region** (lat -23.5 to -10, lon -65 to -60) | **1,012** | **2.8%** |
| Bolivia/Peru (lat -23.5 to -10, lon < -65) | 997 | 2.8% |

**30.2% of train_audio is from the Northern Hemisphere** — completely outside the BC2026 test domain (Pantanal, Southern Hemisphere). For some species, this is the BULK of training data.

### Species heavily contaminated with N-hemisphere recordings

| Species | N hemisphere % | Type |
|---|---:|---|
| **209233 Feral Horse** | **100%** | Domestic animal (only recorded in NH) |
| **74113 Bos taurus (Highland)** | **100%** | Domestic cow (only recorded in NH) |
| osprey | 98% | Migratory raptor |
| houspa (House Sparrow) | 94% | Domestic / urban worldwide |
| 47144 Domestic Dog | 93% | Domestic |
| redjun (Red Junglefowl) | 92% | Domestic chicken ancestor |
| greyel (Greater Yellowlegs) | 91% | Migratory shorebird |
| shshaw (Sharp-shinned Hawk) | 87% | Migratory raptor |
| bbwduc (Black-bellied Whistling Duck) | 85% | Mostly NH range |
| 22985 (a frog) | 83% | (?!) — possibly mis-labeled |

The DOMESTIC ANIMALS (horse, cattle, dog, chicken) are recorded almost exclusively in N hemisphere — but they're PRESENT in Pantanal farms. The Pantanal test recordings of these species should match if the species sounds the same regardless of region.

## 2. Pantanal-region species concentration

Species with highest Pantanal-region fraction:

| Species | n_total | n_Pantanal | Pantanal % |
|---|---:|---:|---:|
| magant1 Mato Grosso Antbird | 63 | **40** | **63%** |
| 738183 White-coated Titi | 5 | 3 | 60% |
| hyamac1 Hyacinth Macaw | 65 | **39** | **60%** |
| 24321 Mato Grosso Snouted Tree Frog | 2 | 1 | 50% |
| whlspi1 | 59 | 28 | 47% |
| rufcac2 Rufous Cacholote | 28 | 12 | 43% |
| chacha1 Chaco Chachalaca | 99 | 35 | 35% |
| pluibi1 | 68 | 24 | 35% |

The **Pantanal specialists** (magant1, hyamac1, chacha1, etc.) have high-quality regional data. Models should handle these well.

## 3. Spectral confusion matrix (per-species mean PSD on 5 train_audio samples each)

Computed cosine similarity across 206 species.

### Cross-class confusion risks (similarity > 0.94)

Same acoustic profile, different class:

| Pair (sim) | Species 1 | Species 2 |
|---|---|---|
| 0.960 | Dwarf Tree Frog (65380) | Sooty-fronted Spinetail (Aves) |
| 0.959 | Dwarf Tree Frog (65380) | **House Sparrow** (Aves) |
| 0.957 | Dwarf Tree Frog (65380) | Rufous-fronted Thornbird (Aves) |
| 0.954 | Dwarf Tree Frog (65380) | Spix's Spinetail (Aves) |
| 0.943 | Paraguayan Swimming Frog (66971) | House Sparrow |
| 0.939 | Dwarf Tree Frog | Gilded Hummingbird |
| **0.927** | **Prionacris erosa (Insecta)** | **Nanday Parakeet (Aves)** |
| 0.927 | Mustached Frog (22956) | Buff-necked Ibis |

The Dwarf Tree Frog (65380, 333 occurrences in labeled soundscape — most common class!) is SPECTRALLY INDISTINGUISHABLE from spinetails, sparrows, and other small chirpy birds. The model can only distinguish them via TEMPORAL/RHYTHMIC features (call duration, repetition pattern).

### Within-class confusion (Amphibia)

| Pair (sim) | Species 1 | Species 2 |
|---|---|---|
| **0.967** | Dwarf Tree Frog (65380) | Paraguayan Swimming Frog (66971) |
| 0.934 | Whistling Grass Frog (22973) | Paraguayan Swimming Frog (66971) |
| 0.902 | Whistling Grass Frog | Dwarf Tree Frog |
| 0.877 | Usina Tree Frog (555123) | Paraguayan Swimming Frog |

The "PANTANAL FROG CHORUS TRIO" (65380, 66971, 22973) is **spectrally CONFUSABLE** — they sound identical at the spectral level. The Jaccard 0.4+ co-occurrence (ROUND 20) is CONSISTENT with them being acoustically the same texture.

### Spectrally distinct same-class pairs (easy to discriminate)

| Pair (sim) | Species |
|---|---|
| 0.001 | undtin1 ↔ wesfie1 |
| 0.002 | 1595929 Uruguay Harlequin Frog ↔ 476521 Cuyaba Dwarf Frog |
| 0.003 | astcra1 Ash-throated Crake ↔ wesfie1 |

## 4. Within-species spectral variance (which species are hardest to model)

Species with most-variable spectral content (need more data/augmentation):

| Species | within_var | Class |
|---|---:|---|
| rebscy1 Red-billed Scythebill | 0.302 | Aves |
| whtdov White-tipped Dove | 0.300 | Aves |
| **244024 Giant Cicada** | **0.287** | **Insecta** (variable calls!) |
| dwatin1 Dwarf Tinamou | 0.276 | Aves |
| whbant2 Antshrike | 0.265 | Aves |
| 43435 Black Howling Monkey | 0.223 | Mammalia |

Species with most-CONSISTENT calls (easiest to model):

| Species | within_var | Class |
|---|---:|---|
| hyamac1 Hyacinth Macaw | 0.071 | Aves |
| oliwoo1 Olivaceous Woodcreeper | 0.071 | Aves |
| saffin Saffron Finch | 0.074 | Aves |
| **1595929 Uruguay Harlequin Frog** | 0.078 | **Amphibia (single-note)** |
| 24287 Brown-bordered Tree Frog | 0.081 | Amphibia |

### Per-class average within-species variance

| Class | mean | std |
|---|---:|---:|
| Amphibia | 0.150 | 0.036 | **most consistent** |
| Mammalia | 0.157 | 0.056 |
| Insecta | 0.164 | 0.107 | high variance |
| Aves | 0.164 | 0.049 | **most variable** |

**Birds are the hardest class to model** (variable calls with multiple variants). Frogs are easiest (single repeating note). The model should allocate more capacity / augmentation to bird species, especially the high-variance ones.

## 5. Concrete plan additions

### Tier-A drop-in
- **Drop / down-weight N-hemisphere recordings for non-domestic species** (mostly affects 26 species with >50% NH data)
- **Boost training for variable-call species**: rebscy1, whtdov, 244024 Giant Cicada, dwatin1 — apply 2-3x oversampling or stronger augmentation
- **For confusable species pairs (sim > 0.94)**, apply **per-class label smoothing** so predictions don't overcommit

### Tier-B (refactoring)
- **Acoustic-similarity-aware ensemble**: Train one model per acoustic CLUSTER (frog texture cluster, bird chirp cluster, etc.) — each model only discriminates within its cluster
- **Use temporal features explicitly**: For the confusable frog↔sparrow pairs, the model needs to know about CALL DURATION and REPETITION RATE (frogs ~ regular interval, sparrows variable)

### Tier-C (research)
- **Recurrence-aware loss**: penalize the model differently for confusable pairs (frog vs sparrow) vs distinct pairs (frog vs hawk)
- **Within-species spectral variance as data weight**: high-variance species get more training steps per sample

## 6. Sources

All derived from local data analysis:
- `/home/user/opencode/birdclef-2026/data/`
- `soundfile.read`, `scipy.signal.welch`, numpy cosine similarity
- 5-sample-per-species mean PSD over 206 species
