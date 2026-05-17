# BirdCLEF+ 2026 — ROUND 22: site×hour species priors + the labeled-vs-pseudo divergence

This round computes the conditional P(species | site, hour) priors from BOTH the labeled set (66 files = 739 windows) AND the pseudo-cache (10,592 files = 127,104 windows). The divergence between them is illuminating.

## 1. Per-site species composition (labeled set)

### S22 — THE NIGHT PANTANAL FROG CHORUS SITE (2,068 species-windows, 65% of all labeled)
Top species:
```
65380   Dwarf Tree Frog              321
517063  Southern Orange-legged Frog  269
555146  Chaco Tree Frog              209
22973   Whistling Grass Frog         181
24279   Lesser Snouted Tree Frog     171
23158   Pale-legged Weeping Frog     169
24321   Mato Grosso Snouted Frog     167
66971   Paraguayan Swimming Frog     149
22967   Marbled White-lipped Frog    120
1491113 Guaraní leaf-litter Frog     53
```

### S08 — SONOTYPE-RICH SITE (322 species-windows)
**9 of 10 top species are SONOTYPES** (cluster 4 + cluster 6):
```
47158son25  48  (texture cluster 4)
47158son17  43  (texture cluster 4)
47158son13  24  (texture cluster 6)
47158son22  24  (texture cluster 6 — alias of son23)
47158son23  24  (texture cluster 6 — alias of son22)
47158son21  22  (texture cluster 6)
chacha1     17  (Chaco Chachalaca)
47158son15  12  (cluster 4 — alias of son16)
47158son16  12  (cluster 4 — alias of son15)
47158son03  12  (cluster 2)
```

### S15 — DAWN BIRD CHORUS SITE (213 windows, hour=06)
```
47158son07  48  (the LOW-FREQ sonotype that's solo 90%)
whtdov      48  (White-tipped Dove)
chvcon1     35  (Chestnut-vented Conebill)
chacha1     32  (Chaco Chachalaca)
orwpar      13
undtin1     12
```

### S19 — NIGHT INSECT + FROG SITE (189 windows)
```
47158son11  24
47158son24  24
326272      23  (Weeping Frog)
22973       20
22967       12
25073       12  (Chiasmocleis mehelyi — MISSING FROM TRAIN!)
```

### S23 — SONOTYPE + HOWLING MONKEY SITE (172 windows)
```
47158son25  36
47158son10  25
47158son06  18
47158son04  12
47158son03  12
43435       12  (Black Howling Monkey)
47158son14  12  (the "monkey alias" sonotype)
chacha1     11
```

## 2. Per-hour P(species) from labeled set

### Hour 01:00 (test sample's hour)

| Species | P(species at hr=01) | Common name |
|---|---:|---|
| **517063** | **0.843** | Southern Orange-legged Leaf Frog |
| **65380** | **0.608** | Dwarf Tree Frog |
| 24279 | 0.529 | Lesser Snouted Tree Frog |
| 23158 | 0.471 | Pale-legged Weeping Frog |
| 555146 | 0.255 | Chaco Tree Frog |
| 1491113 | 0.235 | Guaraní leaf-litter frog (MISSING!) |
| 22961 | 0.235 | Pointedbelly Frog |
| 22967 | 0.235 | Marbled White-lipped Frog |
| 22973 | 0.235 | Whistling Grass Frog |
| litnig1 | 0.216 | Little Nightjar |
| 25092 | 0.196 | (a frog) |
| trsowl | 0.176 | Tropical Screech-Owl |

**These hour-conditional priors give a floor for predictions** — at hour=01:00, the model should AT LEAST predict P(517063) ≥ 0.5, P(65380) ≥ 0.4, etc.

### Hour 03:00 (sonotype-dominant)

```
47158son25  0.667 (24/36)
47158son13  0.444
47158son22  0.444
47158son23  0.444
47158son21  0.407
```

### Hour 06:00 (dawn — bird chorus + son07)

```
47158son07  0.706
whtdov      0.706
chvcon1     0.515
chacha1     0.471
orwpar      0.191
```

## 3. The labeled-vs-pseudo divergence

**Labeled set at hour=01**: dominated by FROGS (517063=84%, 65380=61%)
**Pseudo cache (Perch v2) at hour=01**: dominated by BIRDS (compot1=39%, compau=38%, trsowl=31%)

| Species | Labeled P | Pseudo P | Divergence |
|---|---:|---:|---:|
| 517063 | **0.843** | not in top 8 | LABELED ↑↑ |
| 65380 | 0.608 | 0.291 | LABELED ↑ |
| 24279 | 0.529 | (low) | LABELED ↑ |
| compot1 | not in labeled top | **0.392** | PSEUDO ↑↑ |
| compau | not in labeled top | **0.384** | PSEUDO ↑↑ |
| trsowl | 0.176 | 0.311 | PSEUDO ↑ |
| undtin1 | 0.118 | 0.277 | PSEUDO ↑ |
| litnig1 | 0.216 | 0.156 | both modest |

**Explanation**: The labeled set is dominated by S22 (60+ files at hours 18-23 + 00-02) where the FROG CHORUS is THE dominant sound. The pseudo cache covers ALL sites including S01, S02, S07, S10, S12 where NIGHTJAR/POTOO calls dominate (Perch v2 detects birds well, frogs poorly).

**Best practice**: combine the two priors:
```python
combined_prior = 0.4 * labeled_hour_prior + 0.6 * pseudo_hour_prior
```
This balances the labeled-set's S22 frog bias with the pseudo-cache's broader site coverage.

## 4. S05-SPECIFIC pseudo-cache priors (most relevant for test sample)

The test sample is at site S05, hour=01:00. Direct S05 night-time pseudo-cache (60 windows):

| Species | P(present) at S05 night | Common name |
|---|---:|---|
| **24279** | **0.834 (83%)** | Lesser Snouted Tree Frog (DOMINANT!) |
| 65380 | 0.291 | Dwarf Tree Frog |
| compau | 0.263 | Common Pauraque |
| limpki | 0.198 | Limpkin |
| compot1 | 0.193 | Common Potoo |
| watjac1 | 0.174 | Wattled Jacana |
| 22973 | 0.165 | Whistling Grass Frog |
| whtdov | 0.158 | White-tipped Dove |
| 555146 | 0.147 | Chaco Tree Frog |
| trsowl | 0.133 | Tropical Screech-Owl |
| 23158 | 0.105 | Pale-legged Weeping Frog |

**THIS is the strongest prior** for the test sample. **517063 (the labeled-set top with 84%) is NOT in the S05 top species** because 517063 is an S22 specialty.

**For the test sample BC2026_Test_0001_S05_20250227_010002:**
- Expected dominant species: **24279 Lesser Snouted Tree Frog** (83% from S05 pseudo data)
- Secondary: 65380, compau, limpki, compot1
- A good baseline prediction would heavily weight 24279 and other S05-typical frogs

## 5. Saved priors (CSV files in meta_analysis/)

- `hourly_species_priors.csv` — P(species | hour) from labeled set (13 hours × 75 species)
- `site_species_priors.csv` — P(species | site) from labeled set (9 sites × 75 species)
- `site_hour_species_priors.csv` — P(species | site, hour) joint priors (25 cells × 75 species)
- `duplicate_train_audio.csv` — 200 train_audio files with detected duplicates

A model can READ these CSVs and combine with the model's own predictions:
```python
hourly_p = pd.read_csv('hourly_species_priors.csv').set_index('hour')
# For each test row's (site, hour):
final_pred = 0.8 * model_pred + 0.2 * hourly_p.loc[test_hour].values
```

## 6. Concrete plan additions

### Tier-A drop-in
- **Use S05-specific pseudo prior as baseline for any S05 test file** — heavily weight 24279, 65380 predictions
- **Combine labeled-prior (0.4) + pseudo-prior (0.6)** for general hour-conditional baseline
- **At hour=01:00, set P(517063) floor = 0.4**, P(65380) floor = 0.3 (matches labeled-set probabilities scaled down)

### Tier-B
- **Site embedding** — train model with site index as input feature; let the model learn site-conditional logit shifts
- **Cross-validate priors using HELD-OUT labeled site** (e.g., remove S15 from training prior, validate at S15)

### Tier-C
- **Hierarchical mixture-of-experts**: route test predictions to different "expert" sub-models based on site/hour (frog-chorus expert, sonotype expert, dawn-bird expert)

## 7. Sources

All derived from local data analysis. Hourly priors verified against pseudo-cache (`backtracking/birdclef2026-pseudo-cache-v1`).
