# Code Review — Nina EoS.3 (0.947) vs the EDA

The submitted pipeline is `nina2025/birdclef-2026-eos-3` (downloaded into
`submission_code/eos-3/birdclef-2026-eos-3.ipynb`). Configured ensemble:

```python
solutions = {'type_add': 'direct',
 'Models': [
   {'Model':'Model_3','subm':'subm_3.csv','weight':0.015,'xSED':[],          'LB':'0.928'},
   {'Model':'Model_9','subm':'subm_9.csv','weight':0.985,'xSED':[0.605,0.395],'LB':'0.947'},
 ]}
```

So the score is essentially **Model_9 + a 1.5% dash of Model_3.**

## Architecture summary (Model_9, the 0.947 driver)

- **Embeddings**: Google **Perch** (ONNX) for clip-level audio embeddings (1536-d).
- **Auxiliary heads**: **BirdNET** (ONNX) for additional logits, **SED** model
  (TF→ONNX) for fine-grained 5-s frame predictions.
- **Top model**: `LightProtoSSM(d_input=1536, d_model=128, d_state=16,
  n_classes=234, n_windows=12)` — a prototypical State-Space Model fed by
  Perch + perch_logits + site_ids + hours.
- **Second pass**: `ResidualSSM(d_input=1536, d_scores=234)` corrects first-pass errors.
- **Probes**: Per-class sklearn `MLPClassifier`s (vectorized into a single
  `nn.Module` for fast inference) with PCA-reduced features.
- **Post-processing pipeline**, in order:
  1. Per-taxon temperature: `TAXON_TEMPS = {"Aves":0.90, "Amphibia":1.10, "Insecta":1.15, "Mammalia":1.00, "Reptilia":1.00}`
  2. File-level confidence scaling (top-K pooling)
  3. Rank-aware scaling (rank^power within file)
  4. Adaptive delta smoothing across 12 windows
  5. Per-class threshold sharpening
  6. **Sonotype mirroring** (max-pool across visually-identical insect sonotype groups)
- **Site/hour priors**: `build_prior_tables` from labeled soundscapes;
  `apply_prior(scores, sites, hours, tables, lambda_prior=0.4)` adds a soft prior.
- **TTA**: `temporal_shift_tta` with shifts `[0, ±1, ±2]` seconds.

Settings everywhere: `SR=32_000`, `WINDOW_SEC=5`, `N_WINDOWS=12`, `N_CLASSES=234`. ✅

---

## EDA finding → code-side check

| # | EDA finding | What code does | Verdict |
|---|---|---|---|
| 1 | `train_soundscapes_labels.csv` has every row duplicated (1,478 → 739) | `sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()` at L263 | ✅ Handled |
| 2 | 234 classes, only 206 in `train_audio`; 28 missing are 25 insect sonotypes + 3 others | "Sonotype mirroring" max-pools `(son15,son16)`, `(son09,son12)`, `(son02,son14)`, `(son13,son21,son22,son23)`. **Other 17 sonotypes are NOT mirrored.** Other 3 missing taxa (`1491113`, `25073`, `517063`) have no explicit handling. | ⚠️ Partial |
| 3 | All audio 32 kHz mono | Hard-coded `SR = 32_000` throughout | ✅ |
| 4 | Test soundscapes 60 s; 12 × 5-s windows | `N_WINDOWS = 12`, `WINDOW_SEC = 5` everywhere | ✅ |
| 5 | Class imbalance: Aves 98% of clips | `build_class_freq_weights(Y, cap=10.0)` + focal loss (43 mentions) + per-taxon temperature scaling | ✅ but `cap=10` may be too soft given **the time-imbalance is 100–330× by hours** (D5) |
| 6 | Domain shift: only 2.38% of training clips in Pantanal box | Site/hour prior tables from labeled soundscapes (`apply_prior(lambda_prior=0.4)`) | ✅ for sites/hours; **❌ no geographic feature (latitude/longitude unused; "Pantanal" mentioned 1×, lat/lon 0×)** |
| 7 | 89% of labeled segments are multi-label (mean 4.2 species, max 10) | Multi-label sigmoid BCE/focal — appropriate | ✅ |
| 8 | 99.4% of soundscapes are unlabeled | Used as prior table source; no explicit pseudo-label / noisy-student loop visible (Model_9 doesn't train on unlabeled soundscapes). Model_3 has `cosine restart`, OOF cross-validation but no pseudo-label. | ❌ Big lever left on table |
| 9 | Submission columns match taxonomy exactly | Uses `PRIMARY_LABELS` derived from taxonomy.csv — fine | ✅ |
| 10 | TTA used by top BirdCLEF 2025 solutions | `temporal_shift_tta(shifts=[0,1,-1,2,-2])` ✅ | ✅ |

---

## New issues exposed by the deeper EDA

### 1. Audio-time imbalance is much worse than clip-count imbalance (D5)

Total hours of training audio per class (file-size proxy, bytes/sec from sample):

| class | hours | ratio vs Aves |
| :- | --: | --: |
| Aves     | **330.3** | 1× |
| Amphibia |   3.3 | 1:100 |
| Insecta  |   1.5 | 1:220 |
| Mammalia |   1.0 | 1:330 |
| Reptilia |   ~0  | — |

A `cap=10.0` class-frequency weight is *much* too gentle. The model gets >300×
more bird-seconds of supervision than mammal-seconds. Recommendation:
**uncap (or raise to 50–100×), and add stratified sampling by class so each
mini-batch contains non-Aves classes**. Also: train_audio for Reptilia is **essentially zero** — that class can only be learned from soundscape labels + maybe a Perch fine-tune on iNat data.

### 2. Labeled soundscape coverage is **9 of 23 sites and excludes all daytime hours** (D6, D9)

- Sites in `train_soundscapes_labels.csv`: only **S22 (40 files), S08 (5),
  S09 (5), S15 (4), S19 (3), S23 (3), S13 (2), S18 (2), S03 (2)**.
- Sites S01, S02, S04–S07, S10–S12, S14, S16, S17, S20, S21 are **never labeled.**
  The "site prior" learned by the code therefore only covers ~half the sites.
- **Hours covered: only 00–07 and 18–23.** Daytime (08–17) has zero labels.
- Test soundscapes can be at any site, any hour. The prior will silently
  fall back to a global prior for the unseen 14 sites and the entire daytime
  window — and may even *hurt* there. Recommendation: **check that `apply_prior`
  falls back gracefully on unseen sites/hours**, and consider weighting the
  prior down (`lambda_prior < 0.4`) for OOD slices.

### 3. Sonotype mirroring covers only 9 of 25 sonotypes (D2/D3 + grep)

The code has these mirror groups:
- `(son15, son16)`, `(son09, son12)`, `(son02, son14)`, `(son13, son21, son22, son23)`

That's only 9 sonotypes. The other 16 sonotypes (`son01, son03–08, son10–11,
son17–20, son24–25`) are not mirrored. From D3 the co-occurrence heatmap of
sonotypes shows several other strong pairs / chorus groups
(`son17` + `son25` co-occur heavily, `son24` solos at 24 segments, etc.).
Recommendation: **fit mirror groups data-driven from `soundscape_cooccurrence.csv`**
— e.g., merge sonotypes with Jaccard > 0.8 in the labeled set.

### 4. `latitude` / `longitude` are entirely unused (D4)

EDA shows 119/202 species have ≥1 Pantanal-box clip, **only 31 have ≥10**, and
only 4 have ≥50% in-Pantanal. Code:
- 0 occurrences of `latitude` / `longitude`
- 1 occurrence of `Pantanal` (a comment)

Easy lever: **weight training clips by geographic distance to the Pantanal
centroid** (or simply upsample the 31 species with rich in-region coverage as
proxy anchors). The pretrained Perch already encodes acoustic content, but
giving a geographic prior to species-level heads is essentially free.

### 5. Only 4,372/35,549 train rows (12.3%) have secondary labels — and **NONE of the 28 missing species ever appear as a secondary label** (D7)

Important confirmation: the 28 missing species really only exist in the labeled
soundscape segments. Any model relying on `train_audio` (i.e., the prototype
init in `LightProtoSSM.init_prototypes`) cannot learn them from train_audio at
all. The code's prototype init reads Perch embeddings of train_audio clips →
**28 classes will get zero/random prototypes** unless they fall back to
soundscape-segment embeddings.

I did not see explicit fallback logic for that in Model_9. **Worth verifying
`init_prototypes` includes embeddings of labeled 5-s soundscape segments for
classes with no train_audio.** If not, those 28 columns are blind on the
Perch+ProtoSSM branch and rely entirely on SED + BirdNET to score.

### 6. No semi-supervised / noisy-student loop (vs. BirdCLEF 2025 1st place)

BirdCLEF 2025 1st was won by iterating pseudo-labels on unlabeled soundscapes.
EDA shows **99.4% of soundscapes are unlabeled** here (10,592 unlabeled vs 66 labeled).
Model_9 in EoS.3 doesn't run an iterative noisy-student loop on those. This is
the largest unexploited lever in the current 0.947 baseline, and aligns
exactly with the user's V100-series "pseudo" kernels (V100 reached 0.919 val_auc on
its own; V101 blended V73+pseudo for 0.938; later blends went 0.941–0.942). The
pseudo experiments are happening but didn't make it into the 0.947 baseline.

### 7. The Model_3 contribution is tiny (1.5% weight) — likely noise

Removing Model_3 and refunding weight to Model_9, or using a rank-based blend
with a properly tuned weight, is worth A/B testing. The current `direct` add
with 1.5% weight is unlikely to be statistically meaningful given how much
better Model_9 is.

### 8. EoS.3 has 9 model branches but only uses Model_3 + Model_9

The notebook includes Model_2, Model_4, Model_5, Model_61, Model_62, Model_7,
Model_8 — all gated by `if 'Model_X' in _ensemble_models`. Worth verifying
whether any of these were beating Model_9 on OOF and got cut from the config
by mistake.

---

## Concrete recommendations (prioritized)

1. **Verify prototype init covers 28 missing classes.** If it doesn't, fix
   `init_prototypes` to use Perch embeddings of labeled 5-s soundscape segments
   for any class absent from `train_audio`. **(Biggest correctness win.)**
2. **Run a noisy-student round** on the 10,592 unlabeled soundscapes using
   Model_9 as teacher. Train a student on labeled + pseudo, ensemble. This is
   the BirdCLEF 2025 1st-place recipe and your V100s are already attempting it.
3. **Data-driven sonotype mirror groups** from `stats/soundscape_cooccurrence.csv`
   (Jaccard > 0.8 in the labeled set).
4. **Site/hour-aware prior with graceful OOD fallback** — confirm `apply_prior`
   returns a neutral prior (not a learned but unsupported one) for sites
   S01/S02/S04–S07/S10–S12/S14/S16/S17/S20/S21 and daytime hours 08–17.
5. **Loosen `cap=10.0`** in `build_class_freq_weights` for the time-imbalance
   reality (try `cap=50` then `cap=None`), or switch to per-batch
   class-stratified sampling so every mini-batch sees non-Aves species.
6. **Add a geographic feature** to the LightProtoSSM (distance-to-Pantanal,
   one-hot continent, lat/lon embedding) or simply down-weight train_audio
   clips from species that have no Pantanal-box clips. Lat/lon are currently
   unused.
7. **Sanity-check the Model_3 (1.5%) inclusion** — try Model_9 standalone vs
   blended; if the gain is < 0.001 LB, the blend cost (runtime + variance)
   isn't worth it.
8. **Use a rank blend instead of direct add** between Model_3 and Model_9
   since their scales differ; `rank_1_add2()` exists in the code but
   `type_add` is set to `direct`.
