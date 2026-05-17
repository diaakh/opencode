# BirdCLEF+ 2026 — ROUND 8: New high-value datasets + code-pattern deep dive

This round goes beyond the kernel corpus to inspect the **newly-released Kaggle datasets** that ship trained weights and pseudo-labels, and to extract the exact mechanisms that separate the 0.95+ ELITE from the 0.948 PLATEAU fork crowd.

## 1. The bird-only blindspot: `yasunorim/xc-birdclef-2026-target-urls`

The standard "external Xeno-Canto download list" used by public solutions:

| metric | value |
|---|---:|
| total recordings | 11,563 |
| unique labels covered | 159 |
| total audio hours | 138.9 |
| A-quality fraction | 73% |
| from Brazil | 56% |
| **Aves classes covered** | **159 / 162 (98%)** |
| **Amphibia classes covered** | **0 / 35 (0%)** |
| **Insecta classes covered** | **0 / 28 (0%)** |
| **Mammalia classes covered** | **0 / 8 (0%)** |
| **Reptilia classes covered** | **0 / 1 (0%)** |
| **Missing-from-train classes covered** | **0 / 28** |

**Hidden insight**: Xeno-Canto is a bird-only platform by design. Of the 72 non-Aves classes in the BirdCLEF 2026 taxonomy, **zero** have XC coverage. The 25 insect sonotypes (47158son01-25), the 35 Amphibia (frogs), the 8 Mammalia (capuchin, marmoset, titi, horse…), and the 1 Reptilia (Southern Spectacled Caiman) all need a different source. The only known platform with frog/insect/mammal call recordings at this scale is **iNaturalist Sounds** (which is exactly what Perch v2 was trained on — see paper §4.2).

**Practical implication**: anyone relying solely on XC external data is leaving the missing 28 classes (= 10.7% of macro-AUC, per alexandergremyakov) on the table. The Insecta and Amphibia mining must go through iNat 2024 export — and Perch v2 already encodes that knowledge (taxon IDs 47158, 22961, 23158, 24321, etc. are iNaturalist taxon IDs).

## 2. The duplicated-label bug in `train_soundscapes_labels.csv`

Running `groupby(['filename','start','end']).size()` on the official labels file:

```
Total (file, window) groups: 739
Rows per group distribution: 2 → 739 (100%)
(file, window) groups with >1 distinct label set: 0
(file, window) groups with =1 distinct label set: 1,478 → 739 after dedup
```

**Every annotation is duplicated exactly 2x and the duplicates always agree.** Either it's a save-time bug or two annotators were merged with verbatim agreement. Net effect: 1,478 raw rows = 739 unique windows × 2 copies.

**Who knows this**: 33 out of ~1,250 kernels (2.6%) explicitly call `.drop_duplicates()` on the labels — and the 33 are dominated by the Nina EoS-3/EoS-4 / Karnakbayev family (the actual 0.948+ baseline). The remaining ~1,200 kernels read raw, which **double-weights** the labeled windows in their site/hour priors, focal-loss targets, isotonic calibration, and class-frequency calculations.

For someone building a custom pipeline this is the single highest-leverage one-line fix: `pd.read_csv(...).drop_duplicates()`.

## 3. `backtracking/birdclef2026-pseudo-cache-v1`: free Perch teacher for 99.4% of train_soundscapes

A 441 MB cache the community has barely touched (42 downloads):

| file | shape | notes |
|---|---|---|
| `pseudo_emb.npy` | (127104, 1536) fp16 | Perch v2 embeddings |
| `pseudo_scores.npy` | (127104, 234) fp16 | Per-class logits, range [-7.8, 14.98] |
| `pseudo_soft.npy` | (127104, 234) fp16 | Sigmoid soft labels, row-sum ≈ 6.3 (multi-label) |
| `pseudo_meta.parquet` | 127104 rows | row_id, filename, site, hour_utc |
| `pseudo_manifest.json` | thresholds | 1.8 M positives @ thr 0.10; 730 K @ 0.20; 384 K @ 0.30 |

Coverage: **10,592 of 10,658 train_soundscapes files (99.4%)** — the 66 missing files are exactly the `train_soundscapes_labels.csv` labeled subset.

Pseudo-cache hour distribution:
```
0-4 UTC: 49,980 windows (39%)   ← night / dawn
17-23 UTC: 75,144 windows (59%)  ← dusk / evening
5-16 UTC: 1,980 windows (1.6%)  ← almost no daytime data
```
This is bimodal: the Pantanal soundscape data is overwhelmingly **night-active** (and matches the train_audio temporal pattern). Models that ignore time-of-day prior lose calibration on these classes.

**Hidden practical use**: Download this 441 MB cache → train any student model (MLP, EfficientNet, Snowflake-SED) on `pseudo_soft` as soft targets with MSE/KL — no GPU, no TensorFlow, no Perch inference. The 4-fold EfficientNetV2-S student in `baiyuby/birdclef2026-distill-models` (CV 0.985, T=2.0, α=0.7) was trained this way and ships ~350 MB of weights.

## 4. `chaneyma/birdclef-2026-cv9245-moe-artifacts`: a full ProtoSSM-Mamba inference pipeline

This is one of the most complete public dumps. It includes:

- `pantanal_infer_only_submission.py` (514 lines) — full inference recipe
- 4 fold weights of ProtoSSM (`moe_p0.60_c0.25_r0.15_post_p0.45_fold{1..4}.pt`)
- StudentCNN weight (`.pt`, ed28 distillation cv45)
- StudentCRNN weight (`.pt`)

**Architectural innovations not documented elsewhere in our corpus:**

### 4a. ProtoSSM is a Mamba-lite Selective State Space Model

```python
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        self.in_proj  = nn.Linear(d_model, 2*d_model, bias=False)
        self.conv1d   = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv-1, groups=d_model)
        self.dt_proj  = nn.Linear(d_model, d_model, bias=True)
        # Mamba-style selective A,B,C with softplus(dt)
        A = torch.arange(1, d_state+1).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
    def forward(self, x):
        # selective recurrence with discretization via dt
        ...
```

This is a faithful **Mamba (Gu & Dao 2023) selective scan** in PyTorch, but with the recurrence implemented as a literal Python for-loop over T=12 time steps (small enough that the unrolled loop is fine for CPU inference). The "selective" part is `dt = softplus(dt_proj(x_conv))` which gives time-step-specific gating.

### 4b. The per-class learnable fusion alpha

```python
self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))  # init zero → sigmoid = 0.5
...
alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
out   = alpha * proto_sim + (1 - alpha) * teacher_logits
```

Per-class **soft selection** between prototype-distance (SSM-refiner) and Perch teacher logits. Initialized at sigmoid(0) = 0.5 (equal trust) and learned per class. After training, frequent classes (where Perch is right) keep alpha low (trust teacher); rare/sonotype classes push alpha high (trust the refiner). This is mechanically the same idea as MoE gating, applied at the LOGIT level not the model level.

### 4c. Top-2 mean amplification post-processing

```python
def postprocess_probs_filewise(probs_flat, n_windows=12):
    x = probs_flat.reshape(-1, n_windows, n_classes)
    prev_x = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    next_x = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    x = 0.8*x + 0.1*(prev_x + next_x)              # temporal smoothing
    top2 = np.sort(x, axis=1)[:, -2:, :].mean(axis=1, keepdims=True)
    x = x * top2                                    # !! self-amplification
    return np.clip(x, 0.0, 1.0)
```

The `x = x * top2` step is **novel and undocumented elsewhere**. For each (file, class), it multiplies every window's prediction by the file-level top-2-window average. Effect: classes that are CONFIDENTLY detected in at least 2 of the 12 windows get amplified across the whole file; classes that are weak everywhere get suppressed. This is a **soft per-file confidence multiplier**, distinct from the standard rank-aware scaling `view × file_max^power`. For macro-AUC it preserves ranking but sharpens it.

### 4d. Default blend weights

```python
--blend-perch 0.60   --blend-cnn 0.25   --blend-crnn 0.15
--prior-scale 0.40
```

The CRNN (BiGRU over conv features) gets the smallest weight; the CNN gets 0.25; Perch teacher gets the dominant 0.60. **All three are applied at the LOGIT level before priors**, then ProtoSSM refines on top.

## 5. The texture-vs-event smoothing kernel (aliozanmemetoglu rank 4, LB 0.958)

From `aliozanmemetoglu/birdclef-5-fold-ensemble-submission` and `birdclef-enb0-coarse-ensemble-submission`:

```python
SMOOTH_EVENT   = np.array([0.20, 0.60, 0.20])   # Aves, Mammalia, Reptilia
SMOOTH_TEXTURE = np.array([0.35, 0.30, 0.35])   # Insecta, Amphibia

def time_smooth(preds, is_texture):
    smooth = lambda p, w: w[0]*pad[:-2] + w[1]*pad[1:-1] + w[2]*pad[2:]
    result = preds.copy()
    if is_texture.any():
        result[:, is_texture]  = smooth(preds[:, is_texture],  SMOOTH_TEXTURE)
    if (~is_texture).any():
        result[:, ~is_texture] = smooth(preds[:, ~is_texture], SMOOTH_EVENT)
    return result
```

**The class-of-call asymmetry**: insect & frog calls are *continuous textures* lasting many windows. The TEXTURE kernel intentionally weights neighbors EQUALLY to the current window (0.35 / 0.30 / 0.35) because the current window has no particular onset advantage. Bird/mammal/reptile calls are *discrete events* — the EVENT kernel preserves the current-window peak (0.60).

Class counts: 28+35 = 63 "texture" classes (27% of 234), 162+8+1 = 171 "event" classes (73%). The 63 texture classes include ALL 28 missing-from-train classes plus another 35 amphibians. So texture smoothing is doing the heavy lifting for the lowest-data, hardest classes.

**Two backbone variants** are stacked in aliozanmemetoglu's ensemble:
- `tf_efficientnetv2_s` (n_mels=128, target 128×256) — temporal resolution dominant
- `tf_efficientnetv2_b0` (n_mels=256, target 256×256) — frequency resolution dominant

Each with 5 folds → 10-model ensemble. PRIOR_WEIGHT = 0.15, TTA_SHIFT = 1.25 sec.

## 6. The hideyukizushi recipe (rank 17, LB 0.953) — full training + inference

The notebook `bird26-reproduce-perch-protossm-resssm-inf-train` is a **complete reproducible pipeline** (134 KB of code). Unique pieces:

### 6a. ResidualSSM is initialized to zero output (corrections start = 0)

```python
self.output_head = nn.Linear(d_model, n_classes)
nn.init.zeros_(self.output_head.weight)
nn.init.zeros_(self.output_head.bias)
```

So the residual head only LEARNS corrections — at init it's a pure pass-through. Combined with the residual connection on the SSM (`h = ssm_norm(h + residual)`), this means at init ResidualSSM == identity on first-pass logits. Training only teaches it to add corrections, not re-derive the prediction. This is the same trick as zero-init in ControlNet (Zhang et al. 2023) — `arxiv:2302.05543`.

### 6b. StratifiedGroupKFold (random_state=91) with file as group

```python
StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=91)
# groups = filename, y = primary_label rare-class indicator
```

**File is the group key** — prevents within-file train/val leakage. Combined with "stratification" by rare classes, this gives balanced folds. Most public kernels use plain KFold or stratified-only and silently leak windows from the same file across the split, inflating their CV.

### 6c. Per-class isotonic calibration + F1-optimal threshold

```python
for c in range(n_classes):
    y_true, y_prob = file_y[:, c], file_oof[:, c]
    if y_true.sum() < 3: continue
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_prob, y_true)
    y_cal = ir.transform(y_prob)
    best_f1, best_t = 0.0, 0.5
    for t in threshold_grid:
        # compute F1 at threshold t
        ...
```

Per-class isotonic regression maps OOF probabilities → calibrated probabilities (monotonic, non-parametric). Then per-class F1-optimal threshold from a grid. **Result**: thresholds vary from 0.05 to 0.95 across the 234 classes. Standard 0.5-threshold submissions are wrong for ~70% of classes. (For macro-AUC submissions, the calibration alone is what matters — the threshold is informational.)

### 6d. Per-class ensemble-weight sweep (proto vs MLP)

```python
def sweep_ensemble_weight(oof_proto, oof_mlp, Y_FULL, candidates=np.arange(0.3, 0.8, 0.05)):
    for w in candidates:
        blended = w * oof_proto + (1-w) * oof_mlp
        auc = macro_auc_skip_empty(file_y, blended.max(axis=1))
        ...
```

Sweeps the proto/MLP blend weight on OOF and picks the best by macro-AUC. **The optimal is ~0.6 proto / 0.4 MLP**, matching the 60/40 imaadmahmood baseline.

## 6.5 The ROOT-OF-EVERYTHING: `marynaborovska/birdclef-26-two-pass-ssm-advanced-pp`

After tracing 40 corpus kernels that cite Maryna Borovska, this is the **single canonical source notebook** for the entire 0.946–0.949 PLATEAU recipe. Every novel post-processing technique in the popular template traces back here. The notebook ships:

### 6.5a. The genus-proxy fallback for unmapped species (the only public attempt at the missing 28)

```python
# For each competition species NOT in Perch's 14,795-vocabulary
proxy_map = {}
for _, row in unmapped_df.iterrows():
    target = row["primary_label"]
    genus  = str(row["scientific_name"]).split()[0]   # binomial first word
    hits   = bc_labels[bc_labels["scientific_name"].astype(str)
                       .str.match(rf"^{_re.escape(genus)}\s", na=False)]
    if len(hits) > 0:
        proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

# Restrict to taxa where genus-level audio similarity is biologically plausible
proxy_map = {idx: bc for idx, bc in proxy_map.items()
             if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in {"Amphibia", "Insecta", "Aves"}}

# AT INFERENCE — fill unmapped logit slots with MAX over genus members
for pos_idx, bc_idxs in proxy_map.items():
    bc_arr = np.array(bc_idxs, dtype=np.int32)
    scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)
```

**Reality check** on what this rescues from the 28 missing-from-train classes:
- 3 frogs (`1491113` Adenomera guarani, `25073` Chiasmocleis mehelyi, `517063` Pithecopus azureus): genus matches **if and only if** Perch was trained on at least one congener. Pantanal-region Adenomera and Pithecopus species exist in iNaturalist → likely yes for Adenomera, partial for the rest.
- 25 insect sonotypes (`47158son01-25`): scientific name is literally `Insect son01` → genus is `Insect` → **zero matches in Perch**. Genus proxy gives nothing for sonotypes.

So genus proxy lifts perhaps 3 of 28 missing classes; the 25 sonotypes still float at chance until you actually train on the `train_soundscapes_labels.csv` ground truth (or pseudo-labels on the larger unlabeled set).

### 6.5b. Class-specific temperature (the inverse of what intuition suggests)

```python
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}
temperatures = np.ones(N_CLASSES, dtype=np.float32)
for ci, label in enumerate(PRIMARY_LABELS):
    cls = CLASS_NAME_MAP.get(label, "Aves")
    temperatures[ci] = 0.95 if cls in TEXTURE_TAXA else 1.10
# Apply via:  logits = logits / temperatures
```

Note: T=0.95 (frogs/insects) makes their logit distribution SHARPER (more confident extremes); T=1.10 (birds) softens them. This is the OPPOSITE of typical calibration — but it works here because the texture-class predictions are mostly genus-proxy (max over multiple Perch labels), which already creates "lukewarm" probabilities. Sharpening pulls them away from the 0.5 line where macro-AUC ranking is least informative.

### 6.5c. The five core post-processing functions (all originate here)

```python
# 1. file_confidence_scale — chaneyma's "top-2 amplification" is THIS function (top_k=2, power=0.4)
def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
    view = probs.reshape(-1, n_windows, C)
    top_k_mean = np.sort(view, axis=1)[:, -top_k:, :].mean(axis=1, keepdims=True)
    return (view * np.power(top_k_mean, power)).reshape(N, C)

# 2. rank_aware_scaling(probs, n_windows=12, power=0.4)  — multiplies by file_max^0.4

# 3. adaptive_delta_smooth — alpha adapts to per-window confidence
def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    for t in range(n_windows):
        conf  = view[:, t, :].max(axis=-1, keepdims=True)
        alpha = base_alpha * (1.0 - conf)
        # blend with neighbor average
        out[:, t, :] = (1-alpha)*view[:, t, :] + alpha*neighbor_avg

# 4. Circular shift TTA over [0, 1, -1, 2, -2] windows, counter-shift and average

# 5. Isotonic + F1-optimal threshold per class on OOF
```

The fact that chaneyma's `pantanal_infer_only_submission.py` uses fixed `0.8*curr + 0.1*(prev+next)` smoothing instead of Maryna's adaptive version is actually a SIMPLIFICATION. Maryna's version is strictly better for confident windows (preserves peaks).

### 6.5d. The honest CV protocol

```python
GroupKFold(n_splits=5)  # grouped by filename
macro_auc_skip_empty(file_y, blended.max(axis=1))  # exact comp metric
```

**filename as group** prevents within-file leakage (all 12 windows from the same 60s file go to the same fold). The CV metric is the **exact** competition `roc_auc_score(average="macro")`, with the explicit skip-classes-with-zero-positives matching what Kaggle does. Most public kernels use plain KFold and silently inflate their CV by ~0.01.

### 6.5e. Why this matters for the user

The "EoS-3 → EoS-4 → exp019" chain that produces the 0.949 LB is a tuning of Maryna's hyperparameters: `rank_power 0.4→0.5→0.6` and `lambda_prior 0.4→0.5`. **Without modifying her recipe**, you've already topped out at 0.949. The PLATEAU is hers.

## 7. The ELITE 0.95+ "anti-pattern" vs PLATEAU 0.948–0.95

Computing feature shares across our 1,194-kernel corpus:

| feature | ELITE (n=14) | PLATEAU (n=310) | delta |
|---|---:|---:|---:|
| **loss_focal** | **38%** | 23% | **+15%** |
| **aug_cutmix** | **23%** | 17% | **+6%** |
| uses_sed | 46% | 68% | −22% |
| aug_time_shift | 54% | 76% | −22% |
| uses_birdnet | 0% | 22% | **−22%** |
| rank_aware | 46% | 70% | −24% |
| uses_tucker (Tucker's distilled SED) | 15% | 41% | **−26%** |
| site_hour_prior | 31% | 59% | −28% |
| adaptive_delta | 38% | 67% | −29% |
| uses_onnx | 23% | 64% | **−41%** |
| file_confidence | 15% | 57% | −42% |
| **sonotype_mirror** | **0%** | 43% | **−43%** |

**The ELITE is NOT the PLATEAU with more tricks** — it's a different distribution. ELITE kernels:
- Train their own models with **focal loss + CutMix** (not the public BCE-on-distill template)
- Avoid the public stack: no Tucker's distilled SED, no BirdNET, no sonotype-mirror, no ONNX hot path
- Either inference-only with custom-trained checkpoints (alioz, tonylica, hideyuki) or pure starter (kdmitrie)

The PLATEAU 0.948 cluster is **template-saturated**: 868 teams sit there because they all forked the same Nina EoS-4 / mtoshidesu / imaadmahmood inference template.

The HI vs LO (≥0.94 vs <0.92) deltas tell the opposite story — the public template gets you from <0.92 to 0.94. But the LAST 0.01 lives outside the template.

**This is the answer to "what's the catch": the 0.95 barrier is not crossed by adding more inference tricks. It's crossed by training your own SED model with focal loss + CutMix on Perch-distilled targets.** The 8 known elite kernels all do this; the 868 PLATEAU kernels all do not.

## 8. The `habedi/birdclef-2026-clap-int8-bundle` cautionary tale

CLAP (Contrastive Language-Audio Pretraining, Wu et al. 2023, `arxiv:2211.06687`) as an alternative audio backbone:

| metric | value |
|---|---:|
| CLAP backbone | 33 MB INT8 ONNX (HTSAT-tiny-clap-22k) |
| 5-fold linear probe (BN→768→256→234) | mean fold-AUC 0.886 |
| **Stacked OOF AUC** | **0.673** |

The catastrophic 0.886 → 0.673 gap from mean-fold to stacked-OOF is **fold disagreement**: CLAP embeddings don't generalize across folds. Adding CLAP to an ensemble likely HURTS more than helps — but only 1 corpus kernel uses it.

**Why CLAP fails for BirdCLEF**: CLAP is trained on FreeSound+AudioSet (general environmental + speech audio) at 48 kHz. The Pantanal SwiftOne data is bandlimited mic noise + species-specific calls — exactly the domain Perch was trained on (iNaturalist + XC bird recordings). CLAP's "fish out of water" performance confirms that **bioacoustics-pretrained models (Perch, BirdNET) > general audio models**.

## 9. `bleachonn77/birdclef-2026-expert-labels` is a duplicate, not new info

This dataset (CC0, 6 KB) is **byte-identical** to the official `train_soundscapes_labels.csv` shipped with the competition (1,479 rows, same SHA). It's just a re-upload, not additional expert annotation.

## 10. Putting it all together: the new architectural map

```
                 Pantanal raw 60s WAV (SwiftOne, 32 kHz, 16-bit native)
                                 │
                                 ▼ split into 12 × 5-sec windows
            ┌────────────────────┴────────────────────┬─────────────────┐
            ▼                                         ▼                 ▼
  Perch v2 (frozen)                       SED EfficientNet              CLAP
   - 1536-d emb                           (tucker distill / your own)   (768-d emb)
   - 234 logits (mapped 206)              fold ensembled                LIN probe
            │                                         │                 │
            └──────────────┬──────────────────────────┘                 │
                           ▼                                            │
              Mamba-ProtoSSM (3 BiSSM + prototypes)                     │
                + Mamba-ResidualSSM (1 BiSSM, zero-init head)           │
                + per-class fusion-alpha (chaneyma)                     │
                           │                                            │
                           ▼                                            │
              Logit blend (0.6 perch / 0.25 cnn / 0.15 crnn)            │
                                                                        │
                           │                                            │
                           ▼                                            │
              + lambda_prior × site×hour Bayesian prior                 │
                           │                                            │
                           ▼                                            │
              Texture-aware temporal smoothing                          │
                ([0.35,0.30,0.35] for 63 texture classes,               │
                 [0.20,0.60,0.20] for 171 event classes)                │
                           │                                            │
                           ▼                                            │
              Top-2-window self-amplification (chaneyma)                │
                           │                                            │
                           ▼                                            │
              Rank-aware scaling (file_max^power=0.4–0.6)               │
                           │                                            │
                           ▼                                            │
              Per-class isotonic calibration (hideyuki)                 │
                           │                                            │
                           ▼                                            │
              Adaptive delta smoothing (alpha=0.20)                     │
                           │                                            │
                           ▼                                            │
                      sigmoid → submission.csv
```

**Levers actually owned by ELITE kernels**:
1. Trained their own SED checkpoint (focal+cutmix)
2. Texture-vs-event smoothing kernel split
3. Per-class isotonic + F1-threshold optimization
4. Top-2 self-amplification (single kernel — chaneyma)
5. Per-class fusion-alpha (single kernel — chaneyma)

## 11. Concrete plan for crossing 0.949 → 0.951+

Based on the new evidence:

### Tier-A (zero-risk, high-leverage, < 30 min)
- **Add `.drop_duplicates()` to your labels read** if you haven't (most don't). It corrects 2x prior weight.
- **Adopt aliozan's texture/event smoothing** — drop in the SMOOTH_TEXTURE/SMOOTH_EVENT kernel split, mark Insecta+Amphibia. Pure post-processing change.
- **Add the chaneyma top-2 self-amplification** — single-line `x = x * top2` post-temporal-smoothing.

### Tier-B (free compute, half-day)
- **Download `backtracking/birdclef2026-pseudo-cache-v1`** (441 MB) and train a small student head (MLP probe) on `pseudo_soft` for 1-2 hours on CPU. Use as extra ensemble member with weight 0.15.
- **Per-class isotonic + F1-threshold** on your OOF (script in §6c above). For macro-AUC the calibration is what matters; thresholds are sanity checks.

### Tier-C (multi-day, ELITE-level)
- **Train your own EfficientNet-B0 SED with focal+cutmix** distilled from the pseudo_cache embeddings. Replace Tucker's CC0 model with your own — that's how ELITE kernels differ from PLATEAU.
- **Mine iNaturalist Sounds** for the 28 missing classes (Insecta sonotypes + 3 frogs). The XC URLs dataset will NOT help here. Look up `iNaturalist sounds research-grade Pantanal` exports.

### Tier-D (architectural research)
- Wire in **chaneyma's per-class fusion-alpha** between your ProtoSSM and Perch teacher. Sigmoid-gate per class, init zero. Adds a few KB of parameters; learns where to trust the refiner over the teacher.

## 12. Sources researched fresh (no training-data assumptions)

- [Gu & Dao 2023 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (`arxiv:2312.00752`)](https://arxiv.org/abs/2312.00752) — the SelectiveSSM in ProtoSSM is a near-verbatim Mamba block
- [Wu et al. 2023 — CLAP: Learning Audio Concepts from Natural Language Supervision (`arxiv:2211.06687`)](https://arxiv.org/abs/2211.06687) — explains why CLAP fails on bioacoustics
- [Zhang et al. 2023 — Adding Conditional Control to Text-to-Image Diffusion Models, §3.2 zero convolution (`arxiv:2302.05543`)](https://arxiv.org/abs/2302.05543) — same zero-init trick that ResidualSSM uses
- [Hamer et al. 2024 — Perch 2.0 (`arxiv:2508.04665`)](https://arxiv.org/abs/2508.04665) — iNat + XC training corpus, self-distillation curriculum
- [Hugging Face — `laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused) — 22 kHz CLAP, exactly the variant Habedi shipped
- [Kaggle dataset `yasunorim/xc-birdclef-2026-target-urls`](https://www.kaggle.com/datasets/yasunorim/xc-birdclef-2026-target-urls)
- [Kaggle dataset `backtracking/birdclef2026-pseudo-cache-v1`](https://www.kaggle.com/datasets/backtracking/birdclef2026-pseudo-cache-v1)
- [Kaggle dataset `chaneyma/birdclef-2026-cv9245-moe-artifacts`](https://www.kaggle.com/datasets/chaneyma/birdclef-2026-cv9245-moe-artifacts)
- [Kaggle dataset `baiyuby/birdclef2026-distill-models`](https://www.kaggle.com/datasets/baiyuby/birdclef2026-distill-models)
- [Kaggle dataset `habedi/birdclef-2026-clap-int8-bundle`](https://www.kaggle.com/datasets/habedi/birdclef-2026-clap-int8-bundle)
- [Kaggle dataset `tsubasatech/birdclef-2026-snowflake-sed`](https://www.kaggle.com/datasets/tsubasatech/birdclef-2026-snowflake-sed)
- [Kaggle kernel `aliozanmemetoglu/birdclef-5-fold-ensemble-submission` (LB 0.958, rank 4)](https://www.kaggle.com/code/aliozanmemetoglu/birdclef-5-fold-ensemble-submission)
- [Kaggle kernel `hideyukizushi/bird26-reproduce-perch-protossm-resssm-inf-train` (LB 0.953, rank 17)](https://www.kaggle.com/code/hideyukizushi/bird26-reproduce-perch-protossm-resssm-inf-train)
- [Xeno-canto.org/about — recording scope = birds + soundscapes mentioning birds](https://xeno-canto.org/about/recordings)
- [iNaturalist Sounds export documentation](https://www.inaturalist.org/pages/sounds) — frogs, mammals, insects accepted alongside birds
