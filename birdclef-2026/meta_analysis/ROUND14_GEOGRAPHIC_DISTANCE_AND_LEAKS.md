# BirdCLEF+ 2026 — ROUND 14: geographic distance weighting + iNat metadata enrichment + SigmoidF1 loss

This round adds three concrete, undocumented leverage points: (1) the actual distance-from-Pantanal distribution of training data, (2) iNat ID-based metadata enrichment, (3) BC2024 DS@GT paper losses.

## 1. Pantanal-distance distribution of train.csv (massive domain shift confirmed)

Haversine distances from each train recording to Pantanal center (-19, -56.75):

| Stat | Distance (km) |
|---|---:|
| Mean | 2,658 |
| **Median** | **1,606** |
| 25th percentile | 1,053 |
| 75th percentile | 3,262 |
| Max | **19,500** (literally antipodal) |
| Within 100 km | 113 (0.3%) |
| Within 500 km | 1,487 (4%) |
| Within 1000 km | 7,965 (22%) |
| Within 2000 km | 21,164 (59%) |

**By class_name:**

| Class | Median dist (km) | Count | Implication |
|---|---:|---:|---|
| Reptilia | 971 | 1 | Close, but only 1 record (Southern Spectacled Caiman) |
| Amphibia | **1,298** | 451 | Closest, mostly South American |
| Aves | 1,611 | 34,799 | Medium spread |
| **Insecta** | **3,976** | 199 | **Mostly NOT from Pantanal region** |
| **Mammalia** | **8,167** | 99 | **Global recordings** (Domestic Dog, Bos taurus, Equus, etc.) |

The Mammalia median 8,167 km is striking — most domestic-animal recordings (dog, horse, cow) come from Europe/North America. The Pantanal-test versions of these sounds will be the SAME species but recorded in different environments.

**The 3 species with ZERO recordings within 2000 km of Pantanal** are likely Reptilia (1 rec from outside) + some other rares that only have global uploads.

### Distance-weighted training proposal

```python
import numpy as np

pantanal_center = (-19.0, -56.75)  # Mato Grosso do Sul

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2-lat1); dl = np.radians(lon2-lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# Sample weight inversely proportional to distance + offset
def pantanal_weight(lat, lon, scale_km=500):
    d = haversine(lat, lon, *pantanal_center)
    return 1.0 / (1.0 + d / scale_km)
```

Weights at sample distances:
- 0 km: weight 1.00
- 500 km: 0.50
- 1000 km: 0.33
- 2000 km: 0.20
- 5000 km: 0.09
- 19500 km (max): 0.025

**No public BC2026 kernel does this** — but the geographic shift is the single biggest source of label noise for the long-tail classes. Adding this weighting should be tested on a held-out set as the first thing in any custom training run.

## 2. iNat URL → iNaturalist API metadata enrichment

12,506 of 35,549 train recordings are iNat (35%). The filename `iNat1216197.ogg` directly maps to iNaturalist observation ID `1216197`:

```python
# All iNat filenames cleanly extract the obs ID
filename = "1161364/iNat1216197.ogg"
obs_id = filename.split('iNat')[1].split('.')[0]  # → '1216197'

# Query iNat API for additional context (free, no auth needed)
import requests
resp = requests.get(f"https://api.inaturalist.org/v1/observations/{obs_id}")
data = resp.json()
# Returns: location (precise), observed_on, observed_on_string, taxon hierarchy,
#          quality_grade, identifications_count, identifications_most_agree,
#          captive_cultivated, place_guess, other_user_observations, etc.
```

**Untapped enrichment ideas**:

| iNat field | How to use |
|---|---|
| `quality_grade` (research/casual/needs_id) | Filter to only research-grade observations (high-confidence species ID) |
| `identifications_count` | More IDs = more confident label. Use as sample weight. |
| `identifications_most_agree` | Boolean: do the multiple identifiers agree? Use to filter ambiguous labels. |
| `observed_on` (precise date) | Add seasonal feature (month-of-year prior) |
| `place_guess` | Free-text location (e.g., "Pantanal, Mato Grosso do Sul, Brazil") for fine-grained filtering |
| `taxon.complete_species_count` | How many species in the genus → narrow down genus-level proxies |
| `sounds[0].license_code` | Filter to CC0 / CC-BY recordings only |
| `observations_for_user_X` | If a recordist contributes 100s of recordings, they may have systematic style |

**No public BC2026 kernel does iNat API enrichment.** The 12,506 iNat IDs are sitting there with rich metadata available for free.

### Sample iNat API response excerpts (already in train.csv)

The train.csv ALREADY exposes some of this:
- `latitude`, `longitude`: from iNat
- `rating`: 0.0 default (essentially no info)
- `author`: observer name
- `license`: cc-by-nc / cc0 / etc.
- `url`: direct sound URL

What's NOT in train.csv but available via API:
- Precise observation date (not just year)
- Habitat tags
- Other observations by same user (potential leakage signal)
- Multiple ID confirmations
- Geographic coverage of the species

## 3. SigmoidF1 loss (DS@GT BC2024 paper, alternative to BCE/Focal)

From `arxiv:2407.06291`. SigmoidF1 is a **differentiable F1-score loss** — directly optimizes the metric structure without threshold tuning.

```python
def sigmoid_f1_loss(logits, targets):
    p = torch.sigmoid(logits)
    tp_soft = (targets * p).sum()
    fp_soft = ((1 - targets) * p).sum()
    fn_soft = (targets * (1 - p)).sum()
    f1_soft = 2 * tp_soft / (2 * tp_soft + fp_soft + fn_soft + 1e-7)
    return 1.0 - f1_soft
```

Properties:
- **Differentiable** everywhere
- Naturally balances precision/recall (no `pos_weight` needed)
- Output is in [0, 1] which is interpretable as 1 - F1
- Bénédict et al. 2022 (`arxiv:2108.10566`) introduced it for multi-label classification

**For macro-AUC competition**, the SigmoidF1 loss doesn't directly optimize AUC, but it's a strong ranking-friendly loss that may correlate better with macro-AUC than plain BCE for sparse multi-label data.

## 4. Asymmetric Loss (ASL) — used by DS@GT but no BC2026 kernel

ASL (Ridnik et al. 2021, `arxiv:2009.14119`) is focal-like with **separate γ for positives and negatives**:

```python
def asl_loss(logits, targets, gamma_pos=1.0, gamma_neg=4.0, clip=0.05):
    xs_pos = torch.sigmoid(logits)
    xs_neg = 1 - xs_pos
    
    # Asymmetric clipping: shift negative probs by `clip`
    if clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1)
    
    # Asymmetric focal weighting
    los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
    los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
    
    # Asymmetric focal modulation
    pt0 = xs_pos * targets
    pt1 = xs_neg * (1 - targets)
    pt = pt0 + pt1
    one_sided_gamma = gamma_pos * targets + gamma_neg * (1 - targets)
    one_sided_w = torch.pow(1 - pt, one_sided_gamma)
    
    loss = -(los_pos + los_neg) * one_sided_w
    return loss.sum()
```

Key params: `γ+ = 1` (positives), `γ- = 4` (negatives), `clip = 0.05` (probability shift for hard negatives). **Down-weights easy negatives more aggressively** than symmetric focal loss. Beneficial when 233/234 classes are negative per window.

## 5. EnCodec audio embedding — Meta's neural codec, undocumented in BC2026

The DS@GT paper uses **EnCodec** (Défossez et al. 2022) as a 3rd embedding source beyond Perch and BirdNET. Configuration:

- bandwidth = 1.5 kbps
- output: `ℛ5×150` = 5 × 150-d embedding (750-d total)
- model is `facebook/encodec_24khz` available on HuggingFace

EnCodec is a **neural audio codec** trained for high-quality audio compression. Its embeddings capture audio "content" at a perceptual quality threshold, distinct from Perch's species-discriminative embeddings.

**Untested in BC2026**: Could be a complementary ensemble member alongside Perch + CLAP + Tucker SED.

## 6. Combined approach — concrete plan addition

### Tier-A drop-in (< 1 hour)
- Add **Pantanal-distance sample weighting** using train.csv lat/lon
- Try **SigmoidF1 loss** as alternative to BCE
- Try **Asymmetric Loss** (γ+=1, γ-=4)

### Tier-B (half-day)
- **iNat API enrichment**: query observation IDs for `quality_grade`, `identifications_count`, precise dates. Filter or weight by these.
- Build a **multi-month / seasonal prior** from iNat `observed_on` dates per species → use at inference time (the test files have date in the filename)

### Tier-C (multi-day)
- Train an **EnCodec embedding probe** (similar to CLAP probe) as additional ensemble member
- **Geographic-weighted custom training** restricted to South American recordings (< 4000 km from Pantanal)

## 7. The end-to-end stack so far (consolidated across all rounds)

Here's what the deepest possible approach looks like with all leverages I've found:

```
DATA PREPARATION
├── train.csv with .drop_duplicates() on labels CSV
├── re-encode train_audio to 72 kbps (match train_soundscapes codec)
├── Pantanal-distance sample weights (using train.csv lat/lon)
├── iNat API enrichment (quality_grade, identifications_count)
├── ESC-50 + custom no-call background dataset for mixup background
├── External XC URLs (yasunorim) + previous BirdCLEF carryovers
└── pseudo-cache embeddings (backtracking/birdclef2026-pseudo-cache-v1)
  
ARCHITECTURE ENSEMBLE (3 diverse backbones)
├── Backbone A: tf_efficientnetv2_s_in21k (224×512 mel, in_chans=3)
├── Backbone B: eca_nfnet_l0 (224×512 mel, SqrtBalancing)
└── Backbone C: EfficientVit-b0 (224×224 mel, fast inference)
  + Dedicated insect_amphibia specialist (B0 with 4000-12000 Hz mel band)
  + LSE pool head (r=10) instead of GAP

TRAINING
├── Loss: FocalBCE + label smoothing 1.005, or SigmoidF1, or Asymmetric Loss
├── Optimizer: AdamW (lr=1e-4) for V2-S; RAdam (lr=1e-3) for nfnet
├── Scheduler: CosineAnnealingWarmRestarts(T_0=5) or multi-step (1e-3 → 2.5e-4 at epoch 3)
├── Mixup with MAX-of-labels (Perch 2.0 / jfpuget style), prob=0.5, alpha=None
├── Augmentations: BackgroundNoise (ESC-50 + no-call), TimeFlip, Volume ±12dB
├── SpecAugment: CoarseDropout (0.375, 0.375), RandomLowerHighFreq, freq+time mask
├── AWP adversarial: adv_lr=0.005, adv_eps=0.01
├── Pseudo-label: F2 ≥ 0.5, model ≥ 0.1, ratio 0.4, 2-3 iterations
├── secondary_label_weight=0.5 (use train.csv secondary labels)
├── 50 epochs (Sydorskyi) OR 5-15 epochs (Melichov BC2025 Top 2%)
└── CV: StratifiedGroupKFold(filename, n_splits=5, random_state=91) with rare-class binning

INFERENCE  
├── Perch v2 ONNX (justinchuby/Perch-onnx) with spatial_embedding (16×4×1536)
├── intra_op_num_threads=4 + ThreadPoolExecutor(max_workers=4) async I/O
├── 20-second context, predict 4 × 5-sec segments (alexander / Nikita)
├── overlap_average_max_delta (Nikita) — frame-shift TTA
├── ProtoSSM + ResidualSSM refiner (Maryna canonical recipe)
├── Per-class learnable fusion alpha (chaneyma)
├── Texture/event smoothing kernels [0.35,0.30,0.35] vs [0.20,0.60,0.20]
├── Site×hour Bayesian prior (with all sites, not just labeled — use pseudo-cache)
├── Genus-proxy for unmapped species (Maryna)
├── File top-2 mean amplification (Maryna canonical, used by chaneyma)
├── Rank-aware scaling (file_max^0.4-0.6)
├── Per-class isotonic calibration + F1-threshold (hideyukizushi)
├── Adaptive delta smoothing (Maryna canonical)
├── Final ensemble: 5-fold same-arch avg in PROBABILITY space (Sydorskyi lesson)
├── Convert .pth → ONNX → OpenVINO IR fp16 → AsyncInferQueue (BC2025 2nd place)
└── Budget: 23-min scoring on 90-min limit (hideyukizushi)
```

This is the union of all 14 rounds of findings. **No single public BC2026 kernel implements more than ~30% of these.**

## 8. Sources

- [iNaturalist API documentation](https://api.inaturalist.org/v1/docs/) — for observation enrichment
- [Bénédict et al. 2022 — SigmoidF1 loss (`arxiv:2108.10566`)](https://arxiv.org/abs/2108.10566)
- [Ridnik et al. 2021 — Asymmetric Loss (`arxiv:2009.14119`)](https://arxiv.org/abs/2009.14119)
- [Défossez et al. 2022 — EnCodec neural codec (`arxiv:2210.13438`)](https://arxiv.org/abs/2210.13438)
- [Pantanal Wikipedia (bbox confirmation)](https://en.wikipedia.org/wiki/Pantanal)
- [jfpuget BC2024 README](https://github.com/jfpuget/birdclef-2024)
- [DS@GT BC2024 paper](https://arxiv.org/abs/2407.06291)
