# The Hidden Catch — Distribution Shift WITHIN the Soundscape Set

After multiple forensic passes, this is the finding that is **most consequential
and that none of the public 0.948+ kernels visibly address.**

## What I found

The 66 labeled `train_soundscapes` files are NOT a representative sample of
the 10,658 train_soundscapes total. They are **systematically cleaner audio**:

| metric (per 60-s file) | labeled (n=66) | unlabeled sample (n=200) | ratio |
| --- | ---: | ---: | ---: |
| mean RMS | **0.068** | 0.139 | **2.04× louder** |
| mean peak amplitude | 0.487 | 0.752 | 1.54× higher |
| 99th-percentile amplitude | 0.192 | 0.375 | 1.95× higher |
| samples clipped at >0.999 | **59** | **17,144** | **291× more** |
| samples clipped at >0.99 | 66 | 19,457 | **293× more** |
| samples clipped at >0.95 | 96 | 25,322 | 264× more |
| files with heavy clipping (>1% saturated) | **0 / 66** | **22 / 200 (11%)** | — |
| files with extreme clipping | none | up to **64% saturated** | — |

Verified end-to-end:
- Sampled 200 random unlabeled files + ALL 66 labeled files (no overlap)
- Decoded full 60-s PCM, computed clipping at three thresholds
- Worst unlabeled file: `BC2026_Train_1164_S01_20230218_040000.ogg` —
  **64.4% of samples are at full saturation** (1,237,435 of 1,920,000)
- Best labeled file vs worst labeled file: 0% to maybe 0.05% clipping

## Where the clipping lives

Per-site average % of samples clipped at >0.99 (unlabeled, sample only):

| site | avg % clipped | n labeled here |
| --- | ---: | ---: |
| **S01** | **2.30%** | **0** |
| **S13** | **2.28%** | 2 (lightly labeled) |
| S10 | 2.60% | 0 |
| S11 | 0.14% | 0 |
| S02 | 0.06% | 0 |
| S22 | 0.11% | 40 |

**S01 and S13 are the clipping-heavy sites** — and S01 has **zero** labels.
S13 has 1,873 train_soundscapes total but only 2 are labeled, and those 2
are from the **clean** portion of S13's recordings.

## Why this matters for your 0.949 submission

The test set is drawn from the same recorder deployments as `train_soundscapes`.
So a non-trivial portion of test files will look like the **unlabeled side**:
**louder, heavily clipped, S01/S13-style audio.**

When you train on:
- `train_audio` (clean XC/iNat at 86 kbps, no clipping)
- `train_soundscapes_labels.csv` (clean, no clipping, low RMS)

…the model has **NEVER SEEN** a 64%-saturated 60-second clip during training.
At inference time, Perch's embedding head will see an audio waveform where
high-amplitude frames are flat-topped at ±1.0 — a distribution it was never
trained on, since Perch was trained on standard iNat / xeno-canto data
(also non-saturated). Output embeddings on saturated audio collapse.

## What none of the top public kernels do

Read line-by-line, the 0.948 / 0.949 kernels
(`nina2025/eos-4`, `karnakbaevarthur/power-optimization`, `pilkwang/948-…`,
`safar1/lb-score-0-948`, `sunderekkiz/exp019-eos4-rank-power-06`):

- ✅ Have audio-augmentation in TRAINING (SpecAugment, mixup, time shifts)
- ❌ **Do NOT apply audio-saturation / clipping augmentation**
- ❌ **Do NOT detect clipped test soundscapes** and route them through a
  different pipeline (e.g., clip-robust embeddings, BirdNET-only, or PCM
  un-clipping)
- ❌ **Do NOT re-encode train_audio to 72 kbps** (the codec gap finding
  from the previous round)

## How to fix this — concrete actions

### Action 1: Clipping augmentation during training

Insert into your audio pipeline before mel-spectrogram computation:

```python
def random_clipping_aug(x: np.ndarray, p: float = 0.3) -> np.ndarray:
    """Random gain boost + hard clip. Mimics test-set saturation."""
    if np.random.rand() > p:
        return x
    # gain ratio sampled to roughly match unlabeled distribution
    gain = np.random.uniform(2.0, 8.0)
    x = np.clip(x * gain, -1.0, 1.0)
    return x
```

Apply with probability ~0.3 to training waveforms. This makes the model see
saturation distortion. Don't apply to validation — the OOF should still be
on clean audio so you can compare to the existing OOF dashboard.

### Action 2: Pre-amplify the LABELED training segments

For the 66 labeled-soundscape files specifically (they're 2× quieter than
unlabeled), apply a **fixed +6 dB gain** before extracting embeddings:

```python
labeled_x = np.clip(labeled_x * 2.0, -1.0, 1.0)
```

This brings them closer to the unlabeled (and test) RMS level.

### Action 3: Detect clipped test files at inference, route differently

At test time, count clipped samples per file. If a file has >1% clipping:

- Option A: De-clip first using cubic spline interpolation over clipped
  regions (`numpy.interp`)
- Option B: Skip ProtoSSM (Perch-based) for that file, use SED + BirdNET
  weights of (0, 0.5, 0.5) since both SED and BirdNET handle saturation
  better than Perch's whole-clip embedding head

### Action 4: A combined "domain bridge" augmentation

Combine actions 1+2+ the codec re-encoding from the previous round:

```python
# Pseudocode for training time
def augment(x):
    # 1. Random gain (sometimes inducing clipping)
    if random < 0.3:
        x = np.clip(x * uniform(2, 8), -1, 1)
    # 2. Codec round-trip to 72 kbps to match soundscapes
    if random < 0.5:
        x = codec_simulate(x, bitrate=72000)
    # 3. Standard SpecAugment etc afterwards
    return x
```

This bridges three distribution shifts at once:
- amplitude / saturation (this finding)
- codec quality (previous round)
- ambient context (handled by site/hour priors already)

## Expected impact on score

This is the kind of finding that moves macro-AUC by **+0.001 to +0.005** on
private LB if the test set really does include clipped files. Hard to verify
in advance, but the evidence is:

1. **22 of 200 unlabeled files are heavily clipped** — they exist in the data
2. **Test files are sampled from the same recorder deployments** (the rules
   confirm "some sites overlap between train and test soundscapes")
3. **The model has never seen clipping** under standard training
4. **None of the top 7 public kernels handle clipping** — so this is an open
   lane to exploit

If you implement Actions 1+2 above, on a CV with synthetic clipping injected
into the validation, I'd expect to see a measurable improvement specifically
on the **clipped-validation slice**. That's where you should A/B.

## Caveat — what I can't verify

- **I don't have the test set**, so I cannot directly measure clipping rate
  in test data. The argument is by analogy: test ≈ unlabeled distribution.
- **The labeled-vs-unlabeled clipping difference might be partly site-driven**
  rather than deliberate curation. Either way the model-side fix is the same.
- **The S15 site has the OPPOSITE pattern** (labeled RMS 0.039 vs unlabeled
  RMS 0.034 → labeled is slightly louder there). The clipping anomaly is
  driven by S01/S13/S10, which have no labels.

## Other smaller findings from this round (documented for completeness)

### Within-site PCM near-duplicates exist but are rare

Of 50 random within-(site, date) unlabeled pairs:
- 0/50 had |PCM correlation| > 0.5
- Median |corr| = 0.011 (random)

So the "internal duplicate" hypothesis from the previous round is **not
widespread** — only the specific S19 cases (idx 10610↔10611, 10578↔10615
etc.) are truly duplicate-like. The rest of train_soundscapes is recordings
of similar acoustic environments but distinct audio.

### Most train_soundscapes are NOT in long sessions

Using a 70-sec gap threshold to detect continuous recording sessions:
- **10,357 distinct sessions** from 10,658 files
- **Median session length: 1 file** (10,298 isolated files)
- Only 17 sessions of 6+ files; only 2 sessions of 21+
- So the recorder mostly saves isolated 60-s snippets, not continuous longer
  recordings

### Sample test file shares date with train_soundscapes

Sample submission row 1: `BC2026_Test_0001_S05_20250227_010002` (site S05,
date 2025-02-27, time 01:00 UTC).

Train_soundscapes at the **same site + same date**: 4 files at S05 on
2025-02-27, all at time 17:00:04. These could be the same recording
session sampled at a different hour — confirming that test sites/dates
DO overlap with train_soundscapes.

Note: this is the **example** row from the data description, not the real
test data, so the actual test file may or may not be at S05.

### Energy band differences (4-16 kHz is 3-5× louder in unlabeled)

Verified high-frequency energy is dramatically higher in unlabeled vs
labeled — consistent with the clipping finding (clipped audio creates
high-frequency harmonics).
