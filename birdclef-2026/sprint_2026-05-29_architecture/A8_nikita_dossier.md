# A8 — Nikita Babych Dossier (the BirdCLEF+ 2026 #1) — 2026-05-30

GOAL: go BROADER + DEEPER than A7 on the person behind the 0.965 public-LB #1. Profile, full
BirdCLEF track record, what his 2026 #1 run actually is (he's published almost nothing public for
2026 — this dossier separates confirmed-disclosed from inferred), and a ranked, transferable
technique list to push us 0.950 → 0.96.

Method: Kaggle REST (Bearer) for kernels/datasets/leaderboard (worked); `r.jina.ai` proxy for the
JS-rendered Kaggle profile / writeups / discussion (worked); WebSearch for LinkedIn/identity (LinkedIn
itself returns HTTP 999 to bots, so identity facts are corroborated via search snippets + the proxy).
Tags: **[CONFIRMED]** = directly read from a primary source; **[INFERRED]** = reasoned, not stated.

---

## (a) PROFILE — who he is

| Field | Value | Source / tag |
|---|---|---|
| Kaggle handle | `nikitababich` | kaggle.com/nikitababich [CONFIRMED] |
| Real name | Nikita Babych | [CONFIRMED] |
| Kaggle tier | **Competitions Master** (NOT grandmaster yet) | search snippet of profile + LinkedIn self-desc "Kaggle Competitions Master & Solo Competition Winner" [CONFIRMED] |
| Competition medals | **9 total: 3 gold, 4 silver, 2 bronze** | proxy of /nikitababich/competitions [CONFIRMED] |
| Highest rank | **1st / 2031 — BirdCLEF+ 2025** (solo) | same [CONFIRMED] |
| Country | **Ukraine** (Kyiv; from Kharkiv) | 2025 writeup "final submissions from a shelter"; LinkedIn ua/Kyiv [CONFIRMED] |
| Employer | **Bridgewise** (fintech/AI equity-research, IL/UA) — Data Scientist / DS Engineer | LinkedIn snippet [CONFIRMED] |
| Education | **V. N. Karazin Kharkiv National University** | search snippet [CONFIRMED] |
| LinkedIn | linkedin.com/in/nikita-babych-ab406622a (also ua.linkedin.com/in/nikita-babych) | [CONFIRMED] (page blocks bots) |
| GitHub | **None found for the Kaggle competitor.** `github.com/Nikita-Babich` is a different person (recreational C++/chess/esolang, 0 audio/Kaggle repos) — do NOT treat as his. He ships code via Kaggle kernels, not GitHub. | [CONFIRMED he has no public BirdCLEF GitHub] |
| Twitter/X / personal site | none found | [CONFIRMED absent] |

**Full competition record** (proxy of /competitions, 19 comps):

| Comp | Rank | Medal | domain |
|---|---|---|---|
| BirdCLEF+ 2025 | **1 / 2031** | 🥇 | audio SED |
| Eedi – Mining Misconceptions in Math | **6 / 1446** | 🥇 | LLM / retrieval (NLP) |
| Jigsaw – Agile Community Rules | **10 / 2445** | 🥇 | NLP classification |
| Child Mind Inst. – Detect Sleep States | 16 / 1877 | 🥈 | time-series |
| RSNA 2024 Lumbar Spine | 19 / 1874 | 🥈 | medical imaging |
| HMS – Harmful Brain Activity | 20 / 2767 | 🥈 | EEG / spectrogram |
| Santa 2023 (Polytope Puzzle) | 27 / 1054 | 🥈 | optimization |
| CSIRO Image2Biomass | 88 / 3805 | 🥈 | tabular/CV |
| CZII CryoET | 63 / 931 | 🥉 | 3D CV |
| Deep Past (Akkadian→EN) | 257 / 2674 | 🥉 | NLP |
| **BirdCLEF 2024** | **270 / 974** | — (unmedaled) | audio |
| (ISIC 2024, AES 2.0, Drawing-with-LLMs, MABe, etc.) | various | — | mixed |

**ML style read [INFERRED but well-supported]:** A versatile, *self-training/ensemble-first*
practitioner who wins across modalities — audio SED, LLM-retrieval (Eedi/Jigsaw gold), medical
spectrogram/imaging (HMS, RSNA), time-series (CMI). Strengths: pseudo-labeling / noisy-student,
heavy multi-backbone ensembling, inference-budget engineering (ONNX/OpenVINO). **BirdCLEF 2024 he
was only 270th** — the multi-iterative noisy-student recipe is what took him from mid-pack to #1 in
2025, and he is re-running it in 2026. He works alone (solo wins) and validates on public LB, not CV.

---

## (b) BIRDCLEF TRACK RECORD

- **2023:** not in his competition list → **did not compete** [CONFIRMED absent].
- **2024:** **270 / 974, unmedaled** [CONFIRMED]. No public writeup. (His public "Basic EDA +
  EfficientNet-B0 [LB 0.12]" notebook, 124 votes, is from the 2024 ISIC-era, not a strong bird sub.)
- **2025: 1st / 2031 SOLO** — "Multi-Iterative Noisy Student Is All You Need." This is the canonical
  recipe (already in A7; expanded with new precision in (d) below). Primary sources:
  - Writeup: kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
  - Inference kernel: kaggle.com/code/nikitababich/birdclef2025-1st-place-inference (Apache-2.0, 154 votes, runs ~1m57s, OpenVINO+ONNX ensemble, private 0.93067)
  - YouTube "Kaggle Winners Walkthroughs: BirdCLEF 2025 with Nikita Babych" (youtube.com/watch?v=jivW1JBxV8s)
- **Consistent across his bird solutions:** SED head (2021 4th-place style) + GeM freq-pooling +
  3×-repeated mel as 3-channel; multi-backbone EffNet/RegNet/NFNet ensemble; CE loss; MixUp@0.5;
  noisy-student pseudo-labels with power transform (no threshold); validate on public LB only;
  OpenVINO/ONNX inference with neighbor-chunk smoothing + delta-shift TTA; a dedicated small model
  for the non-Aves / zero-train classes; Xeno-Canto external data for rare taxa.

---

## (c) HIS BIRDCLEF+ 2026 #1 APPROACH (right now)

**Standing [CONFIRMED, REST leaderboard 2026-05-30]:** Rank **#1, public 0.965** (the brief said
0.964; the live API shows his best sub = **0.965**, 369 entries). Top-10 is tight:
1. Nikita Babych 0.965 · 2. Yannan Chen 0.963 · 3. Ali Ozan Memetoglu 0.963 · 4. "more exp is all
you need" 0.962 · 5. BirdCLEF+ 2026 Team 0.962 · 6. ggkush-anilclaw 0.961 · 7. YK 0.961 ·
8. Sinan Calisir 0.960 · 9. Arunodhayan 0.960 · 10. Takoi 0.959. Public LB is **~34% of test**.

**What is PUBLIC for 2026:** essentially nothing. He has **0 public kernels and 0 public datasets
tagged to birdclef-2026** [CONFIRMED via `kernels/list?competition=birdclef-2026` → count 0; his
datasets/list shows only 2025 assets]. No 2026 writeup exists (he never writes one until a comp ends).

**What he HAS disclosed for 2026 [CONFIRMED, his own forum comments 2 days ago, ~2026-05-28]:**
- He is running an **ensemble of 2 seeds**, not single models.
- **~0.935 LB without pseudo-labels → ~0.955 LB with pseudo-labels.** This is the single most
  important number in this dossier: **pseudo-labeling alone is worth ~+0.020 for him in 2026**,
  same magnitude as his 2025 noisy-student gain (0.909→0.930).
- He **has not yet tried LLM-agent automation** ("metadata algorithm dev + iterating over models for
  ensembles") — i.e. his 0.965 is hand-built, classic pipeline, with headroom he himself hasn't tapped.

**How 0.965 differs from the ~0.950 public plateau [INFERRED, high-confidence]:** the public plateau
sits at the *post-pseudo-label single-pipeline* level (~0.948–0.955). His extra ~0.010–0.015 over the
plateau is **the same delta his 2025 stack had over the field**: (i) *multi-iteration* noisy-student
(not one pseudo-label pass), (ii) the **7-way heterogeneous backbone ensemble** with SED+GeM, (iii)
the **dedicated zero-train (Insecta/Amphibia) specialist** blended via zero-matrix insertion, (iv)
inference polish (neighbor smoothing + delta-shift TTA). The 0.935→0.955 he quoted is *one* pseudo
pass; iterating it (his 2025 signature) is what separates the plateau from his #1. **His progression
version-history is not visible** (private kernels), so per-version deltas can't be scraped; the
0.935 / 0.955 / 0.965 waypoints above are the only disclosed progression.

---

## (d) RANKED ACTIONABLE TECHNIQUES — copy these to push 0.950 → 0.96

Format: technique | source | how to apply to us | expected impact | confidence.
(Hyperparameters below are the *exact* 2025 values, now confirmed verbatim from the writeup — several
are MORE precise than A7's capture: optimizer, scheduler, drop_path, XC counts, padding scheme.)

1. **Multi-iteration noisy-student, soft power-transform PL, NO threshold** —
   src: BC2025 1st writeup + his 2026 forum comment ("0.935→0.955 with PL").
   How: replace any single-pass / thresholded pseudo-labeling with **4 self-train iterations**, each
   regenerating pseudo-labels from the current ensemble; apply **power transform** to pseudo-probs with
   per-iter powers **it1=1.0, it2=1/0.65, it3=1/0.55, it4=1/0.6** (≈ p^1.0 / p^1.54 / p^1.82 / p^1.67);
   **never** normalize labels to sum-to-1; **CrossEntropy** loss. His LB per iter: 0.909→0.918→0.927→0.930.
   Impact: **+0.015–0.020** (his quoted 2026 PL gain is +0.020). Confidence: **HIGH**.

2. **100% pseudo×labeled MixUp at fixed blend 0.5** — src: BC2025 1st writeup.
   How: in self-train stages, MixUp EVERY training sample with a random pseudo-labeled soundscape
   sample, **constant weight 0.5 (Beta=∞)** — not random Beta, not a partial ratio. In supervised
   stage, MixUp p=0.5 on absmax-normalized raw audio, equal species sampling weight.
   Impact: part of the +0.02 above; ablating it costs him materially. Confidence: **HIGH**.

3. **7-model heterogeneous SED ensemble: EffNet-B0/B3/B4 + RegNetY-008/016(×2) + ECA-NFNet-L0**,
   all with **SED head (2021 4th-place) + GeM freq-pool + 3×-repeated mel** — src: BC2025 1st writeup.
   How: add backbone diversity beyond our current stack; equal ensemble weights won his best private
   sub. Exact timm ids: `tf_efficientnet_b0/b3/b4.ns_jft_in1k`, `regnety_008.pycls_in1k`,
   `regnety_016.tv2_in1k`, `eca_nfnet_l0.ra2_in1k`.
   Impact: **+0.005–0.010** (ensemble diversity over a single backbone). Confidence: **HIGH**.

4. **Dedicated zero-train specialist (the 28 in 2026): EffNet-B0 on train + Xeno-Canto <60s** —
   src: BC2025 1st writeup (Insecta/Amphibia model). How: train a *separate* B0 on
   **17,844 samples / 700 spp, min 1 sample/spp, 40 ep, bs128** = competition data + XC clips
   (**16,218 Insecta / 544 spp + 979 Amphibia / 113 spp**, max 200/spp, <60s). **Insert its
   predictions into a zero matrix and blend.** Raising min-samples to 5 *degraded* it — keep min=1.
   Impact: **+0.005–0.015** for us (2026 has 28 zero-train vs fewer in 2025 → larger headroom). Conf: **HIGH** method / MED magnitude.

5. **Mel front-end tuned for narrow-band Insecta/Amphibia + 20s chunks** — src: BC2025 1st writeup.
   How: **n_mels=224, n_fft=4096, hop=1252, fmax=16k, top_db=80, sr=32k, 0–1 norm, 20-second chunks**
   (beat 5/10/15/30s), output (3,224,512). **Pad shorter clips on the LEFT with 0 so the right always
   overlaps**; at inference center the first/last chunk and drop padding-region predictions.
   Impact: **+0.002–0.006** (compounds #4). Confidence: MED-HIGH.

6. **WeightedRandomSampler by soundscape pseudo-label confidence** — src: BC2025 1st writeup.
   How: sample weight = **sum of per-soundscape max-label probs** → prioritizes soundscapes whose
   pseudo-labels are reliable. Impact: +0.002–0.005 (stabilizes self-training). Confidence: MED-HIGH.

7. **Inference polish: neighbor-chunk averaging + smoothing kernel [0.1,0.2,0.4,0.2,0.1] + delta-shift
   TTA** — src: BC2025 1st writeup + inference kernel. How: average framewise preds across overlapping
   neighbor chunks, convolve a class's time-series with the 5-tap kernel, add delta-shift TTA (2023
   2nd place). Reuse one spectrogram across all models. Impact: +0.002–0.005, free at inference. Conf: HIGH.

8. **Training schedule (exact, now confirmed):** AdamW, **wd=1e-4**, LR **5e-4 → 1e-6**,
   **CosineAnnealingWarmRestarts (5-ep cycles)**, **drop_path_rate=0.15**, bs=64 (128 for the
   specialist), 5 folds min-1-sample/label, stage-1 supervised 15 ep, self-train stages 25–35 ep.
   src: BC2025 1st writeup. Impact: baseline reproduction fidelity. Confidence: HIGH.

9. **Validate on PUBLIC LB, not CV** — src: writeup ("no CV/LB correlation found"). How: use our LOSO
   only as a leak-detector; rank/select final subs by public LB + worst-fold robustness. (Reinforces
   A7 CLUE 3; his quoted 2026 single numbers are all LB, never CV.) Impact: avoids a shake-DOWN. Conf: HIGH.

10. **OpenVINO (no quant) only if the ensemble overflows 90 min; else torch-jit-trace** — src: writeup
    (he used OpenVINO, no quant) + A7 CLUE 8 (OV slightly worse scores). His 7-model run = ~1m57s.
    How: don't pre-pay the OV accuracy tax; reserve for fitting a big ensemble. Impact: avoids silent
    score loss / enables more models. Confidence: HIGH.

---

## (e) PUBLICLY-USABLE ASSETS (exact slugs)

All are **2025** assets (nothing public for 2026). License where known.

| slug | type | size | license | use |
|---|---|---|---|---|
| `nikitababich/birdclef2025-1st-place-inference` | kernel (Apache-2.0) | — | open | **Reference inference codebase**: SED model class, spectrogram code, ensemble + smoothing + delta-shift TTA, OpenVINO/ONNX loading. Adapt directly. |
| `nikitababich/birdclef2025-1st-place-ensemble` | dataset | 436 MB | Unknown | The 7 trained 2025 model weights (ONNX/OpenVINO). Class set is 2025-specific (won't transfer to 2026 labels) but the **architectures/heads** are reusable. |
| `nikitababich/birdclef2025-1st-place-extra-data` | dataset | **7.47 GB** | **CC0** | **The Xeno-Canto external data** — Insecta(16,218/544spp) + Amphibia(979/113spp) <60s clips + target-species clips. **Directly usable** to seed our 28-class specialist (CC0, no restriction). HIGHEST-VALUE asset. |
| `nikitababich/runtimes-onnx-openvino` | dataset | 97 MB | Unknown | ONNX/OpenVINO runtime wheels for offline (internet-off) inference. Convenience only. |

Not public / not available: any 2026 kernel, 2026 dataset, 2026 weights, his version history, any
GitHub repo. His 2026 #1 pipeline weights are private.

---

### Sources
- Leaderboard (REST): api/v1/competitions/birdclef-2026/leaderboard/view → Nikita Babych 0.965 #1, 369 entries.
- Profile / competitions (proxy): r.jina.ai/https://www.kaggle.com/nikitababich/competitions (9 medals, Master, 19 comps).
- Identity: LinkedIn nikita-babych-ab406622a (HTTP 999 to bots; corroborated via search) — Bridgewise, Kyiv, Karazin Kharkiv Univ.
- 2025 writeup: kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n (full recipe, hyperparams).
- 2025 inference kernel: kaggle.com/code/nikitababich/birdclef2025-1st-place-inference.
- 2026 self-disclosure: r.jina.ai/https://www.kaggle.com/nikitababich/discussion ("0.935 w/o PL → 0.955 w/ PL, 2-seed ensemble; hasn't tried LLM agents yet").
- Datasets (REST): api/v1/datasets/list?user=nikitababich (4 slugs above).
- YouTube walkthrough: youtube.com/watch?v=jivW1JBxV8s.
- Builds on A7_discussion_deepdive.md (this sprint dir) — not re-derived.
