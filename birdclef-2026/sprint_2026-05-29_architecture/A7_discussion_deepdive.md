# A7 — Discussion Deep-Dive (deeper than A5's scan) — 2026-05-29

GOAL: concrete clues toward LB ≥ 0.96 (top 0.964; public plateau ~0.950).
Method: enumerated the BirdCLEF+ 2026 forum (`/discussion?sort=votes` + `?sort=hotness`) and
DEEP-READ each high-signal thread via the `r.jina.ai` text-extraction proxy (Kaggle forum pages are
JS-rendered → raw WebFetch sees only the title; the proxy renders them). Cross-read the BC2025 1st
(Nikita Babych), 2nd (VSydorskyy), 5th (myso1987) writeups + the Perch SFDA/NOTELA paper. The Kaggle
REST token still lacks `discussions.list` scope (404/HTML on every endpoint variant incl. internal
`discussions.DiscussionsService/*`); the proxy was the working path.

---

## A. FORUM INVENTORY (ranked by votes; competition forumId not exposed, IDs harvested from URLs)

| votes | cmts | thread | id | signal |
|---:|---:|---|---|---|
| 139 | 114 | [placeholder] Claude-Code Results for BirdClef2026 (Tom Capybara) | 681146 | HIGH — process log + CV-leak admission |
| 66 | 33 | An example of training process (HGNetV2-B0 Baseline) (Tawara) | 683822 | HIGH — CV↔LB gap numbers |
| 63 | 16 | warping perchv2 inside pytorch for training (hengck23) | 685318 | MED — trainable Perch head |
| 59 | 111 | Is everyone using LLM tools? (Chris Deotte) | 684207 | LOW — meta/noise |
| 50 | 169 | (Beginner Q&A — disc 681358) | 681358 | LOW |
| 42 | 10 | open KaggleClaw — let's build it (lhwcv) | 685794 | LOW — tooling |
| 37 | 17 | train_soundscapes_labels.csv has duplicated records (Tawara) | 681297 | MED — CV hygiene (every row dup'd 2×) |
| 36 | 8 | Sharing baseline LB.928 reproducible+fast (yukiZ) | 686457 | MED — public floor |
| 32 | 29 | (thread, 29 cmts) | ~ | — |
| 30 | 4 | Compare Inference Speed Torch/jit/OpenVINO (Tawara) | 689012 | HIGH — 90-min budget math |
| 28 | 14 | Most useful ideas come from me… AI (coolz) | 692610 | LOW |
| 25 | 19 | [placeholder] tricks in BirdCLEF+ 2026 (hengck23) | 684148 | HIGH — label-blocks + NOTELA |
| 24 | 8 | 5 dataset observations in 2026 Pantanal Data (OpPrime) | 683879 | HIGH — the 28 + inverse class dist |
| 18 | — | Export models to serialized formats (Kate Reed, organizer, pinned) | — | MED — working-note/format |
| 11 | 1 | How to get started + official Discord (organizer) | 680267 | LOW |
| 7 | — | Working Notes — how to submit (Stefan Kahl, organizer, pinned) | — | LOW |
| 3 | 35 | What's the limit without any Perch involved (sghwr) | 700763 | HIGH — Perch-free ceiling |
| ~2 | ~3 | How to label the unlabeled soundscapes? (OpPrime) | 694815 | MED — soft-PL consensus |
| 0 | 3 | ConvNeXt+EfficientNet ensembles worth it under 90min? | 703295 | LOW |
| 0 | — | Struggling with low AUC for Insecta in local validation (Keita Usui) | — | MED (signal: even the 25 insect sonotypes tank local CV for everyone) |
| — | — | Meet the hosts (organizer) | 680383 | LOW |

No organizer post discloses a method or per-class test breakdown. No "I found the secret" thread
contains a real disclosed technique (Tom's 681146 is a Claude-Code automation log, not a recipe).

---

## B. RANKED CLUES toward 0.96  [clue | source | actionable change | impact | confidence]

### CLUE 1 — Nikita's EXACT noisy-student recipe (the +0.03 historical jump), now fully captured
- **Source:** BC2025 1st writeup "Multi-Iterative Noisy Student Is All You Need"
  (kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n).
  Nikita is **LB #1 in 2026 at 0.964**, re-running this recipe.
- **The recipe (every number):**
  - **4 self-training iterations**, LB progression **0.909 → 0.918 → 0.927 → 0.930**.
  - **Power transform on pseudo-probs** (not a threshold): apply power **>1** to suppress noise while
    keeping confident signal. Per-iter powers: it1=1.0, it2=1/0.65, it3=1/0.55, it4=1/0.6
    (≈ p^1.54, p^1.82, p^1.67). This is the "power `p·(p>0.3)+p²`" idea but **cleaner: pure power,
    NO hard threshold** — our WAR_ROOM plan uses TH=0.3 which Nikita does NOT.
  - **Data mix:** stage-1 supervised (labeled only, 15 ep). Stages 2–4 self-train (25–35 ep) with
    **EVERY training sample MixUp'd with a random pseudo-labeled sample**, constant blend weight 0.5
    (Beta=∞). "100% pseudo-labeled mixup" beat partial ratios. **Soft labels throughout, labels NOT
    normalized to sum to 1.**
  - **Sampler:** WeightedRandomSampler, weight = sum of per-soundscape max-label probs.
  - **Loss = CrossEntropy** (beat BCE and Focal — "CE handles imbalanced labels better").
  - **Backbone stack (7):** EffNetB4, EffNetB3, 2×RegNetY-016, ECA-NFNet-L0, RegNetY-008,
    + **1×EffNetB0 dedicated Amphibia/Insecta model**. All share **SED head (2021 4th-place style) +
    GeM freq-pooling + 3× repeated mel as 3-channel input.**
  - **Inference:** OpenVINO (no quant), spectrogram reuse across models, neighbor-chunk averaging,
    smoothing kernel **[0.1,0.2,0.4,0.2,0.1]**, delta-shift TTA.
- **Actionable change:** Switch our NS engine from {TH=0.3 + p·(p>0.3)+p²} to **pure power-transform
  soft pseudo-labels (no threshold) + 100% pseudo×labeled MixUp at blend 0.5 + CE loss**. Run 4
  iterations. This is the validated path to +0.03.
- **Expected impact:** +0.018 → +0.021 over 4 iters (his absolute gain), realistically +0.01–0.02
  net for us on top of the 0.950 base. **Confidence: HIGH** (exact recipe, author is #1 in 2026).

### CLUE 2 — The 28 zero-train classes: Nikita's dedicated EffNetB0 + the Pantanal class-distribution inversion
- **Source:** BC2025 1st writeup (dedicated model) + OpPrime "5 dataset observations" (disc 683879).
- **Two stacked facts:**
  1. **OpPrime confirms the 28 = 25 Insect sonotypes + 3 Amphibians, ZERO focal train_audio.** BUT
     **labeled soundscape activity is INVERTED vs focal clips**: Amphibia **4,174** mentions, Insecta
     **1,136**, Aves only **824**. The classes Perch can't score are the *most abundant* in the
     soundscapes. → there IS abundant in-domain signal for the 28; it's just not in `train_audio`.
  2. **Nikita's fix (and it's cheap):** a **separate EffNetB0** trained on train + **Xeno-Canto clips
     <60s** for Amphibia (+113 spp) and Insecta (+544 spp), **17,844 samples / 700 spp, min 1
     sample/spp** (raising the minimum to 5 *degraded* it). 40 ep, bs128. **Predictions inserted into
     a zero matrix and blended** → "0.002–0.003 LB boost" *in 2025* — and 2025 had far fewer zero-train
     classes than our 28, so the 2026 headroom is larger (Riya's 1/234≈0.0043-per-class math).
- **Sonotypes are acoustic-signature defined** (rhythmic pulses / narrow freq bands, OpPrime obs #4)
  → a spectrogram CNN specialist + template/prototype matching is the *right* tool; Perch's
  species-semantic embedding is the *wrong* tool. This validates our B0-specialist column.
- **Actionable change:** Build the **EffNetB0 Insecta/Amphibia specialist** seeded from (a) cut 5s
  positive segments out of the labeled train_soundscapes (rich: 1,136 + 4,174 mentions), (b) any
  Xeno-Canto insect/frog audio for the 544+113 extra spp at min-1-sample, then (c) noisy-student
  pseudo-positives from the 10,592 unlabeled. Blend via zero-matrix insertion. **Use n_mels≥224 and
  large hop** (narrow-band calls — see CLUE 6).
- **Expected impact:** +0.005–0.015 (more than 2025's +0.003 because 28≫2025's gap). **Confidence:
  HIGH** on the method, MED on the magnitude.

### CLUE 3 — CV is genuinely anti-correlated with LB for EVERYONE (don't trust local OOF; trust public LB rank)
- **Source:** Tawara HGNetV2 baseline (disc 683822) + Nikita 1st writeup + Tom 681146.
- **Hard numbers (Tawara, same model, three heads):**
  | head | **CV** | **LB** |
  |---|---:|---:|
  | Linear | 0.9574 | 0.856 |
  | AttnSED | 0.9626 | 0.859 |
  | LSEHead+TTA | 0.9624 | **0.888** |
  → **Higher CV ≠ higher LB.** The +0.03 LB jump (0.859→0.888) came from a head whose CV was
  *slightly lower*. **Nikita explicitly: "did not find a good CV/LB correlation," validated EXCLUSIVELY
  on the public LB.** Tom (681146): "Keep improving" reached **0.999 CV via severe data leakage.**
- **This upgrades our WAR_ROOM gate.** Our plan's central pillar is "select by your own CV / LOSO,
  not public LB." **The #1 and the public baseline-builders do the OPPOSITE** — they trust public LB
  because local CV is a liar here (our own Spearman −0.157 says the same). The honest reconciliation:
  use LOSO only to *reject* leakage (kill any column whose gain is retrieval/self-label leak), but
  **rank-select final subs by public LB + worst-fold robustness, NOT by mean CV.**
- **Actionable change:** Demote "best-CV" as a selection criterion. Use the public LB (and an exp019
  rank-agreement check) as the primary ranker, LOSO only as a leak-detector. Of the 2 final subs, make
  one the **max-public-LB blend**, not a max-CV blend.
- **Expected impact:** avoids a shake-DOWN (picking a high-CV/low-LB sub). Indirect but **decisive**.
  **Confidence: HIGH** (three independent confirmations incl. the #1).

### CLUE 4 — NOTELA: source-free domain adaptation, purpose-built for exactly this (focal→soundscape) and shipped in the Perch repo
- **Source:** hengck23 tricks thread (disc 684148) recommends the Perch-repo SFDA paper; paper =
  "In Search of a Generalizable Method for SFDA," ICML 2023 (Boudiaf et al.); blog
  research.google/blog/in-search-of-a-generalizable-method-for-source-free-domain-adaptation;
  code in `chirp/projects/sfda` of google-research/perch.
- **What it is:** **NOTELA = NOisy student TEacher with Laplacian Adjustment.** It does denoising
  teacher-student PLUS manifold regularization: **enforce that feature-space NEAREST NEIGHBORS get
  similar pseudo-labels** (Laplacian/cluster assumption) while adding student noise. The paper's
  *headline benchmark is literally adapting a focal-trained bird classifier to passive geographic
  soundscapes* — our exact problem. It beats SHOT/Tent/NRC/DUST/plain-pseudo-label, which "collapse"
  on bioacoustic shift.
- **Why it matters here:** our noisy-student plan is plain self-distillation. NOTELA is a strictly
  better pseudo-label engine for *domain shift* (focal train_audio → 23-site Pantanal soundscape),
  AND its neighbor-consistency term is a principled way to propagate labels onto the **28 zero-train
  classes** that only exist in soundscapes (their nearest neighbors in embedding space carry the
  signal). It's a no-GPU-needed adaptation at pseudo-label time, runs on CPU embeddings.
- **Actionable change:** Replace/augment the pseudo-label generation step with the **NOTELA update**
  (Laplacian neighbor-smoothed pseudo-labels) on Perch/SED embeddings of the 10,592 unlabeled
  soundscapes, before feeding the student. Reference impl exists in `chirp/projects/sfda`.
- **Expected impact:** +0.005–0.015 over vanilla pseudo-labels (it's the SOTA on this exact shift).
  **Confidence: MED-HIGH** (perfectly matched method, but needs porting; impact on our base unproven).

### CLUE 5 — Perch-free ceiling is ~0.948; the marginal gains there are XC-pretrain + pseudo-label + sliding-window TTA
- **Source:** "What's the limit without Perch" (disc 700763): Arunodhayan (LB #9) **0.946 single model
  no Perch**; antoinemasq **0.943 EffNetV2-B0 → 0.948 ensemble**, recipe = **XC-pretrained EffNetV2_B0
  + pseudo-labeling + sliding-window TTA with smoothing**; MengYe: 0.90+ with HGNet + asymmetric loss.
- **Implication for orthogonality (Marcus's thesis):** a from-scratch SED/EffNet stack reaches ~0.946
  WITHOUT touching Perch → it is a **genuinely independent error structure** from our Perch column.
  This is the de-correlated branch we want to blend, and it's near-parity, so the blend gain is real
  (not the tonylica ρ~0.9 trap). antoinemasq's "XC-pretrain → pseudo-label → sliding-window TTA" is the
  cheap path to a strong non-Perch column.
- **Actionable change:** Ensure the noisy-student SED/EffNet column is trained **independent of Perch**
  (no Perch features as input) so the blend is orthogonal; pretrain the backbone on Xeno-Canto first.
- **Expected impact:** the orthogonal blend partner worth +0.003–0.008 on top of Perch.
  **Confidence: MED-HIGH** (two LB-top authors quote ~0.946–0.948 Perch-free).

### CLUE 6 — Mel front-end for narrow-band Insect/Amphibia calls: high n_mels + large hop, 20s chunks
- **Source:** Nikita 1st writeup (rationale verbatim): n_mels=**224**, n_fft=**4096**, hop=**1252**,
  fmax=16k, top_db=80, output (3,224,512), **20-second chunks** (beat 5/10/15/30). Reason: *"some
  species (especially Amphibia and Insecta) have calls within narrow frequency ranges"* → needs more
  mel bins + larger hop. Tawara's HGNetV2 used n_mels=256/n_fft=2048 and also did well on CV.
- **Contrast with our pipeline:** the public 0.950 stack and our windows are **5s / 12-window**. For
  the 28 narrow-band insect/frog targets specifically, a **higher-resolution mel + longer (≥20s)
  context** is materially better; the metric is per-5s but the *specialist* can score longer context
  then map back.
- **Actionable change:** Train the B0 Insect/Amphibia specialist (CLUE 2) on **224–256 mel bins,
  n_fft 4096, hop ~1250, 20s chunks**, not the 5s/64-128-mel front-end used for Aves.
- **Expected impact:** lifts per-class AUC on the 25 insect sonotypes (compounds CLUE 2). +0.002–0.006.
  **Confidence: MED** (specific to the specialist column).

### CLUE 7 — Label-block structure: co-occurring sonotypes come in long continuous blocks (exploit for the 28 + post-proc)
- **Source:** hengck23 tricks (disc 684148): labels come in *"longer continuous blocks"* with the
  *"exact label across 10…40-sec intervals,"* and **sonotypes co-occur in correlated groups** (his
  example: `47158son13;son17;son22;son23;son25` all together). OpPrime obs #2: every label row is
  **duplicated exactly 2×** (1,478 rows / 739 unique).
- **Two actions:**
  1. **Temporal post-proc:** because true events persist 10–40s, **median/max-pool a class's
     probability across a ±N-window block within a file** (stronger than the [.1,.2,.4,.2,.1] kernel
     for the slow insect/frog drones). Cheap, inference-only.
  2. **Co-occurrence prior for the 28:** the insect sonotypes fire in correlated clusters → a learned
     **sonotype co-occurrence / max-pool across the visually-similar son-group** (our `sonotype_mirror`)
     is justified by the data, and can lift a whole correlated group when one member is detected.
  3. **CV hygiene:** dedupe the 2× duplication before any fold split or you double-count.
- **Expected impact:** +0.001–0.004 (post-proc), de-risks CV. **Confidence: MED.**

### CLUE 8 — Inference budget: OpenVINO ≈2× Torch but slightly LOWER scores; jit-trace is the accuracy-safe speedup
- **Source:** Tawara speed thread (disc 689012). 7,200 windows (600 files × 12): Torch ~1m36s,
  **torch-jit-trace ~1m23s**, OpenVINO ~57.6s (8 workers). Caveat verbatim: *"the outputs of OpenVINO
  differ from those of Torch, and the scores may be slightly worse… better to use torch-jit-trace
  rather than OpenVINO if your models do not take so long."* (Nikita DID use OpenVINO, no quant, and
  won — so OV is fine if you can't otherwise fit the ensemble.)
- **Actionable change:** Inference time is NOT the binding constraint (a single model is ~90s, so the
  90-min budget fits ~50+ model-passes). **Prefer torch-jit-trace for accuracy; reserve OpenVINO only
  if the ensemble would otherwise overflow 90 min.** Don't pay the OV accuracy tax prematurely.
- **Expected impact:** avoids a silent score loss; frees us to ensemble more models. **Confidence:HIGH.**

### CLUE 9 — Trainable Perch-in-PyTorch only reaches ~0.885–0.889 (the missing gated-fusion head is NOT a breakthrough)
- **Source:** hengck23 "warping perchv2" (disc 685318): ONNX-Perch wrapped in PyTorch, stop-gradient
  MLP head → **LB 0.889**; direct spatial-feature MLP → 0.885. Lixin73 confirms ONNX↔Torch embedding
  cosine = 0.99999999. Outputs `[B,16,4,1536]` spatial + `[B,1536]` global.
- **Implication:** our "missing trained Perch-logit gated-fusion" gap is **low-value** — fine-tuning a
  head on Perch embeddings caps ~0.889 on its own; the 0.950 comes from the ProtoSSM/SED wrapper, not
  from making Perch trainable. **Do NOT spend the GPU window on the gated-fusion head.** (Useful only
  as the spatial-feature input `[B,16,4,1536]` for a downstream SED/SSM, which we already approximate.)
- **Expected impact:** negative (a time-sink to avoid). **Confidence: MED-HIGH.**

### CLUE 10 — SoftAUCLoss (5th place) directly optimizes the ranking metric and resists overfit on soft labels
- **Source:** BC2025 5th (myso1987) overview: **SoftAUCLoss** = pairwise prob differences + log-loss,
  *"resistant to overfitting, supports soft labels."* The metric is macro-ROC-AUC (pure ranking) →
  a pairwise-AUC surrogate is more aligned than BCE/CE for the *final* head, and tolerates the soft
  pseudo-labels from noisy-student.
- **Actionable change:** For the student/ensemble head, try **SoftAUCLoss as an auxiliary or final
  loss** (Nikita used CE for training stability; a pairwise-AUC fine-tune on top may add a hair).
- **Expected impact:** +0.001–0.003. **Confidence: LOW-MED** (worth a cheap ablation, not core).

---

## C. WHAT CONTRADICTS / UPGRADES OUR CURRENT PLAN

1. **UPGRADE the NS recipe (CLUE 1):** drop the `TH=0.3` hard threshold from WAR_ROOM item 1. Nikita
   uses **pure power-transform soft labels, no threshold, + 100% pseudo×labeled MixUp at 0.5 + CE.**
   Our planned `0.7·pseudo+0.3·hard` blend is also NOT what won — he MixUps full pseudo with full
   labeled, doesn't convex-blend the targets. Align to his exact recipe.
2. **CONTRADICTS our "select by CV not LB" pillar (CLUE 3):** the #1 author and the public-stack
   builders found **CV anti-correlates with LB** (Tawara's table is damning: +CV, −LB). Use LOSO only
   to *detect leakage*, and **rank final subs by public LB + worst-fold**, not by mean CV. Our own
   Spearman −0.157 already agreed; this confirms it across the field.
3. **NEW LEVER not in any of our docs (CLUE 4): NOTELA / SFDA** — the SOTA method for *this exact*
   focal→soundscape shift, in the Perch repo, recommended by hengck23. Strictly better pseudo-labeler
   than vanilla self-distillation AND a principled neighbor-propagation route to the 28. Port it.
4. **DE-PRIORITIZE the Perch gated-fusion head (CLUE 9):** WAR_ROOM notes it as "a known gap";
   evidence says trainable-Perch caps ~0.889, so it's not the gap that matters — don't burn GPU on it.
5. **CONFIRMS our B0-specialist + sonotype-mirror plan (CLUES 2,6,7)** with exact hyperparameters and
   the surprising **class-distribution inversion** (the 28 are the *most abundant* in soundscapes →
   the headroom is real and reachable, not a lost cause).
6. **Inference: jit-trace > OpenVINO for accuracy (CLUE 8)** — don't pre-pay the OV accuracy tax.

## D. ON A PUBLIC ASSET/LEAK STILL UNDEREXPLOITED
- No NEW public weight leak surfaced in the forum beyond what A5 already mapped (ali SED folds,
  tonylica model, sgkfk, raw-pseudos). The forum's underexploited "asset" is **methodological**, not a
  weight file: the **NOTELA reference implementation** (`chirp/projects/sfda`) and **Nikita's full
  2025 inference notebook** (`nikitababich/birdclef2025-1st-place-inference`, referenced by 7 kernels)
  — the latter is the literal source of CLUE 1's recipe and is reusable as a code base.
- The **Xeno-Canto Amphibia/Insecta data** (Nikita's 17,844-sample <60s set, 544 insect + 113 amphibian
  spp) is an external-data lever for the 28 that nobody in the 2026 public crowd is using.

---

### Sources (threads deep-read this scan)
- BC2026 forum: disc 681146, 683822, 685318, 684148, 683879, 700763, 689012, 694815, 684693, 681297
  (via r.jina.ai proxy of kaggle.com/competitions/birdclef-2026/discussion/<id>).
- BC2025 1st: kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
- BC2025 2nd: github.com/VSydorskyy/BirdCLEF_2025_2nd_place (+ ceur-ws.org/Vol-4038/paper_256.pdf)
- BC2025 5th: github.com/myso1987/BirdCLEF-2025-5th-place-solution (SoftAUCLoss)
- NOTELA: ICML 2023 Boudiaf et al., arxiv 2302.06658; research.google SFDA blog; google-research/perch
  `chirp/projects/sfda`.
