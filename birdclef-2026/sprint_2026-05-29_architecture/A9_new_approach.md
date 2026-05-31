# A9 — New Approach + Tunnel-Vision Check (fresh, LB→0.96) — 2026-05-31

GOAL: a NEW noisy-student-LIKE method recently surfaced + a tunnel-vision check on what we're
missing, given our failures (NS-distillation students correlate with the public SED → hurt; 28
son-IDs unreachable; V237 base at the 90-min cliff; best=0.950).

METHOD this scan: pulled FULL SOURCE of the current top public kernels via authenticated Kaggle REST
(`kernels/pull`, Bearer token — the reliable path; the r.jina proxy is IP-blocked on the
discussion-DETAIL API but renders the discussion HTML pages with header `x-respond-with: markdown`).
Pulled the real NOTELA implementation from the archived Perch snapshot tag. Live leaderboard via REST.

CONFIDENCE TAGS: **[CONFIRMED]** read from primary source (code/forum/REST this scan); **[INFERRED]**.

---

## 0. HEADLINE REFRAME — we are AT the public ceiling, not below it [CONFIRMED]

- **Live LB 2026-05-31 (REST):** Nikita **0.966** (up from 0.965), Yannan Chen 0.963, Ali Ozan
  Memetoglu 0.963. Public LB ≈ 34% of test. Competition **ends ~June 4 ("4 days to go")**.
- **The highest LB any PUBLIC kernel claims is 0.948** (`youssefmo942009/lb-0-948` 131 votes ran today;
  `safar1` 0.948; `mtoshidesu` 0.947 "Public Pipeline Reproduced"; several 0.946). **Nothing public
  has crossed 0.95, let alone 0.952.** [CONFIRMED — scanned all 80 top-voted kernel titles + pulled
  source of the top 5.]
- ⇒ **Our 0.950 (V237/EoS9 replay) is already ~+0.002 ABOVE the best public kernel.** The 0.95→0.966
  zone is entirely private (Nikita-tier). The brief's framing ("stuck at the 0.950 public ceiling")
  is slightly off: there is no public 0.95 ceiling we're under — we're at/above the public frontier
  and the gap to chase is the *private* methodology, not a public kernel we failed to match.

---

## (a) THE NEW NOISY-STUDENT-LIKE METHOD: **NOTELA** (now with the exact code/recipe) [CONFIRMED]

### What it EXACTLY is (read from the real source, not the paper abstract)
Source code: `google-research/perch` **archived snapshot tag** `sfda-codebase-snapshot`,
`chirp/projects/sfda/methods/notela.py` (+ `configs/notela.py`). The code is NOT on `main` — it was
archived; raw URL that works:
`raw.githubusercontent.com/google-research/perch/sfda-codebase-snapshot/chirp/projects/sfda/methods/notela.py`
Paper: Boudiaf et al., "In Search of a Generalizable Method for SFDA," ICML 2023 (PMLR v202);
blog research.google/blog/in-search-of-a-generalizable-method-for-source-free-domain-adaptation.

NOTELA = **NOisy TEacher-student with Laplacian Adjustment**. It is **test-time / source-free domain
adaptation**: at inference, with NO labels, it adapts the model on the *test soundscapes themselves*.
The teacher-step pseudo-label rule (verbatim Eq.3, `teacher_step`, line ~190) is:

```
pseudo_label_i  =  p_i^(1/alpha)  ·  exp( (lambda_/alpha) · (W·P)_i / denom )      then row-normalize
```
where `p_i` = current model softmax on sample i, `W` = mutual-kNN adjacency in **embedding space**,
`P` = the matrix of all samples' probabilities. The student then matches these pseudo-labels through a
**noisy (dropout) forward pass**, updating **ONLY BatchNorm params** (`TrainableParams.BN`, Adam,
lr=1e-4). Offline mode recomputes pseudo-labels once per epoch.

**The "aha":** the first factor `p^(1/alpha)` is **literally Nikita's power-transform pseudo-label**
(he uses powers 1.0 / 1.54 / 1.82 / 1.67 = his `1/alpha`). NOTELA = *Nikita's power-transform PL*
**× a feature-space nearest-neighbour consistency term** (the Laplacian/cluster assumption: samples
that are kNN-neighbours in Perch embedding space should get similar labels). It is the principled
generalization of what the 2025 winner already does, plus neighbour-smoothing, plus a BN-only TTA
student. The paper's **headline benchmark is adapting a focal-trained bird classifier to passive
soundscapes — our exact shift** — and it beats SHOT/TENT/NRC/DUST/plain-PL, which collapse on
bioacoustic shift. [CONFIRMED from code + paper.]

### Exact AUDIO recipe (from `configs/notela.py get_audio_config`) [CONFIRMED]
`knn=5`, `lambda_=1.0`, `alpha=1.0`, `use_mutual_nn=True`, `online_pl_updates=False`
(recompute PL once/epoch), `normalize_pseudo_labels=True`, `use_dropout=True`,
`update_bn_statistics=False`, **trainable = BN params only**, Adam lr=1e-4 (no decay, wd=0),
**num_epochs=10**. (Image config differs: knn=15, lambda=2.0.)

### Applicability to US — this is the lever we have NOT tried [INFERRED, high-conf]
Our three documented failures were all about **adding members to a fixed base**. NOTELA is a
different axis entirely: it **adapts the base model's BN stats to the 600 test soundscapes at
submission time, no labels, no extra ensemble member, no runtime member-blend** (so it sidesteps the
90-min member-blend cliff — it's a few BN-only epochs on the already-cached test embeddings/specs).
- It directly attacks the **focal→soundscape domain shift** that made our round-0 focal CNN net-
  negative (0.944). Instead of training a NEW member that's weak on the test domain, it makes the
  EXISTING strong model *better on the test domain*.
- The kNN-consistency term is a principled way to propagate signal onto rare/weak classes via their
  embedding neighbours (helps the long tail among the 206 mapped classes; will NOT save the 25
  anonymized `son##` sonotypes — those are dead in the teacher, see (c)).
- **No public BirdCLEF-2026 kernel implements NOTELA or any TTA** [CONFIRMED — REST kernel search for
  notela/tent/"test time adaptation"/"domain adapt" returns only keyword-coincidence EDA notebooks].
  So it is a genuine, un-mined differentiator.

### Expected impact + the catch
- Paper-scale SFDA gains on this shift are large, but on a *strong already-pseudo-labeled ensemble*
  the marginal headroom is smaller. **Estimate +0.003 to +0.010** on our 0.950 base. **Confidence:
  MED** (method perfectly matched; impact on an ensemble unproven; needs a careful BN-only port).
- **CATCH 1 — it's a TRAINING-at-inference op on a no-internet, 90-min CPU/GPU notebook.** The Perch
  ONNX export has no trainable BN graph; you'd apply NOTELA to a **PyTorch member you control** (the
  SED CNN or the Perch-in-PyTorch warp), not to the frozen Perch ONNX. Budget the BN-adapt loop
  (5–10 epochs over 600 files of cached features) inside the runtime — feasible because it's BN-only
  and embeddings are already cached.
- **CATCH 2 — don't adapt a model that's correlated with the rest of the blend**; adapt the most
  independent strong member (the non-Perch EffNetV2 SED, see (c)#1), so the lift is orthogonal.
- **Cheaper first step (do this before full NOTELA):** the `p^(1/alpha)` factor ALONE, applied to our
  pseudo-labels with Nikita's per-iter powers and NO threshold, is the 80/20 — our docs note we used
  `TH=0.3`, which Nikita does NOT. The Laplacian term is the extra 20%.

---

## (b) CURRENT TOP-PUBLIC-CODE DELTA (last ~3–5 days, full source pulled) [CONFIRMED]

The public stack everyone forks = **Perch v2 ONNX embeddings → ProtoSSM (12-window seq model) +
ResidualSSM + MLP probes + ecological priors (site/hour/site×hour) → distilled-SED branch → rank
blend + safety gates**. Authors: Vyanktesh Dwivedi (base), Imaad Mahmood (ProtoSSM 0.925), Tucker
Arrants (distilled-SED). Current public peak **0.947–0.948**. What's NEW vs our V237 pipeline:

1. **BirdNET v2.4 TFLite as an independent THIRD branch** (`mtoshidesu` 0.947 "Public Pipeline
   Reproduced", v4 "+BirdNET Third Branch"). 3s chunks mapped back to 5s windows; blend becomes
   **50/30/20 Proto/SED/BirdNET** rank-blend (falls back to 60/40 if BirdNET absent), plus a "Gate 3b:
   BirdNET spike preservation." **This is an independent, test-domain-strong member that is public,
   CC-licensed, and runs cheap (TFLite)** — exactly the orthogonal-but-good member we lacked when our
   focal CNN (0.944) and b0 NS-student (0.945) failed. [CONFIRMED from pulled source.]

2. **A pile of post-processing "tweaks" the public 0.948 kernel stacks that our 12-window pipeline may
   not all have** (`youssefmo942009/lb-0-948`, full source) [CONFIRMED]:
   - **Per-class ensemble weights:** `ENSEMBLE_W_PER_CLASS = where(MAPPED, 0.60, 0.35)` — give the
     SED/BirdNET branches MORE weight on unmapped/rare classes, less on well-mapped Perch classes.
   - **`file_confidence_scale`**: multiply each window by `(top-2-mean of the file)^0.4` — pushes up
     files that clearly contain a species (rank-metric trick).
   - **`rank_aware_scaling`**: multiply each window by `(file_max)^0.4` (the "2025 Rank-3" trick;
     `needless090` uses power 0.4–0.5).
   - **`adaptive_delta_smooth`** (base_alpha 0.20) + **circular Gaussian smoothing of the HOUR prior**
     (sigma 1.5, wrap-aware over 24h) so dusk/dawn peaks don't leak across hour buckets.
   - **TTA:** delta-shift `shifts=[0,1,-1,2,-2]` + **temporal-flip** as an extra pass, on the test set.
   - Per-taxon **temperature** (0.95 for "texture" taxa = insect/frog drones, 1.10 otherwise);
     label_smoothing 0.03.
   - 5 safety gates: noise suppression, temporal continuity, SED spike preservation, **sonotype
     mirroring**, rare-class thresholding.

3. **Genus-level "frog/insect proxies"** (`needless090/...iter-pseudo` 0.934, full source) [CONFIRMED]:
   for each UNMAPPED *named* target species, find all Perch-classifier species sharing the **genus**,
   and assign `max`(or mean) of their logits as a proxy score for the unmapped target.
   - This reaches the **3 Amphibia + any named unmapped Aves** that have a scientific name and a
     same-genus sibling in Perch's 14,795-label head. **It does NOT reach the 25 anonymized `son##`
     sonotypes** (no scientific name → explicitly filtered out as `unmapped_non_sonotype`). So it
     CONFIRMS our "25 sonotypes unreachable" finding but reveals a **small headroom on the ~3
     amphibians + named-unmapped species that our pipeline treats as zeros.**

4. **Non-Perch EffNetV2-B0 branch is now a real, near-parity, low-correlation member** (disc 700763,
   updated 6d ago) [CONFIRMED forum]: multiple authors report **XC-pretrained EfficientNetV2-B0 +
   pseudo-labels + sliding-window TTA + smoothing → 0.943–0.946 single model, 0.948 ensembled**, "10
   min inference," and explicitly *"the Perch-embedding notebooks and the SED-distillation one aren't
   very correlated."* This is the orthogonal member that is GOOD on the test domain.

**What the public crowd does NOT do (still open):** NOTELA / any test-time adaptation; multi-iteration
noisy-student (they do single-pass PL); a from-scratch heterogeneous backbone ensemble. The
0.948→0.966 gap is exactly those three, all private to the leaders.

---

## (c) TUNNEL-VISION CHECK — ranked "near solutions we haven't tried", concrete how-to

Our scoreboard tried: add NS-distillation students (hurt, correlated), add focal CNN (hurt, weak on
domain), add BirdMAE (≈flat after base-trim). Common thread of every failure = **"bolt a member onto
a fixed, runtime-maxed V237 base."** The near-wins below break that pattern.

### RANK 1 — Add the PUBLIC **BirdNET v2.4 TFLite** branch (NOT a SED distillation) [HIGH leverage]
- **Why it's different from our failures:** BirdNET is a *fully independent* foundation detector
  (Cornell's own model, TFLite, ~cheap), NOT a distillation of the public SED that's already in our
  base — so it is NOT in the correlated-students trap that killed our b0/effv2s members. The public
  0.947 kernel gets a real lift from it via 50/30/20 rank-blend + a spike-preservation gate.
- **How:** lift `mtoshidesu/birdclef-2026-0-947-...` BirdNET branch verbatim (model
  `shadiakiki.../BirdNET TFLite`, Stefan Kahl). 3s chunks → map to 5s windows; rank-blend at
  **w≈0.20**, with the SED-style spike-preservation gate so BirdNET spikes aren't smoothed away.
- **Runtime:** TFLite is light; budget a scale-test, but it's far cheaper than another SED fold.
- **Expected: +0.003 to +0.008** (public authors see it; it's the orthogonal-AND-good member we
  never actually had). **Confidence: MED-HIGH.** ← **single highest-leverage untried thing.**

### RANK 2 — Strip the power-transform PL threshold + the public post-proc "tweaks" onto OUR base
- We used `TH=0.3` hard-threshold PL; Nikita and NOTELA both use **pure `p^(1/alpha)`, no threshold**.
  And the public 0.948 kernel stacks `file_confidence_scale^0.4`, `rank_aware_scaling^0.4`,
  per-class ensemble weight (mapped 0.60 / unmapped 0.35), circular-Gaussian hour-prior smoothing,
  delta-shift+temporal-flip TTA, per-taxon temperature. **These are inference-only, free, and
  independently validated public.** Audit which our V237 pipeline is missing and add them.
- **Expected: +0.002 to +0.006**, zero runtime member cost. **Confidence: MED-HIGH.**

### RANK 3 — NOTELA / BN-only test-time adaptation on our INDEPENDENT member (see (a))
- Adapt the non-Perch EffNetV2 SED (or Perch-in-PyTorch warp) on the 600 test soundscapes, BN-only,
  10 epochs, knn=5/lambda=1/alpha=1, then blend. Sidesteps the member-blend runtime cliff (it's an
  adapt-in-place, not a new ensemble pass). **Nobody public does this.**
- **Expected: +0.003 to +0.010. Confidence: MED** (needs a careful port; do RANK 2's `p^(1/alpha)`
  first as the cheap subset).

### RANK 4 — Swap the BASE off the V237 90-min cliff to a leaner, equal-or-stronger public base
- Our binding constraint is that **V237 alone runs ~85 min**, so any member tips over 90. But the
  public `mtoshidesu` 0.947 / `youssefmo` 0.948 pipelines run **well under budget on CPU/ONNX**
  (the 0.948 kernel papermill duration ≈ 9 min on the public 600-file set; the EffNetV2 non-Perch
  branch is "10 min"). **Our base may simply be inefficient, not maximal.** If a public 0.947 base
  leaves 60+ min of headroom, you can add BirdNET + NOTELA + the EffNetV2 branch and STILL fit —
  the thing we kept concluding was impossible (members net-flat after base-trim) is an artifact of an
  over-heavy V237, not a law. **Re-benchmark: is V237's 85 min buying anything over a 9-min public
  0.947 base?** If not, rebase. **Expected: unlocks RANKS 1+3 simultaneously. Confidence: MED**
  (must verify real 600-file runtime of the public base on our account).

### RANK 5 — The ~3 named Amphibia + named-unmapped via genus proxies (small, cheap, real)
- Add `needless090`'s genus-proxy: for each unmapped *named* target, proxy = max of same-genus Perch
  logits. Inserts non-zero preds where we currently emit zeros. Won't touch the 25 `son##` (dead for
  everyone). **Expected: +0.001 to +0.003. Confidence: MED.**

### RANK 6 — Pure-power multi-ITERATION (not single-pass) PL, if a non-Perch student is retrained
- Our NS students were single-teacher distillations of the public SED → correlated. If we instead run
  Nikita's **4-iteration** self-train on the **non-Perch EffNetV2** (XC-pretrained), the resulting
  member is both strong-on-domain AND decorrelated. This is the public 0.943–0.946 recipe. Multi-day;
  only if RANKS 1–4 land and time remains. **Expected: the real path to a strong orthogonal member,
  +0.005–0.010, but high time cost. Confidence: MED.**

### Things CONFIRMED to SKIP (failed-methods thread disc 701938, BNBU Anpeng Yuan, + our own results)
- **ASL / Focal / CE for the SED head all underperform — BCE is best.** (Author + commenter; matches
  our pipeline.) Don't re-tune loss.
- **sumix+cutmix+mixup combos are WORSE than mixup alone.** Use mixup alone.
- **Longer chunks (15–20s) hurt CLIP-level (non-frame) models** (predicts species outside the center
  5s). 20s only helps if the head is frame-level SED. Our per-5s metric → keep 5s unless frame-level.
- Adding another **Perch-based** member ≈ +0.001 (noise) — confirmed by disc 700763 author. Diversity
  must come from NON-Perch (BirdNET / EffNetV2), not more Perch.

---

## SOURCES (this scan)
- LB + kernel source: Kaggle REST (Bearer) — `competitions/birdclef-2026/leaderboard/view` (Nikita
  0.966), `kernels/list?competition=birdclef-2026&sortBy=voteCount`, `kernels/pull/<ref>` for
  `youssefmo942009/lb-0-948`, `mtoshidesu/birdclef-2026-0-947-lb-public-pipeline-reproduced`,
  `needless090/birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s`,
  `marynaborovska/birdclef-26-two-pass-ssm-advanced-pp`,
  `ulyanovantonamaranta/birdclef-2026-gate-fake008-head0015`.
- NOTELA code: github.com/google-research/perch tag `sfda-codebase-snapshot`,
  `chirp/projects/sfda/methods/notela.py` + `configs/notela.py` (audio: knn=5, lambda=1, alpha=1,
  BN-only, 10 ep). Paper: Boudiaf et al. ICML 2023 (PMLR v202); research.google SFDA blog.
- Forum (r.jina proxy, `x-respond-with: markdown` on the discussion HTML pages): disc 701938
  (failed-methods, BNBU Anpeng Yuan), disc 700763 (non-Perch limit: EffNetV2-B0 + XC + PL + TTA →
  0.943–0.946 single, 0.948 ens, "not very correlated" with Perch/SED).
- Builds on A7_discussion_deepdive.md, A8_nikita_dossier.md, 03_SUBMISSION_RESULTS.md (this sprint).
