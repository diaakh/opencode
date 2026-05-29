# A4 — SED head + post-proc math that directly optimizes MACRO ROC-AUC

**Charter:** find post-processing & head-architecture levers that *directly* optimize
the competition metric — **macro-averaged ROC-AUC** (skips classes with no positives).
Companion runnable code: [`a4_rankblend.py`](./a4_rankblend.py) (rank-blend + local
macro-AUC eval + weight search; self-tests on real repo OOF — see §6).

---

## 0. The one fact that reframes everything

Macro ROC-AUC is computed **per class, then averaged**. Per class it is the
Wilcoxon–Mann–Whitney statistic: the probability a random positive row outranks a
random negative row, *within that class column, across all row_ids*. Consequences,
proven empirically in §6:

- **Only the within-column ranking matters.** Any strictly-monotone *per-class* map
  (sigmoid, temperature `T`, per-class affine `a·x+b`, isotonic calibration,
  rank-transform) leaves that column's AUC **exactly unchanged**. Verified on exp019:
  raw / rank-transformed / temperature(T=2) all give macro-AUC **0.96429** to 5 dp.
- Therefore **per-class CALIBRATION is worthless for this metric** unless it changes
  cross-row order. "Per-class temperature" only helps if a *single* `T` is shared
  across the operations that DO mix columns (i.e. blending) — on its own it is a no-op.
- The metric **never compares columns to each other**. Class A's scores can live in
  [0.9,0.99] and class B's in [0.0,0.01]; macro-AUC is identical to any rescaling.
  This is precisely why **rank-transform before any cross-model averaging** is the
  correct normalization (§2).
- It is **threshold-free**: `adaptive_delta`, argmax, top-k decisions are irrelevant
  *except* insofar as a row's score feeds the ranking of *other* rows (it doesn't —
  rows are independent). The repo's "adaptive_delta +0.626" correlation is a proxy for
  "this kernel is in the elite cluster", not a causal AUC lever.

So the entire post-proc surface that helps AUC reduces to: **(a) better per-class
cross-row ordering** (better model / ensemble) and **(b) correct cross-model fusion
(rank-space) — never calibration.**

---

## 1. Ranked technique table

Legend — **lever**: expected macro-ROC-AUC LB delta over the exp019 0.949 anchor.
**feas-now**: prototypeable today on existing OOF (T1) with no GPU.

| # | technique | track | expected LB lever | feas-now | math / impl | source |
|---|-----------|-------|-------------------|----------|-------------|--------|
| 1 | **Rank-blend public 0.958 SED 5-fold (aliozanmemetoglu) with exp019, rank-space, w≈0.3, power 0.5** | T1 | **+0.004 … +0.009** (single biggest) | partial (need to mount checkpoints; blend math runnable now) | per-class `rankdata/n`, then `Σ wᵢ·Rᵢ`; output ranks directly (AUC ignores scale). Orthogonal model is the only thing that moves a ranking metric. | meta §"Unexploited gem"; [slot5_blender.py] |
| 2 | **Rank-transform per class BEFORE averaging (vs prob-average)** | T1 | enabler; +0.001–0.003 vs prob-blend when models have different score scales | ✅ yes | strictly-monotone per column → AUC-invariant *within* model, but equalizes scales *across* models so a peaky model can't dominate the mean. | [features.py `_rank_matrix`]; WMW theory |
| 3 | **Local site-balanced macro-AUC proxy + greedy weight search (no LB)** | T1 | meta-lever: stops blind blending, avoids −0.01 regressions | ✅ yes (built & tested) | proxy = `site_mean_AUC − 0.5·max(gap,0)`; rank-order trustworthy, absolute not. Greedy add-helper only if it beats anchor-alone. | [METRIC_FINAL.md]; [a4_rankblend.py] |
| 4 | **SED attention-pooling head (framewise→clipwise) on top of Perch/EffNet** | T2 | +0.005–0.015 (the 0.948→0.955 step) | ❌ needs GPU train | clipwise = Σₜ softmax(att)·sigmoid(cls); jointly learns *which frames* score the clip → sharper per-class cross-clip ranking than mean-pool. | [PANNs Cnn14_DecisionLevelAtt]; meta uses_sed +0.609 |
| 5 | **Temporal-flip TTA (NEW, free)** | T1 (infer) | +0.001–0.003, ~0 cost | partial (needs model fwd; logic now) | average `score(x)` and `score(reverse_time(x))`; for SED also flip the time axis of the mel. Free diversity, no retrain. | TTA std practice; [GPS arXiv 2002.09103] |
| 6 | **Multi-window / multi-crop TTA (12×5 s already in 0.95+ kernels)** | T1 (infer) | +0.001–0.002 | partial | mean (or rank-mean) over overlapping windows per clip. | meta n_windows=12, window=5 |
| 7 | **Per-taxon time-smoothing** (Insecta/Amphibia heavy, Aves light) | T1 | +0.001–0.003 | ✅ on OOF | weighted moving avg over a clip's consecutive 5 s rows `[.1,.2,.4,.2,.1]`; or max-with-neighbors. Continuous insect/frog calls → smoothing fixes a true cross-row ranking error. | [BirdCLEF24 3rd/4th]; meta sonotype |
| 8 | **Pairwise / ROC-star rank surrogate loss (training head)** | T2 | +0.002–0.005 vs BCE, *iff* multi-label per-class pairs | ❌ GPU train | BCE/focal optimize calibrated likelihood (a *proxy*); replace/augment with per-class pairwise logistic/hinge `Σ_{i∈+,j∈−} σ(s_j−s_i)` → directly maximizes WMW=AUC. Use as fine-tune head loss; keep BCE warmup. | [JMLR sort-surrogate 21-0751]; [Hinge Rank Loss/AUC] |
| 9 | **Site/hour prior as a per-class RANK-SHIFT (not multiplicative calib)** | T1 | +0.001–0.003 if priors transfer; **overfit risk high** | ✅ on OOF | logit shift `+w·log prior[site,hour]` reorders rows of a class only where prior varies across rows → genuinely changes AUC. Multiplicative-by-class-constant = no-op (see §0). Use tiny `w≈0.05`; dead-hour→global mean. | [_shared_postproc_v2.py]; meta site_hour +0.539 |
| 10 | **Sonotype mirroring / max-pool visually-similar insect classes** | T1 | +0.001–0.002 on the 28 unmapped | ✅ on OOF | `s[a]=max(s[a], 0.9·s[b])` for alias pairs → lifts ranking on classes Perch can't see. | [postproc_v2 patch_sonotype_aliases] |

**Flagged low-priority (<+0.001 or no-op for AUC):** per-class temperature *alone*,
isotonic/Platt calibration, threshold/argmax tuning, softmax renorm — all monotone
per-class ⇒ macro-AUC-invariant. Use them only for human-readable outputs, never to
chase LB.

---

## 2. Rank-blend vs probability-blend under AUC

- **Why rank-blend wins:** averaging *probabilities* lets a model with extreme,
  poorly-scaled outputs (e.g. one peaked at 0.999) dominate the column mean, which can
  *reorder* rows badly. Averaging *per-class ranks* puts every model on an identical
  uniform[0,1] support per column, so each model contributes equal ordering weight.
  The averaged ranks then re-rank the rows — and AUC only reads that final order.
- **Optimal rank-power:** `R = (rank/n)^p`. The repo's 0.95+ cluster converged on
  **p≈0.5** (meta: `rank_power 0.4–0.5, median 0.5`). `p<1` compresses the high-rank
  tail so confident-but-wrong top scores of a weak model can't pull the blended top;
  `p>1` sharpens. Search p∈{0.5,1.0} per blend.
- **Weight search WITHOUT a leaderboard:** use the local site-balanced macro-AUC proxy
  (§3, §5) as the objective. Greedy coordinate ascent: anchor (exp019) weight 1.0, add
  each helper at the grid weight that maximizes the proxy, accept only if it beats
  anchor-alone. This is implemented and tested in `a4_rankblend.py::search_blend_weights`.

## 3. SED head architecture (PANNs lineage)

The 0.948→0.955 step is the **SED attention head** (`uses_sed +0.609`). Design:
backbone → framewise features `f_{t}` → two parallel 1×1 convs producing, per class c:
classification `σ(y_{t,c})` and attention `softmax_t(a_{t,c})`; **clipwise** score
`Σ_t softmax(a)·σ(y)`. This `DecisionLevelAtt` pooling (PANNs Cnn14) learns *which
frames* matter, giving a sharper, less mean-diluted per-class clip score than global
mean-pool → directly improves cross-clip ranking = AUC. Framewise output is also used
for the per-taxon time-smoothing in #7. (T2: requires GPU to train the head; the public
aliozanmemetoglu SED checkpoints (#1) give this for free at inference.)

## 4. Site/hour/geographic priors as a rank-shift

A class-constant multiplier or per-class temperature is a **no-op** for AUC (§0). A
prior is only an AUC lever when it **varies across rows of the same class** — e.g.
`logit += w·log prior[site,hour]` reorders that class's rows by site/hour. Risk: priors
are estimated from pseudo-labels on the *same 23 sites* as test (good) but can overfit
train-audio site contamination (the repo's `gap` term). Mitigations already in
[`_shared_postproc_v2.py`]: tiny weight (w≈0.05, NOT v1's 3.0 which randomized 20% of
rows), dead-hour (11–16) → global mean fallback. Expected +0.001–0.003, high variance —
gate it behind the §5 proxy before submitting.

## 5. The biggest T1 win — a robust LOCAL macro-ROC-AUC validation

The repo's documented failure ([model_zoo_report.md], [METRIC_REALITY_CHECK.md]):
plain `labeled_macro_auc` has **Pearson +0.035 / Spearman −0.157** with LB — i.e.
*anti*-correlated; while `agreement_exp019` (rank-agreement with the known-good anchor)
has **Spearman 1.0**. Naive OOF AUC is inflated by train-audio site contamination.

**Robust proxy that tracks LB transfer (rank-order, not absolute):**
1. compute **per-site** macro-AUC, average → `site_mean` (kills S22 single-site
   domination);
2. `gap = overall_auc − site_mean` (positive gap ⇒ site contamination ⇒ won't transfer);
3. `proxy = site_mean − 0.5·max(gap,0)`; **use its RANK across candidates**, never its
   absolute value (cross-pipeline refit RMSE was 0.05). Cross-check with rank-agreement
   to the exp019 anchor (the only feature that hit Spearman 1.0).

This is implemented as `local_lb_proxy` / `search_blend_weights` in `a4_rankblend.py`
so blend search is **no longer blind**.

## 6. Verification (ran on real repo OOF: exp019_aligned.npz, 739×234, 75 active classes)

```
invariance:  raw=0.96429  rank=0.96429  temperature(T=2)=0.96429   (calibration = no-op ✓)
greedy search refuses weak helpers (best_oof/mlp/aves all proxy<<exp019) → keeps exp019 alone ✓
injected a genuinely-orthogonal helper → search picks w=0.05, proxy 0.9134→0.9247,
                                          true macro-AUC 0.96429→0.96567 ✓
```
The search correctly (a) rejects helpers that don't transfer and (b) accepts and weights
a helper that does — exactly the discipline needed for the #1 lever (real public SED).

---

## Sources
- Repo: `meta_analysis/FINAL_META_FINDINGS.md`, `analysis/model_zoo_transfer/model_zoo_report.md`,
  `analysis/entropy_tta/METRIC_FINAL.md` & `METRIC_REALITY_CHECK.md`,
  `inference_notebooks/slot5_blender.py`, `_shared_postproc_v2.py`,
  `analysis/model_zoo_transfer/features.py`.
- [PANNs / Cnn14_DecisionLevelAtt](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- [Optimizing ROC with a sort-based surrogate loss (JMLR 24, 21-0751)](https://www.jmlr.org/papers/volume24/21-0751/21-0751.pdf)
- [Hinge Rank Loss and the Area Under the ROC Curve](https://www.researchgate.net/publication/221112400_Hinge_Rank_Loss_and_the_Area_Under_the_ROC_Curve)
- [Greedy Policy Search: learnable TTA (arXiv 2002.09103)](https://arxiv.org/pdf/2002.09103)
- [BirdCLEF 2024 3rd-place solution (predict smoothing)](https://zenn.dev/yuto_mo/articles/53ed2b27c1f52b)
