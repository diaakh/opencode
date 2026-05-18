# Multi-Agent Analysis of BirdCLEF 2026 Effort

10 independent Opus 4.7 agents (5 with prescribed angles, 5 with no prescribed task — read CONTEXT.md fully then decide) audited the corpus and reported back. All reports preserved verbatim below.

Generated: 2026-05-18.

---

## Agent K1 — Data forensics + validation (killed batch, prescribed: verify CONTEXT.md claims)

**Verified contradiction in CONTEXT.md:** The table at line 22-32 lists sub3/sub4 LB as "~0.85" — but Kaggle shows 0.914/0.920 (sub3) and 0.920 (sub4). CONTEXT.md is wrong by 0.06-0.07 LB points. The actual scores are much closer to the buggy sub1 (0.920), suggesting the dead-hour bug bites less than the analysis claims.

### VALIDATED CLAIMS (n=7)
- **1478→739 dedup of train_soundscapes_labels.csv**: exactly reproduces (`/tmp/bc26/train_soundscapes_labels.csv`).
- **Dead-hour gap in pseudo_hour_priors.csv at hours 11-16**: confirmed all-zero rows (`birdclef-2026/meta_analysis/pseudo_hour_priors.csv`).
- **9 sites in labeled OOF + S22 dominance**: S22=11 of 25 site-hour combos (44%); confirms S22-night bias.
- **Sonotype Jaccard=1.0 for son15/son16 and son22/son23**: reproduced from labels (|A|=|B|=12 and 24 respectively); no other Jaccard≥0.95 pair exists.
- **perch_calibration ratios are mathematically sound**: 517063 ratio=106.66 (labeled_prev=0.424, pseudo_prev=0.004); 24321 ratio=197.45. All have labeled_prev > 0.012, so n>9 of 739; not statistically meaningless but small.
- **Bruce+priors sub2 LB=0.755**: confirmed on Kaggle submission history.
- **Best personal LB=0.947 (Raunak v13)**: confirmed.

### DISPUTED/UNVERIFIED CLAIMS (n=6)
- **exp019 baseline = 0.949**: borrowed from a public kernel (sunderekkiz); the team has NEVER personally submitted exp019. Their best self-submission is 0.947. The "+0.019 to 0.968" projection rests on a baseline they haven't measured themselves.
- **OOF AUC 0.9586 / v4 sim 0.9691**: CANNOT reproduce — `meta_corpus/datasets/teacher_oof_predictions.npz` and `data/sample_submission.csv` (the inputs to `benchmark_v3.py`) do not exist in the repo or on disk. All sim numbers are markdown-only.
- **"Bruce alone scores 0.755 on LB"** (CONTEXT.md TL;DR): MISLEADING. The 0.755 submission included the BUGGY w=3.0 hour_prior. True Bruce-alone (no priors) was never submitted.
- **sub3/sub4 scored "~0.85"** (CONTEXT.md table): WRONG. Kaggle shows 0.914 and 0.920. CONTEXT.md is off by 0.06-0.07 LB points. This significantly weakens the "dead-hour crush primary cause" narrative — sub3 (w=2.0) ≈ sub1 (w=3.0).
- **"20% dead-hour exposure" sim is honest**: SUSPECT. The simulation rewrites random row_ids to hour 11; the same OOF is used to derive priors AND benchmarks. The "labeled hybrid" hour prior (13 hours × 75 cls) covers hours 0-4, 6-7, 18-23 — exactly the OOF distribution. Test-LB hours that fall in 5, 8-17 ALL hit pseudo or global-mean fallback. The 20% synthetic dead-hour mix understates real exposure.
- **2025 "14.7% silence" regime claim**: based on **n=237** files (vs 1925-3598 in other years) and the per-year acoustic profile is from "10 files per year" — sample-size noise plausibly dominates.

### NEW FINDINGS (n=5)
- **`hourly_species_priors.csv` covers only 13 of 24 hours** (missing 5, 8-17), and only 75/234 classes. CONTEXT.md frames this as "hybrid with pseudo fallback," but for hours 5, 8-10, 17 the LABELED side ALSO has nothing. Hybrid v4 falls back to pseudo across more of the test set than acknowledged.
- **pseudo_hour_priors aggregation appears buggy**: ROUND8 reports 1,980 windows at hours 5-16 in the cache, yet pseudo_hour_priors.csv has exact 0.0 sum at hours 11-16. Some upstream filter zeroed real data — worth re-running the aggregation.
- **Same-date temporal leak at S05**: ROUND19 notes test row `BC2026_Test_0001_S05_20250227_010002` has 4 same-date training files (S05 17:00 same day). This is logged but NOT exploited in v4-v6 recipes.
- **Submission history shows sub3/sub4 ≈ sub1**: 0.914-0.920 cluster. This suggests the "dead-hour crush" theory only explains sub2 (Bruce-alone, 0.755) — for exp019-based subs, dead-hour bug costs ~0.029 LB, NOT the 0.075+ implied by the simulation.
- **Untested integrations**: V5 (TTA Bruce) and V6 (codec+hpf+softclip) are built locally but never pushed. Codec-proxy +0.019 measured on `fold0.onnx`, but that ONNX is NOT the Perch v2 used in exp019 — transferability is pure speculation.

### OPEN QUESTIONS
- **Will hour-11 even appear in test?** Hidden test set; sample_submission.csv reveals only one hour (01:00). The "20% dead-hour" framing is an assumption about test-set distribution that has no public evidence.
- **What is the actual Perch v2 baseline on the team's own kaggle account?** Could be measured cheaply by submitting exp019_fast (already pushed, never submitted) as v4-1 control. CONTEXT.md treats 0.949 as bedrock — it is not measured.
- **Why does pseudo_hour_priors zero out hours 11-16 when ROUND8 says the cache has data there?** Re-running aggregation may unlock free coverage of the disputed dead hours.
- **Is the v4 OPTIMAL 0.9691 sim reproducible?** Cannot verify without the OOF npz. Need to either restore from a backup, regenerate from Bruce's published bundle, or treat 0.9691 as unverified.

**Immediate action priority for next 5 submissions**: slot 1 v4-1-control is the MOST valuable shot — it's the team's first independent measurement of exp019. Without it, every "+Δ LB" projection has no anchor. Slot 2 v4-2-optimal carries unverified +0.019 OOF claim. Recommend slot 3 changes to either exp019_fast (also tests if the speedup math is identical end-to-end) or a Bruce-alone-no-prior submission to anchor the true Bruce LB.

---

## Agent K2 — Recent Kaggle activity scout (killed batch, prescribed: pull last 7 days deltas)

### NEW KERNELS (n=8)
- **nina2025/birdclef-2026-eos-5** (May 18, 95 votes) | Nina rank unknown | New: simple linear blend = 0.0305 × hideyukizushi(0.928) + 0.9695 × sunderekkiz exp019(0.949). Same exp019 base, just adds 3% hideyukizushi diversity. | **Med** — sanity check the small-weight blend trick (≈0.001 expected lift)
- **mtoshidesu/birdclef-2026-visual-cpu-inference** (May 18, 41 votes) | mtoshidesu rank unknown | Pure refactor / visualization of the public 0.948 EoS-4 pipeline; no algorithm change. | **Low**
- **meenalsinha/birdclef-2026-improved** (May 18, 59 votes) | unknown | Single-cell exp019 fork; explicitly includes **Tweak F = temporal flip TTA** (reverses time axis, runs SSM both ways, averages). Uses IsotonicRegression, PCA, MLPClassifier. Has BirdNET 85 refs. | **High** — temporal-flip TTA is NEW, not in CONTEXT.md dead-ends list. Adds zero training cost.
- **karnakbaevarthur/gated-rank-fusion-pipeline** (May 17, 22 votes) | Karnakbayev (CONTEXT lists power-optimization 0.948) | "Gated rank fusion" — refers to sunderekkiz exp019 rank-power 0.6 base with the Karnakbayev power-optimization branch (already in CONTEXT). Cell 9 credits Pilkwang Kim additions. | **Low** — derivative of known pipeline.
- **zeyadmohamadezzat/birdclef-2026-eos-parity-inference** (May 17, 15 votes) | unknown | Pure CPU port/parity of EoS-4 inference. | **Low**
- **eslamelokpy/birdclef2026-v21d** (May 17, 17 votes) | unknown | exp019-style CPU inference fork. | **Low**
- **damianleandrotamburi/20260329-birdclef** (May 18, 4 votes) | unknown | EDA + Perch v2 embeddings + LogisticRegression baseline. Beginner level. | **Low**
- **anthonytherrien/birdclef-ensemble-of-solutions** (May 16, 27 votes) | unknown | Direct blend of Model_3 (hideyukizushi 0.928, w=0.032) + Model_4 (mtoshidesu 0.947, w=0.968). Same "small weight diversity" trick as EoS.5. | **Low** — same approach as EoS.5.

### NEW DATASETS (n=4)
- **alexycactus/birdclef-2026-cnn-fold-checkpoints** (May 17, 86 MB, 3 dl) — 5-fold EffNet-B0 SED checkpoints. Already mentioned in CONTEXT § ROUND 8. Skip.
- **sergeytata/birdclef26-code** (May 17, 10 KB, 1 dl) — likely code only, very small.
- **samuelzxu/bc26-iter1-perch-cache v2** (May 16, 826 MB) — v2 of the iter-1 pseudo Perch cache (v1 in CONTEXT § ROUND 15). 232/234 classes covered.
- **irinafayzrakhmanova/birdclef2026-full-spec-cache** (May 16, 10.6 GB, 4 dl) — precomputed log-mel n_mels=224, n_fft=4096, hop=1252 over all train audio in float16. **Training-time only** — useless for our 90-min inference budget. Skip.

### LB CHANGES SINCE MAY 17
- **NEW Rank 2: "BirdCLEF+ 2026 Team🤗🤗🤗" jumped 0.957 → 0.960** (tonylica/shtljw/yiheng team). Was rank 7 in CONTEXT. They have public inference + 2 weights datasets; could squeeze further with ensemble.
- Top 3 (Yannan 0.962, more-exp 0.959, Nikita 0.959) unchanged.
- Several new 0.955-0.958 entrants (Sinan Calisir, Diptyajit Das, "less submissions", Mythos of Sisyphus, Arunodhayan, BUET_Perceptron, aicon_team) — likely team consolidations / fine-tunings, no public kernels above 0.949.
- Tom Capybara LB now 0.955 (rank 13). Not directly in CONTEXT but consistent.

### NEW FORUM POSTS WORTH READING
- **Disc 681146 (Tom Capybara log) — May 15 update**: "Just found the secret. Claude took 1 month to discover it. I believe if I'm coding by myself I can discover that earlier." No technique disclosed; comments joke "the secret is to say 'make no mistakes'". 2 votes. Not actionable.
- **Disc 694815** (already in CONTEXT): noisy-student LB drop discussion. No new posts > 5 votes.
- Topic-ID enumeration in 694815-705000 range yielded 404s — no significant new high-vote topics queryable via CLI.

### NEW PAPERS/REPOS
- None. WebSearch returned only the BC2025 papers/repos already in CONTEXT (Sydorskyi BC2025 2nd place, Perch 2.0 paper, Babych 1st place writeup). No BC2026-specific arxiv preprints indexed yet.

### TOP-3 RECOMMENDED ACTIONS for May 19 submission window
1. **Keep the v4-2-optimal submission as slot-1 priority** (queued exp019 + w_sh=0.025 + w_site=0.20, sim B-AUC 0.9691). Nothing scouted in the last 7 days beats this — every new public kernel is still a 0.948-0.949 derivative of the same EoS pipeline.
2. **Add Meenal Sinha's "Tweak F = temporal flip TTA" to v6 augmentation stack** before submitting. It's NOT in CONTEXT's dead-ends list (time_shift TTA hurt, but time-axis FLIP is structurally different — feeds SSM in reverse, then flips back to average). Cheap inference-time addition that may stack with codec+hpf augs for slot-3 or slot-5.
3. **Consider replacing v4-5-strong with a 0.97/0.03 weighted blend (v4-2-optimal + hideyukizushi Model_2)** mirroring nina2025 EoS.5's recipe. EoS.5 is at 95 votes in one day, suggesting the community treats this micro-blend as the new "free +0.001". Cost: trivial; gives ensemble diversity without changing the dominant model. If "BirdCLEF+ 2026 Team" jumped 7→2 with a similar approach, this could be the secret-sauce technique not yet in CONTEXT.

**No need to chase**: Tom Capybara's mystery secret (likely just prompt engineering, no signal). The 10 GB full-spec cache (training-only, can't use in 90-min budget). Most new kernels are pure forks of the same exp019/EoS-4 lineage CONTEXT already exhaustively dissected.

---

## Agent K3 — Top-3 competitor forensics (killed batch, prescribed: reverse-engineer Yannan/cudacoding/Babych)

### Yannan Chen (yannan90, 0.962, rank 1)
- **New evidence**: 10 public kernels, ALL NLP/LLM (LoRA, BitsAndBytes, 16-head classifier on math misconceptions). NO audio kernels, NO datasets, NO models published. Last NLP kernel Oct 2025 was a complex "16-head + cand_map + freq_dist + lambda fusion" classifier on a 37-class problem with `loss_alpha_local`/`loss_beta_global`/`smart_label_smoothing` — a **per-class-aware fusion head** style. Profile is "Yannan Chen" (likely not Rist GM "Chen Jian" who is a different person).
- **Most likely approach**: Brings NLP-grandmaster ensemble craft to BC2026 — likely a **custom multi-head SED with per-class learned fusion alphas + label smoothing tuned per-class** (much more sophisticated than the global rank-power 0.5/0.6 plateau crowd uses), trained on Perch v2 embeddings + iterative pseudo. The 0.962 single-team gap over the 0.948 fork plateau ≈ +0.014, consistent with a ProtoSSM-class single-model upgrade.
- **Confidence**: low–med (no direct audio evidence)
- **What to steal**: per-class learned fusion alpha head on top of exp019's rank-power outputs (vs our current single global w_sh).

### cudacoding (Boredom, 0.959, rank ~3)
- **New evidence**: 2 public kernels, ALL on **CSIRO Biomass** (image regression), Chinese comments. Notebooks reveal SOTA tooling: **DINOv3-large**, ConvNeXt, **dual-stream architecture** (separate single+dual inference engines run on 2 GPUs), **Stage1→Stage2 "online training"** (test-time training/SWA on stage1 predictions to fine-tune for stage2), threshold-clip post-processing `target = 0.8·s1 + 0.2·s2; clip(x<0.1→0)`. Member of `cudacoding/newdata-0901` (NIPS competition).
- **Most likely approach**: Two-stage **test-time training** pipeline — Stage 1 = ensemble inference, Stage 2 = on-the-fly online finetune using stage1's high-confidence pseudos against the soundscape set, applied at submit time. This is unique vs everyone else and explains the 297-submission count (TTT requires sweeping hyperparams). Likely uses DINOv3 or ConvNeXt backbones rather than EfficientNet.
- **Confidence**: med (kernel idioms transfer cleanly to audio; "speedup-ttt-fold2" filename explicitly says TTT)
- **What to steal**: at submit time, train a tiny linear probe on the top-K confident exp019 predictions in the unseen test set, then use it as a 7th model in the ensemble. Fits in <10 min CPU.

### Nikita Babych (nikitababich, 0.959, rank ~4)
- **New evidence**: BC2025 1st-place writeup explicitly named **"Multi-Iterative Noisy Student Is All You Need"**. Confirmed iteration chain (per medium/tekkix): `pseudo I + Mixup + StochDepth: 0.872→0.898 → power scaling + pseudo II (4 rounds): 0.898→0.930 → separate amphibia+insecta pipeline: 0.930→0.933 → TTA ±2.5s: 0.91→0.922`. His **`birdclef2025-1st-place-extra-data`** Kaggle dataset (last updated Jun 2025, 7.8 GB) contains 979 frog files + 16,218 grasshopper files across 504+ Neotropical species, **including the genera Pithecopus, Chiasmocleis, Dendropsophus that match BC2026's missing classes 517063 (Pithecopus azureus), 25073 (Chiasmocleis mehelyi), 1491113 (Guarani leaf-litter frog)**. His BC2025 inference uses EffNet-B0 trained on `incest_amphibia` (sic — "insect+amphibia") subset with 397 multilabels mapped down — same pattern likely re-applied to BC2026's 234 classes. NO BC2026-specific public datasets, indicating private training.
- **Most likely approach**: Re-running his exact BC2025 pipeline on BC2026 with the same extra-species data already in hand — `tf_efficientnet_b4/b3/b0/eca_nfnet_l0/regnety_016/008` ensemble at (224×512), n_fft=4096, duration=20s, framewise SED head, sampler_maxsum balancing, 3–4 pseudo iterations, TTA ±2.5s, **plus separate B0_insecta_amphibia head for the 28 missing classes**. Confidence: high — he likely has a 1-2 month head start on the sonotype problem nobody else has solved.
- **What to steal**: Attach `nikitababich/birdclef2025-1st-place-extra-data` as a Kaggle dataset and use its frog/insect Neotropical genus audio as test-time matched-genus priors / pseudo-cache enrichment for missing classes (no retraining needed).

### SURPRISING DISCOVERIES
- **`aliozanmemetoglu/raw-pseudos` + `pseudo-text-init-iter-0`** (Apr 15–25, 2026): 127k-row × 234-class OOF CSVs from a **5-fold EfficientNetV2-S** pseudo-labeling iteration **executed on BC2026 publicly**. Filename "iter-0" + identical Babych "Multi-Iterative Noisy Student" frame. CONTEXT.md flagged Ali's 5-fold SED bundle as "0 other importers gem" — but missed that Ali has **publicly released his iter-0 pseudo-label OOF**. Anyone can use this as a noisy-student teacher signal without retraining. Only 10–13 downloads.
- **Babych's BC2025 extra-data covers BC2026 missing-class genera** (Pithecopus, Chiasmocleis, Dendropsophus, Scinax, Rhinella, Leptodactylus). CONTEXT.md ROUND24 treats all 28 missing as "use Perch / hour prior / broadcast" — never considered using genus-congener audio from another Kaggle dataset for direct labels.
- **Yannan Chen ≠ "Chen Jian" of Rist** (the well-known recent GM #1). Yannan90 is a separate, lower-tier NLP-focused Kaggler; his BC2026 0.962 is even more anomalous than CONTEXT.md implied — first audio competition for him.
- **EoS.5 still scores 0.949** despite being a blend of yukiZ 0.928 + Derek's exp019 (the same baseline we use). The blend weight is just 0.05/0.95 — meaning Nina has plateaued exactly where we have. The 0.962 ceiling is a SOLO gap, not a public-kernel gap.

### ACTIONABLE STEALS
- **High-value**: Attach `aliozanmemetoglu/raw-pseudos` (`pseudo_labels_v2s_oof_raw.csv`, 346 MB) to our inference kernel. Use as a v2s-derived prior signal that's blendable with exp019's rank-power outputs. Different model family than Perch+ProtoSSM → ensemble diversity. The CSV is 5-fold OOF so it's leakage-safe for train_soundscapes calibration.
- **High-value**: Attach `nikitababich/birdclef2025-1st-place-extra-data` and extract frog/insect genus audio for 25 sonotypes + 3 frogs. Even without retraining, you can compute Perch v2 embeddings of these clips and use them as **per-missing-class prototype matching** in addition to the existing hour priors. Likely +0.005–0.010 macro-AUC on the missing-class subset (which is ~12% of macro per ROUND7).
- **Med-value**: Inspect `mauriciooffermann/birdclef-2026-exp-034-sparse-fusion-safe-bundle` (Mar 22, 907 MB) — newer than CONTEXT.md ROUND29's `anchored-bundle-20260319`. "Sparse-fusion-safe" name suggests **safety-aligned sparse late fusion**, a different paradigm than anything inventoried.
- **Med-value**: `brucewu1200/birdclef-2026-cvlb-assets-0911` (Mar 29, 470 MB) — Bruce's newer "CV/LB assets". CONTEXT.md ROUND28 only has the older clip_student_bundle.pkl.
- **Low-value**: cudacoding TTT pattern (Stage1 inference → Stage2 online train on confidence-clipped Stage1 predictions → 0.8/0.2 blend) — adaptable but heavy for 90-min CPU.

Files of note (downloaded to /tmp on this host):
- `/tmp/ali_raw/pseudo_labels_v2s_oof_raw.csv` (346 MB, all 234 classes × 127k rows × 5 folds)
- `/tmp/ali_iter0/pseudo_labels_v2s_oof.csv` (98 MB filtered version)
- `/tmp/babych_csv/birdclef2025_extra_species_data.csv` (19 MB, 504-species index incl. Neotropical frogs/insects)
- `/tmp/babych_bc25/birdclef2025-1st-place-inference.ipynb` (full Babych BC2025 inference pipeline with verbatim 7-model ensemble config + weights `[0.133,0.166,0.133,0.133,0.166,0.133,0.133]`, gauss_convolve smoothing weights `[0.1,0.2,0.4,0.2,0.1]`, TTA, OpenVINO runtime)

---

## Agent K4 — Inference budget engineering (killed batch, prescribed: verify exp019_fast & find more wall-clock wins)

### VERIFIED WALL-CLOCK BUDGET (from exp019_fast Kaggle log)
- **Total measured**: 297 s (~5 min) for **20 dry-run files**, NOT the 600-file submission set. The 85→55-75 min figures in `exp019_fast/README.md` are projections, not measurements.
- **Per-stage breakdown from log timestamps**:
  - Imports + ONNX install + ASSET search: 0→116 s = **116 s fixed overhead** (will not scale)
  - Perch ONNX inference (20 files): 126→169 s = **42.5 s** → ~2.1 s/file → **~21 min on 600 files**
  - ProtoSSM training: **12.5 s** (file-count-invariant — uses cached 708-row labeled set)
  - ResidualSSM training: **1.6 s** (file-count-invariant)
  - Tweak-C grid + MLP probes: ~13 s (fixed)
  - SED Model_7 (5 folds, batched, 20 files): 248→297 s = **49 s** → 2.45 s/file → **~25 min on 600 files**
  - **BirdNET did NOT execute** in this log (no `"BirdNET inference:"` line — dry-run skipped it). Original spec: ~360 s for 600 files (6 min) per README, so ~6 min once enabled.
- **Projected 600-file wall**: 116 + 21 + 0.5 + 25 + 6 ≈ **~70 min** (fits 90-min budget with ~20 min slack)
- v1 OOMed (commit 581ce59) at SED_BATCH_FILES=8 + workers=4. v2 dropped to BATCH=2 + workers=2 + gc.collect — explicitly **leaves throughput on the table** to satisfy 13 GB RAM cap.

### DROP-IN ACCELERATIONS (low-risk, ≤1 day to implement)
1. **OpenVINO conversion for SED 5 folds** (per disc 689012, HGNetV2/EffNet get ~2× on CPU; SED here is **EfficientNet-B0 backbone** — line 257 `tf_efficientnet_b0.ns_jft_in1k`, exactly the BC2025 2nd-place case where +30-50% applied). Wheels ship in `nikitababich/runtimes-onnx-openvino` (per ROUND9). **Est savings: 25 min → 13 min = -12 min**. Risk: low (BC2025 2nd-place verified) but ProtoSSM stays on Torch (no BN to fuse, NFNet-like).
2. **Raise SED_BATCH_FILES from 2 to 4 with explicit `del batch_mel` post-call** (current peak mel = 4 files × ~4 MB ≈ 16 MB, not 100 MB as commit message claimed — verified from n_mels=256, hop=512, 5s @ 32k → 313 frames × 256 × 4 B = 320 KB/window × 12 = 3.85 MB/file). Real OOM driver was **5 SED ONNX sessions held simultaneously + Perch ONNX still resident + ProtoSSM PyTorch + BirdNET TFLite**, not the mel tensor. Free Perch session via `del perch_sess; gc.collect()` BEFORE entering SED loop (already happens partially at line 1059, 5771). **Est savings: 4 min.** Risk: medium (must re-verify peak RSS).
3. **Async-pipeline BirdNET TFLite** (the chunk loop is per-file, with 20 chunks/file × 600 files = 12k TFLite invokes — currently serial). Use `tflite_runtime` with `num_threads=8` (Kaggle has 8 logical cores, current is 4 — line 5967) and overlap the Aves-mapping post-loop with the next file's load. **Est savings: 2 min** (out of 6 min). Risk: low.
4. **Hoist Bruce CLIP Ridge into the existing Perch pass** — Perch ONNX is called ONCE per 5s window for both Model_3 (ProtoSSM) and (proposed) Bruce. Currently sub2 Bruce re-runs Perch from scratch (~21 min). Share the (708, 1536) embedding tensor → Bruce Ridge predict adds **~5 s** total. **Est savings: 20 min** vs running sub2 standalone.

### ENSEMBLE-ADDITION OPPORTUNITIES (using freed budget)
- After OpenVINO SED (-12 min) + shared Perch (-5 min adds Bruce predict ~0): wall drops to ~55 min → **35 min free**.
- **Add Bruce Ridge as 8th model** at logit weight 0.1-0.3 → est **+0.005 LB** (Bruce alone gives 0.9586 with prior; as diversity partner of a 0.949 ensemble it adds at most diversity-bonus).
- **Add aliozanmemetoglu 5-fold SED** (LB 0.958 standalone, 0 other importers, called out as biggest unexploited gem in FINAL_META_FINDINGS): runs in ~10 min batched with OpenVINO. Est **+0.003-0.008 LB** via diversity. Total: still <90 min.
- **Add 3-shift TTA on Bruce-Perch path** (the AUGMENTATION_HONEST_LIMITS doc said only Bruce had budget): TTA itself HURTS per V6 benchmark, so SKIP. **Codec_proxy 16k augmentation path (+0.019 on fold0.onnx)** would add ~7 min and is the strongest single audio-side gain.

### EQUIVALENCE-TEST GAPS
- **Test 1 is fully synthetic** — random tensors, not real audio features. Doesn't exercise the actual SED ONNX export's behavior under cross-file batching.
- **Test 2 uses `fold0.onnx`** (a different model — the v6 augmentation fold) as a *stand-in* for the actual `sed_fold[0-4].onnx` from `tuckerarrants/bc2026-distilled-sed-public`. The runtime probe on Kaggle confirmed batching works (log line 75) but local equivalence was never verified on the real SED weights.
- **No edge cases**: files <60 s, all-silent windows (S04 is 78% silent per ROUND17), heavily-clipped files (S13 29%, S01 16% clipped).
- **Per-file Gaussian smoothing (`gaussian_filter1d` mode="nearest")** runs identically on per-file slabs in both paths — confirmed in code (lines 1412, 5885) — but the test never asserts bit-identity on the actual `sed_fold*.onnx` outputs with realistic mels.
- **Probe uses only `fold_sessions[0]`** (line 1372) — if folds 1-4 have different reshape ops, probe gives false-positive. No per-fold probe loop.
- **gc.collect() inside inner loop** (line 1441) adds 50-100 ms overhead per batch × 300 batches = up to 30 s — negligible but unnecessary; once-per-N-batches would suffice.

### TOP-3 RECOMMENDED ENGINEERING MOVES
1. **Convert the 5 SED ONNX folds to OpenVINO IR offline, ship as a private dataset, swap inference engine to `openvino.runtime.AsyncInferQueue`.** Largest verified win (~12 min). Reference: ROUND9 (Nikita's runtimes wheel), disc 689012 (~2× on EffNet). Stay on Torch for ProtoSSM (no BN → no benefit). Validate with a strengthened equivalence test that loads BOTH `onnxruntime` and `openvino` and asserts `max_abs_diff < 1e-4` on first 5 dry-run files.
2. **Strengthen `equivalence_test.py` to use real SED weights on real audio** before trusting math-identity for batched path. Add tests for: per-fold batching probe (not just fold 0), shorter-than-60s files, all-silent input, and OpenVINO-vs-ONNX numeric drift. Currently zero of these are tested.
3. **Share the Perch embedding across Model_3, Model_7, and a newly-added Bruce-Ridge branch**, then bolt Bruce Ridge on as ensemble member with log-space weight ~0.15. Free move (sub-second compute once embeddings are in RAM) and brings the independent CLIP-Ridge signal into the same kernel — V4 priors already cover the prior side. Combined expected: v4-2-optimal (sim 0.9691) → +0.002-0.005 from Bruce diversity → projected LB 0.970-0.973, well within the v6-best-case ceiling.

Additional flag: `MODE = "submit"` on line 1516 vs `MODE = "infer"` on line 217 — there are TWO MODE assignments in the same file (Model_2 block and Model_7 block); the Model_7 one overrides and is what actually runs. The dry-run-vs-real switch (line 5825) keys on `len(test_paths) == 0`, so real 600-file behavior cannot be inferred from this log alone — push a forced-batch test against `train_soundscapes` (10 files) before May-19 reset.

---

## Agent K5 — Creative skeptical hypothesis generator (killed batch, prescribed: contrarian view)

### DISPUTED ASSUMPTIONS (n=6)

- **Claim: "50% transfer rate from OOF AUC → LB AUC."**
  - Why suspect: It's a **back-fit from a single point**. The only real OOF→LB datapoint is exp019 (OOF 0.9586 → LB 0.755 with bugs = -85% transfer, or 0.755 absolute). The "50%" is derived ex-post by assuming bugs explained the gap. The doc says "Simulation predicts the actual LB drop within 0.003" — that's *suspiciously perfect* for a model fitted to explain the very phenomenon. With n=1 the variance is unbounded; v4 could land anywhere from 0.93 to 0.97.
  - Alternative: There may be a third bug we haven't found. Check column order in sample_submission vs prior_df alignment (postproc_v4.py reindexes by class name — but if class order differs between `pseudo_hour_priors.csv` and `sample_submission.csv`, all labels shift). Check whether `align()` actually preserves order across all 234 classes.

- **Claim: "Labeled prior > pseudo prior" (V4 hybrid → 0.9691 vs 0.9585).**
  - Why suspect: ROUND25 shows the labeled hour prior **alone is 0.489 macro-AUC LOO-site** — *worse than random*. The labeled set is 61% S22 (frog-night). When you replace 75 of 234 pseudo columns with labeled values, you're injecting S22-night bias into all 9 sites and 24 hours. The "+0.0106" sim win came from Bruce's S22-night-biased OOF — i.e., the eval set has the same bias as the prior. **Circular evaluation.**
  - Alternative: V4 might LOSE 0.01 on LB because the true test (Pantanal, 2025, daytime-heavy per ROUND19) doesn't match labeled S22 night. Hour-only labeled prior at w=0.025 may transfer better than the hybrid; ranking could be v4-3 > v4-2 > v4-4.

- **Claim: "Dead-hour fix: use global mean for hours 11-16."**
  - Why suspect: 2025 train_soundscapes (ROUND19) are *daytime-recorded* (peaks at 06:00, 10:00). Test is 2025 → likely has substantial daytime content. The "global mean" is dominated by night (hours 0-4, 18-23 = 80% of labeled). Using night-dominated mean as the "daytime prior" likely UNDER-shifts daytime bird species (chacha1, whtdov, chvcon1) that DO occur 11-16.
  - Alternative: Hours 11-16 should use a **2025-only daytime prior** rebuilt from the 237 2025 files (15-25% of which fall in 06-11). Or: use a zero-shift (w=0) for hours not in pseudo cache, leaving exp019 unmodified at those rows.

- **Claim: "exp019 ceiling without retraining is ~0.965."**
  - Why suspect: The 0.965 ceiling assumes 50% OOF→LB transfer of the +0.038 sim gain. But test files distribution may be MORE daytime-skewed than labeled (test is 2025-recent only). If priors trained on 2021-2024 night-data are misaligned to 2025-day test, the transfer could be NEGATIVE (priors hurt). Conversely if test is more night-like than labeled, transfer could be 80-100% and ceiling is 0.98.
  - Alternative: True ceiling is bimodal — either 0.95 (priors break on daytime test) or 0.97+ (priors help if test=night).

- **Claim: "75 labeled classes ⊂ test set."**
  - Why suspect: Never verified. The 75 labeled classes are derived from train_soundscapes_labels, which is dominated by S22 frogs. Test is private; we don't know which species actually have positive labels in test (macro-AUC skips classes with no positives). If 30 of the 75 labeled species have ZERO test positives, those 30 contribute NOTHING but their bias still corrupts neighboring predictions on the same rows.
  - Alternative: Apply prior only to classes with non-zero pseudo-cache mass across multiple sites (proxy for "real species in Pantanal").

- **Claim: "Bruce standalone LB 0.755 = bugs."**
  - Why suspect: Sub2 was Bruce standalone with same w=3.0 bug; simulated 0.78, actual 0.755. But what if Bruce alone simply scores ~0.85-0.88 on LB and the bug subtracted only -0.10, not -0.13? Then "Bruce alone with corrected w=0.05" should reach ~0.85-0.88 LB, not 0.93. **The Bruce 0.93 OOF assumes Bruce sees the full test distribution well — but Bruce was trained on labeled 739-row set, 61% S22.**

### UNTRIED IDEAS RANKED BY (gain × probability) / cost (n=8)

1. **Per-file top-K post-processing** (`final[file,win,c] *= mean(top-K of final[file,:,c])`)
   - Mechanism: 2nd-place BC2025 trick (per alexycactus's May-18 notebook); boosts confident-within-file classes, suppresses noise. Preserves within-file ranking but changes cross-file ranking (which is what macro-AUC measures).
   - Cost: 1 line; 1 submission slot.
   - Est gain: +0.005 to +0.01 LB (top teams reported).
   - Risk: Low — applies multiplicatively on raw probs, not rank-power.
   - Why nobody (in our corpus) has tried: only surfaced today (May 18) in a public kernel. V3 benchmark didn't include it.

2. **Multi-iteration prior refit on v4 predictions**
   - Mechanism: Run v4-2-optimal once → get predictions on test_soundscapes-style hold-out → refit pseudo_hour_priors from those predictions → re-stack. The pseudo-cache is iter-1 from Perch alone; doing iter-2 on a better backbone (exp019+labeled prior) closes the noisy-student loop. Sydorskyi/Babych BC2025 winners did 3-4 rounds.
   - Cost: One local pass; no Kaggle submission needed; could refit priors before reset.
   - Est gain: +0.002–0.008 expected (smaller than first iter, but signal is sharper).
   - Risk: Medium — could overfit to v4's biases.
   - Why nobody has tried: corpus says "we have ONE iteration" — but it's not even verified that the pseudo-cache covers daytime hours after refit.

3. **Multi-modal gating from imaadmahmood (0.946 LB)** — three conditional rank-blend gates (`fake_only`, `proto_cont`, `sed_only`) trigger on cross-model agreement patterns.
   - Mechanism: Instead of fixed blend weights, use disagreement signal between exp019 and Bruce to selectively trust one over the other.
   - Cost: ~30 LOC; integrate into v4 final step.
   - Est gain: +0.003-0.008 (this is what got imaad to 0.946).
   - Risk: Tuned on imaad's pipeline, may not transfer.
   - Why nobody (in corpus) has tried: imaad kernel was downloaded but its gates weren't extracted as a technique.

4. **Daytime-specific prior table from 2025 train_soundscapes**
   - Mechanism: Build a separate prior from ONLY the 237 2025 files (37 at hr 0-4, 157 at hr 6-11). Use that for daytime test rows; use the existing pseudo prior for night rows.
   - Cost: One table; pre-computable; ~10 LOC change.
   - Est gain: Unknown but plausibly +0.003-0.010 because it fixes the dead-hour problem CORRECTLY (rather than using night-mean as a fake daytime prior).
   - Risk: Medium — 157 daytime files is small.
   - Why nobody has tried: The corpus globalizes the 11-16 fix without recognizing the 2025-daytime regime shift.

5. **Submit v4-1-control + raw exp019 + Bruce-only as 3 different submissions** (max diversity, not bracket)
   - Mechanism: instead of submitting 5 variants of "exp019 + labeled prior" tuned on different weights, submit (a) v4-2-optimal, (b) raw exp019 (control), (c) v4-2 with HOUR-ONLY (no site), (d) Bruce-only-fixed-w=0.05, (e) top-K post-proc on exp019. Final-2 pick gets to choose maximally diverse.
   - Cost: Same 5 slots; reorders queue.
   - Est gain: Doesn't increase max LB, but increases probability of getting at least one above 0.96.
   - Risk: Low. Tom Capybara's "inverse guidance" principle.
   - Why nobody has tried: Current plan brackets around v4-2-optimal — assumes the optimal weight is the safe bet.

6. **Sonotype FULL-SHARE for Jaccard=1.0 clusters (not max-pool)**
   - Mechanism: For son15==son16==son02==son14==son22==son23, ALL members of the group get the AVERAGE (not max) of the cluster. Currently CONTEXT.md only does max-pool broadcast on missing classes. Imaadmahmood does `group_max` — but if the labels are duplicated by construction (Jaccard=1.0 confirmed in ROUND24), then mean is unbiased while max introduces a positive bias.
   - Cost: 5 LOC.
   - Est gain: +0.001–0.003 (sonotypes = 10.7% of macro-AUC per ROUND 7).
   - Risk: Tiny.
   - Why nobody has tried: The corpus says broadcasting is no-op on rank-power — but it's no-op only when broadcast is *static lift*. Within-class re-ranking via mean-share could shift ranks.

7. **Per-class temperature on LOGIT-space outputs (NOT rank-power)**
   - Mechanism: Apply per-class temperature scaling BEFORE the rank-power transform. The corpus dismissed temperature scaling because it's a no-op on rank-power; but exp019 internally has logit-space outputs before its rank_power(0.6) transform. If we hook into exp019 PRE-rank-power, temperature works.
   - Cost: Requires modifying exp019 kernel internals — high cost.
   - Est gain: Could be +0.005-0.010 on rare classes per ROUND27 forensics.
   - Risk: Medium — needs careful kernel edit.
   - Why nobody has tried: Everyone treats exp019 as black-box.

8. **CRITICAL bug check: column-order mismatch between priors CSVs and sample_submission**
   - Mechanism: `postproc_v4.load_priors_hybrid` reindexes by class name, but if any of the 234 classes have inconsistent string keys (e.g., trailing whitespace, case differences, "47158son01" vs "47158_son01"), the prior could silently misalign for some columns. This would manifest as random poor performance on certain classes.
   - Cost: 5-min audit.
   - Est gain: If a bug exists, +0.005 to +0.02 (one of the silent killers).
   - Risk: Just diagnostic.
   - Why nobody has tried: assumed correct because OOF eval used same code path; but Bruce 739-row eval uses CSV→array conversion in a different order.

### CONTRARIAN TAKE

The whole "stack priors on exp019" approach assumes test ≈ labeled. **It doesn't.** Labeled is 61% S22, mostly 2021-2024, night-recorded; test is 2025-only, 23 sites, hour-distribution unknown (probably more daytime than labeled because 2025 train_soundscapes peak at 06:00 and 10:00). Every prior we've built leaks 2021-2024-night-S22 bias into a 2025-daytime-multi-site target. The reason the public 0.948 plateau exists with 321 kernels is precisely that these priors saturate at the labeled-set ceiling, and going higher (Yannan 0.962, Nikita 0.959) requires **different training data**, not better post-processing. **A radical alternative**: use exp019 untouched as one slot, drop ALL priors, and devote the other slots to (a) Bruce-with-w=0.05 standalone — it might actually beat exp019 on the 30% of test rows where exp019 is wrong, (b) top-K post-proc only, (c) imaad's three-gate blend, (d) a 2025-only daytime prior. The 0.968 simulated ceiling is a **mirage from circular eval**; the true ceiling without retraining is probably 0.953-0.957, and the diversity strategy maximizes probability of landing there.

### ONE EXPERIMENT BEFORE MAY-19 RESET

**Run the v4-2-optimal pipeline locally on the 30 labeled validation files used in V6_AUG_BENCHMARK, stratified by hour bin (night vs daytime), and report per-bin AUC delta vs exp019-untouched.** Cost: one local run (no Kaggle slot). Output: if daytime-bin AUC is FLAT or NEGATIVE while night-bin is +0.04, we know the prior helps night and hurts daytime — meaning v4-2 will likely UNDERPERFORM exp019 on a test set that's more daytime-skewed than labeled. The fix is then to gate the prior application by hour (skip 06-16, apply 18-05). This 30-min experiment de-risks the entire May-19 batch.

---

## Agent F1 — Fresh unbiased exploration (no prescribed task)

### 1. The single most material discrepancy in CONTEXT.md

**The corpus's "0.949 baseline" is a fiction the team never personally established.** Every section of CONTEXT.md treats `sunderekkiz/birdclef-2026-exp019-eos4-rank-power-06` as their own 0.949 baseline that they're "moving from." But the team's actual best public-LB submission (verified via Kaggle API on team `adkasd` / "dia") is **0.947**, achieved four separate times (May 13–14) by replaying various public kernels. They never ran vanilla exp019 themselves, never submitted a 0.948+ score, and currently sit at **rank 538** with 0.947 (0.015 below leader Yannan Chen at 0.962).

Implication: every "+0.005 improvement over 0.949 → expected 0.965 LB" narrative in CONTEXT.md (V3/V4 docs, ROUND 30/31 ensemble sims, etc.) is anchored to a baseline they don't actually hold. Their realistic delta is from 0.947, not 0.949.

### 2. The dead-hour bug is real and verified, but the post-mortem omits a second bug

I reproduced the dead-hour gap directly from `inference_notebooks/priors_bundle/pseudo_hour_priors.csv`:

```
Hour:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
Sum:  ~6 ~6 ~6 ~6 ~6 10 10  9  7  8  9  0  0  0  0  0  0  7  7  6  6  6  6  6
```

Hours 11–16 have **literal zeros** for every class. With `EPS=1e-7` clip + `w=3.0` weight, the patch shifts every class logit by `−48.3` for any test row at those hours. That collapses predictions to ~`σ(−48) ≈ 0` uniformly and destroys class ranking → random per-class AUC for ~20% of test rows. INVESTIGATION_LB_DROP.md correctly diagnoses this.

However, the post-mortem's "weight too aggressive on rank-power outputs" claim is **probably also right but for the wrong reason**. Inspecting `sub1_hour3.py`:

- The patch is applied to the **already-post-processed** `submission.csv` written by exp019. That CSV is in PROBABILITY space, not rank-power space. The values are *outputs* of rank-power scaling fed through sigmoid clamping.
- Exp019's submission.csv has values clamped to roughly `[0.477, 0.555]` (per V3 finding).
- Converting `p=0.477` → logit `≈ −0.092`; `p=0.555` → logit `≈ +0.221`. The whole class-discrimination *logit range* is ~0.31.
- Then we add `w * log(prior)`. For the smallest non-dead-hour prior values (e.g. `1e-5`), `log(1e-5) ≈ −11.5`, so `w=3.0` adds shifts up to `−34.5`. **Even at non-dead hours, the patch overwhelms the actual signal** because the signal is in a 0.31-logit band and the prior shift is 100× larger.

So the failure is not specifically a "dead-hour" issue — **the same patch with `w=3.0` would hurt at every hour because exp019's output dynamic range is far too compressed for additive logit shifts of that magnitude**. The dead-hour case is just the extreme.

The V3 fix (`w=0.05` + `site_shrinkage`) addresses the magnitude problem, but the V3/V4 OOF benchmarks are run on Bruce's **uncompressed** logits, not on exp019's rank-power output. The OOF→LB transfer assumption could still fail.

### 3. The OOF benchmark distribution itself is broken

The 739 labeled-window OOF set Bruce Wu derived from `train_soundscapes_labels.csv` has **zero rows from 14 of 23 sites and zero rows from hours 8–17 inclusive**. CONTEXT.md acknowledges this (ROUND 17, sec 5; ROUND 22), but never resolves it. The team:

1. Built 5 priors from this biased 739-row set
2. Validated 5 post-processing variants on the same 739-row set
3. Got "0.9586 macro-AUC, +0.028"
4. Submitted → LB 0.755–0.920 (−0.029 to −0.194)

The pattern is textbook **Goodhart drift on the proxy metric**. The V4 "+0.0125 over V3" claim (0.9691 on hybrid prior) is measured on the *same biased OOF* — the very metric the team has demonstrated is uncorrelated with LB direction. There's no reason to expect V4 to actually beat V3 on LB, despite the optimism in `V4_LABELED_PRIORS.md` ("LB 0.965–0.978 expected").

### 4. Yannan Chen / cudacoding may not be using the "iterative noisy-student" recipe everyone assumes

The corpus repeatedly assumes the top-3 (Yannan Chen 0.962, "more exp is all you need" 0.959, Nikita Babych 0.959) are using BC2025-style multi-iter noisy student. Verifying against the live leaderboard:

- Yannan Chen has 241 submissions in ~2 months — that's ~4/day average, *exactly* the submission daily limit. Heavy iteration but not necessarily noisy-student-shaped.
- "BirdCLEF+ 2026 Team🤗🤗🤗" jumped to **rank 2 at 0.960** (corpus had them as rank 7 at 0.957) — the team is `shtljw, tonylica, yiheng`. Tonylica was already at rank 7 alone; the jump from 0.957 → 0.960 came from a 3-person merger and likely an ensemble of three independent pipelines, **not** noisy-student innovation.
- Nikita Babych is at 0.959 with 305 subs — he's reusing his BC2025 pipeline (the corpus' claim is correct here).

The "0.96+ requires private noisy-student" framing is likely overfit to one observable (Babych's published BC2025 recipe). A 3-way team merge of strong public-tier pipelines may explain the rank-2 spike, suggesting **team mergers** (deadline May 27) may matter more than additional training rounds for the next ~10 LB points.

### 5. The V4 "labeled prior" hybrid likely won't help

The corpus' V4 strategy replaces 75 of 234 prior columns with labeled-derived values, fills the rest with pseudo. The hybrid OOF jumps from 0.9585 → 0.9691. But:

- The 75 labeled-covered classes are exactly the classes that dominate the labeled OOF set (by construction — they appear in `train_soundscapes_labels.csv`).
- Replacing those columns in the prior, then evaluating on the same labeled set, **introduces label leakage**: the model's "prior" now contains the answer key for the test rows that count.
- On real test data, this should give ~zero advantage over pseudo (the labeled prior is built from a tiny S22-night-dominated 739-window sample; pseudo is built from 127k windows across all sites).

I'd predict V4-optimal scores **worse than V3** on the real LB, not better. The V4 OOF gain is largely a leakage artifact.

### 6. Verified-correct claims worth acting on

A few corpus claims that withstand verification:

- **Bruce Wu's `clip_student_bundle.pkl`** is genuinely the strongest no-training public baseline (OOF 0.93 on Perch-frozen features). Could be directly added as ensemble diversity at near-zero CPU cost.
- **`train_soundscapes_labels.csv` does have block-doubled rows** (1478 → 739) and the organizers' acknowledgment has not been pushed yet — `.drop_duplicates()` is a free win.
- **Hour 11–16 zero coverage in pseudo-cache** is real (verified).
- The **codec bitrate shift** (train_audio 86 kbps vs train_soundscapes 72 kbps) and **clipping concentration at S01/S13/S10** are real — these are training-time fixes the team correctly identifies as out-of-scope for inference patches.
- **Team is at rank 538** with 16 days until deadline. To reach prize zone (top 5 / 0.957+) they need +0.010 LB which the corpus' tooling-only path cannot deliver — the 0.948 plateau represents Perch's transfer ceiling.

### 7. Highest-leverage suggestions the corpus underweights

Given the realistic position (rank 538, 0.947, 16 days, no GPU training pipeline mentioned, 5 subs/day):

a) **Submit vanilla exp019 untouched** as a baseline sanity check. The corpus assumes it scores 0.949; this is unverified. If it actually scores 0.949, that's free +0.002 over their 0.947 best. If it scores lower, the entire V3/V4 narrative collapses and they should stop spending submission slots on post-processing.

b) **Build an ensemble of the three best public 0.948+ kernels in rank space**: nina2025/eos-4, sunderekkiz/exp019, and karnakbaevarthur/power-optimization. Rank-averaging diverse strong baselines typically gains +0.001–0.003 with zero risk. The corpus' May 17 "Raunak 0.947 + V73 5-fold blend" attempts (LB 0.931–0.932) suggest the team has been blending in *probability* space with weights that don't preserve rank — rank-blending with equal weights is more robust.

c) **Use the team-merger deadline (May 27)**. The corpus dismisses team mergers in passing; rank-2's recent jump to 0.960 came from a 3-person merge. The team is currently a solo (1 member). Finding 1–2 collaborators at 0.948–0.950 willing to merge would beat any post-processing effort.

d) **Stop optimizing for the 739-window OOF metric.** The corpus has demonstrated empirically that this proxy is anti-correlated with LB direction for the patches it scores highest. Future patch validation should use a *held-out* slice that mirrors test-hour distribution (i.e., sample some hours 11–16 by site-shifting), or just bypass OOF entirely and use direct LB feedback (1 sub/day budget).

### 8. Files that load-bear the inaccuracies

- `/home/user/opencode/birdclef-2026/CONTEXT.md` — multiple sections describe "exp019 0.949 baseline" as if it were the team's own (analysis/CODE_REVIEW.md line ~941–950; meta_analysis/ROUND7_DEEP_FINDINGS.md line ~2517; inference_notebooks/ throughout). None of these reference an actual `adkasd` submission scoring 0.948+.
- `/home/user/opencode/birdclef-2026/inference_notebooks/V4_LABELED_PRIORS.md` — the +0.0106 OOF gain is plausibly leakage; LB transfer not predicted.
- `/home/user/opencode/birdclef-2026/inference_notebooks/sub1_hour3.py` line 108–112 — the `EPS=1e-7` clip in the original patch is what produces the −48 logit shift in dead hours. The fix is in `postproc_v3.py` line 90–98, which the team has not yet pushed as a submission (sub1 v3 was submitted but actually used `w=3.0` per the description — they pushed the *interface* fix but kept the buggy weight).

### TL;DR

The corpus is meticulously researched but built on a baseline (`exp019 LB 0.949`) the team never actually achieved on Kaggle. Their best is 0.947. After spending 5 daily submissions on May 18 chasing OOF-validated patches, they dropped 0.027–0.194 LB instead of gaining. The proxy OOF metric (Bruce Wu's 739-row labeled set) is empirically anti-correlated with LB direction for prior-based patches, and V4's claimed advance over V3 is plausibly OOF leakage. With 16 days to deadline and the team ranked 538th, the corpus' tooling-only strategy is unlikely to reach the prize zone. The single most valuable submission slot at this point is **untouched vanilla exp019** to verify whether the 0.949 baseline even exists for them.

---

## Agent F2 — Fresh unbiased exploration (no prescribed task)

### TL;DR

The corpus's **central optimistic claim** — that a `pseudo + labeled hour-prior` post-processing stack will lift exp019 from LB 0.947 → predicted LB **0.963-0.975** — is **largely an artifact of training-set leakage** in how the labeled prior was benchmarked. The pseudo-only weight-corrected variant (v3 w=0.05) may yield a modest **+0.001 to +0.010** real lift; the labeled-hybrid variant (v4) is **closer to +0.000 to +0.005** despite the simulation showing +0.0125 over v3. The "expected 0.965-0.975 LB" target is not grounded.

### What I verified (high confidence)

| Claim from CONTEXT.md | Verified | Notes |
|---|---|---|
| `train_soundscapes_labels.csv` has 1478 raw rows → 739 unique (every row doubled) | YES (independently from Kaggle) | Confirms ROUND8/10 |
| 66 labeled files, 9 sites (S03/S08/S09/S13/S15/S18/S19/S22/S23), hours 00-07 + 18-23 only | YES | Exact match |
| `pseudo_hour_priors.csv` hours 11-16 are all zero (dead-hour problem) | YES (Σ priors = 0.0000 across all 234 classes for hr 11-16) | Confirms INVESTIGATION_LB_DROP.md root cause |
| User's actual current LB best | **0.947** (team `dia`/`adkasd`, rank 538/3617, 140 submissions) | CONTEXT.md repeatedly cites 0.949 — this number is **not** their own; it belongs to `sunderekkiz/exp019` |
| May 18 0:00-0:15 UTC: 5 submissions (sub1-sub4 v3/v4) scored 0.755-0.920 | YES | Confirms the LB-drop catastrophe |
| Corrected v3 (w=0.05) and any v4 (w=0.025) variants exist in repo as `.py` plans only | YES | No corrected variants ever pushed to Kaggle / submitted to LB |

### The new finding — leakage in the v3/v4 simulation

`birdclef-2026/inference_notebooks/priors_bundle/hourly_species_priors.csv` is **literally** the per-hour observed frequency of each species in `train_soundscapes_labels.csv`. Concrete proof:

- Hour=1, species 517063 (Southern Orange-legged Frog): file says **0.843137**; recomputed P(species | hour=1) from the labels CSV: **43/51 = 0.843137**. Identical to 6 dp.

The `benchmark_v3.py` / `sweep_v3.py` and the postproc_v4 "hybrid prior" experiments then evaluate `Bruce-OOF + λ·log(hour_prior)` on the *same 739 windows* that produced the prior. This is training-on-test.

Magnitude of the leakage, measured directly:

```
Naive (on-train) macro-AUC of labeled hour-prior alone: 0.9449   (75 classes)
LOO-file honest macro-AUC of labeled hour-prior alone:  0.6227   (75 classes)
Leakage delta:                                         +0.3222   AUC points
```

Even with leakage corrected, P(species | hour) does carry **some** real signal (0.62 > 0.50 random), so the prior is not worthless — but the +0.0125 v4-over-v3 advantage in the corpus is **essentially the leakage signature**, not an honest gain. The corpus author flagged the related issue in INVESTIGATION_LB_DROP.md ("OOF distribution must match LB distribution") but only diagnosed the dead-hour case; they did not catch the priors-from-eval-set leakage.

### Other discrepancies / things worth questioning

1. **"0.949 baseline"**: CONTEXT.md uses this throughout but the user's true best is **0.947** (~+0.002 worse). All gain projections should be shifted down by 0.002.

2. **Pseudo-prior is also leaky**, but more weakly. The pseudo-cache embeddings (`backtracking/birdclef2026-pseudo-cache-v1`) are Perch v2 predictions on `train_soundscapes`. The 66 labeled files are a subset of `train_soundscapes`. So even the pseudo-only v3 simulation has some test-train overlap, though Bruce's 3-fold CV provides partial mitigation.

3. **Bruce's 0.9304 OOF**: corpus reports per-fold 0.9214 / 0.9396 / 0.8813 — the >0.05 spread between folds, especially fold 2's 0.88, suggests folds are imperfectly stratified. The "0.93" headline is fair-ish for Bruce in isolation (the model is trained per-fold) but the **stacked** numbers (0.96+) are not — they're held to the wrong yardstick.

4. **The user's CONTEXT.md is contradictory about what's been pushed**: the V4_LABELED_PRIORS.md "expected LB 0.965-0.975" reads as forward-looking. PUSHED_KERNELS.md only shows the v1 (w=3.0) buggy push. No corrected v3/v4 variants appear in `kaggle competitions submissions`. So **the entire v3/v4 LB lift remains untested on Kaggle.**

5. **Dead-hour fix may be unnecessary anyway**: train_soundscapes (the SwiftOne duty-cycle deployment) has **near-zero data at hours 11-16**, suggesting the recorder doesn't record during local daytime. The sample test file is at 01:00 UTC (night). It is *plausible* that test_soundscapes is night/dawn/dusk-only, in which case the dead-hour fill is a no-op. The investigation document assumed "test rows hit hours 11-16" without verifying. Easy to check by sampling test_soundscapes filenames at submission time (or reading them from sample_submission once it's populated).

### What's strongest in the corpus (worth keeping)

- **Per-class Perch calibration finding** (ROUND23): 517063 under-predicted 107×, compot1 over-predicted 80×. These are computed from the *same* labels but the correction direction is structural (Perch's training corpus problem), not idiosyncratic — likely transfers to test.
- **Sonotype aliases** (ROUND17/20): son15≡son16, son22≡son23, 43435≡son14. Topological coupling, not statistical. These are safe to apply at inference and the corpus correctly notes they have ~0 AUC effect because the alias pairs likely co-rank already.
- **Codec domain shift** (ROUND13): train_audio 86 kbps vs train_soundscapes 72 kbps. Verifiable from OGG headers; not a leakage claim.
- **Sites without labels** (S01, S02, S04-S07, S10-S12, S14, S16, S17, S20, S21): 14/23 sites. Site-prior won't help for these regardless of weight; the existing graceful fallback is the right design.
- **The 0.948 PLATEAU is real**: 868 teams crowd 0.945-0.95; rank ~538 confirms this. Crossing it requires non-template work (custom-trained SED, multi-iter pseudo-label, dedicated insect-amphibia model — all per ROUND 9 / ROUND 34).

### Recommended next steps (research-only, no code changes)

1. **Re-benchmark v3 and v4 with leave-file-out cross-validation** before pushing more LB submissions. The current `benchmark_v3.py` uses naive evaluation. A clean script: hold out 13 random files (~20%), refit priors on remaining 53 files, score on held-out. Repeat 5×. Compare honest CV macro-AUC vs the leaky 0.9585/0.9710 numbers. Expected: collapse to ~0.93-0.94.

2. **At submission time, log the first 50 test filenames** to confirm whether test hours fall in 11-16 (dead band). If they don't, the dead-hour fix is irrelevant and v3 can simplify.

3. **Try the v3 w=0.05 patch on one slot only** as a low-risk probe. Expected real LB: 0.946-0.952 (vs the corpus's 0.963). Treat the OOF projection as a ceiling, not an estimate.

4. **Stop trusting Bruce-OOF as an LB proxy**: it under-represents daytime hours, over-represents S22 night, and is the data the labeled priors were fit on. Build a held-out validation split from `train_soundscapes` files not in the labeled subset (use Perch+self-distillation pseudo-labels as soft truth, evaluate macro-AUC on classes where pseudo labels have high confidence). That's the cleanest "almost-test" surface available.

5. **The real lever is still elite-tier training**, per ROUND 34: a 5-fold SED EfficientNet with multi-iter noisy-student + Mauricio's anchored-stage2 + Nikita's dedicated insect_amphibia model. Inference-only patches are post-hoc and largely saturated.

Key files referenced:
- `/home/user/opencode/birdclef-2026/inference_notebooks/priors_bundle/hourly_species_priors.csv` — leaky prior (matches labels CSV exactly)
- `/home/user/opencode/birdclef-2026/inference_notebooks/postproc_v4.py` — hybrid prior loader, layers leakage on top of v3
- `/home/user/opencode/birdclef-2026/inference_notebooks/benchmark_v3.py` — benchmarking script, evaluates priors on data used to fit them
- `/home/user/opencode/birdclef-2026/inference_notebooks/INVESTIGATION_LB_DROP.md` — correct root-cause analysis of v1 LB drop, but didn't catch the deeper leakage issue
- `/home/user/opencode/birdclef-2026/inference_notebooks/V4_LABELED_PRIORS.md` — author's optimistic 0.965-0.975 LB projection that I'm challenging

---

## Agent F3 — Fresh unbiased exploration (no prescribed task)

### What I verified

- **CONTEXT.md is internally coherent and largely accurate** as of its generation time (2026‑05‑18 08:01 UTC). I sanity-checked the most consequential numbers:
  - Sample submission: 234 species columns, baseline 0.004274 (= 1/234), row format `BC2026_Test_<id>_<site>_<date>_<time>_<endsec>` ✅ matches CONTEXT
  - Taxonomy: 162/35/28/8/1 (Aves/Amphibia/Insecta/Mammalia/Reptilia) ✅
  - `pseudo_hour_priors.csv` has dead-hour bug: rows for hours 11–16 are all zeros — exactly as INVESTIGATION_LB_DROP.md describes (verified by summing class columns per row at `/home/user/opencode/birdclef-2026/inference_notebooks/priors_bundle/pseudo_hour_priors.csv`)
  - `hourly_species_priors.csv` (labeled) covers only hours **0–7 and 18–23** (no daytime). So V4's "hybrid" prior also has no daytime coverage.
  - sub2's bug at line 189 of `inference_notebooks/sub2_bruce_standalone.py`: `np.clip(hour_prior_arr, EPS=1e-7, 1.0)` followed by `log()` and `+ 3.0 * log_hour_prior` → **−48 logit shift on every class for hours 11–16**

### Where CONTEXT.md is overstated or wrong

**1. The "0.949 baseline" framing is misleading.**
The user's actual public LB best is **0.947**, not 0.949. CONTEXT.md repeatedly says "your 0.949 baseline (exp019)" and treats it as the user's score. From `kaggle competitions submissions birdclef-2026`:
- 4 user submissions at 0.947, 4 at 0.946, several at 0.944–0.945
- **No user submission ever scored 0.948 or 0.949**
- The 0.949 number is Derek (sunderekkiz)'s public exp019 kernel score — a different person's submission

This matters because every "+expected delta" framing in V3/V4 docs is anchored to a number the user hasn't actually achieved. The realistic baseline for the user is **0.947**, and "matching 0.949" would already be a +0.002 unsolved gain.

**2. The leaderboard has shifted since CONTEXT was written.**
A few hours after CONTEXT generation:
- New rank 2: **"BirdCLEF+ 2026 Team🤗🤗🤗" at 0.960** (May 18 07:11 UTC) — this team is **not mentioned anywhere in CONTEXT.md**. CONTEXT lists tonylica's team as "BirdCLEF+ 2026 Team🤗" rank 7 at 0.957 — likely the same handle, since promoted; or a new team. Worth checking before relying on the "Yannan 0.962 / cudacoding 0.959 / Nikita 0.959" top-3 framing.
- **`nina2025/birdclef-2026-eos-5`** dropped (May 18 05:37 UTC, already 98 votes). I pulled it: it's `Model_2 (LB 0.928, weight 0.0305) + Model_5 (Derek's exp019 LB 0.949, weight 0.9695)`. So EoS-5 = EoS-3-style ensemble around Derek's 0.949 instead of around Karnakbayev's 0.948. CONTEXT only saw up to EoS-4.

**3. The "V4 hybrid prior fix" was never tested on real LB.**
The CONTEXT V4_LABELED_PRIORS.md document claims OOF macro-AUC 0.9691 (vs V3 0.9585) and "expected LB 0.965–0.975". But checking `kaggle competitions submissions`: **only `sub2 v4` (standalone Bruce + buggy w=3) was submitted and it scored 0.755**. The actual V4 fix (w_hour=0.025, fill-dead-hours-with-global-mean) was never submitted. So the entire V4 "labeled-prior is the missing piece" narrative is **simulation-only**, never validated against the LB.

### The genuinely interesting unexploited finding (my contribution)

**The LB drop magnitudes on sub1 (0.949→0.920) and sub2 (~0.85→0.755) imply the test set contains substantial daytime (hours 11–16 UTC) recordings — which is a category that is entirely absent from training data.**

Decomposition assuming the dead-hour rows get reduced to near-random ranking:
- For sub2 (Bruce standalone, real OOF ~0.93, LB 0.755): if dead-hour rows go to AUC ≈ 0.5, daytime fraction in test ≈ **(0.93 − 0.755) / (0.93 − 0.5) ≈ 41%**. Even with a more conservative Bruce-on-test baseline of 0.85, fraction ≈ 27%.
- For sub1 (exp019 0.949, LB 0.920): the rank-power-transformed scores limit the per-row damage, but the drop still implies **at least 10–15%** of test rows are in dead hours.

**This is significant because:**
- `train_soundscapes` (10,658 files): **zero files** at hours 11–16 UTC (verified via CONTEXT.md ROUND 18, where pseudo_cache hour distribution shows 0 windows in those hours)
- `train_soundscapes_labels.csv` (66 labeled): also zero daytime
- Pseudo-cache (Perch predictions): same — 0 daytime windows
- **Every public prior table in the corpus has zero daytime coverage**
- Yet 10–40% of the test set is daytime — recorders ARE configured to record at those hours, the organizers just chose not to release them as `train_soundscapes`

**Implications:**
1. The site×hour Bayesian prior — central to every 0.948+ kernel — silently falls back to global-prior for 10–40% of test rows. This is a giant unaddressed gap.
2. The "labeled prior > pseudo prior" V4 conclusion is false for daytime rows (both are empty). V4's "+0.011 over V3" gain is entirely confined to the night-time slice that's already easy.
3. Any kernel that explicitly **fills daytime priors with a global-mean fallback** (rather than zero) would mechanically avoid the dead-hour collapse and capture a free win.
4. To genuinely improve, you'd want to **extrapolate a "what's likely calling in Pantanal at noon" prior** — perhaps from:
   - iNat observation timestamps for Pantanal-region species (CONTEXT ROUND 14 noted iNat API exposes `observed_on` — never used)
   - Birds with diurnal activity patterns (most Aves call during daytime, the OPPOSITE of frogs/cicadas)
   - eBird/iNat seasonal+hourly priors for the 162 Aves classes

5. The single highest-leverage fix the user could ship in their next submission slot is dead-simple:
```python
# In any post-processing cell, BEFORE applying log(prior):
for h in range(24):
    if hour_prior_df.loc[h].sum() == 0:
        # Fill with global mean across covered hours
        hour_prior_df.loc[h] = hour_prior_df[hour_prior_df.sum(axis=1) > 0].mean(axis=0)
```
plus dropping the weight from 3.0 → 0.05. CONTEXT acknowledges this in INVESTIGATION_LB_DROP.md, but the user spent the next several days (V3/V4/V5/V6) elaborating on synthetic OOF benchmarks instead of just running the corrected submission. **The corrected post-proc was never actually submitted as of CONTEXT generation.**

### Secondary observation: the work has drifted from highest-EV actions

CONTEXT documents 5 unused daily submission slots that, instead of being used to:
- Re-test the buggy submissions with the dead-hour fix (1 slot, ~0 risk, +0.005 to +0.015 expected)
- Push `exp019_fast` to verify the I/O speedup hasn't broken math (1 slot)
- Submit `aliozanmemetoglu/birdclef-sed-fold-1..4` (LB 0.958, **zero external users** per CONTEXT) ensembled in (1 slot — biggest documented unexploited public lever)

...were spent generating more OOF simulations on the same 739-window labeled set. CONTEXT itself flagged this: "OOF distribution must match LB distribution — Bruce's 739-row labeled set was site-S22-night-biased." Yet V3, V4, V5, V6 are all benchmarked against the same biased OOF.

### One final concrete recommendation the corpus misses

CONTEXT ROUND 28 documents Bruce Wu's `clip_student_bundle.pkl` and `teacher_oof_predictions.npz`. The OOF includes **per-class AUC**. CONTEXT didn't compare per-class which classes Bruce is **worst on at night** to evaluate whether daytime test rows would have any chance at all. Spot-check: Bruce's worst night-time classes are 65377, son08, 74113 (per CONTEXT). For daytime test rows, the natural answer would be high-AUC daytime bird species (Tucker's distilled SED head was trained on full train_audio including diurnal species). So ensemble-wise:
- For night test rows: trust ProtoSSM+Bruce (frog-heavy)
- For day test rows: trust Distilled SED+BirdNET (bird-heavy)

This hour-conditional ensemble routing is missing from every public kernel. A single boolean on hour ∈ {11..16} could swap blend weights between mapped frogs/insects (suppress) and mapped birds (boost). Estimated worth: +0.005 to +0.015 LB and addresses the exact failure mode underneath the recent submission collapses.

---

## Agent F4 — Fresh unbiased exploration (no prescribed task)

**Critical observation**: Today (2026-05-18), 5 submissions have already been used and ALL FIVE FAILED to improve over the 0.947 exp019 baseline:
- sub1 (w=3.0): 0.920 (-0.027)
- sub2 (Bruce standalone): 0.755 (-0.192)
- sub3 (alias+blind): 0.914
- sub4 (alias): 0.920
- sub2 v4 (with fix): 0.755

The "v4 batched bruce hour_prior" still got 0.755 — meaning the dead-hour fix didn't help. The user has hit their daily limit (5/day). v4 kernels are READY but cannot submit until tomorrow.

### 1. The v4 plan's expected gain (+0.014 to +0.019 LB) is likely too optimistic. The corpus glosses over critical OOF→LB evidence already in hand.

**Direct evidence from the actual submission history** (`kaggle competitions submissions birdclef-2026`):

| Submission | OOF claim | Actual LB | Delta vs claim |
|---|---|---|---|
| exp019 base | (n/a) | 0.947 | — |
| sub1 (w=3.0) | OOF +0.028 | 0.920 (-0.027) | -0.055 swing |
| sub2 (Bruce standalone + hour) | OOF 0.93 | 0.755 | -0.175 absolute |
| sub2 v4 (with dead-hour fix) | OOF 0.93 | 0.755 | -0.175 absolute |
| sub3 (alias+blind, w=2) | sim +0.028 | 0.914 (-0.033) | sign-flipped |
| sub4 (alias only, w=3) | sim +0.028 | 0.920 (-0.027) | sign-flipped |

The corpus (V3_FINDINGS_STACK.md) proposes a "halve the simulated gain" heuristic to project LB. That heuristic was already invalidated by v1: it predicted +0.014 LB, actual was -0.027 LB. **Transfer was -100%, not 50%.** The corpus's `INVESTIGATION_LB_DROP.md` blames the weight (w=3.0 too large) and dead-hour bug, but the v4 fix retained the same simulation framework. There is no evidence that fixing the two bugs converts the transfer from negative to positive — only simulation suggests it.

**The Bruce-standalone result is a stronger warning**: even after the dead-hour fix (sub2 v4), Bruce + hour-prior scored 0.755 LB despite OOF 0.93. **That's a 0.18 OOF→LB gap that the corpus never reconciles.** The corpus dismisses Bruce as "weaker than exp019" but doesn't analyze why Bruce's OOF over-claims by 0.18. Likely cause: Bruce's Ridge was fit on labeled S22-night-dominated data. On test data (likely diverse sites/hours), the learned coefficients generalize poorly.

By analogy, the v4 stack benchmark on Bruce OOF (predicting LB 0.965–0.978) inherits the same overfitting risk because it tunes weights on the same 739-row labeled set.

**Concrete recommendation**: Run **v4-1-control** (untouched exp019) FIRST tomorrow to verify the kernel runs end-to-end (~0.947 expected). Then use **slot 2** for `w_hour=0.01` (10x smaller than v4-2's 0.05) as a careful probe. If +0.001-0.005, scale up. If 0 or negative, **stop** and try a different lever (don't burn slots on more weights of the same prior).

### 2. The site×hour prior has a glaring hole at the only test row we can sanity-check.

Sample test file `BC2026_Test_0001_S05_20250227_010002` = **site S05, hour=01**.

Verified locally with `pseudo_site_hour_priors.csv`:
- S05 hour=01 row: **does not exist** (S05 only has hour=03 and hour=17 rows)
- Of 21 sites in the pseudo cache, **only 7 have hour=01 rows** (S01, S02, S12, S13, S16, S18, S22)
- 14 sites have NO hour=01 entry: S03, S04, S05, S06, S07, S09, S10, S11, S14, S15, S17, S19, S20, S21

The fallback chain (`p_site_hour_prior` → `hp[h]` hour prior) is logically sound, but **for any test row at a site/hour not in pseudo cache, site_hour patches collapse to the bare hour prior**, identical to `p_hour_prior` alone. The v3/v4 "stack" benchmark thus measures the marginal gain of site_shrinkage (w=0.2) ON TOP of hour, not a genuine site×hour interaction.

If test data is dominated by sites without hour=01 pseudo coverage (very plausible given how unbalanced unlabeled coverage is), the v4-2 "optimal" stack is functionally equivalent to v4-3 "hour-only" for those rows. So the +0.0011 OOF gap between v4-2 (0.9691) and v4-3-equivalent (0.9637) likely reflects S22-specific overfitting, not a real edge.

**Recommendation**: Confirm v4-3-hour-only and v4-2-optimal converge in real LB. If yes (likely), drop the site_shrinkage component — it's complexity for no real benefit.

### 3. Bruce's OOF dataset is unrepresentative — the corpus identifies this but doesn't act on it.

Confirmed from local files:
- Labeled set: 66 files at 9 sites, dominated by S22 (61% of labeled, 40 of 66 files).
- Bruce's 739 evaluation rows: 0 rows at hours 11-16, 100% at hours 0-7 + 18-23.
- Hidden test set: per organizers, ~600 files across all 23 sites at unconstrained hours.

The whole v3/v4 OOF benchmark stack uses Bruce's OOF predictions, transformed via `to_rank_power_like_exp019()` (a synthetic transform that compresses Bruce's logits into the [0.477, 0.555] range to mimic exp019's actual outputs). **There is no validation that exp019's real outputs behave like rank-power(Bruce)** — they have similar dynamic range but might encode entirely different per-row uncertainty. The "scenario B" 20% synthetic dead-hour injection is also arbitrary — true test dead-hour fraction is unknown.

**This is the single largest methodological weakness.** The corpus mentions it once in INVESTIGATION_LB_DROP ("OOF distribution must match LB distribution") but doesn't fix it. A real validation would require: (a) running exp019 on the 66 labeled soundscape files locally and computing actual OOF, then (b) tuning weights against that.

The lack of an exp019 OOF means **the v4 simulation is fundamentally unanchored to the actual pipeline being patched**. This is why sub1's simulated +0.028 became actual -0.027.

### 4. The "labeled hybrid prior" boost from v3→v4 has a hidden risk worth stress-testing.

Verified locally:
- `hourly_species_priors.csv` covers 75 species over 13 hours (hours 5, 8-17 missing)
- Labeled S22 hour=01 top species: 517063 (84%), 65380 (61%), 24279 (53%)
- Pseudo (all-site) hour=01 top species: compot1 (39%), compau (38%), trsowl (31%), 65380 (29%)

**These tops disagree substantially.** Labeled prior is S22-biased (because S22 = 61% of labeled data). At a test file from S01/S02/S05/etc., the labeled prior may boost the wrong species. Pseudo data for S05 nights (from corpus round 22) showed 24279 (Lesser Snouted Tree Frog) dominant, but labeled prior says 517063 — different species.

This means v4's hybrid replaces a noisy-but-broad pseudo prior with a clean-but-narrow labeled prior. If test sites match the labeled coverage (S22, S08, S15, etc.), v4 helps. If test sites are S01/S02/S04-S07/S10-S12/S14/S16/S17/S20/S21 (14 of 23 sites have no labeled data), the labeled prior may actively HURT vs pseudo on those rows.

**Recommendation**: Add a per-site gate. For sites in `{S22, S08, S09, S15, S19, S23, S13, S18, S03}` (labeled coverage), use labeled prior. For other sites, use pseudo only (potentially with lower weight). The current code already has a site-aware fallback in `p_site_hour_prior` but applies labeled hybrid uniformly via `pseudo_hour`.

### 5. Existing untried levers that the corpus identifies but doesn't pursue.

The corpus lists 14 unused alternative model bundles (ROUND 15, 29, 34). Three are most actionable RIGHT NOW given a 90-min CPU budget and the fact that the user has 5 daily submission slots tomorrow:

a. **`aliozanmemetoglu/birdclef-sed-fold-1..4`** — author is LB rank 4 (0.958), checkpoints publicly available, **used by 0 other kernels** (corpus finding ROUND 8). The team has been chasing post-processing gains while completely ignoring a top-4 author's published weights.

b. **`majkel1337/long-convnextv2-tiny-onnx`** — single-pass 60s → 12×234 inference (uploaded 2026-05-15, 16 downloads). Corpus ROUND 15 measured its OOF at 0.6303 — too low alone but adds architectural diversity (no other public kernel uses ConvNeXtV2).

c. **`tonylica/birdclef-2026-model` LB872.pt** — corpus ROUND 9 fully decoded the config; it's a 2-stage finetune of `aidensong123/bestfold`. **24 importers** means it's been tried, but the corpus shows no attempt at ensembling its rank with exp019 in adkasd's pipeline.

**The current 0.947 → ~0.95 strategy via priors looks brittle (max LB gain ~+0.003 by my analysis, possibly negative).** A simple rank-blend `0.85 × exp019 + 0.15 × rank(alioz_5fold)` is likely +0.003 to +0.008 — comparable expected gain with structurally different risk (ensemble diversity vs prior tweaks).

### Bottom line

**The team's primary plan (v4-2-optimal) is plausible but oversold.** Its OOF→LB transfer projection is built on the same framework that has empirically transferred at -100% over the past two days. The pseudo-cache priors have a real coverage gap at S05/14 other sites + hours 11-16 that the v4 fixes only partially address (global-mean fallback is noise, not signal). Recommended tomorrow:

1. Slot 1: v4-1-control (sanity, confirm 0.947).
2. Slot 2: v4-2-optimal but at **w_hour=0.01** not 0.05 (5x smaller probe).
3. Slot 3: Pure rank-blend of exp019 + aliozanmemetoglu's 5-fold SED (15% weight) — the largest unexploited public lever.
4. Slot 4: v4-2-optimal at the original w_hour=0.05 (the team's actual plan, but as a follow-up not a primary).
5. Slot 5: hold for diagnostic depending on results of slots 1-4.

---

## Agent F5 — Fresh unbiased exploration (no prescribed task)

### The hypothesis I want to test

**Claim:** the corpus's "v4 OPTIMAL" recipe (`w_hour=0.025`, `w_site=0.20`, hybrid labeled prior) will deliver +0.014-0.020 LB.

**My counter-hypothesis:** the v4 OOF simulation is biased the same way the previous one was, and the recipe will plateau much closer to baseline (or hurt slightly), because the **fundamental coverage gap** — daytime hours and out-of-labeled sites — is unresolved. The v4 "fix" is a weight reduction, not a coverage fix.

### Evidence for the counter-hypothesis (all verified from the priors CSVs in the repo)

#### 1. Labeled hour prior covers only nighttime (verified)

`hourly_species_priors.csv` has rows for hours **{0, 1, 2, 3, 4, 6, 7, 18, 19, 20, 21, 22, 23}** — 13 nighttime/dawn/dusk hours. Hours **5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17** are completely absent. The v4 code (`postproc_v4.py` line 38-43) fills the missing hours with `lab_hour.mean(axis=0)`. That mean is dominated by S22 nighttime frog chorus, because **60% of labeled files (40 of 66) are S22 night-time**. Filling a noon test row with this prior pushes `P(Southern Orange-legged Frog | noon) ≈ 0.36` when the biological reality is ~0.

#### 2. Pseudo hour prior has 6 dead hours (verified)

`pseudo_hour_priors.csv` sums across classes at each hour:
```
hours 0-10, 17-23: prior sums in range 5-10 (data present)
hours 11, 12, 13, 14, 15, 16: prior sums = 0.000 (completely empty)
```
The v3 fix fills those 6 dead hours with `df.loc[covered].mean(axis=0)`. The covered hours are heavily night-skewed (train_soundscapes are duty-cycled to night/dawn/dusk per ROUND6a), so the daytime prior fill is also night-biased.

#### 3. Pseudo site×hour prior is extremely sparse (verified)

`pseudo_site_hour_priors.csv` has 120 rows total. Of 23 sites × 24 hours = 552 possible (site, hour) cells, only 120 are populated. Some sites have only 2-4 hours covered (S03: hours 19, 21; S04: 0, 4, 8, 17; S05: 3, 17). **S08 and S23 are missing entirely** from pseudo_site_hour, even though they are critical labeled sites for sonotypes.

#### 4. Labeled site×hour prior is 25 (site,hour) keys, all at S22 + S08 + S15 + S19 + S13 + S18 + S03 + S09 + S23 at specific hours (verified)

Of 552 possible cells, only **25 are covered by labels = 4.5% coverage**. If the private test draws uniformly from 23 sites × 24 hours, **95.5% of test rows fall back to the (also-biased) hour-only prior**.

#### 5. The S05_h01 example test row has ZERO labeled (site,hour) data (verified)

The sample submission row `BC2026_Test_0001_S05_20250227_010002` is at S05, hour=01. S05 has zero labeled keys, and hour=01 has only S22 and S18. So even the **example test row from the data description** would get a fallback prior, not a real one.

#### 6. Bruce's OOF is the same S22-night-biased distribution that produced the failed simulation

The corpus admits "Bruce's 739-row labeled OOF: zero rows at hours 11-16, dominated by S22." Yet **the v4 simulation that produces 0.9691 uses the same OOF**. The 20% "synthetic dead-hour exposure" added in `benchmark_v3.py` reassigns rows to hour 11 by changing the *prior lookup hour*, not by simulating actual daytime species distribution. **This is a methodology error**: it tests "is the prior lookup robust to a missing-hour fallback?" but not "is the prior content correct for daytime?" The simulation cannot detect the systematic daytime bias because the **ground-truth Y matrix never changes** — it's still all S22-night species.

#### 7. Macro-AUC math: why ranking can still degrade

Macro-AUC depends on within-class ranking. For nocturnal-only species (compot1, 517063, 47158son25), applying a strong positive logit shift at noon test rows pushes those windows UP in the ranking. If most of the noon test rows have **zero positives** for these species, the within-class ranking corrupts: nocturnal windows (which are correctly high) are mixed with daytime windows (now also high due to prior). The AUC of those nocturnal classes drops, dragging macro-AUC.

### The clean test the corpus is missing

A defensible OOF that would predict LB: **stratify the OOF by held-out hour groups and held-out sites**. Specifically, evaluate the recipe under three slices:

| Slice | Expected v4 behavior | Test |
|---|---|---|
| S22 night-only (Bruce-style) | OOF claims +0.025 | matches simulation |
| held-out site, labeled hour (e.g. S15 at h=20) | OOF should still gain | partially tests site coverage |
| held-out hour (e.g. any site at h=12) | OOF should *neutralize* the prior | would expose the daytime-bias bug |

The third slice is impossible to simulate without daytime ground-truth labels, which the corpus correctly notes don't exist (no labeled daytime files).

### What the corpus *should* have considered

A **safe** fix preserves the gain at known-good cells and **neutralizes** at unknown cells:

```python
# When the (site, hour) cell or the hour bucket is NOT in the original (unfilled) coverage,
# skip the prior shift entirely OR shift toward a class-frequency-neutral target.
HOURS_WITH_DATA = {0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,23}  # pseudo coverage
LABELED_HOURS = {0,1,2,3,4,6,7,18,19,20,21,22,23}                # labeled coverage

def p_hour_prior_safe(prob, hours, hp_arr, w):
    lp = _logit(prob)
    shift = np.zeros_like(prob)
    # Only shift for hours where we actually had data — otherwise leave untouched
    valid = np.isin(hours, list(HOURS_WITH_DATA))
    shift[valid] = w * np.log(np.clip(hp_arr[hours[valid]], EPS, 1.0))
    return _sigmoid(lp + shift)
```

For test files at hours 11-16, this returns `prob` unchanged (best worst-case behavior) instead of applying a night-biased prior.

### Predicted outcome of the corpus's planned v4 submission

If the private test set is **S22-night-dominated** (matching the labeled set distribution), v4 will deliver a moderate gain, perhaps **+0.005 to +0.010** LB. If it has any non-trivial daytime or non-labeled-site exposure (the rules say "sites overlap" but never claim hour distribution is night-only, and the test set is **600 files** spanning real PAM deployments which by `recording_location.txt` cover diverse conditions), v4 will land closer to **baseline ± 0.005**.

The corpus's predicted +0.014-0.020 LB gain is plausible *only* under the optimistic assumption that the test distribution matches the labeled OOF distribution. The submission history shows previous predictions of this kind failed by 0.05-0.20 LB.

### Most actionable single recommendation

Before consuming submission slots on `sub_v4_*`, the user should add **a "gated" hour-prior variant** (`safe_v4`) that:
1. Applies the hour shift **only when hour ∈ {0-7, 18-23}** — i.e., where pseudo or labeled data exists pre-fill.
2. Returns `prob` unchanged for hours 8-17.

Then submit one slot with `safe_v4` alongside one slot with the existing `v4_2_optimal`. The LB delta between them is a direct measurement of how much daytime test data exists, with no further OOF simulation needed. That single experiment resolves the largest open uncertainty — and it costs only 1 marginal submission slot.

### Other findings worth noting

- The **fork_lineage** finding from FINAL_META_FINDINGS that `aliozanmemetoglu/birdclef-sed-fold-1..4` (LB 0.958, rank 4) is imported by 0 other kernels remains the highest unexploited public lever. Rank-blending these folds with the user's exp019 base is independently verifiable by a single submission and has never been tried by the user per the submission history.
- The **codec re-encoding** finding (`train_audio` at 86 kbps vs `train_soundscapes` at 72 kbps) is a *training-time* fix and the user cannot apply it for the exp019-based path. But for the `sub2` standalone Bruce path, applying a real codec round-trip to the test audio at 72 kbps before Perch inference is a true distribution-matching step that's never been tested. ROUND 9 V6_AUG_BENCHMARK shows `codec_proxy 16k` gave +0.019 on a single model — though transferability to Perch is unverified.
- The current LB top is **0.962** (Yannan Chen). The corpus's "0.968" prediction is **6 points higher than the global LB leader** — physically implausible unless v4 unlocks something nobody else has, which the evidence does not support.

---

# Consolidated Cross-Agent Synthesis

10 independent agents converged on substantially the same diagnosis:

## Load-bearing facts that contradict CONTEXT.md

1. Team's actual best is **0.947**, not 0.949. The account `adkasd` ("dia") has never submitted exp019 vanilla. Every "+Δ vs 0.949" projection is unanchored.
2. **v4 has never been submitted to LB.** All v4 claims are simulation-only. The OOF benchmark scripts cannot be reproduced from the repo (input npz files missing).
3. **The v3/v4 OOF benchmarks are training-on-test by construction.** `hourly_species_priors.csv` is literally P(species | hour) computed from the same 739 labeled windows that are then used to evaluate it. Measured leakage: **+0.32 macro-AUC** (naive 0.9449 vs LOO-honest 0.6227).
4. **The "20% dead-hour exposure" sim is methodologically broken** — it changes only the prior-lookup hour, not the ground-truth Y matrix. Cannot detect daytime-bias problems by design.
5. **OOF→LB transfer rate has been -100% on every comparable submission** (sub1 sim +0.014 → actual -0.027, sign-flipped). The "halve the simulated gain" heuristic was empirically falsified by v1.
6. **Predicted v4 LB 0.968 is 6 points above world leader (Yannan 0.962)** — physically implausible.
7. sub3/sub4 LB scores in CONTEXT.md are **off by 0.06-0.07** (actual 0.914-0.920, corpus says ~0.85).

## Mathematical decomposition (Agent F3)

LB drops imply **10-40% of test set is in hours 11-16**, where all priors have literally zero coverage. Test is bird-heavy in those hours (Aves is diurnal — opposite of frog/insect texture classes everyone optimizes for).

## Caveat (Agent F2)

SwiftOne's duty-cycle deployment skips daytime in `train_soundscapes`. If `test_soundscapes` follows the same duty cycle, daytime fraction may be near zero and the dead-hour fix is a no-op. This DIRECTLY CONTRADICTS Agent F3's decomposition. Slot-1 should be designed to resolve this — log first 50 test filenames at submission time.

## Top-3 competitor intel (Agent K3)

- **Yannan Chen** (0.962) is an NLP grandmaster, BC2026 is his first audio competition. Likely uses NLP-style per-class learned fusion head, not noisy-student.
- **Rank 2** spiked 0.957 → 0.960 from a **3-person team merger** (merger deadline May 27).
- **Babych's `birdclef2025-1st-place-extra-data` (7.8 GB, public)** covers exact missing-class genera (Pithecopus, Chiasmocleis, Dendropsophus). Never explored.
- **`aliozanmemetoglu/raw-pseudos` (346 MB)** — public iter-0 pseudo OOF, 5-fold v2s, ~10 downloads. Free signal nobody is using.
- **`aliozanmemetoglu/birdclef-sed-fold-1..4`** — LB 0.958 author, 0 other importers. Largest documented unexploited lever per FINAL_META_FINDINGS.
- **cudacoding** does test-time training (Stage1 inference → Stage2 online finetune on confident pseudos).

## Convergent recommendations across agents

Multiple independent agents arrived at the same plan:

1. **Slot 1: vanilla exp019 untouched** — first time the team measures their actual exp019 LB.
2. **Slot 2: a logging variant** that prints test-file hour distribution → resolves the Agent F2 vs F3 conflict.
3. **Slot 3: rank-blend exp019 + aliozanmemetoglu's 5-fold SED (15% weight)** — biggest untapped lever.
4. **Slot 4: `safe_v4`** (priors gated to hours where data exists, return prob unchanged elsewhere) — Agent F5's proposal.
5. **Slot 5: v4-2-optimal** as the corpus's actual plan, but as a follow-up not primary. The LB delta between 4 and 5 measures daytime bias directly.

## Untried high-leverage actions

- **Hour-conditional ensemble routing** (Agent F3): night rows → ProtoSSM+Bruce, day rows → Distilled SED+BirdNET.
- **Multi-iter prior refit** on v4 predictions (closes the noisy-student loop without retraining).
- **Per-file top-K self-amplification** (alexycactus May 18 trick).
- **imaadmahmood's 3-gate blend** (got him to 0.946, never extracted as a technique).
- **2025-only daytime prior** rebuilt from the 237 2025 train_soundscapes files.
- **Per-class fusion-alpha** head (chaneyma's ROUND8 trick).
- **Temporal-axis flip TTA** (meenalsinha May 18, NOT in dead-ends list — distinct from time-shift TTA which hurts).
- **EoS-5 micro-blend pattern**: `0.0305 × hideyukizushi + 0.9695 × exp019` (95 votes in one day; community treating it as free +0.001).

## Engineering wins (Agent K4)

- **OpenVINO conversion** of 5 SED folds: -12 min wall-clock (verified path via nikitababich/runtimes-onnx-openvino).
- **Share Perch embedding** across Model_3, Model_7, and Bruce Ridge → adds Bruce as 8th model nearly free.
- **exp019_fast's equivalence_test.py is synthetic** — never tested on real `sed_fold[0-4].onnx` with realistic mels.

## Strategic reframe

The whole "stack priors on exp019" approach assumes test ≈ labeled. The labeled set is 61% S22 night, 2021-2024. Test is 2025-only, 23 sites, hour distribution unknown. The 0.948 plateau exists with 321 kernels precisely because priors saturate at the labeled-set ceiling; going higher (Yannan 0.962, Babych 0.959) requires **different training data**, not better post-processing. The corpus's tooling-only ceiling is likely **0.953-0.957**, not 0.968 — and the rational strategy is diversity + team merger before May 27, not more weight tuning.
