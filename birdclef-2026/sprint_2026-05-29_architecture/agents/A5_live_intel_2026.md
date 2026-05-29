# A5 — LIVE Intelligence Delta (scan 2026-05-29, comp ends 2026-06-03)

Fresh scan vs the previous meta (last scan ~May 18–19, FINAL_META_FINDINGS.md + AGENT_REPORTS K1/K2/K3).
Method: Kaggle REST (Bearer token) — `leaderboard/download` (full 4,088-team CSV), `kernels/list?competition=...&sortBy=dateRun` (300 kernels, 3 pages), `datasets/view` per-slug (existence probe), `kernels/pull` metadata-mount scan (90+ recent kernels), plus WebSearch. The token lacks `models.get`/`discussions.list` scope (those 403 / return HTML), so models + forum confirmed indirectly via mount-graph and web.

---

## 1. LEADERBOARD — top 32 (full LB, 4,088 teams) — **NEW**

| rk | score | team | members | subs | vs May-18 |
|---:|---:|---|---|---:|---|
| 1 | **0.964** | Nikita Babych | nikitababich | 366 | ↑ from 0.959 — **now #1** (NEW) |
| 2 | 0.963 | Yannan Chen | yannan90 | 295 | ↑ from 0.962 (was #1) |
| 3 | **0.963** | Ali Ozan Memetoglu | aliozanmemetoglu,berringurler,denizegememetoglu | 253 | ↑ from 0.958 — **+0.005, biggest mover into top-3** (NEW). MERGE: now 3-person team |
| 4 | 0.962 | more exp is all you need | cudacoding | 357 | ↑ from 0.959 |
| 5 | 0.962 | BirdCLEF+ 2026 Team🤗🤗🤗 | shtljw,tonylica,yiheng | 335 | ↑ from 0.960 |
| 6 | 0.961 | ggkush - anilclaw | nlztrk | 183 | **NEW to top (Tarık nlztrk)** |
| 7 | 0.961 | YK | evgeniimaslov2,xyzdivergence | 313 | MERGE — 2-person, **NEW** |
| 8 | 0.960 | Arunodhayan | arunodhayan | 269 | ↑ (was ~0.955) |
| 9 | 0.959 | Takoi | takoihiraokazu | 338 | **NEW** (known GM) |
| 10 | 0.958 | Sinan Calisir | snnclsr | 175 | ↑ |
| 11 | 0.958 | [Deleted] | — | 19 | deleted team |
| 12 | 0.958 | Diptyajit Das | diptyajitdas | 252 | ↑ |
| 13 | 0.958 | coolz | cooolz | 302 | NEW |
| 14 | 0.957 | Jack Van Dyke | jackvd | 261 | NEW |
| 15 | 0.957 | Youssef Ouertani | youssefouertani | 292 | — |
| 16 | 0.957 | The Mythos of Sisyphus | alturutin | 303 | ↑ |
| 17 | 0.957 | mirandora | mirandora | 310 | NEW (known GM) |
| 18 | 0.957 | Duck said:"Quack, quack!" | pursueml,zejunfool | 326 | MERGE NEW |
| 19 | 0.957 | Herra Huu | herrahuu | 96 | NEW |
| 20 | 0.957 | Tom Capybara | tom99763 | 380 | ↑ from 0.955 |
| 21 | 0.956 | BUET_Perceptron | musaturfarazi,nufayerreza | 249 | MERGE |
| 22 | 0.956 | cmasch | cmasch | 332 | — |
| 23 | 0.956 | 🐣yukiZ🐣 | hideyukizushi | 348 | ↑ from 0.953 (rank-17 public author) |
| 24 | 0.956 | less submissions | 5-person CN merge | 262 | MERGE |
| 25 | 0.955 | Jiacheng Ma | jakkma | 164 | — |
| 28 | 0.955 | Antoine Masq | antoinemasq | 393 | — |
| 29 | 0.955 | Konstantin Dmitriev | kdmitrie | 393 | perch-starter author |
| 30 | 0.955 | Iliamna | calibrator,decotoj | 394 | MERGE |
| 31 | 0.955 | Prompt is all you need | 4-person CN merge | 281 | MERGE |

**Headline deltas:**
- **Nikita Babych retook #1 at 0.964** (was 0.959/#3-4). His BC2025-1st noisy-student recipe + extra-data is paying off late.
- **aliozanmemetoglu jumped 0.958→0.963 and MERGED into a 3-person team** (added berringurler + teammate denizegememetoglu, the SED-checkpoint co-owner). The top is now a single team. This is the most material structural change.
- Whole field compressed upward ~+0.002–0.005; the 0.955–0.957 band is now ~12 teams of mostly team-merges (YK, Duck, BUET, Iliamna, "less submissions", "Prompt is all you need").
- **No public kernel scores above 0.952** (see §2). The 0.957–0.964 tier is still 100% private. The public/private gap remains ~+0.012.
- **No public kernel crossed the old 0.948 Perch plateau with a *disclosed* method** — the plateau merely shifted to ~0.950/0.952 via a new public base stack (below), not via a disclosed breakthrough.

---

## 2. NEW PUBLIC KERNELS / DATASETS / MODELS (last ~10 days, sortBy dateRun)

### The public plateau MOVED: 0.948 → ~0.950/0.952 (**NEW base stack**)
The May-27 voted-kernel crowd no longer mounts exp019 / EoS-4. The uniform new base (17–18 of 35 recent voted kernels mount ALL of these) is:
- `tuckerarrants/bc2026-distilled-sed-public` (dataset)  ← **now the dominant SED base**
- `tuckerarrants/perch-v2-no-dft-onnx` (dataset)
- `tuckerarrants/birdclef-2026-waveform-cache` (dataset)
- `hideyukizushi/sgkfk-202604041716` (dataset, rank-23 author weights) — **still public, verified**
- KERNEL: `hideyukizushi/bird26-reprod-perch-proto-residualssm-train-s7177` (the new ProtoSSM+ResSSM train base)
- MODEL: `google/bird-vocalization-classifier/.../perch_v2_cpu/1` (Perch v2, unchanged)

This is the "0950/0952" lineage you see all over the kernel list (`bc2026-p949-*`, `r0952-*`, `eos7sz`, `eos-9`). It is an *incremental* Perch+ProtoSSM+distilled-SED blend — **not** a new architecture. The 0.948→0.952 gain came from the tuckerarrants distilled-SED + sgkfk weights, both already public.

### Genuinely NEW techniques (not in FINAL_META_FINDINGS.md)
| kernel | date | votes | NEW technique | assessment |
|---|---|---:|---|---|
| **pilkwang/birdclef-2026-eos-oof-gated-pcen** | 05-29 | 97 | **PCEN front-end + OOF-gated blend + BirdNET sidecar + taxonomy(genus/family) smoothing**. Mounts `shadiakiki1/birdnet-analyzer` TfLite + private `pilkwang/birdclef26-sidecar-exp00*`. | **NEW & HIGH** — PCEN per-channel energy normalization as a parallel front-end, OOF-gated weighting, and BirdNET sidecar for the 28 unmapped classes. Closest public thing to a real method delta. Reportedly ~0.952. |
| **pilkwang/birdclef-2026-eos6-pcen-rank-sidecar** | 05-24 | 77 | Same PCEN+rank sidecar lineage (earlier version). | NEW (pilkwang originated the PCEN sidecar line ~May-24). |
| **karnakbaevarthur/hierarchical-taxonomy-post-processing** | 05-27 | 102 | **Hierarchical taxonomy post-processing** — propagate/smooth probabilities up the genus→family tree, then redistribute. 85 `taxonom`/48 `genus` refs in code. | **NEW & MED** — taxonomy smoothing is NOT in the meta findings' dead-ends. Cheap, inference-only, ~+0.001–0.003. Heavily forked (the `*-tax`, `*-hier-tax-*` kernels). |
| **mtoshidesu/test-birdclef-2026-yaroslav-v221-tax** | 05-29 | 91 | "v221" + taxonomy post-proc on the yaroslav/mtoshi base. | derivative of karnak taxonomy idea. |
| **nina2025/birdclef-2026-eos-9** | 05-27 | 343 | EoS line iterated to v9; now built on the tuckerarrants+sgkfk base, taxonomy-smoothed. | iteration, not new method, but highest-voted recent kernel. |
| **meenalsinha/birdclef-2026-improved** | 05-28 | 161 | Updated; temporal-flip TTA + isotonic + taxonomy (carries the K2 temporal-flip finding forward). | known (K2 flagged flip-TTA). |

### Other notable activity
- **Massive "claude-fork" swarm** (`bc2026-claude-*-fork`, `bc2026-p949-*-sedres-*`): dozens of low-vote auto-generated forks (hassan*/sultan*/shahad*/joriahmed/sans6262q/etc.) systematically replaying every public kernel + alpha-sweeping SED-residual blends and probing 2023/2025 models (`syd2025`, `babych2025`, `lihang2023`) via ONNX/OpenVINO. Signal: people are brute-forcing alpha weights and OpenVINO-porting old winners. No disclosed breakthrough.
- HGNetV2 / NFNet backbones appearing (`samejimatink0/...hgnetv2-b0-pretrain`, `starsdaisuki/...henry-nfnet`, `lixinyin/...hgnet-gate`) — backbone diversification beyond EfficientNet. **NEW-ish** but low-vote.
- New DATASETS in window: `pilkwang/birdclef26-sidecar-exp001/002/002b` (the PCEN-sidecar weak-audio caches — private/own), `brendancarlin/birdclef2026-models`, `baiyuby/birdclef2026-distill-models`. Nothing topping the tuckerarrants base.

---

## 3. DISCUSSIONS (last ~10 days)

Token lacks `discussions.list` scope and Kaggle forum pages are JS-rendered (WebFetch sees titles only), so this is partial — sourced from kernel-naming signal + web search.

- **"the secret" thread (Tom Capybara, disc 681146)** — still no concrete technique disclosed as of this scan. Tom Capybara rose 0.955→0.957 but shared nothing. Treat as noise (consistent with K2's read). **Already-known, no change.**
- **Submission-selection / overfitting consensus (NEW, late-comp framing):** the recurring late-comp guidance (Kaggle strategy playbook + BC2021 lore widely re-cited): pick final subs by **your own CV / a personal mini-leaderboard, not the public LB** — the canonical example is the BC2021 team that jumped 7th→1st on private by trusting CV over public LB. With a hidden private LB and a compressed 0.955–0.964 public band (deltas < the noise floor), public rank is low-signal.
- **PCEN + taxonomy-smoothing are the openly-circulating "tricks" this week** (pilkwang + karnakbaevarthur), evidenced by the fork swarm adopting `-pcen-sidecar` and `-tax`/`-hier-tax` suffixes. These are the concrete late-comp insights that surfaced. **NEW.**
- No organizer clarification of note surfaced in-window.

---

## 4. THE "FREE LEVER" — re-verified (importable assets) — **CRITICAL UPDATE**

Existence probed via `datasets/view` (returns full JSON if public; `403 datasets.get denied` if private/deleted — discriminator confirmed against known-good slugs).

| asset | type | status NOW | exact slug | mount path | importers (last 10d) |
|---|---|---|---|---|---|
| aliozanmemetoglu SED 5-fold | Kaggle **Models** | **UNCONFIRMED** (token lacks models.get; `models/list` returns empty for all owners). NOT seen mounted by any recent kernel. Author now LB-#3 & merged → **may have privatized**. | `aliozanmemetoglu/birdclef-sed-fold-1..5` (per FINAL_META) | `/kaggle/input/birdclef-sed-fold-N/...` | **0** |
| aliozanmemetoglu raw pseudo-OOF | dataset | **PUBLIC ✅** | `aliozanmemetoglu/raw-pseudos` (140.8 MB, upd 2026-04-25) | `/kaggle/input/raw-pseudos/` (`pseudo_labels_v2s_oof_raw.csv`) | 0 |
| aliozanmemetoglu pseudo text-init iter-0 | dataset | **PUBLIC ✅** | `aliozanmemetoglu/pseudo-text-init-iter-0` (151.3 MB, 2026-04-15) | `/kaggle/input/pseudo-text-init-iter-0/` | 0 |
| needless090 SED v5-trio | dataset | **GONE ❌ (now PRIVATE/deleted)** — `403 datasets.get denied`. Was public May-18. | `needless090/birdclef2026-sed-v5-trio` | — | n/a |
| needless090 SED ensemble | dataset | **GONE ❌** (403) | `needless090/birdclef2026-sed-ensemble` | — | n/a |
| needless090 perch-tflite | dataset | PUBLIC ✅ (349 MB, 2026-03-31) — still there, but not the SED trio | `needless090/birdclef2026-perch-tflite` | `/kaggle/input/birdclef2026-perch-tflite/` | — |
| hideyukizushi sgkfk weights | dataset | **PUBLIC ✅** (26 MB) — now the universal base | `hideyukizushi/sgkfk-202604041716` | `/kaggle/input/sgkfk-202604041716/` | 17 of 35 voted |
| tonylica model (rank-5 team) | dataset | **PUBLIC ✅** (775 MB, v2) | `tonylica/birdclef-2026-model` | `/kaggle/input/birdclef-2026-model/` | not in recent voted set |

**Key changes vs FINAL_META_FINDINGS:**
1. **needless090's SED v5-trio AND sed-ensemble are no longer public (403).** The "2nd-best SED resource" from the meta is dead — drop it from the plan.
2. **aliozanmemetoglu's SED fold *Models* could not be confirmed** (no models.get scope; zero recent importers). Given the author is now LB-#3 and merged, assume possibly privatized; **verify in a live Kaggle session by attaching `aliozanmemetoglu/birdclef-sed-fold-1` before relying on it.** Their *datasets* (`raw-pseudos`, `pseudo-text-init-iter-0`) ARE still public and still mounted by **0** other kernels.
3. **The "free lever" is STILL unexploited:** across 90 recent kernels, **zero** mount ali SED, needless, tonylica, raw-pseudos, or pseudo-text. tonylica/birdclef-2026-model (rank-5 team weights, 775 MB) is public and unused by the current voted crowd.
4. The community's *new* free base is `tuckerarrants/bc2026-distilled-sed-public` + `hideyukizushi/sgkfk-202604041716` — these are what moved the plateau to ~0.952. If our pipeline isn't on this base yet, that alone is the cheapest catch-up.

---

## 5. LATE-STAGE STRATEGY (5 days left, hidden private LB)

- **Public LB is near-useless for ranking now.** Top 32 spans 0.964→0.954 (0.010) with most teams at 366–394 submissions — heavily probed; deltas are below the test-set noise floor. Compressed band + merge churn = public rank is not a reliable private proxy.
- **Consensus (NEW framing for this window): select final subs by your own CV / mini-LB, not public score.** Canonical BC lore: the 7th→1st private jump comes from trusting CV. With 234 classes / macro-ROC-AUC and a 23-site hidden test, a calibrated ensemble that's robust across folds beats the single highest public score.
- **Submission-selection hygiene:** pick 2 finals that are *decorrelated* — e.g. (a) best-CV ensemble, (b) a robustness-hedged variant (taxonomy-smoothed + per-class temperature + conservative priors, no aggressive dead-hour weighting). Avoid choosing two near-identical public-LB-max subs.
- **Overfitting risk is concentrated in:** the dead-hour/site-hour prior weights (K1 showed these are fit on the same OOF that benchmarks them → optimistic) and any rank-power tuned to public LB. Keep prior weights conservative for the final sub.
- **Highest-EV cheap moves in the time left:** (1) rebase onto the tuckerarrants distilled-SED + sgkfk public stack (~0.952 floor); (2) add **taxonomy (genus/family) smoothing** (karnak) — cheap, inference-only; (3) add a **BirdNET sidecar with up-weighting on the 28 unmapped classes** (pilkwang PCEN-sidecar idea); (4) ensemble in `tonylica/birdclef-2026-model` (public rank-5 weights, 0 importers) for free diversity; (5) if a live session confirms ali's SED fold Models are still public, add them — but **verify first**, don't assume.

---

### Sources
- Kaggle REST: `competitions/birdclef-2026/leaderboard/{view,download}`, `kernels/list?competition=birdclef-2026&sortBy=dateRun`, `kernels/pull`, `datasets/view`, `datasets/list?user=`.
- WebSearch: BirdCLEF strategy playbook (lamsade.dauphine.fr Birdclef_2026.pdf); BC2025 top-2% write-up (Max Melichov, Medium); ferariz/birdclef2026 ablation repo (PCEN, ProtoSSM, post-proc).
- Token scope note: `models.get` and `discussions.list` denied — models/forum confirmed indirectly (mount-graph + web). Recommend a quick authenticated UI check of `aliozanmemetoglu/birdclef-sed-fold-*` before relying on it.
