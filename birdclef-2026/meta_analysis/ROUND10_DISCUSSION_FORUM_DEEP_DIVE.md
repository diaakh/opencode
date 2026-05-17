# BirdCLEF+ 2026 — ROUND 10: Discussion forum forensics + LSE head + Hengck's chain

This round pulls the actual content of high-vote BC2026 discussion threads (via `kaggle competitions topic-messages` API, not the rendered web page). Many useful insights live ONLY in the forum, not in published kernels.

## 1. Discussion 681297 — Duplicate labels bug confirmed by organizers (37 votes)

**ttahara reported on March 17:**
```python
train_ss_labels = pd.read_csv(".../train_soundscapes_labels.csv")
print(len(train_ss_labels))                  # 1478
print(train_ss_labels.duplicated().sum())    # 739
```

**Competition organizer official response (10 votes):**
> "Oh, ok, good find, not sure what went wrong there; we'll probably do a quick dataset update within the next few days to deal with these quirks you folks found."
> "We don't have additional labels, we'll remove the duplicate ones."
> "Yes, seems like every entry has a duplicate which we'll remove"

**Current data state**: I just re-verified, our local copy still has 1478 rows. **The organizers acknowledged the bug but the dataset has NOT been re-published with the fix.** This means anyone forking the BC2026 data today is still affected. The 33 kernels (2.6% of corpus) that call `.drop_duplicates()` are still doing the right thing; the 1,200+ that don't are silently 2x-weighting these annotations.

## 2. Discussion 683822 — Hengck23's HGNetV2-B0 + LSE pool ladder (61 votes)

This is the **single most informative public progression** for what brings a single SED model from 0.86 to 0.90+.

Hengck23 (rank 1639 on BC2026 despite being a Kaggle grandmaster, because he publishes everything publicly) shared:

```python
# LSE Pool (LogSumExp): smooth differentiable alternative to max pool, for MIL
def lse_pool(x, dim=1, r=10.0):
    T = x.size(dim)
    return torch.logsumexp(r * x, dim=dim) / r - math.log(T) / r

def forward(self, ...):
    last = self.backbone(spec)              # (B, 2048, 8, 8)
    last = last.mean(dim=2)                 # (B, 2048, T)
    last = last.transpose(1, 2)             # (B, T, 2048)
    time_logit = self.head(last)            # (B, T, 234)
    logit = lse_pool(time_logit, dim=1)     # (B, 234) → BCE loss
```

**HGNetV2-B0 (4-fold) head comparison (all other things equal):**

| Head | Public LB |
|---|---:|
| ImageNet GAP (global avg pool) | 0.860–0.863 |
| **LSE head (r=10)** | **0.876** (EMA 0.874) |
| Time-gated SED | 0.851–0.855 |

LSE wins by +0.015 over GAP on the same backbone. **No corpus kernel I've seen uses LSE pool**; everyone uses either GAP or attention SED.

**The full improvement chain documented in the thread**:

| Step | LB |
|---|---:|
| HGNetV2-B0 + LSE head (4-fold) | 0.876 |
| + aliozanmemetoglu post-processing (TTA, temporal filter, site/time prior) | 0.883 |
| + Perch distillation (from disc 685318) | 0.898 |
| + Hengck's texture/event smoothing | 0.891 (from 0.876 base) |
| + segment_sec 5 → 10 (longer training crops) | 0.90+ |

The **distillation step alone** added +0.022 to a self-trained HGNet (0.876 → 0.898). This is the same +0.022 that tuckerarrants documented for EfficientNet-B0.

**The clip-pool formula** (frame → clip prediction):
```python
clip_logits = (torch.logsumexp(alpha * frame_logits, dim=1) - math.log(T)) / alpha
# alpha=1 used by Hengck
```
This is the LSE pool with α=1 (smoother than r=10).

**Perch 2.0 paper note on mixup**: According to Hengck citing the paper, "mixup labels are NOT weighted" — Perch concatenates labels with OR instead of weighting by the mixup α. This is unusual compared to standard mixup which interpolates labels.

## 3. Discussion 685318 — Hengck23's PyTorch Perch v2 + distillation experiments (62 votes)

`hengck23/pytorch-differentiable-perchv2` — a fully PyTorch port of Perch v2 with weights, verified to match ONNX:

| metric | value |
|---|---|
| embedding cosine mean (vs ONNX) | 0.9999999999995 |
| embedding cosine min | 0.9999999999992 |
| embedding MAE | 7.8e-08 |
| embedding max abs error | 6.1e-07 |
| spectrogram correlation | 0.9999999999999 |

Hengck explained the **stop-gradient distillation mechanism** intuition:
> "the prediction of 1536-d distilled embedding vector is not accurate at first with high mse... yet the prediction is trained to make predictions with embedding+noise... as training proceeds, the mse gets less... in effect, we are using mse loss as regularization to smooth the loss landscape of prediction head"

He also revealed:
> "ONNX Perch from justinchuby/Perch-onnx returns BOTH `embedding` (B, 1536) AND `spatial_embedding` (B, 16, 4, 1536)"

**This `spatial_embedding (B, 16, 4, 1536)` exposes time×freq×channel structure** — 16 time bins, 4 frequency bands, 1536 channels per cell. NO corpus kernel uses the spatial embedding; everyone uses just the mean 1536-d. **Distilling on the spatial embedding (16×4×1536 = 98,304-d teacher target) preserves temporal-spectral localization** that the global mean discards.

**Hengck's ideal future approach** (not yet implemented):
- SSL on 10k unlabeled soundscapes
- Predict next 5-sec window from 5 previous windows (multi-layer latent prediction)
- Few-shot probe on the 66 labeled soundscapes
- This is the "Bittern Lesson" approach inverted — use SSL pretraining + supervised fine-tune

## 4. Discussion 686457 — hideyukizushi's reproducibility + ONNX async loading (36 votes)

The reproducibility fixes Hideyukizushi documented:

1. **Torch model init randomness** — unspecified `torch.randn` calls produce different weights per run
2. **Mixup/CutMix randomness** — `np.random` without fixed seed inside `mixup_cutmix` / `mixup_files`
3. **Dropout randomness** — `nn.Dropout(dropout)` with PyTorch's default RNG
4. **MLPClassifier(random_state=42)** — they explicitly fix sklearn's RNG

```python
# Recommended fix (his code)
torch.manual_seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)
# In dropout layers, ensure explicit p
# In MLPClassifier, set random_state=42
```

**ONNX Perch with async audio loading** — single biggest CPU speedup:

```python
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

_so = ort.SessionOptions()
_so.intra_op_num_threads = 4  # Kaggle CPU has 4 cores / 8 threads
ONNX_SESSION = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so, providers=["CPUExecutionProvider"])

# Async double-buffered audio loading
executor = ThreadPoolExecutor(max_workers=4)  # 4 = safer than 8 (avoids thrashing)
```

**Result**: 90-min CPU budget became 23-min total scoring time. **The "double buffering" pattern** — disk I/O of next file runs in parallel with current file inference — gives 30-40% wall-clock speedup over sequential.

**SGKF strategy with rare-class binning**:
```python
y_strat = np.argmax(Y_SC, axis=1)
unique_classes, counts = np.unique(y_strat, return_counts=True)
rare_classes = unique_classes[counts < n_splits]
y_strat[np.isin(y_strat, rare_classes)] = -1  # bin rare classes
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=91)
```

This is the key SGKF trick — rare classes (fewer files than n_splits) get binned to "-1" so they don't cause `StratifiedGroupKFold` to fail. Without this, SGKF rejects the data.

## 5. Discussion 689012 — ttahara's OpenVINO vs Torch benchmark (30 votes)

Concrete timings on 600 `.ogg` files (7200 windows) with HGNetV2-B0 + LSE head:

| Method | num_workers | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---:|---|---|---|---|
| Torch | 1 | 4m 8s | 2m 58s | 2m 19s | 3m 8s |
| Torch | 2 | 3m 13s | 2m 28s | 1m 48s | 2m 21s |
| Torch | 4 | (continued) | | | |
| Torch-jit-trace | varies | faster than Torch | | | |
| **OpenVINO** | varies | **~2x faster than Torch** | | | |

For NFNet (eca_nfnet_l0), **OpenVINO gives NO speedup** because NFNet has no BatchNorm, so OpenVINO's conv+BN fusion can't apply. **For HGNetV2-B0 and EfficientNet, OpenVINO is the right choice**; for NFNet, stay with Torch-jit-trace.

The pipeline: `.pth → torch.onnx.export → ov.convert_model → openvino IR (.xml/.bin) → core.compile_model → AsyncInferQueue`.

## 6. Discussion 690887 — Domain-matched external background data (14 votes)

The cleverest augmentation idea I haven't seen anywhere in the corpus:

> "1. Train a binary classifier on the Kaggle dataset for 2 classes: XC focal audio vs Pantanal soundscape (can use unlabeled set)
> 2. Download external data — especially external PAM (Passive Acoustic Monitoring) soundscapes — such that the binary classifier gives high confidence that it is same domain as Kaggle soundscape AND does NOT contain Kaggle bird classes
> 3. Use these as **background for mixup augmentation**"

**Why this works**: standard mixup uses train_audio clips as mixup partners, but those are focal recordings (different domain). Mixing focal+focal doesn't teach the model the Pantanal noise floor. Using domain-matched-but-no-target-class external soundscapes as background teaches the model to ignore Pantanal-style background.

**Companion idea (also 14 votes)**: Use frozen Perch + cosine similarity to MEASURE domain similarity between candidate external data and the Pantanal soundscape pool. This gives an automatic domain filter.

## 7. Discussion 694815 — Pseudo-labeling pitfall (Multi-Iterative Noisy Student)

**Critical warning from a participant**:
> "I'm trying to follow closely to last year's winning Noisy Student approach... But, I'm noticing a fairly large LB reduction (supervised only single fold LB ~0.85-0.88, using pseudo labels + labelled data with same model single fold ~0.79)."

**Naive pseudo-labeling drops LB by 0.06-0.09 points**. The user used their 0.943 LB ensemble to generate pseudo-labels but the student under-performs the teacher single-fold.

**What's the fix**: (from another reply, 3 votes)
> "From what I understood from previous write-up's is that usually they precompute the pseudo labels and select which ones should be used and then only train their model with it. Personally I do the pseudo-labeling on the fly. If your model is well calibrated and is not making too confident predictions, using soft labels should attenuate the noise from wrong pseudo labels. Also doing on the fly, gives you more diverse data as you can simply take random crops instead of precomputed windows."

**On-the-fly soft-label pseudo-labeling > precomputed hard pseudo-labels**.

## 8. Discussion 681146 — Tom Capybara's Claude-Code automated experimentation log (134 votes)

Tom Capybara (tom99763, rank 13 LB 0.955) is publicly running a Claude-Code agent on this competition. He posts daily updates of HTML reports. His "Noisy classmates" extension of noisy student:

> "[2026/4/7] Start working on a very interesting approach which I name it **'Noisy classmates'**... it's the extension of noisy student."

Concept: instead of teacher → student (one-to-one), **multiple peer students exchange information during training** (like classmates exchanging notes before an exam). Each student is the teacher for the others on different subsets.

His progression:
```
2026/3/15: 0.849 LB (best single)
2026/3/17: 0.893 LB (added distillation)
2026/3/19: 0.918 LB (Claude-code submission)
2026/3/20: 0.921 LB (new approach)
2026/3/22: 0.926 LB (53 rounds, 1099 methods tested)
... eventually 0.955 LB (rank 13)
```

His distillation prompt template (from hengck23):
> "throw in a bunch of wave files (e.g. unlabeled soundscape files from this and previous competitions). extract their embeddings and make a database. distill to your favourite pytorch models and enjoy!"

Tom's **"inverse submission guidance"** strategy:
> "Claude Code proposes a submission, and you intentionally use only a single submission to test that hypothesis. If the leaderboard (LB) result comes back negative, it provides a strong corrective signal — a high-value 'reward' in terms of learning — because it prevents the agent from continuing to optimize in the wrong direction."

Information-gain-driven submission picking, not score-maximization.

**Tom's CV-LB inconsistency observation**:
> "Keep improving the result approach: This led to excessive blending and aggressive tuning of weighted sums. While it achieved an impressive CV score of 0.999, it ultimately suffered from severe data leakage."
> "Keep developing and extending the current best notebook with ~80% confidence toward a 0.938–0.94 LB approach: This strategy produced more reasonable and robust results, and importantly, the improvements translated well to the LB."

**0.999 CV is a RED FLAG for leakage**. This matches our finding that hideyukizushi's val_auc=0.979 vs LB=0.953 = 2.6pp gap, alexander val=0.979 vs LB=0.950 = 2.9pp gap, tonylica val=0.996 vs LB=0.957 = 3.9pp gap. **A val-LB gap >0.025 means your CV is over-fit or leaky.**

## 9. Cross-cutting findings from BC2025 carry-overs

From BC2025 thread 568886 (107 votes, kdmitrie — now BC2026 Rank 15):
> "Almost all CSA recordings contain human voice" (BC2025-specific issue)
> "143 of train_soundscapes (1.5%) contain human voice" (BC2025)

**BC2026 status**: I checked. BC2026 doesn't use the CSA collection (only XC + iNat). 586 of 35,549 recordings (1.6%) have CSA-style author names (Spanish-speaking), but they're spread across:
- 582 Aves recordings
- 3 Insecta
- 1 Amphibia

Author JAYRSON ARAUJO DE OLIVEIRA contributes 2,874 recordings (8% of all data) but covers 155 species with no >50% concentration in any single species. **No author-leakage concern in BC2026.**

## 10. The "kaggle competitions topic-messages" API — undocumented goldmine

The way I'm reading these threads:
```bash
kaggle competitions topic-messages birdclef-2026 <topic_id> -n -1 --csv
```

This dumps the full thread content as CSV (with author, post date, vote count, HTML body). The Kaggle web page lazy-loads and JS-renders, so most users can't easily scrape it. The API returns raw HTML body for every post.

Discussion IDs referenced in BC2026 corpus kernels (verified live):
- 681146: Tom Capybara's Claude-Code log (134v)
- 681297: Duplicate labels bug (37v + organizer ack)
- 683822: Hengck's LSE+HGNet ladder (61v)
- 685318: Hengck's distillation (62v)
- 686457: hideyukizushi's reproducibility (36v)
- 689012: OpenVINO benchmark (30v)
- 690887: Domain-matched external data (14v)
- 694815: Pseudo-labeling pitfall (3v)

Plus BC2025 carryovers:
- 568886: Human voice removal (107v) — kdmitrie's approach
- BC2025 1st-place writeup with Multi-Iterative Noisy Student

## 11. What this round adds to the action plan

### Tier-A immediate adds (< 1 hour)
- **Replace your GAP/attention SED head with LSE pool head** (r=10, BCE on clip logits) — gains +0.015 over GAP on HGNetV2
- **Drop `.drop_duplicates()` on labels read** (still needed — organizer fix not pushed)
- **Switch to ONNX Perch + `intra_op_num_threads=4` + `ThreadPoolExecutor(max_workers=4)`** double-buffered audio loading — gets you to 23-min scoring on 90-min budget

### Tier-B same-day
- **Distill from Perch's `spatial_embedding` (B, 16, 4, 1536)** instead of just the mean (1536). Use justinchuby/Perch-onnx model output. Spatial distillation preserves time-frequency localization. NO corpus kernel does this.
- **Train domain-binary classifier (XC vs Pantanal-soundscape)** → use to filter EXTERNAL PAM datasets for mixup-background augmentation. Cost: ~30 min training.
- **SGKF with rare-class binning** (set rare-class y_strat to -1) → unblocks SGKF on imbalanced data.

### Tier-C multi-day
- **Implement Tom Capybara's "Noisy classmates"**: train N student models simultaneously; each student is teacher for the others on different fold subsets. Extension of single-teacher noisy student.
- **On-the-fly soft pseudo-labeling** (not precomputed) on random crops, with calibration check. The naive precompute-and-train approach hurts LB by 0.06-0.09 (discussion 694815).
- **Hengck23's SSL pretraining**: predict next 5-sec window from 5 previous windows on 10k unlabeled soundscapes, then few-shot probe on 66 labeled.

### Tier-D submissions discipline
- **Use Tom's "inverse submission guidance"**: pick submissions that maximize information gain about whether your strategy is on the right track, not raw expected-score.
- **Watch the val-LB gap**: gap > 0.025 = leaky validation. Tonylica is at 4pp gap, hideyuki at 2.6pp gap, alexander at 2.9pp gap. Bring the val protocol closer to test (group by file + site + day).

## 12. Sources researched fresh (URLs verified by direct fetch)

- [Discussion 681146 (Tom Capybara, 134v): Claude-Code agentic loop](https://www.kaggle.com/competitions/birdclef-2026/discussion/681146)
- [Discussion 681297 (37v): Duplicate labels in train_soundscapes_labels.csv + organizer ack](https://www.kaggle.com/competitions/birdclef-2026/discussion/681297)
- [Discussion 683822 (61v): Hengck23 HGNetV2-B0 + LSE head ladder](https://www.kaggle.com/competitions/birdclef-2026/discussion/683822)
- [Discussion 685318 (62v): Hengck23 PyTorch Perch v2 + distillation experiments](https://www.kaggle.com/competitions/birdclef-2026/discussion/685318)
- [Discussion 686457 (36v): hideyukizushi reproducibility + ThreadPoolExecutor pattern](https://www.kaggle.com/competitions/birdclef-2026/discussion/686457)
- [Discussion 689012 (30v): ttahara OpenVINO vs Torch benchmark](https://www.kaggle.com/competitions/birdclef-2026/discussion/689012)
- [Discussion 690887 (14v): Domain-binary classifier + external PAM background](https://www.kaggle.com/competitions/birdclef-2026/discussion/690887)
- [Discussion 694815: Pseudo-labeling LB drop warning](https://www.kaggle.com/competitions/birdclef-2026/discussion/694815)
- [BC2025 568886 (107v): kdmitrie human voice removal](https://www.kaggle.com/competitions/birdclef-2025/discussion/568886)
- [justinchuby/Perch-onnx (HF) — spatial_embedding output](https://huggingface.co/justinchuby/Perch-onnx)
- [hengck23/pytorch-differentiable-perchv2 (Kaggle code)](https://www.kaggle.com/code/hengck23/pytorch-differentiable-perchv2)
