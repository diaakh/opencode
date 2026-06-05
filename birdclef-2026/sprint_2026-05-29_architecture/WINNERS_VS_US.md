# WINNERS vs US — BirdCLEF+ 2026 post-mortem diagnosis (2026-06-05)

**Final private LB (confirmed, Kaggle REST `competitions/birdclef-2026/leaderboard/view`):**
1. Nikita Babych **0.96574** · 2. tennogh **0.96013** · 3. kapenon **0.95992** ·
4. BirdCLEF+ 2026 Team **0.95902** · 5. Jiacheng Ma **0.95824** · 6. Sinan Calisir **0.95762** ·
7. 空飛ぶ宝石 0.95715 · 8. kazumax 0.95675 · 9. Yannan Chen 0.95661 · 10. coolz 0.95627 …
**Our team (adkasd): private ≈ 0.9417** (best = public-replay forks).

Tags: **[CONFIRMED]** = read from a primary source this scan; **[INFERRED]** = reasoned.
Note on sourcing: Nikita has **not posted a dedicated 2026 1st-place writeup** (LifeCLEF working-notes
deadline is 2026-06-17). His 2026 disclosures are forum comments; his full method is his **2025**
writeup, which he confirms he re-ran in 2026. The 2nd/3rd/10th/11th writeups ARE published and were
read in full.

---

## 1. THE WINNING RECIPE(S) — concrete, per top team

### Common skeleton shared by the entire top-10 [CONFIRMED across writeups]
Every top solution is the **same shape**, and it is NOT "fork a public Perch notebook":
- **Perch v2 as a TEACHER, distilled into your OWN trained CNN/SED backbones** — not Perch embeddings
  consumed at inference. (This is the single biggest divergence from what we did — see §3.)
- **Multi-round self-training / pseudo-labeling on the unlabeled soundscapes**, soft labels with a
  **power transform, no hard threshold**, teacher = your own current ensemble (regenerated each round).
- **A heterogeneous multi-backbone SED ensemble** (EffNet/NFNet/RegNet/ConvNeXt/SE-ResNeXt + Perch),
  equal-ish weights, members deliberately **decorrelated**.
- **A dedicated zero-train (Insecta/Amphibia) specialist** seeded from Nikita's CC0 XC extra-data.
- **No trustworthy CV — everyone validated on the public LB**, with fast (~2–10 min) inference so they
  could iterate dozens of times.
- **OpenVINO/ONNX inference**, neighbor-window smoothing + delta-shift TTA, site/hour ecological priors.

### 1st — Nikita Babych (0.9657)
- Method = his 2025 **"Multi-Iterative Noisy Student"**, re-run for 2026. [CONFIRMED 2025 writeup;
  CONFIRMED 2026 forum he re-ran it]
- 2026 disclosures (forum, [CONFIRMED]): **"most of my models were distilled from Perch v2 in the
  first training stage"**; **~0.935 LB without pseudo-labels → ~0.955 LB with PL + full preprocessing**;
  running a **2-seed ensemble**. He called the *public Perch notebooks* "the actual trap" (huge
  overfit), NOT Perch itself.
- 2025 recipe (his verbatim, applies in 2026): **4 self-train iterations**, per-iter power transform
  (p^1.0 / p^1.54 / p^1.82 / p^1.67), **no threshold**, **CrossEntropy**, **100% MixUp@fixed-0.5**
  of each sample with a random pseudo-labeled soundscape; **7-backbone SED ensemble**
  (EffNet-B0/B3/B4 + RegNetY-008/016 + ECA-NFNet-L0, SED head + GeM freq-pool + 3×mel); dedicated
  **EffNet-B0 zero-train specialist** on comp + XC (<60s) inserted via a zero-matrix; inference
  neighbor-smoothing + delta-shift TTA, OpenVINO, ~2 min runtime.
  - Source: kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
  - Source: kaggle.com/competitions/birdclef-2026/discussion/704250 (Perch comment) and /683791 (0.935→0.955).

### 2nd — tennogh (0.9601) — the richest published writeup [CONFIRMED]
Source: kaggle.com/competitions/birdclef-2026/writeups/2nd-place-diverse-ensemble-with-pseudo-labeling-a
- Backbones: **EfficientNet-v2-s + EfficientNet-b0 + NFNet (+ public Perch)**. Mel 128/2048/hop512,
  fmax 16k, 5s windows; **GeM hurt — used avg pool**.
- **5 explicit pseudo-label rounds**, each with a NEW teacher ensemble, tracked on LB:
  focal **0.917** → R1 **0.919** (power 1.55, mixup) → R2 **0.922** (stopped mixing, **substitute
  soundscapes 50→60%**, power 1.0) → R3 **0.934** (XC-pretrained backbones, +public Perch/SED labels)
  → R4 **0.938** (softAUC + 0.25·BCE) → R5 0.933. **Soft labels throughout, no hard threshold.**
- External: mostly comp data; **Insecta specialist seeded from Nikita's 2025 CC0 XC extra-data** +
  comp Insecta/Amphibia, 5× upweighted. Specialist lift: **+0.002 (insecta only), +0.004 with 5× up**.
- **No reliable CV** — "all validation strategies unreliable, I used LB as the main signal"; his best
  held-out set had only **~0.2 correlation with LB**.
- Final **7-member ensemble** (Perch 0.936 / distilled-SED-v2s 0.929 / R1-NFNet / R3-v2s / R4-v2s /
  R5-NFNet / Insecta), equal weights minus a haircut for correlated R3/R4/R5. Inter-model Spearman:
  Perch–SED **0.41**, highest CNN pair R3–R4 **0.79**. Post-proc: sonotype mirroring + temporal
  continuity. **Final 0.959 public / 0.960 private.** Compute: **~$200 remote GPU + 2 months Claude.**

### 3rd — kapenon (0.9599) [CONFIRMED]
Source: kaggle.com/competitions/birdclef-2026/writeups/3rd-place-solution; +forum 704250.
- Two families: **`tf_efficientnetv2_s_in21k` + SED AttHead** (2025-2nd-place FT) and
  **`seresnext26t_32x4d` + Distill + SED head** (Perch knowledge-distillation). 234 classes, 32k, 5s,
  **5-fold**.
- **Single-round soft pseudo-labels** (sigmoid probs) on unlabeled soundscapes, **site-bias reduction
  sampling 1/√(site_count)**. Aug: wave mixup / SpecAugment / RandomFiltering / rare-species upsample.
- 3-member blend, **weights 0.4 / 0.2 / 0.4**; OpenVINO IR; TTA **0.4·standard + 0.3·adjacent-shift**;
  post-proc site/hour priors, file-confidence^0.4, rank-aware scaling, adaptive smoothing.
- His teammate **Tucker Arrants** (forum 704250, [CONFIRMED]): **"13 of 14 models in my final
  submission were Perch distilled, and almost all submissions have the same private and public LB"** —
  i.e. distillation gave them a **near-zero shake** (robust generalization).

### 10th — coolz (0.9563) [CONFIRMED]
- **Trainable Perch v2** backbone + linear→RoPE→**4-layer SED Transformer**, BCE, 10-ep cosine,
  mixup@1.0, soundscape ×2.5 weight, class-balanced sampling. 5-fold ONNX/OpenVINO,
  60s→12×5s logits, site/hour + day/night + monthly-Amphibia priors. **Single-backbone** (Perch only) —
  reached 0.956 with **almost no ensemble diversity** by making Perch *trainable*. Forum: "public
  method based on [frozen] Perch is misleading."

### 11th — Salman Ahmed, "without Perch" (0.955 final; **0.959 unsubmitted**) [CONFIRMED]
Source: kaggle.com/competitions/birdclef-2026/writeups/704258.
- **No Perch at all** — pure XC-pretrained EfficientNet SED ensemble (B0_NS/B1_NS/V2B0/LITE), CE loss
  + "nocall" class, mel 480/4096/hop1001 (→20×1s frame probs), staged train→soundscape→XC.
- Cautionary tale: his **Stage-1 ensemble scored private 0.959 (would have been ~3rd) but he didn't
  select it** — picked a 0.955 blend. Confirms selection risk under no-CV, and that a strong
  *independent* (non-Perch) ensemble was fully competitive at the top.

---

## 2. WHERE WE WERE RIGHT (fair and specific)

Our analysis docs (A8/A9/FINAL_RETROSPECTIVE) got a large fraction of the *diagnosis* correct — the
failure was almost entirely **execution/timing**, not understanding. Specifically:

1. **We correctly identified noisy-student / multi-iteration PL as THE winning lever.** A8 nailed
   Nikita's exact recipe (powers, no-threshold, MixUp@0.5, 7-backbone, zero-train specialist) and A9
   correctly singled out "multi-iteration (not single-pass) PL" as the gap. tennogh's published 5-round
   ladder (0.917→0.938) and Nikita's 0.935→0.955 PL gain **confirm this precisely.** [CONFIRMED]
2. **We correctly diagnosed "no CV that tracks LB" and that the winners also had no CV.** tennogh
   ("all validation unreliable, used LB," ~0.2 corr) and Nikita (LB-only) confirm it. Our error was NOT
   misjudging CV — it was the *second half* (fast loop) we missed. [CONFIRMED]
3. **We correctly identified that diversity must be INDEPENDENT, not Perch-correlated, and measured
   correlation as the gate.** tennogh's published Spearman table (Perch–SED 0.41, R3–R4 0.79) and his
   weight-haircut for correlated members is exactly the orthogonality principle we wrote down. [CONFIRMED]
4. **We correctly identified the zero-train specialist + Nikita's CC0 XC extra-data** as the route to
   the non-Aves classes. tennogh literally used that dataset. [CONFIRMED]
5. **We correctly catalogued the inference polish that matters** (power-transform PL with no threshold,
   neighbor smoothing, delta-shift TTA, site/hour priors, file-confidence^0.4, rank-aware scaling) —
   every published top solution uses this exact toolkit. [CONFIRMED]
6. **We correctly concluded a fast base + multiple independent members within budget was the move**
   (A9 RANK 1–4). That is structurally the winners' pipeline. We just reached the conclusion on the
   last day.

**Partly right / important correction:** A9's claim that **the 28 son-IDs are "unreachable for
everyone"** was *operationally* right for us (we couldn't reach them) but **the framing was a cope.**
tennogh and Nikita both built **dedicated Insecta/Amphibia specialists** that bought +0.002–0.004 —
small, but in a field where 1st–10th spans only **0.0095**, that is a medal-moving margin. The 28
weren't "free for everyone"; the winners *worked* them and we wrote them off. [CONFIRMED tennogh +0.004]

---

## 3. WHERE / WHY WE FAILED — the concrete deltas

Mapped to root cause, and split "never had the idea" vs "had it, executed wrong/too late."

### A. We consumed Perch as a frozen inference embedding; winners DISTILLED Perch into their own trained backbones. [had the idea late]
This is the **single decisive technical delta.** Nikita: "most of my models were distilled from Perch v2
in the first training stage." Tucker (3rd): "13 of 14 models were Perch distilled." coolz (10th) made
Perch **trainable**. We forked the **frozen-Perch public ProtoSSM/SED notebook** — the exact thing
Nikita called **"the actual trap … huge overfit."** Distillation turns Perch's knowledge into models
you can then (a) self-train on soundscapes and (b) ensemble diversely; a frozen-Perch inference graph
can do neither. **Root cause: wrong frame** — "fork public Perch + add members" vs "distill Perch into
your own trainable ensemble." We never trained a distilled student as our *base*; we only trained
distilled students as *bolt-on members onto the frozen base*, which made them correlated-by-construction
(our own Cause 3). The winners' identical building block (Perch distillation) was *non-correlated*
because it WAS the base, trained independently per backbone. [CONFIRMED]

### B. Single pseudo-label pass vs multi-round self-training. [had the idea, executed once]
tennogh ran **5 rounds** (0.917→0.938, +0.021); Nikita **4** (+0.020). We ran NS **once**, blended one
checkpoint (0.945), declared "it hurts," quit (our Cause 4). We never measured the iter trajectory. The
+0.02 that PL is worth is *the entire 0.94→0.96 gap*, and it only materializes across rounds.
**Root cause: judged an iterative method on one shot + no feedback loop to afford the rounds.** [CONFIRMED]

### C. No fast inference loop. [never executed]
Winners ran **~2–10 min** inference (Nikita ~2 min OpenVINO; coolz/Salman ~10 min) and iterated dozens
of times on the LB. Our base ran **~85 min**, hit the 90-min cliff, **timed out twice** (zero score),
and forced fold-trimming that ate accuracy. At ~3 h/experiment we got 1–2 bits/day; they got dozens.
**Root cause: slow base never fixed — Job #1 we skipped.** This is *the* multiplier behind B and the
late frame-switch. [CONFIRMED their runtimes; CONFIRMED our timeouts]

### D. Bolt-on members onto a fixed base vs a co-trained diverse ensemble. [had the principle, mis-executed]
Every member we added (focal CNN 0.944, NS-b0 0.945, BirdMAE 0.949) *hurt or was flat*, because they
were weak-on-domain and/or correlated with the frozen SED already in the base. The winners' members
were each **independently trained, self-trained on the soundscape domain, and decorrelated by design**,
then equal-weighted. We had the orthogonality principle in writing (Cause 3) but applied it to the wrong
object (members onto a frozen base, not an ensemble of independent distilled students). [CONFIRMED]

### E. Wrote off the 28 zero-train classes; winners built a specialist. [had a partial idea, abandoned]
We declared the son-IDs "unreachable for everyone." tennogh/Nikita got +0.002–0.004 from a dedicated
Insecta/Amphibia specialist on the CC0 XC data — exactly the asset A8 flagged as "highest-value." In a
0.0095-wide top-10 that margin is real. **Root cause: a convenient ceiling story that stopped us trying.**
[CONFIRMED]

### F. Started the independent-ensemble build on the last day. [right frame, fatally late]
Our retrospective Cause 5 already says this. The winners spent the full window (tennogh: ~$200 GPU +
2 months) iterating rounds. We committed to "Branch B" with ~2.5 days left under GPU cap=2. **Root
cause: stayed in the wrong frame for a month because the slow/anti-correlated loop never surfaced the
error.** [CONFIRMED our timeline]

### G. (Process) Agent-swarm overhead vs a tight personal loop. [execution hygiene]
Our Cause 6. Notably, a **"101st place Pure Claude-Code Solution"** exists on the forum — pure-agent
solutions landed mid-pack, not top. The top teams used Claude as an *assistant to a human-driven fast
loop* (tennogh explicitly: "$200 GPU + 2 months Claude subscription"), not as the driver. [CONFIRMED thread exists]

---

## 4. THE DECISIVE 2–3 DIFFERENCES that separated 0.94 from 0.96

1. **DISTILL PERCH INTO YOUR OWN TRAINABLE BACKBONES, then self-train them on soundscapes.** We fed a
   *frozen* Perch graph at inference (the public "trap"); they made Perch knowledge the seed of an
   independent, domain-adaptable ensemble. This one architectural choice unlocks both #2 and the
   private-LB robustness (Tucker: distilled models had ~zero public→private shake; our fork dropped
   public→private to 0.9417). **[CONFIRMED, biggest delta]**
2. **MULTI-ROUND noisy-student pseudo-labeling (4–5 rounds), not one pass.** Worth ~+0.020 = the whole
   gap. They iterated it; we ran it once and quit. **[CONFIRMED]**
3. **A FAST (~2–10 min) inference loop that let them iterate the above on the LB dozens of times.** Our
   ~85-min base (2 timeouts, fold-trimming) capped us at 1–2 experiments/day and is the root multiplier
   that prevented #1, #2, and the timely frame-switch. **[CONFIRMED]**

One-line verdict: *the winners and our own notes agreed on the recipe; they had a fast loop and a month
to iterate Perch-distilled independent backbones through multi-round self-training, and we had a slow
base, a single shot, and the wrong (frozen-Perch bolt-on) frame until the last day.*

---

## SOURCES
- Final LB: Kaggle REST `api/v1/competitions/birdclef-2026/leaderboard/view` (2026-06-05).
- 2nd: kaggle.com/competitions/birdclef-2026/writeups/2nd-place-diverse-ensemble-with-pseudo-labeling-a (tennogh) [CONFIRMED, full].
- 3rd: kaggle.com/competitions/birdclef-2026/writeups/3rd-place-solution (kapenon) [CONFIRMED, full].
- 10th: kaggle.com/competitions/birdclef-2026/writeups/10th-solution-simple-model-as-always (coolz) [CONFIRMED].
- 11th: kaggle.com/competitions/birdclef-2026/writeups/704258 (Salman Ahmed, "without Perch") [CONFIRMED].
- "Was Perch a Trap?": kaggle.com/competitions/birdclef-2026/discussion/704250 — Nikita (Perch distillation), Tucker Arrants (13/14 distilled), coolz, AK [CONFIRMED].
- "best single model LB": kaggle.com/competitions/birdclef-2026/discussion/683791 — Nikita 0.935→0.955 PL [CONFIRMED].
- 1st-place method (2025 writeup he re-ran): kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n [CONFIRMED]. NOTE: no dedicated 2026 1st writeup yet (working-notes deadline 2026-06-17) [CONFIRMED absent as of 2026-06-05].
- Our docs: FINAL_RETROSPECTIVE.md, 03_SUBMISSION_RESULTS.md, A8_nikita_dossier.md, A9_new_approach.md.
- Related meta-threads (exist, content JS-rendered/not scraped this scan): "The Plateau at 0.950" (Pied Pie LLC), "What we learned by NOT beating 0.950" (Whyme Labs), "Why Shake?" (Tawara).
