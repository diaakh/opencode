# A1 — State-Space Models / Linear-Attention for BirdCLEF+ 2026

**Agent:** A1 | **Date:** 2026-05-29 | **Charter:** SSM / linear-attention SOTA for audio SED, + transferable math from recent LLM-lab work (Mamba/Mamba-2, RWKV-7, GLA, gated DeltaNet, DeepSeek MLA, MoE).

**Bottom line up front:** The "ProtoSSM / residual-SSM" pattern that correlates +0.66 with high LB is **almost certainly NOT a heavy Mamba block**. It is a *tiny* structured-state-space (or diagonal linear-recurrent) **temporal smoother + prototype classifier** sitting on top of a frozen Perch-v2 embedding sequence (12 windows × 5 s = one 60 s file). It is cheap, CPU-friendly, and the "SSM" is doing the same job as the famous BirdCLEF-2024 `[0.1,0.2,0.4,0.2,0.1]` smoothing kernel — but *learned* and *content-adaptive* (the "adaptive_delta" feature, +0.626 corr). The two-pass (forward+backward) variant = bidirectional scan. This is the single highest-confidence structural lever, and crucially **the head trains in minutes and infers on CPU in seconds** — most of it is even T1-feasible if any pretrained head weights exist publicly.

---

## 1. Reverse-engineering "ProtoSSM / residual SSM"

### 1.1 The data shape it operates on (confirmed from meta-findings)
- Perch-v2 → per-5 s-window embedding `e_t ∈ R^D` (Perch embedding D ≈ 1280; Perch-v2 logits dim 1536 / embed 1280).
- A 60 s test file → sequence `E = [e_1 … e_T]`, `T = 12` windows (meta: `n_windows=12`, `window_sec=5`, both constant across the 0.95+ tier).
- Output wanted: per-window logits `Z ∈ R^{T×234}` (then pooled to file-level + per-window for the soundscape submission, which scores each 5 s row).

### 1.2 The "Proto" part — prototype / cosine head (NOT a dense linear layer)
Meta shows `uses_mlp_probes +0.626` and `sonotype_mirror +0.408`. "Proto" = a **prototypical-network classifier**: each of the 234 classes has a learned prototype vector `p_c ∈ R^d` (or a small bank of K prototypes per class), and the logit is a (temperature-scaled) cosine similarity:

```
ẽ_t = W_proj e_t                 # optional low-rank projection D→d, d≈256
z_{t,c} = τ · cos(ẽ_t, p_c) = τ · (ẽ_t·p_c)/(‖ẽ_t‖‖p_c‖)
```

Why prototypes beat a plain linear head here:
- **The 28 unmapped insect/frog classes** (A2's bottleneck) have few/zero Perch-aligned logits. A cosine-to-prototype head + `sonotype_mirror` (max-pool the logit over visually/acoustically similar sonotypes) generalizes to them far better than a softmax linear layer trained on scarce data. This is the standard DCASE few-shot bioacoustic recipe (mean-of-support prototype, cosine/Euclidean match — DCASE'22 task5 ensemble went 41%→60% F-score this way).
- Cosine normalization gives well-behaved, calibratable scores for a **ranking metric (macro ROC-AUC)** — only the *ordering* per class matters, and cosine logits rank-blend cleanly (the `rank_aware +0.651` feature).

If K>1 prototypes per class: `z_{t,c} = τ · max_k cos(ẽ_t, p_{c,k})` (multi-prototype / soft-kNN; handles song-vs-call polymorphism within a species).

### 1.3 The "SSM" part — a tiny diagonal selective state-space temporal mixer
This is the load-bearing piece. Over the T=12 sequence, run a **diagonal/scalar linear recurrence** that smooths and propagates evidence across windows. Minimal Mamba-2-style SSD form (scalar-times-identity decay `a_t`, the cheapest variant):

```
# per channel n = 1..N of a small state (N≈16–64), shared decay a_t (scalar) per step
a_t = exp(-Δ_t · softplus(λ))          # decay in (0,1); Δ_t = "adaptive_delta" (content-dependent step)
Δ_t = softplus(w_Δ · e_t + b_Δ)        # selective/input-dependent step size  ← the "adaptive_delta" feature
h_t = a_t · h_{t-1} + (1-a_t) · (B_t ⊙ u_t)     # B_t, u_t = learned proj of e_t
y_t = C_t · h_t                                  # readout
```

Key points reverse-engineered from the feature names:
- **`adaptive_delta` (+0.626)** = the selective Δ_t (Mamba's input-dependent timescale). When a bird is calling, Δ_t large → state updates fast / lets evidence in; in silence, Δ_t small → state decays / ignores. This is exactly Mamba's "selectivity" and is what makes it beat a fixed conv kernel.
- **`residual_ssm` (+0.656)** = a **residual connection around the SSM**: `out_t = e_t' + α·y_t` (or gated `out_t = e_t' + g_t⊙y_t`). The SSM is a *correction* to the per-window embedding, not a replacement — matches meta `correction_weight ≈ 0.30–0.35` and `alpha_blend ≈ 0.4`. Residual keeps the strong frozen-Perch signal while adding temporal context.
- **Two-pass / bidirectional** (the `marynaborovska/two-pass-ssm` kernel): run the recurrence forward *and* backward and sum, since a 60 s file is not causal — a call in window 7 should inform windows 5 and 9 symmetrically. This is the principled, learned generalization of the BirdCLEF-2024 3rd-place `[0.1,0.2,0.4,0.2,0.1]` symmetric smoothing kernel (which alone gave +0.01 LB).

### 1.4 Full forward pass (per file)
```
E (T×D)  --proj-->  Ẽ (T×d)
         --biSSM--> H  (forward state + backward state, residual-added) → Ẽ'  (T×d)   # temporal context
         --proto--> Z  (T×234) = τ·cosine(Ẽ', P)                                       # per-window logits
post-proc: per-class temperature, time-smoothing, max+mean logit adjustment, rank-blend with Perch/SED/BirdNET
```
Per-window logits `Z` go straight to the soundscape submission rows; file-level confidence (`file_confidence +0.537`) = a pooled summary (mean/max over t) used to re-weight.

### 1.5 Why it correlates so strongly with LB (and the honest caveat)
The +0.664 correlation is *partly confounded*: ProtoSSM kernels are forks of the same high-scoring base notebook, so the feature co-occurs with the whole 0.95+ recipe (Perch-v2 + SED + rank-blend + site-hour prior). The *isolated* lever of the SSM temporal head over plain mean-pooling is realistically **+0.003–0.008** (cf. BirdCLEF-2024 smoothing = +0.01, and the SSM is a strict generalization). Still a BIG lever by sprint criteria, and it is the structural piece that lets you cross the 0.948 Perch-transfer ceiling without retraining Perch.

---

## 2. 2025–2026 SSM / linear-attention advances relevant to a small CPU head

| advance | what it adds | CPU-inference-friendly? | relevance here |
|---|---|---|---|
| **Mamba-2 / SSD** (Dao & Gu 2024) | scalar-decay `a_t` ⇒ state recurrence is just elementwise mul+add; chunked-scan = matmul on hardware. ~30-line PyTorch ref impl. | **Yes** — at T=12 there is no scan-kernel issue at all; a plain Python/torch loop is microseconds. No custom CUDA needed. | Use the **scalar-A SSD** form for the temporal head. Simplest, fastest, trains stably. |
| **Gated Linear Attention (GLA)** (Yang et al., ICLR'25) | data-dependent gating + chunkwise-parallel form; matrix-valued state. | Yes for short T (chunkwise = matmul). | Slightly more expressive than scalar SSD; marginal at T=12. Use only if scalar SSD underfits. |
| **Gated DeltaNet** (Yang/Kautz, ICLR'25) — Mamba-2 + delta rule | delta-rule memory writes (correct/overwrite state) + gating; beats Mamba-2 on recall. | Yes (short T). | Overkill for T=12; delta-rule shines at long-context retrieval, not 12-step smoothing. **Low.** |
| **RWKV-7 "Goose"** (2025) — generalized delta rule, diagonal+low-rank state, vector gating | strong expressivity, implicit positional encoding. | Yes (pure recurrence, CPU-native, no matmul kernels). | Architecturally elegant; for T=12 the extra machinery is unjustified. **Low.** |
| **Log-linear / hierarchical GLA (Fenwick-tree state)** (2025) | multi-scale memory. | Yes. | No payoff at T=12. **Skip.** |
| **Tiled Flash Linear Attention / chunked scan** (2025) | faster training kernels. | GPU-train only. | Irrelevant to a 12-step head; T1/T2 inference is trivial. |
| **State-Space Models for Bioacoustics** (arXiv 2512.03563, Dec'25) — BioMamba | Mamba-2 stack matches Transformer (AVES) on BEANS at ~40% less memory. | inference cheaper than attention | Validates SSMs for bioacoustics; but BioMamba is a full *encoder* (replaces AVES), not a small head — out of scope for the no-GPU sprint, useful as a T2 backbone bet. |
| **Compiler-first SSD / O(1) autoregressive cache** (arXiv 2603.09555) | kernel-free Mamba-2 on CPU/GPU/TPU from one source. | **Yes, explicitly CPU.** | Nice-to-have if we ever ship a deeper SSM; not needed at T=12. |

**Takeaway:** For T=12 windows the entire zoo collapses to "use the **scalar-decay Mamba-2 (SSD) recurrence with selective Δ**, bidirectional, residual, prototype readout." Everything fancier (delta rule, vector gating, log-linear state) targets long-context LLM recall and buys nothing at length 12. CPU latency is a non-issue for any of them at this length.

---

## 3. Concrete head: small bi-SSM + prototype on frozen Perch-v2

### Architecture & budget
- Input: frozen Perch-v2 embeddings, `D=1280`, `T=12`.
- Proj `D→d`, `d=256`. SSM state `N=16`, bidirectional. Prototype bank `P ∈ R^{234×d}` (K=1; optionally K=2).
- **Param count:** proj 1280×256 ≈ 0.33M; SSM (B,C,Δ proj + λ) ≈ 256×(256+16+16) ≈ 0.07M ×2 dirs ≈ 0.15M; prototypes 234×256 ≈ 0.06M. **Total ≈ 0.55M params** (≈1.1 MB fp16). Trivially fits CPU/no-internet rules.
- **FLOPs per file:** proj 12·1280·256 ≈ 4M; SSM recurrence 12·256·16·~6 ≈ 0.3M ×2; proto 12·234·256 ≈ 0.7M. **≈ 5–6 MFLOPs/file** on top of Perch. Negligible vs Perch's embedding extraction.

### CPU latency for ~600 one-minute test files in 90 min
- Budget: 90 min / 600 files = **9.0 s/file** total. Perch-v2 embedding extraction (ONNX, the dominant cost — `uses_onnx +0.535`) is ~0.3–1.5 s/file on CPU for 12 windows depending on threads. The bi-SSM+proto head is **<1 ms/file** (5 MFLOPs). **Head is free; the speed lever is Perch ONNX/OpenVINO, owned by A3.** Even a 5-model ensemble of these heads adds <5 ms/file. ⇒ plenty of headroom; the SSM head does **not** threaten the 90-min budget.

### PyTorch sketch (scalar-SSD, bidirectional, residual, prototype)
```python
import torch, torch.nn as nn, torch.nn.functional as F

class BiSSDProtoHead(nn.Module):
    def __init__(self, D=1280, d=256, N=16, C=234, K=1, tau=16.0):
        super().__init__()
        self.proj = nn.Linear(D, d)
        # selective SSM params (one set per direction)
        self.Bp = nn.Linear(d, N, bias=False)     # input->state proj
        self.Cp = nn.Linear(d, N, bias=False)     # state->output proj
        self.dt = nn.Linear(d, 1)                 # adaptive delta (selective step)
        self.lam = nn.Parameter(torch.zeros(N))   # log-decay rate per state channel
        self.u  = nn.Linear(d, N)                 # gated input value
        self.out= nn.Linear(N, d)                 # state -> residual correction
        self.alpha = nn.Parameter(torch.tensor(0.35))  # residual/correction weight
        self.proto = nn.Parameter(torch.randn(C, K, d))
        self.tau, self.N, self.K = tau, N, K

    def _scan(self, x):                           # x: (B,T,d) -> (B,T,d) correction
        B, T, _ = x.shape
        dt = F.softplus(self.dt(x))               # (B,T,1) selective step  (adaptive_delta)
        a  = torch.exp(-dt * F.softplus(self.lam))# (B,T,N) decay in (0,1)
        bu = F.softplus(self.Bp(x)) * self.u(x)   # (B,T,N) gated input
        C  = self.Cp(x)                           # (B,T,N) readout proj
        h  = x.new_zeros(B, self.N)
        ys = []
        for t in range(T):
            h = a[:, t] * h + (1 - a[:, t]) * bu[:, t]
            ys.append((C[:, t] * h).unsqueeze(1))
        y = torch.cat(ys, 1)                       # (B,T,N)
        return self.out(y)                         # (B,T,d)

    def forward(self, E):                          # E: (B,T,D) frozen Perch embeds
        x = self.proj(E)
        fwd = self._scan(x)
        bwd = torch.flip(self._scan(torch.flip(x, [1])), [1])   # two-pass / bidirectional
        x = x + self.alpha * (fwd + bwd)           # residual_ssm
        xn = F.normalize(x, dim=-1)                # (B,T,d)
        pn = F.normalize(self.proto, dim=-1)       # (C,K,d)
        sim = torch.einsum('btd,ckd->btck', xn, pn)
        logits = self.tau * sim.amax(dim=-1)       # (B,T,C) multi-proto max; per-window logits
        return logits
```
Training (T2, runs in minutes on returned GPU; even CPU-trainable given tiny size): BCE-with-logits multi-label + optional rank/AUC surrogate (defer to A4), `mixup` (+0.417) on embeddings, `time_shift` (+0.634) by rolling the window sequence. Freeze Perch; only the 0.55M head trains. Add `sonotype_mirror` by max-pooling logits across the unmapped-sonotype groups post-hoc.

### Post-processing chain (mostly T1, A4 owns the search)
1. Per-class temperature on logits (standard 0.948 trick).
2. Symmetric time-smoothing over T (learned bi-SSM already does most of this; a residual `[0.1,0.2,0.4,0.2,0.1]` conv on probs is a cheap belt-and-suspenders).
3. `P += (P_max + P_mean − P_max_mean)·0.8` per class (BirdCLEF-2024 3rd-place max+mean adjustment; verify on OOF — it helped public +0.02 but was fragile on private, so gate it on CV).
4. Rank-blend (`rank_power≈0.5`) with Perch logits + SED branch + BirdNET (unmapped classes).
5. Site-hour prior multiply (`lambda_prior≈0.4`).

---

## 4. DeepSeek / Qwen / Kimi transferable math — what realistically applies

| idea | transfers? | verdict |
|---|---|---|
| **DeepSeek MLA (multi-head latent attention, low-rank KV)** | The *math* (low-rank joint projection of the embedding into a latent space before the head) maps to the `D→d` low-rank `proj` we already use. The KV-cache benefit is irrelevant (no autoregressive decode here). | Take only the **low-rank projection** idea (already in the sketch). MLA proper = **no** (no decode loop). **Low as a standalone.** |
| **DeepSeek/Mixtral MoE routing** | A sparse MoE prototype head: route each window-embedding to a few "expert" prototype banks (e.g. by taxonomic group / sonotype). Image-classification MoE shows experts specialize per-class-cluster; could help the 234-class long-tail + 28 unmapped. | Plausible **small** lever: a 4–8 expert top-2 router over prototype groups (Aves/Insecta/Amphibia/...). Adds <0.1M params, near-zero CPU cost. But risk of overfit on rare classes and the `sonotype_mirror` max-pool already captures most of it. **Speculative bet, T2, ~+0.002–0.004 if it lands.** |
| **Qwen/Kimi long-context tricks** | Aimed at 100k+ tokens; T=12 makes them inert. | **No.** |
| **Multi-head / multi-prototype readout** | Multi-head MoE (MH-MoE) splits tokens into sub-tokens for finer routing → maps cleanly to **K>1 prototypes per class** (song vs call). | **Yes, cheap** — already in sketch as `K=2` max-pool. Worth A/B. **Low–medium.** |
| **Muon/normalization & gating lessons (Kimi/DeepSeek training)** | RMSNorm + careful gating init stabilizes tiny SSMs. | Use RMSNorm on `proj` output and init `alpha` small (0.3). Free hygiene, **not a lever.** |

**Honest read:** the LLM-lab headline architectures are built for autoregressive decode over very long contexts. At T=12 windows / 234 classes their distinctive benefits evaporate. The transferable *crumbs* are: (a) low-rank latent projection (MLA → already used), (b) multi-prototype/multi-head readout (MH-MoE → K>1), (c) a small taxonomic-group MoE router as a speculative T2 bet. None is a >+0.005 lock; the SSM temporal head + prototype classifier (Section 1/3) is the real prize.

---

## Ranked findings

| # | technique | track | expected LB lever | feasibility | impl sketch / math | source |
|---|---|---|---|---|---|---|
| 1 | **Bi-directional scalar-SSD (Mamba-2) temporal head + selective Δ + residual** over the 12-window Perch sequence | **T2** (train min; head infers free on CPU). Partial **T1** if any public ProtoSSM head weights load directly | **+0.003–0.008** (generalizes BirdCLEF-24 smoothing +0.01); structural, crosses 0.948 ceiling | High — 0.15M params, ~30-line torch, no custom kernels, <1 ms/file | `h_t=a_t h_{t-1}+(1-a_t)B_t⊙u_t`, `a_t=exp(-Δ_t·softplus(λ))`, `Δ_t=softplus(W e_t)`; fwd+bwd; `out=e+α(y_f+y_b)`, α≈0.35 | tridao.me Mamba-2 SSD blog; arXiv 2512.03563 |
| 2 | **Prototype / cosine classifier head (K-proto, sonotype max-pool)** replacing dense linear | **T2** train; **T1** post-proc for sonotype mirror | **+0.003–0.006**, esp. on 28 unmapped classes (A2 bottleneck) | High — 0.06M params | `z_{t,c}=τ·max_k cos(W e_t, p_{c,k})`; sonotype mirror = max over similar-class group | DCASE'22 task5 protonet ensemble (41→60% F) |
| 3 | **Symmetric time-smoothing + max/mean logit adjustment** post-proc | **T1** (pure post-proc on existing OOF/logits) | **+0.004–0.010** (24-3rd: smoothing +0.01) | Trivial, today, no GPU | conv `[0.1,0.2,0.4,0.2,0.1]` on probs over t; `P+=(P_max+P_mean−P_maxmean)·0.8` (gate on CV) | zenn.dev BirdCLEF-24 3rd solution |
| 4 | **Low-rank latent projection (D→d≈256)** before head (DeepSeek-MLA math) | T2 | enables 1–3; standalone <+0.001 | High (one Linear) | `ẽ=W_proj e`, share across SSM+proto | arXiv 2502.14837 (MLA) |
| 5 | **Multi-prototype / multi-head readout (K=2)** (MH-MoE math) | T2 | **+0.001–0.003** (song/call polymorphism) | High, +0.06M | `max_k cos(·,p_{c,k})` | arXiv 2404.15045 (MH-MoE) |
| 6 | **Taxonomic-group sparse MoE router** over prototype banks | T2, speculative | **+0.002–0.004** *if it lands*; overfit risk on rare classes | Medium — small router, near-zero CPU | top-2 of 4–8 group experts; gate by softmax over group logits | arXiv 2411.18322 (MoE image-cls sweet spot) |
| 7 | Gated DeltaNet / RWKV-7 / GLA / log-linear state | — | **<+0.001 here** (long-context machinery wasted at T=12) | n/a | use only if scalar-SSD underfits | ICLR'25 GatedDeltaNet / RWKV-7 arXiv 2503.14456 / GLA arXiv 2312.06635 |
| 8 | BioMamba full SSM *encoder* (replace AVES/Perch) | T2, big | unknown; needs heavy GPU train; competes with Perch ceiling | Low this sprint (no GPU 7h) | Mamba-2 stack on wav2vec frontend | arXiv 2512.03563 |

Low-priority (<+0.002, flagged): #7 (all long-context linear-attn variants at T=12), MLA KV-cache machinery, chunked-scan training kernels, hierarchical/Fenwick GLA.

---

## Sources
- Mamba-2 / SSD math: https://tridao.me/blog/2024/mamba2-part1-model/ , https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
- SSMs for bioacoustics (BioMamba, Dec 2025): https://arxiv.org/abs/2512.03563
- CPU-native O(1) SSD cache: https://arxiv.org/pdf/2603.09555
- Gated Linear Attention (ICLR'25): https://arxiv.org/abs/2312.06635
- Gated DeltaNet (ICLR'25): https://arxiv.org/pdf/2412.06464
- RWKV-7 "Goose": https://arxiv.org/pdf/2503.14456
- DeepSeek MLA: https://arxiv.org/abs/2502.14837 ; explainer https://planetbanatt.net/articles/mla.html
- Multi-Head MoE: https://arxiv.org/abs/2404.15045 ; MoE for image-cls sweet spot: https://arxiv.org/html/2411.18322
- DCASE'22 few-shot bioacoustic protonets: https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Li_90_5.pdf
- BirdCLEF-2024 3rd post-proc (smoothing + max/mean adj): https://zenn.dev/yuto_mo/articles/53ed2b27c1f52b
- BirdCLEF-2026 ProtoSSM kernels: https://www.kaggle.com/code/imaadmahmood/birdclef-2026-perch-v2-protossm-0-925 ; https://www.kaggle.com/code/marynaborovska/birdclef-26-two-pass-ssm-advanced-pp ; https://www.kaggle.com/code/nina2025/birdclef-2026-onnx-perch-proto-sed ; repo https://github.com/ferariz/birdclef2026

*Caveat: BirdCLEF-2026 Kaggle kernels are JS-rendered; exact ProtoSSM source could not be scraped. Section 1 is reverse-engineered from the meta-analysis feature names (adaptive_delta, residual_ssm, two-pass, n_windows=12, correction_weight≈0.3, alpha_blend≈0.4, rank_power≈0.5), the public SSM literature, and the analogous BirdCLEF-2024 post-processing. Confidence: high on the structure, medium on exact hyperparameters.*
