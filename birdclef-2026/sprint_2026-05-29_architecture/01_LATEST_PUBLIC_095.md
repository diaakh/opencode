# Latest public 0.95 code — pulled 2026-05-29 (raw .ipynb saved in `public_kernels/`)

Pulled the freshest high-vote public kernels via Kaggle API. The current SOTA public
structure (≈0.950) is **pilkwang/birdclef-2026-eos-oof-gated-pcen** (97 votes, today),
which is the "EoS9 anchor" everyone is forking. Reverse-engineered structure below.

## The 0.950 public anchor (what to beat)
Three-branch rank ensemble:
```
p_anchor = 0.967 * p_poweropt_sz   (dominant ProtoSSM+distilled-SED branch)
         + 0.021 * p_poweropt_pssm (small intermediate PSSM)
         + 0.012 * p_yukiZ         (low-weight diversity: Perch+ProtoSSM+ResSSM)
```
Dominant branch:
```
z_sz = G_prior( 0.60 * R(p_proto) + 0.40 * R(p_sed) )
p_sz = G_post( z_sz )
```
- `R` = **class-wise percentile rank** (rank-blend, not prob-blend — confirms A4 thesis).
- `G_prior` = ecological **site/hour priors** + residual correction.
- `G_post` = file-confidence scaling, rank-aware scaling, temporal smoothing, continuity
  gates, rare-class damping, clipping.
- **Taxonomy smoothing** (post-blend): genus α=0.15, class α=0.05
  `p_out = (1-αc)[(1-αg)p + αg·mean_genus(p)] + αc·mean_class(p_tax)`
- **Gated sidecars** (PCEN/ConvNeXt + BirdNET), OFF by default, applied as masked rank
  correction: `p_final = R(p_anchor) + W_class·M·(R(p_sidecar) − R(p_anchor))`,
  gate max weight 0.03. `M` masks which cells a sidecar may move.

Other config: 5s windows, 12 windows/file, file_pred = max over windows, focal BCE
(γ≈2–2.5, label_smoothing≈0.03), mixup/cutmix α≈0.3–0.4 (train side).

## Read of it
- This is **100% post-processing + rank-blend engineering on top of frozen public
  embeddings** (Perch ProtoSSM, Tucker distilled SED). No new model trained. Entirely
  CPU/T1. It is the saturated ceiling of the public approach (~0.949–0.950).
- The **PCEN + BirdNET sidecars are disabled by default** because the author can't get
  them to help via the gated-correction mechanism — i.e. an open problem they haven't
  cracked. That 28-unmapped-class + insect/PCEN gap is exactly where A2 is digging.
- To beat 0.950 we need a genuinely **diverse, independently-trained model** (our SSM-SED
  / noisy-student g124) added as a real ensemble member or a stronger sidecar gate — not
  more reshuffling of the same three public branches.

## Other kernels pulled (saved in public_kernels/)
- `yaroslavkholmirzayev/0950-replay` — clean 0.950 replay of the EoS anchor.
- `karnakbaevarthur/hierarchical-taxonomy-post-processing` (102 votes) — the taxonomy-
  smoothing source.
- `mtoshidesu/test-birdclef-2026-yaroslav-v221-tax` (91 votes) — taxonomy variant.
- `chesteryuan/last-4days-bx-inference-lb-910`, `rauffauzanrambe/...-netpipelene-v2`
  (0.95 baseline) — for diff/diversity.

Loaded external assets seen: perch-meta, tuckerarrants distilled-SED ONNX, custom
`exp002/exp002b` weak-audio PCEN+logmel-ConvNeXt sidecars.
