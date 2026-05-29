# Noisy-Student Self-Distillation + Bidirectional SSM/Prototype Head

Build-Agent B2 | Sprint 2026-05-29 | Items **T2-A** (noisy-student loop) and **T2-B**
(bi-Mamba-2/SSD + prototype head). Commit-ready training code with a CPU smoke path that
fires the instant GPU resets. Sources: `agents/A6_prior_winners.md`,
`agents/A1_ssm_linear_attention.md`.

These are **new modules behind flags**; the default `train_g124.py` EfficientNet/mel path
is unchanged (`--ssm-head` and `--noisy-student-rounds` both default off).

---

## Files

| File | Purpose |
|------|---------|
| `ssm_head.py` | **T2-B.** `BiSSDProtoHead`: low-rank proj (D→d) → bidirectional scalar-decay Mamba-2/SSD selective scan with adaptive Δ + residual → prototype/cosine classifier (K protos/class) → optional sonotype max-pool mirror. Pure-PyTorch scan fallback + optional `mamba-ssm` kernel. `param_count` / `flops_per_file` / `summarize_head`. |
| `noisy_student.py` | **T2-A.** PowerTransform, soft-target mixing, pseudo-ratio cap, ramping noise schedule, ensemble teacher probs, mixup/embedding-noise, and the `run_noisy_student` round-over-round orchestration loop with LB-correlated round selection. |
| `train_g124.py` (edited) | New flags + `train_ssm_noisy_student()` (real-run path on precomputed Perch embeddings), `load_pseudo_parquet()` (B1 schema), `load_perch_embeddings()`, config builders. Dispatched from `train()` when `--ssm-head` is set. |
| `run_ssm_smoke.py` | `--smoke` CPU path. Synthesizes tiny Perch-like embeddings + pseudo targets and runs the full 3-round loop end-to-end in ~2 s, no GPU/audio/Perch needed. |
| `tests/test_ssm_noisy_student.py` | 21 tests mirroring `tests/` style (sys.path insert, plain asserts). |

---

## T2-A: Noisy-student protocol (exactly A6)

```
PSEUDO_ALPHA=0.7  PSEUDO_TH=0.3  PSEUDO_POWER=2  N_ROUNDS=4(3-4)  pseudo-ratio cap<=0.4
power_transform(p) = p*(p>0.3) + p**2, clamp[0,1]      # zero weak, sharpen confident
target            = 0.7*pseudo + 0.3*hard               # unlabeled rows: hard=0
```

* **Soft labels only** (never argmax) — macro ROC-AUC is a ranking metric.
* **Pseudo-ratio cap**: `cap_pseudo_ratio(L, P, c)` keeps `P<=cL/(1-c)`; `subsample_pseudo_frame`
  keeps the highest-`file_confidence` rows (default cap 0.4 → 40% pseudo / 60% labeled).
* **Noisier student than teacher**: `NoiseSchedule.for_round` ramps strength
  `0.1→0.2→0.3→0.4(→cap 0.5)` over rounds — gaussian jitter + embedding dropout +
  temporal roll + beta-mixup, all stronger than the teacher saw (Xie et al. 2019).
* **Rounds configurable** via `--noisy-student-rounds`; student of round *r* becomes the
  teacher of *r+1*; round 0 is supervised (no teacher).
* **Final TSS stage** (`--ns-final-tss`): last round relabels with the stricter
  `tss_threshold=0.7` (5th-place `pseudo_tss_th`) to clean targets.
* **Round selection** uses the repo's LB-correlated `compute_yao_selection_score`
  (0.40·rank_corr + 0.25·value_corr + 0.15·topk_overlap + 0.10·(1-rank_mae) +
  0.10·active_score − 0.05·val_penalty). `select_best_round` keeps the best round's
  fp16 CPU checkpoint.

`run_noisy_student` takes abstract callbacks (`init_student`, `train_one_round`,
`evaluate`, `selection_score_fn`) so the same loop drives both the SSM head and (future)
the EfficientNet backbone, and so the smoke test can pass tiny closures.

## T2-B: Bidirectional SSM + prototype head

Operates on a per-file Perch-v2 embedding sequence `E∈R^{B×T×D}` (`T=12` 5 s windows,
`D=1280`) → per-window logits `Z∈R^{B×T×234}` (straight to the soundscape submission rows).

```
Ẽ = RMSNorm(W_proj E)                                   # D→d=256 low-rank latent (MLA)
Δ_t = softplus(W_Δ Ẽ)        a_t = exp(-Δ_t·softplus(λ))  # selective "adaptive_delta"
h_t = a_t·h_{t-1} + (1-a_t)·(softplus(B_t)⊙u_t)          # scalar-SSD recurrence
y_t = C_t·h_t   ;   corr = W_out y                       # readout
Ẽ' = Ẽ + α·(corr_fwd + corr_bwd)   α≈0.35               # residual_ssm, bidirectional
z_{t,c} = τ·max_k cos(Ẽ'_t, p_{c,k})                     # prototype/cosine classifier
```

* **Prototype/cosine classifier** (not a dense linear head): 234 class prototypes,
  K=1 (optionally 2 for song/call). `seed_prototypes_from_embeddings` seeds prototypes
  from mean labeled-soundscape embeddings; **zero-train_audio classes (the 28) keep
  random init** (detected from the zero input row, not post-projection) and are lifted by
  the optional **sonotype mirror** (`sonotype_groups` max-pools logits across acoustically
  similar classes).
* **Scan backends**: pure-PyTorch sequential scan (default, CPU-native, microseconds at
  T=12) and an optional `mamba-ssm` kernel path used only on CUDA. `--ssm-no-mamba-kernel`
  forces the PyTorch path (used by the smoke test).

### Param / FLOP budget (production: D=1280, d=256, N=16, C=234, K=1, bidir)

```
proj  327,936 | ssm 33,858 | proto 59,904 | norm 256 | alpha 1   => total 421,955 params
MFLOPs/file (T=12): proj 7.86 + ssm 0.99 + proto 1.53           => ~10.10 MFLOPs/file
```

421,955 trainable params (~0.84 MB fp16) — verified `== sum(p.numel())` of the built
module by `test_ssm_head_param_count_matches_module`. The head is **free** vs Perch
embedding extraction; it does not threaten the 90-min CPU inference budget.

---

## How to launch the real run (when GPU is back)

Prereqs from B1 / Perch precompute:
* `perch_embeddings.npz` — `E:(F,12,1280)` float32 + `filenames:(F,)`.
* `pseudo_labels.parquet` — B1's schema (below).

```bash
python train_g124.py \
  --competition-dir /kaggle/input/birdclef-2026 \
  --output-dir /kaggle/working/g124_ssm \
  --ssm-head \
  --perch-embeddings /path/perch_embeddings.npz \
  --pseudo-parquet   /path/pseudo_labels.parquet \
  --noisy-student-rounds 4 --ns-final-tss \
  --ns-pseudo-alpha 0.7 --ns-pseudo-threshold 0.3 --ns-pseudo-power 2 \
  --ns-pseudo-ratio-cap 0.4 \
  --ssm-proj-dim 256 --ssm-state-dim 16 --ssm-protos-per-class 1 \
  --epochs 12 --lr 1e-3
# -> /kaggle/working/g124_ssm/g124_ssm_fold1_fp16.pt
```

CPU smoke (no GPU/audio/Perch, ~2 s):

```bash
python run_ssm_smoke.py --rounds 3
```

---

## Pseudo-label schema (assumed — RECONCILE WITH B1)

`pseudo_label_precompute/README.md` did **not** exist when this was written, so the loader
(`load_pseudo_parquet`) assumes and validates:

| column | type | meaning |
|--------|------|---------|
| `filename` | str | soundscape stem/basename, must match `filenames` in the Perch npz |
| `start_sec` | float | window start (0,5,…,55) — `start_seconds` accepted as alias |
| *(234 cols)* | float | soft scores in [0,1], one per BirdCLEF-2026 class code |
| `file_confidence` | float | per-file confidence for weighting/subsampling (defaults to 1.0 if absent) |

The loader **fails loudly** if `filename`/`start_sec` or any of the 234 class columns are
missing. If B1's schema differs (e.g. emits `row_id` like the legacy `pseudo_csv` path
instead of `filename`+`start_sec`, or a different confidence column name), update
`PSEUDO_PARQUET_META_COLS` and the alias handling in `load_pseudo_parquet`.

---

## Test / scaffold notes

* `tests/test_ssm_noisy_student.py`: **21 passed** on CPU (`torch 2.12.0+cpu`).
* The pre-existing `tests/test_g124_assets.py` is **uncollectable in this checkout** —
  it imports `run_kaggle_evidence_traces`, `build_evidence_traces`, `build_evidence_pseudo`,
  `evidence_trace`, which were never committed (`git ls-files` shows none). This is
  unrelated to B2's work; all public symbols that file imports from `train_g124` /
  `g124_assets` / `infer` / `build_*` still resolve after the edits (verified).
