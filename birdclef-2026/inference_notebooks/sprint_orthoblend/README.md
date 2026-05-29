# SPRINT ORTHO-BLEND — BirdCLEF+ 2026 (Build-Agent B3)

Items **T1-1 + T1-2 + T1-4**: a submittable CPU inference notebook that (a) catches up to
the NEW public base stack (tuckerarrants distilled-SED + sgkfk ProtoSSM/ResidualSSM +
perch-v2-no-dft), (b) adds the **live UNEXPLOITED orthogonal trained models** as low
rank-weight blend members, (c) uses **OpenVINO FP16 + AsyncInferQueue** (fall back to
ONNXRuntime CPU) for speed.

- Notebook: `sprint_orthoblend.py` (cell-structured; push as a Kaggle script kernel)
- Kernel metadata: `dataset-metadata.json` (verified input mounts; `enable_internet=false`)
- Output: `submission.csv` — 234 species cols, `row_id = <filename>_<endsec>`, 12×5s windows.

---

## Verified-live asset slugs + exact mount paths  (Kaggle API, 2026-05-29, all 200 OK)

`datasets/view/<owner>/<slug>` returned 200 for every slug below; file listings read from
the dataset zip central directory (exact names + byte sizes confirmed).

| member / role | slug | mount path | key files |
|---|---|---|---|
| distilled-SED (5 folds) — **dominant** | `tuckerarrants/bc2026-distilled-sed-public` (91 MB) | `/kaggle/input/bc2026-distilled-sed-public/` | `sed_fold0..4.onnx` (each 19.7 MB; in `[B,1,256,313]` → `clip_logits[B,234]` + `framewise_logits`) |
| Perch v2 (no DFT) ONNX | `tuckerarrants/perch-v2-no-dft-onnx` (484 MB) | `/kaggle/input/perch-v2-no-dft-onnx/` | `perch_v2_no_dft.onnx` (+ onnxruntime wheel) |
| ProtoSSM + ResidualSSM — **dominant + diversity** | `hideyukizushi/sgkfk-202604041716` (26 MB) | `/kaggle/input/sgkfk-202604041716/` | `train_proto_ssm_single/models/proto_ssm_best.pt`, `train_proto_ssm_single/models/proto_ssm_history.json`, `ResidualSSM/models/residual_ssm_best.pt`, `perch_cache/full_oof_meta_features.npz` |
| **tonylica orthogonal trained** (rank-5 team, 0 importers) | `tonylica/birdclef-2026-model` (775 MB, v2) | `/kaggle/input/birdclef-2026-model/` | `LB872.pt`, `LB862.pt`, `LB811.pth`, `LB792.pth`, `manifest.json`, `xsed/sed_fold0..4.onnx`, `perch/perch_v2_no_dft.onnx` |
| site/hour priors | `adkasd/birdclef-2026-priors-research` | `/kaggle/input/birdclef-2026-priors-research/` | `combined_hour_prior.csv` |
| Perch base model | `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1` | `/kaggle/input/...perch_v2_cpu/1/` | `assets/labels.csv` |

> Mount discovery in the notebook is path-agnostic (`Path('/kaggle/input').rglob(...)`),
> so a slug attaching at any folder name still resolves. Any missing member is silently
> dropped and the remaining rank-weights renormalize (graceful degrade).

### tonylica note (load-bearing)
`tonylica/birdclef-2026-model` bundles the **0.949 public-notebook inputs** (its `perch/` and
`xsed/` copies are byte-identical to the tuckerarrants assets) **plus its own trained heads**
`LB872.pt / LB862.pt / LB811.pth / LB792.pth`. Those `.pt`/`.pth` are **raw state_dicts with no
published model class**, so they are not safely loadable offline. The portable, runnable
orthogonal signal we ensemble as the `tonylica` member is its **own `xsed/sed_fold*.onnx` fold
set** (same I/O contract as the distilled SED, runs through the identical OpenVINO/ORT path).
This still injects tonylica's independently-trained branch as orthogonal diversity at low
rank-weight, which is the T1-2 goal. (If a published model class for LB872 lands, add it as a
new member with its own loader — the membership dict makes that a one-line change.)

---

## Ensemble config dict  (`ENSEMBLE` in `sprint_orthoblend.py`, cell 0)

Membership is **fully config-driven**: `{name: rank-weight}`. Weights are relative and
renormalized at blend time. Orthogonal trained members enter at low rank-weight (A4).

```python
ENSEMBLE = {
    # --- EoS9 dominant branch (the ~0.950 anchor) ---
    "proto_ssm":     0.60,   # hideyukizushi/sgkfk ProtoSSM   (dominant)
    "distilled_sed": 0.40,   # tuckerarrants 5-fold distilled SED (dominant)

    # --- live UNEXPLOITED orthogonal trained members (T1-2) — LOW rank-weight ---
    "tonylica":      0.10,   # tonylica/birdclef-2026-model (rank-5 team, 0 importers)
    "sgkfk_resssm":  0.08,   # hideyukizushi ResidualSSM diversity branch

    # --- parallel-validation toggles (flip on after orchestrator LOSO lands) ---
    # "birdmae":     0.05,   # BirdMAE  (leave-one-site-out validating now)
    # "perch20":     0.05,   # Perch 2.0
}
```

This mirrors the EoS9 anchor: `R = 0.60·R(proto) + 0.40·R(sed)` dominant branch (rank-space),
the two orthogonal trained members added low (per A4: orthogonal members enter at low
rank-weight, default 0.05–0.15), and **BirdMAE / Perch20 left commented-ready** so the
orchestrator can toggle them the moment leave-one-site-out validation confirms transfer —
without touching inference code.

Post-blend pipeline (reproduces the anchor's `G_prior` / `G_post` / taxonomy tail):
1. per-class **percentile rank-transform**, `power = RANK_POWER = 0.5` (A4 / 0.95+ cluster).
2. **rank-blend** with `ENSEMBLE` weights (rank-space — never probability-space; A4 §2).
3. within-file **Gaussian temporal smoothing** across the 12 windows (`sigma = 0.65`).
4. **site/hour log-prior** rank-shift, `W_PRIOR = 2.0` (conservative; A4 §9 overfit warning).
5. **taxonomy smoothing**: genus `α = 0.15`, class `α = 0.05` (architecture doc).

Other key constants (cell 0): `RANK_POWER=0.5`, `PROTO_W/SED_W=0.60/0.40`, `GENUS_ALPHA=0.15`,
`CLASS_ALPHA=0.05`, `GAUSS_SIGMA=0.65`, `BATCH_FILES=16`, SED mel `n_fft=2048 hop=512 n_mels=256
fmin=20 fmax=16000 top_db=80` (EXACT public 0.950 recipe → mel `[N,1,256,313]`).

---

## Speed: OpenVINO FP16 + AsyncInferQueue (item T1-4)

ONNX models (Perch + the SED fold sets) are wrapped by `OnnxRunner` which:
- **OpenVINO path** (default if `openvino` importable): `read_model` → `compile_model("CPU",
  {PERFORMANCE_HINT: THROUGHPUT, INFERENCE_PRECISION_HINT: f16})` → **AsyncInferQueue(2)** to
  overlap the 2 physical cores across the static batch. FP16 is lossless for a ranking metric.
- **Fallback**: ONNXRuntime CPU, `intra_op=4, inter_op=1, ORT_ENABLE_ALL` (the established
  `sub_v8` session) — automatic if OpenVINO import/compile fails.

**OpenVINO FP16 speedup estimate (A3 lever 1): ~1.5–3× on the heavy ONNX stages** (Perch +
distilled SED), ~0 ROC-AUC cost. This roughly **halves** heavy-stage wall time vs FP32 ORT.

---

## 90-minute budget table  (4 vCPU / 2 physical cores, ~600×1-min files, 7,200 windows)

Per-stage costs are order-of-magnitude from A3, scaled to 600 files. "ON" rows = this
notebook's default `ENSEMBLE`.

| stage | model(s) | FP32 ORT | **OpenVINO FP16** | in default run? |
|---|---|---:|---:|:---:|
| Perch embeddings + logits (substrate for proto/resssm) | `perch_v2_no_dft.onnx` | ~16–18 min | **~8–9 min** | ON |
| distilled SED (5 folds, dominant) | `bc2026-distilled-sed-public` | ~12–15 min | **~6–8 min** | ON |
| tonylica SED (5 folds, orthogonal) | `birdclef-2026-model/xsed` | ~12–15 min | **~6–8 min** | ON |
| ProtoSSM / ResidualSSM heads (reuse cached Perch emb) | sgkfk `.pt` | <1 min each | <1 min each | ON |
| mel front-end (librosa, shared once per batch) | — | ~3–4 min | ~3–4 min | ON |
| rank-blend + priors + taxonomy (numpy) | — | ~1–2 min | ~1–2 min | ON |

| scenario | what runs | est. wall time | fits 90 min? | headroom |
|---|---|---:|:---:|---:|
| A. FP32 ORT, default 4 ONNX fold-sets + heads | Perch + distilled-SED + tonylica-SED + SSM heads | **~50–58 min** | yes | ~32–40 min |
| **B. OpenVINO FP16, default (shipping config)** | same, FP16 heavy stages | **~26–32 min** | **yes** | **~58–64 min** |
| C. B + toggle BirdMAE + Perch20 (post-LOSO) | + 2 orthogonal members | **~36–44 min** | yes | ~46–54 min |

**Headroom:** the shipping default (scenario B) runs in **~26–32 min**, leaving **~58–64 min**
of the 90-min budget free. That headroom is exactly what lets BirdMAE/Perch20 be toggled in
(scenario C, still ~46–54 min headroom) once leave-one-site-out validation lands — no speed
work required to add them.

---

## Validation discipline (carry-over from 02_T1_BLEND_RESULT.md)

NEVER select blend weights in-sample. The orthogonal trained members (tonylica, sgkfk
ResidualSSM, and the pending BirdMAE/Perch20) are the only legitimate levers; weights here
are conservative defaults (0.08–0.10) pending the orchestrator's grouped leave-one-site-out
result. Any member whose solo proxy beats the anchor is leakage-suspect until proven on
held-out sites.

## Push (do NOT auto-commit)

```
kaggle kernels push -p inference_notebooks/sprint_orthoblend/
```
Attaches the 5 verified datasets + Perch model + the proto-residualssm train kernel listed in
`dataset-metadata.json`. `enable_internet=false`, CPU only.
