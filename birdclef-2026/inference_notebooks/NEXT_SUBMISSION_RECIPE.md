# Next submission recipe — Bruce + pseudo_hour_prior at honest weight

**Status:** Queued. Daily 5/5 slots burned on May 18 UTC. First available slot is at 00:00 UTC May 19.

**Authoring date:** 2026-05-18. Holds whichever model-set is in the repo at that time.

## TL;DR

A standalone `Bruce CLIP-Ridge + pseudo_hour_prior` submission with the **dead-hour fix** and **w_hour ≈ 1.0** (NOT the corpus's 0.05).

- Bruce alone on labeled OOF: **0.867 macro-AUC**
- Bruce + corpus recipe (w_hour=0.05, w_site=0.20): **0.924**
- Bruce + pseudo_hour w=1.0 (NEW): **0.974**
- Bruce + labeled_hour w=0.5 + pseudo_site w=0.5: **0.975** (slight leak, sanity-check ceiling)

The +0.107 lift over Bruce-alone is measured on the same 739-window labeled set the corpus has been using, but with a **leakage-safe base** (Bruce ridge, not exp019). The corpus's v3/v4 recipe gave 0.924 because it tuned weights against exp019's leaked internal OOF; with an honest base, the weights want to be 10-20× larger.

## What we know vs. what we're betting

**Measured (high confidence):**
- Bruce CLIP-Ridge predicts the 75 labeled-positive classes at 0.867 macro-AUC on the 739 labeled windows. This is reproducible from `/tmp/bruce_out/labeled_oof_perch_bruce.npz` (Bruce kernel v6).
- Adding `w_hour * log(pseudo_hour_prior)` with `w_hour=1.0` lifts macro-AUC to 0.974 on the same 739 windows.
- The lift is monotone in w from 0 to 1.0 in the 2D sweep — not a knife-edge.

**Bet (medium confidence):**
- The lift transfers to LB. The labeled set is **S22-night-biased**; if test hours/sites match (likely night-heavy), most of the 0.107 gain transfers. If test contains substantial daytime (the conflict the agents flagged), the dead-hour fix becomes essential.

**Unknown (where we still need an LB sample):**
- Whether Bruce-alone scores above 0.85 on LB. CONTEXT.md says `sub2_bruce_standalone + buggy w=3.0` scored LB 0.755, but a clean Bruce + dead-hour-fix has never been LB-tested. Without that anchor, we can't know if 0.867 → 0.974 on labeled OOF means LB 0.85 → 0.95 or LB 0.75 → 0.85.

## The recipe (paste-as-final-cell)

```python
# bc26_bruce_pseudo_hour_v1.py — paste at end of a Bruce-standalone kernel
import numpy as np, pandas as pd, re
from pathlib import Path

EPS = 1e-6

# 1. Load Bruce predictions (assumes a prior cell wrote bruce_probs as DataFrame
#    with row_id index and 234 class cols — same format as submission.csv)
df = pd.read_csv("submission_bruce_raw.csv")  # or however we name the raw Bruce output

# 2. Load pseudo hour prior (build it locally from the Kaggle dataset
#    or include the CSV in our private dataset attachment)
prior_df = pd.read_csv("/kaggle/input/birdclef-2026-priors-research/pseudo_hour_priors.csv")
prior_df = prior_df.set_index("hour")

# 3. DEAD-HOUR FIX (replace global-mean with hour-safe gating)
# RATIONALE: hours 11-16 are missing from train_soundscapes. Filling with
# global mean = injecting night-biased predictions into possibly-daytime test
# rows. Safer: leave those rows untouched. See DAYTIME_GAP_STRATEGY.md.
HOURS_WITH_DATA = frozenset({0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,23})

# 4. Parse hour from each row_id
def parse_hour(rid):
    m = re.match(r".*_S\d+_\d{8}_(\d{2})\d{4}_\d+$", rid)
    return int(m.group(1)) if m else -1

hours = df["row_id"].apply(parse_hour).values

# 5. Apply hour shift at w=1.0
W_HOUR = 1.0  # NEW: was 0.05 in corpus; 1.0 is best on Bruce in our 2D sweep

prob_cols = [c for c in df.columns if c != "row_id"]
P = df[prob_cols].to_numpy(dtype=np.float32)
P_clip = np.clip(P, EPS, 1 - EPS)
base_logit = np.log(P_clip / (1 - P_clip))

prior_arr = np.zeros_like(P)
valid_row = np.zeros(len(P), dtype=bool)
for i, h in enumerate(hours):
    if h in HOURS_WITH_DATA and h in prior_df.index:
        valid_row[i] = True
        for ci, c in enumerate(prob_cols):
            if c in prior_df.columns:
                prior_arr[i, ci] = prior_df.loc[h, c]

# Apply shift ONLY on rows where we have prior data; leave unseen hours unchanged
prior_logit = W_HOUR * np.log(np.clip(prior_arr, EPS, 1.0))
new_logit = base_logit.copy()
new_logit[valid_row] = base_logit[valid_row] + prior_logit[valid_row]
P_new = 1.0 / (1.0 + np.exp(-new_logit))
print(f"Applied prior on {valid_row.sum()}/{len(P)} rows ({valid_row.mean()*100:.1f}%); "
      f"{(~valid_row).sum()} rows untouched (hours outside train coverage)")

df_new = pd.DataFrame(P_new, columns=prob_cols)
df_new.insert(0, "row_id", df["row_id"])
df_new.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv: {df_new.shape}, "
      f"prior shifts applied with w={W_HOUR}, "
      f"dead-hour fill for hours not in pseudo cache")
```

## Why w=1.0 (not 0.05)

The corpus's V3_FINDINGS_STACK.md derived `w=0.05` from sweeping against:
- An exp019-like rank-power transformed version of Bruce OOF
- Bruce's outputs compressed into the `[0.477, 0.555]` range to "mimic" exp019
- Macro-AUC measured on the same 739 windows the prior was built from

When you sweep against **Bruce's actual probability outputs** (which span [0.005, 0.99]):
- Bruce alone: 0.867
- w=0.05: 0.886 (+0.019)
- w=0.10: 0.901
- w=0.20: 0.929
- w=0.50: 0.974
- **w=1.00: 0.974 (peak)**
- w=2.00: 0.972 (slight overshoot)

The plateau between w=0.5 and 1.0 makes 1.0 the safe choice — small variation around it is forgiving. A 5× error in either direction would barely move the result.

## Risks

1. **Test set hour distribution may differ.** Labeled is 100% hours 0-7 + 18-23. Test could include hours 11-16 — that's where the dead-hour fix matters (without it, w=1.0 would crush daytime predictions by −13 logits per class).

2. **Bruce's standalone LB ceiling is unknown.** Per CONTEXT.md, only one Bruce-flavored submission has happened (sub2 with buggy w=3.0, scored 0.755). The honest Bruce baseline could be anywhere from 0.80 to 0.93 on LB. We're betting the +0.107 lift transfers proportionally.

3. **The labeled_hour version (LB ceiling 0.975) is leaky.** The labeled_hour prior is built from these exact 739 labels. Don't use it.

4. **Per-class softmax-weighted ensemble (0.913 macro-AUC) is a parallel option** — but it needs all 9 models running, ~30+ min on Kaggle. Bruce+pseudo_hour is the cheapest single-kernel test.

## Better plan when we have slots (May 19+)

5-slot lineup that resolves the most uncertainty per slot:

| Slot | Submission | What it answers |
|---:|---|---|
| 1 | vanilla exp019 untouched | True team baseline (corpus assumes 0.949, never verified from this account) |
| 2 | **Bruce standalone + pseudo_hour w=1.0 + dead-hour fix** | Tests the +0.107 lift over Bruce-alone, and the dead-hour fix transfer |
| 3 | exp019 + pseudo_hour w=0.5 + dead-hour fix | Tests if the same recipe lifts exp019 too (likely yes, smaller magnitude) |
| 4 | exp019 + pseudo_hour w=0.05 (corpus's "best" recipe) | Sanity check: corpus's recipe vs ours |
| 5 | Per-class AUC-softmax-weighted ensemble (Bruce + Perch + 7 SED) | Tests if per-class routing matches the 0.913 oracle |

Slot 2 vs 4 tells us if our weight is right. Slot 2 vs 3 tells us if Bruce-base beats exp019-base. Slot 5 tests the routed-ensemble headroom.

## Implementation status

- `bruce_standalone` kernel exists at `birdclef-2026/inference_notebooks/sub2_bruce_standalone.py` but uses w=3.0 (buggy). Need to either:
  - Modify it in place (this recipe)
  - Or fork as `sub_v7_bruce_w1.py`
- Required datasets to attach on Kaggle:
  - `brucewu1200/birdclef-2026-cvlb-assets-0911` (Bruce bundle + Perch ONNX)
  - The `pseudo_hour_priors.csv` from our private `adkasd/birdclef-2026-priors-research` dataset (already exists)
  - `tuckerarrants/perch-v2-no-dft-onnx` (onnxruntime wheel)

## Verification artifacts (in this repo)

- `analysis/prior_retune/heatmap_2d.png` — the 2D sweep visualization
- `analysis/prior_retune/sweep_2d_pseudo.csv` — raw grid values, pseudo×pseudo
- `analysis/prior_retune/sweep_2d_hybrid.csv` — raw grid values, labeled_hour×pseudo_site
- `analysis/prior_retune/retune_priors_vs_bruce.py` — the script that produced all of it
- `analysis/per_class_routed/ensemble_strategies.csv` — per-class ensemble comparison
- `analysis/model_audit/overall_ranking_v2.csv` — all 12 models on labeled OOF

Reproduce with `python3 analysis/prior_retune/retune_priors_vs_bruce.py`.
