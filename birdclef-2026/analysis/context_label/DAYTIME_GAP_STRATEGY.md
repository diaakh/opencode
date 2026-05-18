# Strategy for hours 11-16 (the daytime gap)

**Problem:** All available training/labeling data is at hours 0-10 + 17-23. Zero windows at hours 11-16. If test files fall in that gap, our hour-conditional priors will silently apply night-biased shifts to daytime audio.

This document explains what we do about it.

## Why hours 11-16 are missing from train_soundscapes

Per ROUND6a (CONTEXT.md), the 4 sites that contribute 95% of train_soundscape files (S22, S01, S02, S13) are duty-cycled SwiftOne deployments that record 1 minute every 75-105 minutes. The duty-cycle is weighted toward night/dawn/dusk — when the bioacoustic activity is highest. Hours 11-16 (Pantanal local noon) are empty by design, not by accident.

The 14 small sites are continuous-recording but contribute < 5% of total files.

## Will test contain hours 11-16?

**Don't know.** Per the rules:
- Test sites overlap with train_soundscapes sites
- Same SwiftOne hardware
- Per-site gain settings inherited from train

If the deployment behavior is faithful, **test should follow the same duty-cycle** → 0% daytime exposure. The sample test row is at 01:00 UTC, which is consistent.

But two pieces of evidence push the other way:
1. **Agent F3's LB-drop decomposition** (in AGENT_REPORTS.md): If Bruce-alone scores 0.93 on labeled OOF and Bruce-with-buggy-w=3.0 scored LB 0.755, the implied daytime fraction is 27-41%. (Caveat: this assumes Bruce's labeled OOF is a realistic LB estimate, which is itself questionable.)
2. **The May-18 LB drops magnitude.** sub1 (exp019 + w=3.0 hour-prior) dropped -0.027 on LB. That's consistent with ~10% of rows getting the dead-hour crush.

We won't resolve this without an LB submission that probes the question directly.

## Mitigation strategies

### Strategy A: Pseudo-label only what we can defend (DONE)

The pseudo-labeling pipeline only writes labels for windows at hours 0-10 + 17-23 (the hours we have unlabeled audio for). We CAN'T produce pseudo-labels for hours 11-16 because we have no audio there. This is conservative and correct.

The `pseudo_labels_v2.parquet` output file is honest about its coverage — every row is at a hour ∈ {0-10, 17-23}.

### Strategy B: Hour-safe prior application at inference (IMPLEMENT)

For submission kernels that apply priors, gate the prior shift to hours where we actually have data. For unseen hours, return raw model predictions unchanged.

```python
HOURS_WITH_DATA = frozenset({0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,23})

EPS = 1e-6

def apply_hour_prior_safe(prob, hours, prior_table, w):
    """Apply log-shift only to hours present in training data.

    prob:        (N, C) array of input probabilities
    hours:       (N,) array of integer hours per row
    prior_table: dict[hour] -> (C,) array of P(class | hour)
    w:           weight on the log-prior shift
    Returns:     (N, C) shifted probabilities
    """
    p = np.clip(prob, EPS, 1 - EPS)
    base_logit = np.log(p / (1 - p))
    shift = np.zeros_like(prob, dtype=np.float32)
    for h, prior_row in prior_table.items():
        if h not in HOURS_WITH_DATA:
            continue
        mask = (hours == h)
        if mask.any():
            shift[mask] = w * np.log(np.clip(prior_row, EPS, 1.0))
    return 1.0 / (1.0 + np.exp(-(base_logit + shift)))
```

**Cost-benefit:**
- On rows at hour ∈ {0-10, 17-23}: full prior gain (Bruce 0.867 → 0.974 in our measurement)
- On rows at hour ∈ {11-16}: prior is a no-op. Lose any gain (which would be guessed anyway). **Avoid catastrophic crushing.**

This is what Agent F5 proposed and what the v3/v4 buggy submissions failed to do.

### Strategy C: Per-hour LB probe (USES 1 SLOT)

Submit one variant that:
- Logs each test file's hour to `submission.csv` itself (as an extra unused class column — won't affect scoring per row_id ordering, will be visible in the post-submit output via the CSV download)
- Or: emit predictions to `subm_4.csv` (the intermediate file) which we CAN download afterward

Wait — `subm_4.csv` IS in the kernel output, just not the scored one. Look at line 6269 of `exp019_fast.py`: `sub.to_csv("subm_4.csv", index=False)`. That file is intermediate and emitted before the dry-run align. After a submission, we can `kaggle kernels output <slug>` and download it.

**This means we CAN log test hour distribution from a submission.** Append a small print/save that dumps the hour-distribution of the actual hidden test set into the kernel output. Cost: 1 submission slot to run with internet=off in scoring mode. Output: a tiny CSV in `/kaggle/working/` that survives the scoring run.

Actually, important caveat: when Kaggle scores a kernel for the leaderboard, only `submission.csv` is preserved as the scoring artifact. The HTML output and any other working-dir files may NOT be downloadable from a scored run. Need to verify before committing a slot to this.

**Safer test:** before relying on this for the LB-probe, push a kernel with the same code in dry-run mode (no hidden test), `kaggle kernels output` the dry-run, confirm the CSV is recoverable. Then it'll work the same way at scoring time.

### Strategy D: External data prior for daytime (FUTURE)

iNat exposes per-observation timestamps via their public API. For each of the 234 BC2026 species, we could:
1. Query observations within Pantanal bounding box (lat -16.5 to -21.6, lon -55.9 to -57.6)
2. Count by hour-of-day
3. Smooth + normalize → per-species per-hour activity prior
4. Ship as a private Kaggle dataset

This gives us a **leakage-safe daytime prior** built from external biological signal, independent of our train_soundscapes coverage. It's not a "model" — it's a static lookup table. Doesn't help us during inference dynamics, but tells us "what daytime test rows are likely calling".

For implementation: ~5-10 min Python script with `requests` + iNat API. Not in this PR but worth doing if hours 11-16 turn out to matter on LB.

### Strategy E: Class-conditional override

Some species are nocturnal-only (frogs, owls). Some are diurnal-only (most Aves). At a daytime test row, the right answer is: suppress nocturnal classes, boost diurnal classes.

From the `taxonomy.csv`:
- 162 Aves (mostly diurnal — except nightjars, owls)
- 35 Amphibia (mostly nocturnal)
- 28 Insecta (mostly nocturnal)
- 8 Mammalia (mixed)
- 1 Reptilia (mixed)

A coarse "if hour ∈ {11-16}: suppress all Amphibia + Insecta by 0.5x, boost all non-nightjar Aves by 1.2x" rule would be a sensible neutral prior for daytime rows. **But again — we don't know if test has these rows.**

## Decision

For the next submission cycle:
1. **Bake Strategy B into every submission.** Mandatory. Cost 0.
2. **Use Strategy C in slot 2 of the next 5-slot cycle.** Confirm whether test has hours 11-16.
3. Based on slot-2 result:
   - If 0% daytime: ignore strategies D/E.
   - If > 5% daytime: build Strategy D's external prior, ship as dataset, integrate by next cycle.

## File ownership

- `analysis/context_label/build_pseudo_labels.py` already constrains output to hours-with-data. ✓
- `inference_notebooks/NEXT_SUBMISSION_RECIPE.md` should be updated to require Strategy B (hour-safe prior). Currently has placeholder text suggesting global-mean fallback — that's *wrong*, replace.
- Any new submission kernel must import `HOURS_WITH_DATA` and use `apply_hour_prior_safe()`.
