# FINAL RETROSPECTIVE — BirdCLEF+ 2026 (honest root-cause analysis)

**Result:** winner Nikita Babych **0.9657**; top-10 0.956–0.966. **Our best private ≈ 0.9417**
(achieved by *forking public 0.950 kernels*, not by anything we built). Every original artifact
we made scored below the public-replay forks; two submissions timed out to no score. Net: a month
of effort, and our own engineering **subtracted** value. This is why.

---

## The single deepest cause: WE NEVER BUILT A FEEDBACK LOOP
Everything below flows from this. To improve a model you need a fast, trustworthy signal. We had
**neither half**:
- **No CV that tracks LB.** Our local OOF was *anti*-correlated with the LB (Spearman −0.157). So
  every real decision required a Kaggle submission.
- **No fast submission.** Our base ran ~85 min and the scoring queue took 1–2 h. So a single
  experiment cost ~3 h and one of ~5 daily slots.
Result: maybe 1–2 real bits of information per day. You cannot climb a leaderboard at that rate.
**Nikita also had no CV — but his pipeline ran in ~9 min, so he iterated on the LB dozens of times.
He had the loop; we didn't.** Fixing the feedback loop was Job #1 and we never did it.

## Cause 2 — We built on a slow base and band-aided it for a month (tunnel vision)
The V237 base was ~10× too slow (A9 flagged this on contact). Instead of fixing it, I repeatedly
trimmed SED folds (losing accuracy) — and still hit **two timeouts** (wasted slots, zero score).
I optimized the wrong variable over and over. The "fast base" pivot came in the final hours.

## Cause 3 — We chased CORRELATED members for weeks despite knowing better
Our own LOSO test said early: only *independent* models help (BirdMAE +0.004); Perch/SED
distillations are mirages. Yet I spent the bulk of the effort on noisy-student students that
**distilled the public SED already inside the base** — correlated by construction, doomed to hurt.
The one independent member (BirdNET, corr 0.32) wasn't even tried until the last day. **I had the
principle in writing and didn't act on it.**

## Cause 4 — Judged an iterative method on one shot
Noisy-student needs iterations (Nikita 0.909→0.918→0.927→0.930). We ran it once, blended one
checkpoint (0.945), declared "it hurts," and quit — never measuring the iter-1→2→3 trajectory.

## Cause 5 — Wrong FRAME, never deliberately chosen or revisited
We operated the whole month inside the frame *"add small members to a public 0.950 base."* The
winning frame was *"train your own strong independent ensemble and iterate it."* The +0.015–0.024
gap to the top **is exactly the value of an independent ensemble we never committed to** (started
Branch B on the last day). We never wrote the frame down or scheduled a "is this frame still right?"
review — so we stayed in the wrong one until forced out.

## Cause 6 — Orchestration overhead masqueraded as progress
Dozens of sub-agents (many ending prematurely, leaving kernels running, with broken scale-tests),
a 15-min monitor with scores of "no change" heartbeats. High activity, low throughput. Motion ≠
progress. A tight personal loop on a fast pipeline would have beaten the whole agent swarm.

---

## THE WORKFLOW THAT WON'T FAIL THIS WAY (durable checklist)

**PHASE 0 — Build the loop before any modeling (gate: do not pass until both hold):**
1. **Fast submission pipeline < 15 min** (jit-trace/ONNX/OpenVINO from day 1). Measure the REAL
   600-file runtime; if a base can't fit a member, it's disqualified as a base.
2. **A CV validated to track LB.** Submit 3–4 deliberately-different configs, correlate CV vs LB.
   If |Spearman| isn't high & positive, FIX the CV first. No trustworthy CV ⇒ no project.

**PHASE 1 — Choose the FRAME explicitly and write it down.**
- "Fork-public-and-tweak" caps at the public ceiling. "Build-own-independent-ensemble" is the only
  path to top-tier. Decide based on the goal. If the goal is top-3, commit to the hard build in
  WEEK 1, not the last day.

**PHASE 2 — Pre-registered decision gates (no slot without passing):**
- Every candidate member must clear: **orthogonality (corr < ~0.7 vs current blend) AND a positive
  held-out delta** on the validated CV — BEFORE it consumes a submission. We wasted slots on members
  whose failure we could have predicted.
- For iterative methods: measure the **trajectory**, never one shot.

**PHASE 3 — Anti-tunnel-vision discipline:**
- A scheduled weekly "ZOOM OUT": re-read your own findings docs; ask "what's the binding constraint,
  and am I optimizing it or a proxy?" (Our binding constraint was the feedback loop + base speed —
  we optimized fold counts.) Act on the strategic conclusions already in your notes.
- When stuck >2 experiments with no gain, change the FRAME, not the knob.

**PHASE 4 — Execution hygiene:**
- Prefer a tight personal loop over agent sprawl; if delegating, demand a measured result (real
  runtime over ≥200 files, not a 5-file estimate) and a single concrete artifact.
- Track a one-line scoreboard of every submission (public+private) so trends, not vibes, drive calls.

**The one-sentence lesson:** *we spent a month optimizing inside the wrong frame because we never
built the fast, trustworthy feedback loop that would have shown us the frame was wrong — and that
loop, plus an independent strong model, is the entire game.*
