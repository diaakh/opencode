# T1 in-container blend search — result (2026-05-29)

Ran `agents/t1_blend_search.py`: proxy-driven greedy rank-blend of the exp019 anchor
against **106** harvested (739×234) OOF matrices on disk.

## Headline (read with skepticism)
- exp019 alone: true macro-AUC **0.96429**, site-balanced proxy 0.91341.
- Greedy search "achieved" true macro-AUC **0.98799 (+0.0237)** and proxy 0.94042.

## ⚠️ This +0.024 is almost certainly LEAKAGE/OVERFIT, not LB
1. **In-sample selection.** Weights are chosen on the same 739 labeled-soundscape rows
   the score is read from. No grouped held-out split → inflation is guaranteed.
2. **The lift is driven by RAG/kNN-retrieval helpers.** `rag_embeddings:P_ctx_knn`
   has a *solo* proxy of **0.9772 — higher than the SOTA exp019 pipeline (0.9134)**.
   A retrieval model out-ranking exp019 is the textbook signature of **label leakage**:
   context-kNN built from the same labeled soundscapes effectively retrieves its own
   answers. `P_disc`, `db_ridge` similar. These will NOT transfer to the hidden test.
3. Matches the repo's own hard lesson (K1 agent, `METRIC_REALITY_CHECK.md`): naive OOF
   macro-AUC is *anti-correlated* with LB (Spearman −0.157) due to site contamination.

## The legitimately real signal
| helper | Δtrue macro-AUC | Δproxy | independent model? |
|---|---:|---:|---|
| `birdmae_blend:P_birdmae` | **+0.0050** | +0.0008 | ✅ yes (BirdMAE, no retrieval leak) |
| `perch20_blend:P_perch20` | small + | ~0 | ✅ yes (Perch 2.0) |
| RAG/kNN (`P_ctx_knn`, `P_disc`, `db_ridge`) | large + | large + | ❌ leaks labels — discard |

**Interpretation:** the only helpers that lift true macro-AUC *without* inflating the
leaky proxy are genuinely independent models (BirdMAE, Perch20). +0.005 is modest but
real and confirms the sprint thesis — **orthogonal independent models are the only lever**;
in-sample blend reshuffling is a mirage.

## Validation lesson locked in for the rest of the sprint
- NEVER select blend weights in-sample. Use **grouped (by file/site) held-out** or
  leave-one-site-out, and treat any helper whose solo proxy beats exp019 as leakage-suspect
  until proven on held-out sites.
- The trustworthy candidates to actually blend on Kaggle are independent *trained* models
  (BirdMAE, Perch20, and the live `tonylica`/`sgkfk` weights), evaluated leave-one-site-out.
- This is why the real lever is T2 (train a new orthogonal model) + T1-1 (blend live
  independent trained weights), not the on-disk derived OOFs.

---

## Leave-one-site-out (LOSO) validation — `agents/t1_loso_validate.py`
Honest out-of-site test (weight chosen on N-1 sites, scored on held-out site, concatenated):

| helper | LOSO overall macro-AUC | Δ vs exp019 (0.96429) |
|---|---:|---:|
| **birdmae** (independent FM) | 0.96830 | **+0.00401  ✅ TRUSTED** |
| perch20 (independent FM) | 0.96429 | +0.00000 (no help here) |
| P_ctx_knn (leaky control) | 0.98301 | +0.01872 ⚠️ |
| db_ridge (leaky control) | 0.97298 | +0.00868 ⚠️ |

### Important caveat about this test
LOSO here holds out a site for **weight selection only**. It does NOT remove leakage that
is **baked into a precomputed prediction matrix**. Context-kNN (`P_ctx_knn`) retrieves each
row's own neighbours/labels across all 739 rows, so the held-out site's predictions are
*also* pre-leaked → the control does not collapse. Lesson: **a helper built by retrieval over
the labeled set cannot be cleared by LOSO-on-weights**; it must be regenerated with strict
held-out-neighbour retrieval before it can be trusted.

### Actionable
- ✅ **Include BirdMAE** as a real orthogonal rank-blend member (+0.004, survives honest test).
- ❌ Perch20: no lift on this OOF (keep as toggle, may help on full test).
- ❌ kNN/db_ridge derived OOFs: do NOT ship until regenerated leak-free.
- This is the empirical proof of the sprint thesis end-to-end: independent trained/foundation
  models give small-but-real lifts; in-sample/retrieval blends are mirages.
