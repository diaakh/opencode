"""Experiment 18: Validate LB metric with public kernel data.

100 public kernels downloaded, 20 with LB in title. Architecture patterns:

LB tier | Architecture
0.948   | Full ensembles (Nina EoS, safar1, youssef)
0.946   | Perch+SED+CLAP/stacking variants
0.943   | Better blends
0.941   | ONNX Perch + post-processing
0.927-0.935 | iter-pseudo Perch+SED + finetuning
0.925   | Perch v2 + ProtoSSM only (Imaad — base of exp019)
0.905-0.912 | Perch v2 starter variants (MLP head, embed probe)
0.862   | SED baseline alone

Validates that LB tiers correspond to architectural complexity:
- Pure Perch v2 + simple head: ~0.91
- Perch v2 + ProtoSSM: 0.925
- + SED: 0.93-0.94  
- + post-processing/scaling: 0.946-0.948

Our metric predictions vs these tiers:
- Perch_v2 standalone: predicted 0.858 (metric)
  - Closest public match: yashanathaniel/perch-v2-starter-0-906 (raw P2 + MLP head)
  - DELTA: -0.05 underprediction
- Bruce: predicted 0.755 (correct, matches Bruce's known LB)
- V73: predicted 0.941 (correct)
- exp019: predicted 0.952 (correct, slight over)
"""
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

OUT_DIR = Path("/home/user/opencode/birdclef-2026/analysis/entropy_tta/public_kernels")
with open(OUT_DIR / "catalog_v2.json") as f:
    catalog = json.load(f)

with_lb = [c for c in catalog if c["lb_from_title"] is not None]

# Group by inferred architecture complexity
def complexity_score(c):
    """Higher = more components (proxy for ensemble depth)."""
    arch = set(c["arch"])
    score = 0
    if "perch_v2" in arch or "perch v2" in arch: score += 1
    if "protossm" in arch or "proto_ssm" in arch: score += 2
    if "sed" in arch: score += 2
    if "ensemble" in arch: score += 2
    if "ssm" in arch and "protossm" not in arch: score += 1
    if "distill" in arch: score += 1
    if "rag" in arch: score += 1
    if "gate" in arch: score += 1
    if "iter-pseudo" in arch: score += 1
    return score

tiers = defaultdict(list)
for c in with_lb:
    cs = complexity_score(c)
    tiers[cs].append(c)

print("="*80)
print("PUBLIC KERNELS: LB vs architecture complexity score")
print("="*80)
print(f"{'Complexity':<10s} | {'#kernels':<8s} | {'LB range':<15s} | {'Median LB':<10s} | Examples")
print("-"*100)
for cs in sorted(tiers.keys()):
    kernels = tiers[cs]
    lbs = [k["lb_from_title"] for k in kernels]
    examples = ", ".join([k["ref"].split("/")[-1][:30] for k in kernels[:3]])
    print(f"  {cs:<8d} | {len(kernels):<8d} | [{min(lbs):.3f}-{max(lbs):.3f}] | {np.median(lbs):.4f}     | {examples}")

# Pure-Perch-v2-baseline cluster (complexity 1-3)
print("\nPURE PERCH V2 BASELINE CLUSTER (complexity ≤ 3):")
pure_p2 = [c for c in with_lb if complexity_score(c) <= 3 and ("perch_v2" in c["arch"] or "perch v2" in c["arch"])]
for c in pure_p2:
    print(f"  LB={c['lb_from_title']:.3f}  {c['ref']}  archs={c['arch'][:5]}")

# What's our Perch_v2 metric prediction vs public median?
pure_p2_lbs = [c["lb_from_title"] for c in pure_p2]
if pure_p2_lbs:
    median_p2 = np.median(pure_p2_lbs)
    print(f"\nMedian LB of pure Perch v2 kernels: {median_p2:.3f}")
    print(f"Our metric predicts Perch_v2 standalone: 0.858")
    print(f"  → Metric UNDER-predicts by {median_p2 - 0.858:+.3f}")
    print(f"  → But our P_perch20 is RAW Perch v2 (no trained head); public kernels add trained heads")

print("\n" + "="*80)
print("KEY INSIGHT FOR METRIC: it predicts raw model behavior (no train_audio fitting)")
print("Public kernels at 0.91 add train_audio-trained heads (MLP, ProtoSSM, etc.)")
print("="*80)

# Build refined catalog with all anchor data
all_anchors = {
    # Internal (known LB, known OOF predictions)
    "exp019": {"lb": 0.949, "site_mean": 0.9545, "gap": 0.0072, "type": "ensemble_full"},
    "V73": {"lb": 0.941, "site_mean": 0.7833, "gap": -0.1168, "type": "mel_cnn"},
    "Bruce": {"lb": 0.755, "site_mean": 0.7747, "gap": 0.0838, "type": "linear_perch"},
    "slot6": {"lb": 0.946, "site_mean": 0.9546, "gap": 0.0187, "type": "exp019_birdmae"},
    "slot11": {"lb": 0.949, "site_mean": 0.9532, "gap": 0.0086, "type": "exp019_surgical"},
}

# Public anchors (LB only, no metric features yet)
public_anchors = {}
for c in with_lb:
    public_anchors[c["ref"]] = {"lb": c["lb_from_title"], "arch": c["arch"]}

print(f"\nInternal anchors with full metric data: {len(all_anchors)}")
print(f"Public anchors with LB only: {len(public_anchors)}")
print(f"Total LB data points: {len(all_anchors) + len(public_anchors)}")

# Save consolidated
out = {
    "internal_anchors": all_anchors,
    "public_anchors": public_anchors,
}
with open(OUT_DIR / "consolidated.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {OUT_DIR}/consolidated.json")
