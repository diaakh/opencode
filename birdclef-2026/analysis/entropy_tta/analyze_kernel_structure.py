"""Analyze kernel code structure to identify which can be efficiently
reproduced on labeled train_soundscapes.

For each kernel:
- Detect if it uses test_soundscapes or train_soundscapes path
- Check for existing labeled-OOF computation
- Estimate runtime/complexity
"""
import json
import re
from pathlib import Path

OUT_DIR = Path("/home/user/opencode/birdclef-2026/analysis/entropy_tta/public_kernels")
with open(OUT_DIR / "catalog_v2.json") as f:
    catalog = json.load(f)

# Filter to LB-known kernels
with_lb = [c for c in catalog if c["lb_from_title"] is not None]
print(f"LB-known kernels: {len(with_lb)}")

def analyze_kernel(kdir):
    """Read kernel code and detect patterns."""
    files = list(kdir.glob("*.ipynb")) + list(kdir.glob("*.py"))
    if not files:
        return None
    code = ""
    for fp in files:
        try:
            code += fp.read_text()
        except: pass
    # Check for train_soundscapes references (means it already runs on labeled data)
    has_train_sscapes = bool(re.search(r'train_soundscapes', code))
    # Check imports / models
    uses_perch = bool(re.search(r'perch_v2|perch-v2|bird-vocalization-classifier', code, re.I))
    uses_onnx = bool(re.search(r'onnxruntime|\.onnx', code, re.I))
    uses_tf = bool(re.search(r'tensorflow|tf\.saved_model', code, re.I))
    uses_sed = bool(re.search(r'distilled-sed|distilled_sed|\bSED\b', code))
    uses_protossm = bool(re.search(r'protossm|proto_ssm|ProtoSSM', code, re.I))
    # Test paths
    test_path = re.search(r'(test_soundscapes|/kaggle/input/birdclef-2026/test)', code)
    # Estimate complexity by line count
    lines = code.count('\n')
    return {
        "code_lines": lines,
        "has_train_sscapes": has_train_sscapes,
        "uses_perch": uses_perch,
        "uses_onnx": uses_onnx,
        "uses_tf": uses_tf,
        "uses_sed": uses_sed,
        "uses_protossm": uses_protossm,
        "test_path_found": bool(test_path),
    }

# Process LB-known kernels
results = []
for k in with_lb:
    ref = k["ref"]
    safe_name = ref.replace("/", "_")
    kdir = OUT_DIR / safe_name
    info = analyze_kernel(kdir)
    if info is None:
        continue
    info["ref"] = ref
    info["lb"] = k["lb_from_title"]
    info["votes"] = k["votes"]
    results.append(info)

# Sort by simplicity (fewer lines, fewer components = easier to reproduce)
def simplicity_score(r):
    """Lower = simpler/easier to reproduce."""
    s = r["code_lines"]
    if r["uses_sed"]: s += 1000
    if r["uses_protossm"]: s += 500
    if r["uses_tf"]: s += 500
    return s

results.sort(key=simplicity_score)
print(f"\n{'LB':<6s} | {'lines':<6s} | onnx | tf | perch | protossm | sed | ref")
print("-"*100)
for r in results:
    flags = []
    if r["uses_onnx"]: flags.append("O")
    if r["uses_tf"]: flags.append("T")
    if r["uses_perch"]: flags.append("P")
    if r["uses_protossm"]: flags.append("S")
    if r["uses_sed"]: flags.append("E")
    print(f"{r['lb']:.3f}  | {r['code_lines']:<6d} | {'Y' if r['uses_onnx'] else '-':<4s} | {'Y' if r['uses_tf'] else '-':<2s} | {'Y' if r['uses_perch'] else '-':<5s} | {'Y' if r['uses_protossm'] else '-':<8s} | {'Y' if r['uses_sed'] else '-':<3s} | {r['ref']}")

# Save
with open(OUT_DIR / "kernel_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_DIR}/kernel_analysis.json")

# Recommend top 5 candidates for reproduction
print("\n" + "="*80)
print("TOP CANDIDATES FOR LABELED-OOF REPRODUCTION (simplest first)")
print("="*80)
for r in results[:5]:
    print(f"  LB {r['lb']:.3f}  {r['code_lines']:<5d} lines   {r['ref']}")
