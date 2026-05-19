"""Properly extract code from .ipynb files (JSON format)."""
import json
import re
from pathlib import Path

OUT_DIR = Path("/home/user/opencode/birdclef-2026/analysis/entropy_tta/public_kernels")
with open(OUT_DIR / "catalog_v2.json") as f:
    catalog = json.load(f)

with_lb = [c for c in catalog if c["lb_from_title"] is not None]

def read_ipynb_code(ipynb_path):
    """Extract code cells from .ipynb (JSON)."""
    try:
        with open(ipynb_path) as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        code = []
        for cell in cells:
            if cell.get("cell_type") == "code":
                src = cell.get("source", "")
                if isinstance(src, list): src = "".join(src)
                code.append(src)
            elif cell.get("cell_type") == "markdown":
                src = cell.get("source", "")
                if isinstance(src, list): src = "".join(src)
                code.append("# MD: " + src)
        return "\n".join(code)
    except Exception as e:
        return f"# ERR: {e}"

def analyze_kernel(kdir):
    files = list(kdir.glob("*.ipynb")) + list(kdir.glob("*.py"))
    if not files: return None
    code = ""
    for fp in files:
        if fp.suffix == ".ipynb":
            code += read_ipynb_code(fp)
        else:
            try: code += fp.read_text()
            except: pass
    if not code: return None
    
    has_train_sscapes = bool(re.search(r'train_soundscapes', code))
    uses_perch = bool(re.search(r'perch_v2|perch-v2|bird-vocalization-classifier', code, re.I))
    uses_onnx = bool(re.search(r'onnxruntime|\.onnx', code, re.I))
    uses_tf = bool(re.search(r'tensorflow|tf\.saved_model', code, re.I))
    uses_sed = bool(re.search(r'distilled[-_]sed|distilled_sed|tuckerarrants/bc2026-distilled', code))
    uses_protossm = bool(re.search(r'protossm|proto_ssm', code, re.I))
    uses_efficient = bool(re.search(r'efficientnet', code, re.I))
    uses_birdmae = bool(re.search(r'birdmae|bird-mae|bird_mae', code, re.I))
    uses_v73 = bool(re.search(r'v73|fold0\.onnx|fold[1-4]\.onnx', code, re.I))
    uses_iter_pseudo = bool(re.search(r'iter[_-]?pseudo', code, re.I))
    
    return {
        "code_lines": code.count('\n'),
        "code_chars": len(code),
        "has_train_sscapes": has_train_sscapes,
        "uses_perch": uses_perch,
        "uses_onnx": uses_onnx,
        "uses_tf": uses_tf,
        "uses_sed": uses_sed,
        "uses_protossm": uses_protossm,
        "uses_efficient": uses_efficient,
        "uses_birdmae": uses_birdmae,
        "uses_v73": uses_v73,
        "uses_iter_pseudo": uses_iter_pseudo,
    }

results = []
for k in with_lb:
    ref = k["ref"]
    safe_name = ref.replace("/", "_")
    kdir = OUT_DIR / safe_name
    info = analyze_kernel(kdir)
    if info is None: continue
    info["ref"] = ref
    info["lb"] = k["lb_from_title"]
    info["votes"] = k["votes"]
    results.append(info)

def complexity(r):
    cs = 0
    if r["uses_perch"]: cs += 1
    if r["uses_protossm"]: cs += 2
    if r["uses_sed"]: cs += 2
    if r["uses_efficient"]: cs += 1
    if r["uses_birdmae"]: cs += 1
    if r["uses_v73"]: cs += 1
    if r["uses_iter_pseudo"]: cs += 1
    return cs

results.sort(key=complexity)

print(f"{'LB':<6s} | {'cmplx':<5s} | {'lines':<5s} | components | ref")
print("-"*110)
for r in results:
    flags = []
    if r["uses_perch"]: flags.append("Perch")
    if r["uses_protossm"]: flags.append("ProtoSSM")
    if r["uses_sed"]: flags.append("SED")
    if r["uses_efficient"]: flags.append("Eff")
    if r["uses_birdmae"]: flags.append("BMAE")
    if r["uses_v73"]: flags.append("V73")
    if r["uses_iter_pseudo"]: flags.append("ItPseudo")
    print(f"{r['lb']:.3f}  | {complexity(r):<5d} | {r['code_lines']:<5d} | {','.join(flags):<35s} | {r['ref']}")

# Group by complexity
from collections import defaultdict
groups = defaultdict(list)
for r in results:
    groups[complexity(r)].append(r)

print(f"\n{'Complexity':<10s} | {'#':<3s} | {'LB range':<15s} | components")
for cs in sorted(groups.keys()):
    grp = groups[cs]
    lbs = [g["lb"] for g in grp]
    sample = grp[0]
    comp_str = ""
    for k in ["uses_perch", "uses_protossm", "uses_sed", "uses_efficient", "uses_birdmae", "uses_v73"]:
        if sample[k]: comp_str += k.replace("uses_", "")[:5] + "+"
    print(f"  {cs:<8d} | {len(grp):<3d} | [{min(lbs):.3f}-{max(lbs):.3f}] | typical: {comp_str.rstrip('+')}")

with open(OUT_DIR / "kernel_analysis_v2.json", "w") as f:
    json.dump(results, f, indent=2)
