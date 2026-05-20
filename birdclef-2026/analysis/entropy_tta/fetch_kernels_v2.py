"""Download code for 100 BC2026 kernels and extract LB scores.

Strategy:
1. Get top 100 by votes
2. Download each kernel's code  
3. Extract LB from title/ref/description
4. Categorize by architecture
"""
import subprocess
import re
import json
from pathlib import Path

OUT_DIR = Path("/home/user/opencode/birdclef-2026/analysis/entropy_tta/public_kernels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Already have catalog
with open(OUT_DIR / "catalog.json") as f:
    kernels = json.load(f)

print(f"Catalog: {len(kernels)} kernels")

# Download code for each (use kaggle kernels pull)
N_DOWNLOAD = 100
for i, k in enumerate(kernels[:N_DOWNLOAD]):
    ref = k["ref"]
    safe_name = ref.replace("/", "_")
    dest = OUT_DIR / safe_name
    if dest.exists() and any(dest.iterdir()):
        continue  # skip already downloaded
    dest.mkdir(exist_ok=True)
    r = subprocess.run(
        ["kaggle", "kernels", "pull", ref, "-p", str(dest), "-m"],
        capture_output=True, text=True, timeout=60
    )
    if r.returncode != 0:
        print(f"  [{i+1}/{N_DOWNLOAD}] {ref}: FAIL {r.stderr[:100]}")
    else:
        print(f"  [{i+1}/{N_DOWNLOAD}] {ref}: OK")

# Now scan each kernel's code for LB indicators
def extract_lb_from_code(text):
    """Find LB number in code/markdown."""
    text = text.lower()
    # Look for explicit LB statements
    patterns = [
        r'lb[\s:=]+0\.(\d{3})',
        r'public[_\s]+lb[\s:=]+0\.(\d{3})',
        r'leaderboard[\s:=]+0\.(\d{3})',
        r'score[\s:=]+0\.(\d{3})',
        r'auc[\s:=]+0\.(\d{3})',
    ]
    found = []
    for p in patterns:
        for m in re.finditer(p, text):
            v = float(f"0.{m.group(1)}")
            if 0.7 < v < 1.0:
                found.append(v)
    return found

# Also look at architecture keywords
def classify_arch(text):
    text = text.lower()
    has = []
    for kw in ["perch_v2", "perch v2", "protossm", "proto_ssm", "sed", "birdmae", "bird_mae",
              "convnext", "efficientnet", "v73", "hgnetv2", "birdaves", "bird_aves",
              "iter_pseudo", "iter-pseudo", "rag", "knn", "ridge", "mlp", "lightgbm", "lgb",
              "two_pass", "gate", "ensemble", "ssm", "distill", "calp"]:
        if kw in text:
            has.append(kw)
    return has

# Scan all downloaded kernels
catalog_v2 = []
for k in kernels[:N_DOWNLOAD]:
    ref = k["ref"]
    safe_name = ref.replace("/", "_")
    dest = OUT_DIR / safe_name
    files = list(dest.glob("*.ipynb")) + list(dest.glob("*.py")) + list(dest.glob("*-metadata.json"))
    code_text = ""
    for fp in files:
        try:
            code_text += fp.read_text() + "\n"
        except: pass
    lb_in_code = extract_lb_from_code(code_text)
    arch = classify_arch(code_text)
    title_lb = None
    s = (k["title"] + " " + ref).lower()
    m = re.search(r'0[-\.]9(\d{2})\b', s)
    if m:
        title_lb = float(f"0.9{m.group(1)}")
    catalog_v2.append({
        "ref": ref, "title": k["title"], "votes": k["votes"],
        "lb_from_title": title_lb,
        "lb_in_code": lb_in_code[:5],  # top 5 candidates
        "arch": arch,
        "code_size": len(code_text),
    })

with open(OUT_DIR / "catalog_v2.json", "w") as f:
    json.dump(catalog_v2, f, indent=2)

# Print summary
print(f"\n--- Summary ---")
with_lb = [c for c in catalog_v2 if c["lb_from_title"] is not None]
print(f"Kernels with LB from title: {len(with_lb)}/{len(catalog_v2)}")

# Architecture distribution
from collections import Counter
arch_counter = Counter()
for c in catalog_v2:
    for a in c["arch"]:
        arch_counter[a] += 1
print(f"\nArch keyword frequencies:")
for arch, count in arch_counter.most_common():
    print(f"  {arch:<15s}: {count}")

# Print kernels with LB by architecture
print(f"\n--- Kernels with LB grouped by primary arch ---")
for c in with_lb:
    archs = ",".join(c["arch"][:3]) if c["arch"] else "?"
    print(f"  LB={c['lb_from_title']:.3f}  {c['ref']:<55s}  arch=[{archs}]")
