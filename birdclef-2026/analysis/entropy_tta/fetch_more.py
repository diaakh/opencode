"""Fetch more BC2026 public kernels - get pages 6-15 (votes desc) + recent date."""
import subprocess
import re
import json
from pathlib import Path

OUT_DIR = Path("/home/user/opencode/birdclef-2026/analysis/entropy_tta/public_kernels")

# Load existing catalog
existing = set()
with open(OUT_DIR / "catalog_v2.json") as f:
    for c in json.load(f):
        existing.add(c["ref"])
print(f"Existing kernels: {len(existing)}")

# Get more pages from both vote-desc and recent
new_kernels = []
for sort_by in ["voteCount", "scoreDescending", "dateRun"]:
    for page in range(1, 16):
        r = subprocess.run(
            ["kaggle", "kernels", "list", "--competition", "birdclef-2026",
             "--sort-by", sort_by, "-p", str(page), "--page-size", "20"],
            capture_output=True, text=True, timeout=20
        )
        if r.returncode != 0:
            continue
        lines = r.stdout.strip().split('\n')
        for line in lines[2:]:
            line = line.strip()
            if not line: continue
            parts = re.split(r'\s{2,}', line)
            if len(parts) >= 4:
                ref = parts[0]
                if ref in existing: continue
                title = parts[1]
                author = parts[2] if len(parts) > 2 else ""
                try: votes = int(parts[-1])
                except: votes = 0
                existing.add(ref)
                new_kernels.append({"ref": ref, "title": title, "author": author, "votes": votes, "sort": sort_by})

print(f"\nNew unique kernels found: {len(new_kernels)}")

# Extract LB from title
def extract_lb(s):
    s = str(s).lower()
    for p in [r'0[-\.]?9(\d\d)\b', r'lb[-\s]+0\.?(\d{3})']:
        m = re.search(p, s)
        if m:
            d = m.group(1)
            if len(d) == 2: return float(f"0.9{d}")
            if len(d) == 3: return float(f"0.{d}")
    return None

new_with_lb = []
for k in new_kernels:
    k["lb_from_title"] = extract_lb(k["title"]) or extract_lb(k["ref"])
    if k["lb_from_title"] is not None:
        new_with_lb.append(k)

print(f"New kernels with LB in title: {len(new_with_lb)}")
print(f"\nTop 30 new kernels with LB:")
for k in sorted(new_with_lb, key=lambda x: -x["votes"])[:30]:
    print(f"  LB={k['lb_from_title']:.3f}  votes={k['votes']:3d}  {k['ref'][:65]}")

# Save expanded catalog
with open(OUT_DIR / "catalog_expanded.json", "w") as f:
    json.dump(new_kernels, f, indent=2)
print(f"\nSaved {len(new_kernels)} new entries to catalog_expanded.json")
