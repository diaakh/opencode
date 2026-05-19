"""Fetch 100 public BC2026 kernels and extract LB scores from titles."""
import subprocess
import re
import json
from pathlib import Path

OUT_DIR = Path("/home/user/opencode/birdclef-2026/analysis/entropy_tta/public_kernels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Get 100 kernels (5 pages × 20)
kernels = []
for page in range(1, 6):
    r = subprocess.run(
        ["kaggle", "kernels", "list", "--competition", "birdclef-2026",
         "--sort-by", "voteCount", "-p", str(page), "--page-size", "20"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"page {page} failed: {r.stderr[:200]}")
        continue
    # Parse table output
    lines = r.stdout.strip().split('\n')
    for line in lines[2:]:  # skip header
        line = line.strip()
        if not line: continue
        parts = re.split(r'\s{2,}', line)
        if len(parts) >= 4:
            ref = parts[0]
            title = parts[1]
            author = parts[2] if len(parts) > 2 else ""
            try: votes = int(parts[-1])
            except: votes = 0
            kernels.append({"ref": ref, "title": title, "author": author, "votes": votes})

print(f"Got {len(kernels)} kernels")

# Extract LB from title/description with regex
def extract_lb(s):
    # Patterns: "0-925" (in URL), "0.925", "LB 0.925", "0.925 LB"
    s = str(s).lower()
    patterns = [
        r'0[-\.]?9(\d\d)\b',     # 0.9xx, 0-9xx
        r'lb[-\s]+0\.?(\d{3})',  # lb 0925 or lb 0.925
        r'lb[-\s]+(\d{3})',
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            digits = m.group(1)
            if len(digits) == 2:
                return float(f"0.9{digits}")
            if len(digits) == 3:
                return float(f"0.{digits}")
    return None

for k in kernels:
    k["lb_from_title"] = extract_lb(k["title"]) or extract_lb(k["ref"])

with_lb = [k for k in kernels if k["lb_from_title"] is not None]
print(f"\nKernels with LB in title/ref: {len(with_lb)}/{len(kernels)}")

# Save catalog
with open(OUT_DIR / "catalog.json", "w") as f:
    json.dump(kernels, f, indent=2)

# Print summary
print("\nKernels with LB extracted from title/ref:")
for k in with_lb[:30]:
    print(f"  LB={k['lb_from_title']:.3f}  votes={k['votes']:3d}  {k['ref'][:60]}")

# LB distribution
lbs = [k["lb_from_title"] for k in with_lb]
if lbs:
    print(f"\nLB stats: n={len(lbs)}, min={min(lbs):.3f}, max={max(lbs):.3f}, median={sorted(lbs)[len(lbs)//2]:.3f}")
