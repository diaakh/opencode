"""Download outputs for new LB-known kernels."""
import subprocess
import json
from pathlib import Path

OUT_BASE = Path("/tmp/pub_outputs")
OUT_BASE.mkdir(exist_ok=True)

with open("/home/user/opencode/birdclef-2026/analysis/entropy_tta/public_kernels/catalog_expanded.json") as f:
    new_kernels = json.load(f)

# Filter to those with LB in title
to_download = [k for k in new_kernels if k.get("lb_from_title") is not None]
# Sort: prioritize unique LB tiers, then by votes
print(f"Will attempt download for {len(to_download)} new LB-known kernels")

n_with_oof = 0
n_failed = 0
results = []
for i, k in enumerate(to_download[:50]):  # first 50
    ref = k["ref"]
    safe = ref.replace("/", "_")
    dest = OUT_BASE / safe
    if dest.exists() and len(list(dest.glob("*"))) > 1:
        # already downloaded
        files = [f.name for f in dest.glob("*")]
        npz_files = [f for f in files if f.endswith(".npz") or "cache" in str(f)]
        if npz_files:
            n_with_oof += 1
            results.append({"ref": ref, "lb": k["lb_from_title"], "files": files})
        continue
    dest.mkdir(exist_ok=True)
    r = subprocess.run(
        ["kaggle", "kernels", "output", ref, "-p", str(dest)],
        capture_output=True, text=True, timeout=60
    )
    files = [f.name for f in dest.glob("**/*") if f.is_file()]
    has_oof = any(f.endswith(".npz") or "cache" in str(f) for f in files)
    results.append({"ref": ref, "lb": k["lb_from_title"], "files": files, "has_oof": has_oof})
    if has_oof:
        n_with_oof += 1
        print(f"  [{i+1}] OOF FOUND: {ref}  LB={k['lb_from_title']}  files={files[:3]}")
    if r.returncode != 0:
        n_failed += 1

print(f"\nDownloaded: {len(results)}")
print(f"With OOF cache files: {n_with_oof}")
print(f"Failed: {n_failed}")

# Save results
with open("/home/user/opencode/birdclef-2026/analysis/entropy_tta/public_kernels/outputs_v2.json", "w") as f:
    json.dump(results, f, indent=2)
