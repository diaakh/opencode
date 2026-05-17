"""Bulk-pull all 1,250 BirdCLEF 2026 kernels.

Runs in patches with delays to avoid rate-limiting. Resumable: skips files
already pulled. Logs failures.
"""
import pandas as pd, subprocess, os, time, sys, signal
from pathlib import Path

CORPUS = Path(__file__).parent
NB_DIR = CORPUS / "notebooks"
NB_DIR.mkdir(exist_ok=True)
LOG = CORPUS / "pull.log"

PRIORITY = pd.read_csv(CORPUS / "kernel_priority.csv")
print(f"To pull: {len(PRIORITY)} kernels", flush=True)

def slug(ref):
    return ref.replace("/", "__")

t0 = time.time()
ok = fail = skip = 0
fails = []
with open(LOG, "a") as logf:
    for i, row in PRIORITY.iterrows():
        ref = row["ref"]
        s = slug(ref)
        dest = NB_DIR / s
        if dest.exists() and any(dest.glob("*.ipynb")):
            skip += 1; continue
        dest.mkdir(exist_ok=True)
        try:
            r = subprocess.run(
                ["kaggle", "kernels", "pull", ref, "-p", str(dest), "--metadata"],
                capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and any(dest.glob("*.ipynb") or dest.glob("*.py") or dest.glob("*.r")):
                ok += 1
            else:
                fail += 1
                logf.write(f"FAIL {ref}: {r.stderr[:200]}\n")
                fails.append(ref)
        except subprocess.TimeoutExpired:
            fail += 1
            logf.write(f"TIMEOUT {ref}\n")
            fails.append(ref)
        except Exception as e:
            fail += 1
            logf.write(f"ERR {ref}: {e}\n")
            fails.append(ref)
        # Progress every 20
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(PRIORITY) - i - 1) / max(rate, 1e-6)
            print(f"[{i+1:4d}/{len(PRIORITY)}] ok={ok} skip={skip} fail={fail}  "
                  f"elapsed={elapsed/60:.1f}min, ETA={remaining/60:.0f}min", flush=True)
        # Small delay to avoid rate limits
        time.sleep(0.3)

print(f"\nDone. ok={ok} skip={skip} fail={fail}  in {(time.time()-t0)/60:.1f}min")
print(f"Failures logged: {len(fails)} (see {LOG})")
