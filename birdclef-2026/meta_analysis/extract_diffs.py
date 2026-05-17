"""Extract version-to-version DIFF descriptions from kernel markdown.

Many authors describe their experiment changes like 'lambda_prior 0.4 -> 0.5'.
Each such delta is a known A/B probe with a score outcome.
"""
import json, re
from pathlib import Path
import pandas as pd
from collections import Counter

CORPUS = Path(__file__).parent
NB_DIR = CORPUS / "notebooks"

# Pattern: "X 0.4 -> 0.5" or "X 0.4 → 0.5"
ARROW_DELTA = re.compile(r"(\w[\w\s_/]*?)\s*(\d+\.?\d*)\s*(?:->|→|to)\s*(\d+\.?\d*)", re.I)
# Pattern: "score from 0.91 to 0.92" or "from 0.91 -> 0.92"
SCORE_FROM_TO = re.compile(r"(?:score|LB|public).{0,40}(0\.9\d{2,3}).{0,10}(?:->|→|to|→).{0,10}(0\.9\d{2,3})", re.I)
# "v17: 0.946, v18: 0.948" pattern
VERSION_SCORE = re.compile(r"\bv(\d+)[:\s].{0,40}(0\.\d{3,4})", re.I)

def parse(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
    except: return None
    cells = nb.get("cells", [])
    md_text = "\n".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") == "markdown")
    code_text = "\n".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code")

    deltas = []
    for m in ARROW_DELTA.finditer(md_text + "\n" + code_text):
        var = m.group(1).strip()
        v0, v1 = m.group(2), m.group(3)
        try:
            f0, f1 = float(v0), float(v1)
            # Filter reasonable ranges
            if 0.001 <= f0 <= 1000 and 0.001 <= f1 <= 1000 and f0 != f1:
                deltas.append((var[:40], v0, v1))
        except: pass
    versions = []
    for m in VERSION_SCORE.finditer(md_text):
        versions.append((m.group(1), m.group(2)))
    score_progresses = []
    for m in SCORE_FROM_TO.finditer(md_text):
        score_progresses.append((m.group(1), m.group(2)))
    return {
        "n_deltas": len(deltas),
        "deltas_sample": "|".join(f"{v}={a}→{b}" for v, a, b in deltas[:10]),
        "n_versions_mentioned": len(versions),
        "version_scores": "|".join(f"v{v}:{s}" for v, s in versions[:8]),
        "n_score_progresses": len(score_progresses),
        "score_progresses_sample": "|".join(f"{a}→{b}" for a, b in score_progresses[:5]),
    }

def main():
    rows = []
    for d in sorted(NB_DIR.iterdir()):
        if not d.is_dir(): continue
        ipynbs = list(d.glob("*.ipynb"))
        if not ipynbs: continue
        info = parse(ipynbs[0])
        if info is None: continue
        info["ref"] = d.name.replace("__", "/", 1)
        rows.append(info)
    df = pd.DataFrame(rows)
    df.to_csv(CORPUS / "diffs.csv", index=False)
    print(f"Wrote diffs.csv: {len(df)}")
    # Top probes (changes from X to Y)
    print(f"\n=== Kernels with most score-progress mentions ===")
    print(df.nlargest(15, "n_score_progresses")[["ref","n_score_progresses","score_progresses_sample"]].to_string(index=False))
    print(f"\n=== Kernels with most version mentions ===")
    print(df.nlargest(15, "n_versions_mentioned")[["ref","n_versions_mentioned","version_scores"]].to_string(index=False))

if __name__ == "__main__":
    main()
