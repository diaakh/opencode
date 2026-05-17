"""Extract blend / ensemble configs from each notebook.

Look for the 'solutions = {' pattern, ensemble weights, model lists.
"""
import json, re
from pathlib import Path
import pandas as pd

CORPUS = Path(__file__).parent
NB_DIR = CORPUS / "notebooks"

# Pattern: solutions = { 'type_add': '...', 'Models': [ ... ] }
# Extract Model + weight + xSED + LB
MODEL_BLOCK = re.compile(
    r"\{[^{}]*'Model'[^{}]*'weight'\s*:\s*([0-9.]+)[^{}]*'LB'\s*:\s*'(0\.\d{3,4})'[^{}]*\}",
    re.DOTALL
)
SOLUTIONS_BLOCK = re.compile(r"solutions\s*=\s*\{(.+?)\}\s*$", re.DOTALL | re.MULTILINE)
SUBM_NAME = re.compile(r"'subm'\s*:\s*'([^']+)'")
MODEL_NAME = re.compile(r"'Model'\s*:\s*'([^']+)'")

def parse_blends(text):
    """Find blend configurations."""
    blends = []
    # Look for solutions = { ... } blocks
    for m in re.finditer(r"solutions\s*=\s*\{(.*?)\}\s*\n", text, re.DOTALL):
        block = m.group(1)
        # Find all Model entries
        models = re.findall(r"\{[^{}]*'Model'\s*:\s*'([^']+)'[^{}]*\}", block)
        weights = re.findall(r"'weight'\s*:\s*([\d.]+)", block)
        lbs = re.findall(r"'LB'\s*:\s*'(0\.\d{3,4})'", block)
        if models or weights or lbs:
            blends.append({
                "n_models": len(models),
                "models": "|".join(models),
                "weights": "|".join(weights),
                "lbs": "|".join(lbs),
            })
    return blends

def parse_notebook(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
    except: return None
    cells = nb.get("cells", [])
    src = "\n".join("".join(c.get("source", [])) for c in cells)
    blends = parse_blends(src)
    if not blends: return None
    # Combine all blend configs
    result = {"n_blend_configs": len(blends),
              "max_n_models": max(b["n_models"] for b in blends),
              "all_models": "||".join(b["models"] for b in blends),
              "all_weights": "||".join(b["weights"] for b in blends),
              "all_lbs": "||".join(b["lbs"] for b in blends)}
    return result

def main():
    rows = []
    for d in sorted(NB_DIR.iterdir()):
        if not d.is_dir(): continue
        ipynbs = list(d.glob("*.ipynb"))
        if not ipynbs: continue
        info = parse_notebook(ipynbs[0])
        if info is None: continue
        info["ref"] = d.name.replace("__", "/", 1)
        rows.append(info)
    df = pd.DataFrame(rows)
    df.to_csv(CORPUS / "blends.csv", index=False)
    print(f"Wrote blends.csv: {len(df)} kernels with blend configs")
    if len(df) > 0:
        print(f"\nKernels with most models in ensemble:")
        print(df.nlargest(20, "max_n_models")[["ref","max_n_models","all_lbs"]].to_string(index=False))

if __name__ == "__main__":
    main()
