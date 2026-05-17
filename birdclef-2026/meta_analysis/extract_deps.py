"""Extract /kaggle/input/ dependencies from each kernel.

Each kernel pulls pretrained models / data from public Kaggle datasets.
The dependency tree shows the resource graph: who depends on which model
weights / processed datasets / wheels.
"""
import json, re
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

CORPUS = Path(__file__).parent
NB_DIR = CORPUS / "notebooks"

# /kaggle/input/<dataset_slug>/...
INPUT_PAT = re.compile(r"/kaggle/input/([a-zA-Z0-9_\-]+)")
NOTEBOOK_OUT = re.compile(r"/kaggle/input/notebooks/([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)")
DATASETS = re.compile(r"/kaggle/input/datasets/([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)")
MODELS = re.compile(r"/kaggle/input/models/([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)")
COMPETITIONS = re.compile(r"/kaggle/input/competitions/([a-zA-Z0-9_\-]+)")

def parse(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
    except Exception as e:
        return None
    cells = nb.get("cells", [])
    src = "\n".join("".join(c.get("source", [])) for c in cells)
    inputs = set(INPUT_PAT.findall(src))
    notebooks_used = set((u, k) for u, k in NOTEBOOK_OUT.findall(src))
    datasets_used = set((u, k) for u, k in DATASETS.findall(src))
    models_used = set((u, k) for u, k in MODELS.findall(src))
    competitions_used = set(COMPETITIONS.findall(src))
    return {
        "n_inputs": len(inputs),
        "input_slugs": "|".join(sorted(inputs)),
        "n_notebooks_used": len(notebooks_used),
        "notebooks_used": "|".join(f"{u}/{k}" for u, k in sorted(notebooks_used)),
        "n_datasets_used": len(datasets_used),
        "datasets_used": "|".join(f"{u}/{k}" for u, k in sorted(datasets_used)),
        "n_models_used": len(models_used),
        "models_used": "|".join(f"{u}/{k}" for u, k in sorted(models_used)),
        "competitions_used": "|".join(sorted(competitions_used)),
    }

def main():
    rows = []
    for d in sorted(NB_DIR.iterdir()):
        if not d.is_dir(): continue
        ipynbs = list(d.glob("*.ipynb"))
        if not ipynbs: continue
        info = parse(ipynbs[0])
        if info is None: continue
        info["slug"] = d.name
        info["ref"] = d.name.replace("__", "/", 1)
        rows.append(info)
    df = pd.DataFrame(rows)
    df.to_csv(CORPUS / "deps.csv", index=False)
    print(f"Wrote deps.csv: {len(df)} rows")

    # Top dependency datasets
    notebook_count = Counter()
    dataset_count = Counter()
    model_count = Counter()
    for _, r in df.iterrows():
        for nb in str(r["notebooks_used"]).split("|"):
            if nb and nb != "nan": notebook_count[nb] += 1
        for ds in str(r["datasets_used"]).split("|"):
            if ds and ds != "nan": dataset_count[ds] += 1
        for m in str(r["models_used"]).split("|"):
            if m and m != "nan": model_count[m] += 1

    print(f"\n=== Top 30 NOTEBOOK outputs imported (kernels that import another kernel's output) ===")
    for k, n in notebook_count.most_common(30):
        print(f"  {n:4d} × {k}")
    print(f"\n=== Top 30 DATASETS imported ===")
    for k, n in dataset_count.most_common(30):
        print(f"  {n:4d} × {k}")
    print(f"\n=== Top 30 MODELS imported ===")
    for k, n in model_count.most_common(30):
        print(f"  {n:4d} × {k}")

    pd.DataFrame(notebook_count.most_common(), columns=["notebook","n_kernels_using"]).to_csv(
        CORPUS / "dep_top_notebooks.csv", index=False)
    pd.DataFrame(dataset_count.most_common(), columns=["dataset","n_kernels_using"]).to_csv(
        CORPUS / "dep_top_datasets.csv", index=False)
    pd.DataFrame(model_count.most_common(), columns=["model","n_kernels_using"]).to_csv(
        CORPUS / "dep_top_models.csv", index=False)

if __name__ == "__main__":
    main()
