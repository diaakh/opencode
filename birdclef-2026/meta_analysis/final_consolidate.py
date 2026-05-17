"""Final consolidation — combines all extracted tables into ONE master frame.

Inputs:
  - kernel_with_lb.csv (priority list + author LB)
  - features_v2.csv (per-kernel features + claimed scores)
  - deps.csv (kernel dependencies)
  - blends.csv (ensemble configs)
  - diffs.csv (version-history)

Outputs:
  - master.csv (one row per kernel with everything)
  - MASTER_FINDINGS.md (final synthesis)
"""
import pandas as pd, numpy as np, re
from pathlib import Path
from collections import Counter

CORPUS = Path(__file__).parent
OUT = CORPUS.parent / "meta_analysis"

# Load each piece
inv = pd.read_csv(CORPUS / "kernel_with_lb.csv")
print(f"Inventory: {len(inv)}")

# Try each optional table
tables = {}
for name in ["features_v2", "deps", "blends", "diffs"]:
    fp = CORPUS / f"{name}.csv"
    if fp.exists():
        tables[name] = pd.read_csv(fp)
        print(f"  {name}.csv: {len(tables[name])} rows, cols={list(tables[name].columns)[:5]}...")
    else:
        print(f"  {name}.csv: MISSING")

master = inv.copy()
for name, df in tables.items():
    if "ref" not in df.columns:
        continue
    # prefix non-key columns
    other_cols = [c for c in df.columns if c != "ref"]
    df = df.rename(columns={c: f"{name}__{c}" for c in other_cols if not c.startswith(name)})
    master = master.merge(df, on="ref", how="left")
print(f"\nMaster: {len(master)} rows, {len(master.columns)} cols")

# Filter to strong-signal kernels
strong_signal_cols = ["features_v2__config_lb_max", "features_v2__title_score",
                       "features_v2__lb_prefix_max", "best_score"]
strong = master.dropna(subset=strong_signal_cols, how="all")
print(f"With LB signal: {len(strong)}")

# Save master
master.to_csv(OUT / "master.csv", index=False)
print(f"Wrote master.csv")

# ---- Now do the big synthesis ----
findings = []
findings.append("# BirdCLEF 2026 — Meta-analysis of 1,250 public kernels\n")

findings.append("## 1. Corpus and LB coverage\n")
findings.append(f"- **{len(inv)}** unique public BirdCLEF 2026 kernels enumerated via Kaggle API")
findings.append(f"- **{inv['best_score'].notna().sum()}** kernels matched to a public LB user")
findings.append(f"- **{inv['best_score'].notna().sum() / len(inv) * 100:.1f}%** coverage")
findings.append(f"- Author LB best: range {inv['best_score'].min():.3f}–{inv['best_score'].max():.3f}, median {inv['best_score'].median():.4f}")

# Public LB distribution
findings.append("\n## 2. Public LB distribution (3,602 teams)\n")
findings.append("| score range | n teams |")
findings.append("|---|---:|")
# (We have user_lb_scores, but the histogram is from the leaderboard)

# Bucket counts
findings.append("\n## 3. Score buckets in our corpus (best LB by kernel author)\n")
def bucket(s):
    if pd.isna(s): return None
    if s >= 0.955: return "5.top_0.955+"
    if s >= 0.948: return "4.elite_0.948-0.955"
    if s >= 0.940: return "3.high_0.940-0.948"
    if s >= 0.925: return "2.mid_0.925-0.940"
    if s >= 0.900: return "1.low_0.900-0.925"
    if s >= 0.500: return "0.weak_0.500-0.900"
    return "0.no_signal"
inv["bucket"] = inv["best_score"].apply(bucket)
bs = inv["bucket"].value_counts().sort_index()
findings.append("| bucket | count |")
findings.append("|---|---:|")
for b, n in bs.items():
    findings.append(f"| {b} | {n} |")

# Top-50 LB authors with public kernels
findings.append("\n## 4. Top-50 LB authors WITH public kernels\n")
top_kerns = inv.dropna(subset=["best_rank"])
top_kerns = top_kerns[top_kerns["best_rank"] <= 50].sort_values("best_rank")
findings.append("| author | LB | rank | kernels |")
findings.append("|---|---:|---:|---|")
for author, group in top_kerns.groupby("author_user"):
    lb = group["best_score"].iloc[0]
    rank = int(group["best_rank"].iloc[0])
    kerns = group["ref"].tolist()
    kerns_str = "<br>".join(kerns)
    findings.append(f"| {author} | {lb} | {rank} | {kerns_str} |")

# Architecture features by bucket
if "features_v2" in tables:
    feat_cols = [c for c in master.columns if c.startswith("features_v2__feat__")]
    if feat_cols:
        findings.append("\n## 5. Architecture features by score bucket\n")
        master["bucket"] = master["best_score"].apply(bucket)
        agg = master.groupby("bucket")[feat_cols].mean()
        # Top features by hi-vs-low delta
        hi = master[master["bucket"].isin(["5.top_0.955+", "4.elite_0.948-0.955"])]
        lo = master[master["bucket"].isin(["0.weak_0.500-0.900", "1.low_0.900-0.925"])]
        if len(hi) > 5 and len(lo) > 5:
            diff = (hi[feat_cols].mean() - lo[feat_cols].mean()).sort_values(key=abs, ascending=False)
            findings.append("\nTop 25 features by HI (≥0.948) vs LO (<0.925) delta\n")
            findings.append("| feature | hi share | lo share | delta |")
            findings.append("|---|---:|---:|---:|")
            for f, d in diff.head(25).items():
                fn = f.replace("features_v2__feat__", "")
                findings.append(f"| {fn} | {hi[f].mean():.2f} | {lo[f].mean():.2f} | {d:+.3f} |")

# Top public checkpoints/datasets dependencies
if "deps" in tables:
    findings.append("\n## 6. Most-imported public dependencies (kernels using them)\n")
    # Count
    notebook_count = Counter()
    dataset_count = Counter()
    model_count = Counter()
    for _, r in tables["deps"].iterrows():
        for nb in str(r.get("notebooks_used", "")).split("|"):
            if nb and nb != "nan": notebook_count[nb] += 1
        for ds in str(r.get("datasets_used", "")).split("|"):
            if ds and ds != "nan": dataset_count[ds] += 1
        for m in str(r.get("models_used", "")).split("|"):
            if m and m != "nan": model_count[m] += 1
    findings.append("\n### Top notebook outputs imported\n")
    findings.append("| notebook | used by N kernels |")
    findings.append("|---|---:|")
    for k, n in notebook_count.most_common(20):
        findings.append(f"| {k} | {n} |")
    findings.append("\n### Top datasets imported\n")
    findings.append("| dataset | used by N kernels |")
    findings.append("|---|---:|")
    for k, n in dataset_count.most_common(20):
        findings.append(f"| {k} | {n} |")
    findings.append("\n### Top models imported\n")
    findings.append("| model | used by N kernels |")
    findings.append("|---|---:|")
    for k, n in model_count.most_common(20):
        findings.append(f"| {k} | {n} |")

# Write findings
with open(OUT / "MASTER_FINDINGS.md", "w") as f:
    f.write("\n".join(findings))
print(f"\nWrote MASTER_FINDINGS.md ({len(findings)} lines)")
