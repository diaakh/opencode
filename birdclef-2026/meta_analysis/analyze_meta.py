"""Meta-analysis on the corpus feature table.

Inputs: features_v2.csv (from extract_v2.py)
Outputs:
  meta_analysis.md   — human-readable findings
  bucket_features.csv — per-score-bucket presence of features
  param_sensitivity.csv — numeric params vs score
  approach_clusters.csv — k-means cluster assignment
  lineage_graph.csv — fork tree
"""
import pandas as pd, numpy as np, re
from pathlib import Path
from collections import Counter

CORPUS = Path(__file__).parent
OUT = CORPUS.parent / "meta_analysis"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(CORPUS / "features_v2.csv")
print(f"Loaded {len(df)} kernel feature rows")
score_col = "best_score"

# ---------- 1. Score distribution ----------
print(f"\n[1] Score distribution:")
have_score = df[df[score_col].notna()].copy()
print(f"  with score: {len(have_score)}/{len(df)}")
print(have_score[score_col].describe())

# Bucket scores
def bucket(s):
    if pd.isna(s): return "no_score"
    if s >= 0.948: return "5.elite_0.948+"
    if s >= 0.940: return "4.high_0.940-0.948"
    if s >= 0.920: return "3.mid_0.920-0.940"
    if s >= 0.880: return "2.low_0.880-0.920"
    return "1.weak_<0.880"

df["score_bucket"] = df[score_col].apply(bucket)
print(f"\nScore buckets:")
print(df["score_bucket"].value_counts().sort_index())

# ---------- 2. Feature presence per score bucket ----------
feat_cols = [c for c in df.columns if c.startswith("feat__")]
print(f"\n[2] Feature presence per score bucket ({len(feat_cols)} features):")
agg = df.groupby("score_bucket")[feat_cols].mean()
agg.to_csv(OUT / "bucket_features.csv")
# What features distinguish elite (0.948+) from mid (0.92-0.94)?
elite = df[df["score_bucket"] == "5.elite_0.948+"]
mid = df[df["score_bucket"] == "3.mid_0.920-0.940"]
weak = df[df["score_bucket"] == "1.weak_<0.880"]
if len(elite) > 0 and len(mid) > 0:
    print(f"\n  elite (n={len(elite)}) vs mid (n={len(mid)}) feature delta:")
    delta = (elite[feat_cols].mean() - mid[feat_cols].mean()).sort_values(key=abs, ascending=False)
    for f, d in delta.head(20).items():
        print(f"    {f:40s} elite={elite[f].mean():.2f} mid={mid[f].mean():.2f} delta={d:+.2f}")

# ---------- 3. Numeric param sensitivity ----------
num_cols = [c for c in df.columns if c.startswith("num__")]
print(f"\n[3] Numeric parameter sensitivity ({len(num_cols)} params):")
sensit_rows = []
for c in num_cols:
    # Coerce to numeric
    vals = pd.to_numeric(df[c], errors="coerce")
    sub = df.assign(v=vals).dropna(subset=["v", score_col])
    if len(sub) < 5: continue
    # Correlation with score
    if sub["v"].std() > 0 and sub[score_col].std() > 0:
        corr = sub["v"].corr(sub[score_col])
        # Bin by value, report mean score per bin
        try:
            bins = pd.cut(sub["v"], bins=4)
            grp = sub.groupby(bins, observed=False)[score_col].agg(["mean", "count"])
        except: grp = None
        sensit_rows.append({"param": c, "n": len(sub),
                            "v_min": sub["v"].min(), "v_max": sub["v"].max(),
                            "score_corr": corr})
sens = pd.DataFrame(sensit_rows).sort_values("score_corr", key=abs, ascending=False)
sens.to_csv(OUT / "param_sensitivity.csv", index=False)
print(sens.to_string(index=False))

# ---------- 4. Approach clusters (kmeans on features) ----------
from sklearn.cluster import KMeans
print(f"\n[4] Approach clusters (KMeans on features):")
X = df[feat_cols].astype(float).fillna(0).to_numpy()
for k in [4, 8]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    df[f"cluster_k{k}"] = km.labels_
    print(f"\n  k={k} clusters:")
    for cl in range(k):
        members = df[df[f"cluster_k{k}"] == cl]
        mean_score = members[score_col].mean()
        n = len(members)
        n_with_score = members[score_col].notna().sum()
        # Top features in this cluster
        top_feats = members[feat_cols].mean().sort_values(ascending=False).head(5)
        print(f"    cl {cl}: n={n}, n_with_score={n_with_score}, mean_score={mean_score:.4f}")
        for f, v in top_feats.items():
            if v > 0.3:
                print(f"       {f}: {v:.2f}")

df[["slug", "ref", score_col, "score_bucket", "cluster_k4", "cluster_k8"]].to_csv(
    OUT / "approach_clusters.csv", index=False)

# ---------- 5. Lineage / fork graph ----------
print(f"\n[5] Top forked-from kernels (most referenced as a baseline):")
fork_counter = Counter()
for refs in df["fork_refs"].dropna():
    for r in str(refs).split("|"):
        r = r.strip()
        if r: fork_counter[r] += 1
top_forks = fork_counter.most_common(30)
print("\n  Top 30:")
for r, n in top_forks:
    print(f"    {r:60s} referenced {n} times")
pd.DataFrame(top_forks, columns=["referenced", "n_kernels_referencing"]).to_csv(
    OUT / "lineage_graph.csv", index=False)

# ---------- 6. Time evolution ----------
print(f"\n[6] Time evolution: max score per month:")
priority = pd.read_csv(CORPUS / "kernel_priority.csv")
priority["lastRunTime"] = pd.to_datetime(priority["lastRunTime"], errors="coerce")
merged = priority.merge(df[["ref", score_col]], on="ref", how="left")
merged["month"] = merged["lastRunTime"].dt.to_period("M")
month_max = merged.groupby("month")[score_col].agg(["max", "mean", "count"])
print(month_max)

# ---------- 7. Write markdown ----------
md = ["# Meta-analysis of BirdCLEF 2026 public corpus\n"]
md.append(f"Analyzed **{len(df)}** public Kaggle kernels for BirdCLEF 2026.\n")
md.append(f"## Score coverage\n\n")
md.append(f"- Kernels with parsable best_score: **{have_score.shape[0]}** ({have_score.shape[0]/len(df)*100:.1f}%)\n")
md.append(f"- best_score range: {df[score_col].min():.3f} – {df[score_col].max():.3f}\n")
md.append(f"- best_score mean: {df[score_col].mean():.4f}, median: {df[score_col].median():.4f}\n\n")
md.append(f"## Score buckets\n\n")
bs = df["score_bucket"].value_counts().sort_index()
md.append("| bucket | count |\n|---|---:|\n")
for b, n in bs.items():
    md.append(f"| {b} | {n} |\n")
md.append("\n")
md.append(f"## Top features by elite-vs-mid delta\n\n")
if len(elite) > 0 and len(mid) > 0:
    md.append("Comparison: kernels at 0.948+ vs kernels at 0.92–0.94.\n\n")
    md.append("| feature | elite share | mid share | delta |\n|---|---:|---:|---:|\n")
    for f, d in delta.head(20).items():
        md.append(f"| {f} | {elite[f].mean():.2f} | {mid[f].mean():.2f} | {d:+.3f} |\n")
md.append("\n## Numeric param sensitivity (correlation with score)\n\n")
md.append("| param | n | min | max | corr with score |\n|---|---:|---:|---:|---:|\n")
for _, r in sens.iterrows():
    md.append(f"| {r['param']} | {int(r['n'])} | {r['v_min']:.3g} | {r['v_max']:.3g} | {r['score_corr']:+.3f} |\n")
md.append("\n## Top 30 most-forked baselines\n\n")
md.append("| referenced | n_kernels |\n|---|---:|\n")
for r, n in top_forks:
    md.append(f"| {r} | {n} |\n")
with open(OUT / "meta_analysis.md", "w") as f:
    f.write("".join(md))
print(f"\nWrote {OUT / 'meta_analysis.md'}")
