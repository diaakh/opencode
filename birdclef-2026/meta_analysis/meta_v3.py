"""Final meta-analysis combining:
  - Kernel features (from extract_v2)
  - Author's actual public-LB best (from leaderboard join)
  - Kernel's own claimed score (from notebook content)

Outputs:
  meta_v3_summary.md
  per_bucket_features.csv (cleaner)
  hi_lo_diff.csv  (what differentiates 0.95+ kernels from < 0.93)
  fork_lineage.csv (most-referenced baseline kernels)
"""
import pandas as pd, numpy as np, re, json
from pathlib import Path
from collections import Counter

CORPUS = Path(__file__).parent
OUT = CORPUS.parent / "meta_analysis"
OUT.mkdir(exist_ok=True)

# Load extracted features (run extract_v2 first; we just re-use its output)
feat = pd.read_csv(CORPUS / "features_v2.csv")
# Load author LB
inv = pd.read_csv(CORPUS / "kernel_with_lb.csv")
inv = inv[["ref", "best_score", "best_rank", "totalVotes", "lastRunTime"]].rename(
    columns={"best_score": "author_lb_best", "best_rank": "author_lb_rank"})
df = feat.merge(inv, on="ref", how="left")
print(f"Joined: {len(df)} kernels  ({df['author_lb_best'].notna().sum()} with author LB)")

# Use BEST AVAILABLE score: prefer config_lb_max > title_score > lb_prefix_max > author_lb_best > bare
def best_kernel_score(row):
    # The author's LB is the UPPER bound on what any of their kernels can score
    # Kernel-specific claimed score is sharper signal
    for k in ["config_lb_max", "title_score", "lb_prefix_max"]:
        v = row.get(k)
        if pd.notna(v) and 0.5 <= float(v) <= 0.999:
            return float(v)
    # Fall back to author LB best — but this is upper bound, not kernel-specific
    if pd.notna(row.get("author_lb_best")):
        return float(row["author_lb_best"])
    return None
df["score"] = df.apply(best_kernel_score, axis=1)

# Filter out spurious bare-score-only kernels with no other signal but value too high
has_strong_signal = (df["config_lb_max"].notna() | df["title_score"].notna() |
                     df["lb_prefix_max"].notna() | df["author_lb_best"].notna())
df = df[has_strong_signal].reset_index(drop=True)
print(f"After filtering to strong-signal kernels: {len(df)}")

# Score buckets
def bucket(s):
    if pd.isna(s): return None
    if s >= 0.955: return "5.top_0.955+"
    if s >= 0.948: return "4.elite_0.948-0.955"
    if s >= 0.940: return "3.high_0.940-0.948"
    if s >= 0.925: return "2.mid_0.925-0.940"
    if s >= 0.900: return "1.low_0.900-0.925"
    return "0.weak_<0.900"

df["bucket"] = df["score"].apply(bucket)
print(f"\nBucket counts:")
print(df["bucket"].value_counts().sort_index())

# Features
feat_cols = [c for c in df.columns if c.startswith("feat__")]
num_cols = [c for c in df.columns if c.startswith("num__")]
print(f"\nFeatures: {len(feat_cols)}, numeric params: {len(num_cols)}")

# Per-bucket feature share
agg = df.groupby("bucket")[feat_cols].mean().T
agg.columns.name = None
agg.to_csv(OUT / "per_bucket_features.csv")
print(f"\nFeature share by bucket (top 20 most variable across buckets):")
agg["std_across"] = agg.std(axis=1)
top_variant = agg.sort_values("std_across", ascending=False).head(25)
print(top_variant.round(2).to_string())

# Hi-vs-lo diff
hi = df[df["bucket"].isin(["5.top_0.955+", "4.elite_0.948-0.955", "3.high_0.940-0.948"])]
lo = df[df["bucket"].isin(["1.low_0.900-0.925", "0.weak_<0.900"])]
print(f"\nHigh-tier (0.94+) n={len(hi)}, Low-tier (<0.925) n={len(lo)}")
diff_rows = []
for f in feat_cols:
    h = hi[f].mean(); l = lo[f].mean()
    diff_rows.append({"feature": f, "hi_share": h, "lo_share": l, "delta": h - l})
diff_df = pd.DataFrame(diff_rows).sort_values("delta", key=abs, ascending=False)
diff_df.to_csv(OUT / "hi_lo_diff.csv", index=False)
print(f"\nTop 25 features by HI vs LO delta:")
print(diff_df.head(25).to_string(index=False))

# Numerical param sensitivity
print(f"\nNumeric param sensitivity (correlation with score, only params with ≥10 datapoints):")
sens_rows = []
for c in num_cols:
    vals = pd.to_numeric(df[c], errors="coerce")
    sub = df.assign(v=vals).dropna(subset=["v", "score"])
    if len(sub) < 10: continue
    if sub["v"].std() > 0 and sub["score"].std() > 0:
        corr = sub["v"].corr(sub["score"])
        # Per-bucket mean
        per_bucket = sub.groupby("bucket")["v"].agg(["mean", "count"]).round(4)
        sens_rows.append({"param": c, "n": len(sub),
                          "v_min": sub["v"].min(), "v_max": sub["v"].max(),
                          "v_median": sub["v"].median(),
                          "corr": corr})
sens = pd.DataFrame(sens_rows).sort_values("corr", key=abs, ascending=False)
sens.to_csv(OUT / "param_sensitivity_v3.csv", index=False)
print(sens.to_string(index=False))

# Fork lineage
fork_counter = Counter()
for refs in df["fork_refs"].dropna():
    for r in str(refs).split("|"):
        r = r.strip()
        if "/" in r: fork_counter[r] += 1
top_forks = fork_counter.most_common(40)
pd.DataFrame(top_forks, columns=["baseline_kernel", "referenced_by_n_kernels"]).to_csv(
    OUT / "fork_lineage.csv", index=False)
print(f"\nTop 40 most-referenced baseline kernels:")
for r, n in top_forks:
    print(f"  {n:4d} × {r}")

# Best public kernel per author with author_lb_best
print(f"\n=== Top 30 by author_LB (those with public kernels) ===")
top30 = df.dropna(subset=["author_lb_best"]).sort_values("author_lb_best", ascending=False).drop_duplicates("ref").head(30)
print(top30[["ref", "author_lb_best", "author_lb_rank", "score", "totalVotes"]].to_string(index=False))

# Save full joined table
df.to_csv(OUT / "joined_full.csv", index=False)
print(f"\nWrote joined_full.csv with {len(df)} rows")
