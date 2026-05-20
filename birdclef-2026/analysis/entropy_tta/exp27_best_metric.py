"""Experiment 27: Solidify the best metric formula.

LOO winner: overall_auc + site_mean
LOO RMSE: 0.0057
LOO ρ: 0.986

Also try 3-feature models.
Also test on public anchors (expected to fail).
"""
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from scipy.stats import spearmanr, pearsonr

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
feat_df = pd.read_csv(f"{ETT}/anchor_features.csv", index_col=0)
OUR = ["exp019", "V73", "Bruce", "slot6_recon", "slot11_recon", "sub1_v3"]
df_our = feat_df.loc[OUR].copy()
ALL_FEATURES = [c for c in df_our.columns if c != "lb"]

y = df_our["lb"].values
X = df_our[ALL_FEATURES].values

# Fit the BEST 2-feature model on ALL 6 anchors (final coefficients)
i_oa = ALL_FEATURES.index("overall_auc")
i_sm = ALL_FEATURES.index("site_mean")

X_best = X[:, [i_oa, i_sm]]
reg = LinearRegression()
reg.fit(X_best, y)
pred_in = reg.predict(X_best)

print("="*80)
print("BEST METRIC (fit on all 6 anchors):")
print("="*80)
print(f"LB ≈ {reg.intercept_:.4f} + {reg.coef_[0]:.4f}*overall_auc + {reg.coef_[1]:.4f}*site_mean")
print(f"\nIn-sample predictions (training fit):")
for n, p, a in zip(OUR, pred_in, y):
    print(f"  {n}: predicted {p:.4f}, actual {a:.4f}, error {p-a:+.4f}")

# LOO breakdown
print(f"\nLOO predictions (more honest):")
preds_loo = []
loo = LeaveOneOut()
for tr, te in loo.split(X_best):
    r = LinearRegression()
    r.fit(X_best[tr], y[tr])
    p = r.predict(X_best[te])[0]
    preds_loo.append(p)
preds_loo = np.array(preds_loo)
for n, p, a in zip(OUR, preds_loo, y):
    print(f"  {n}: LOO predicted {p:.4f}, actual {a:.4f}, error {p-a:+.4f}")

# Now: try 3-feature models for further improvement
print("\n" + "="*80)
print("3-FEATURE MODELS (top 10 by LOO RMSE)")
print("="*80)
results_3f = []
for triple in combinations(range(len(ALL_FEATURES)), 3):
    if any(np.std(X[:, t]) < 1e-6 for t in triple): continue
    X_sub = X[:, list(triple)]
    preds = np.zeros_like(y)
    for tr, te in LeaveOneOut().split(X_sub):
        r = LinearRegression()
        try:
            r.fit(X_sub[tr], y[tr])
            preds[te] = r.predict(X_sub[te])
        except:
            preds[te] = y[tr].mean()
    rmse = np.sqrt(((preds - y)**2).mean())
    rho, _ = spearmanr(preds, y)
    results_3f.append((tuple(ALL_FEATURES[t] for t in triple), rmse, rho))
results_3f.sort(key=lambda x: x[1])
print(f"{'Features':<60s} | LOO RMSE | LOO ρ")
for feats, rmse, rho in results_3f[:10]:
    print(f"{', '.join(feats):<60s} | {rmse:.4f}   | {rho:+.3f}")

# Validate on public anchors (expected: poor performance)
print("\n" + "="*80)
print("PUBLIC ANCHOR PREDICTIONS (using best 2-feature metric)")
print("="*80)
PUBLIC_ANCHORS = [n for n in feat_df.index if n not in OUR]
df_pub = feat_df.loc[PUBLIC_ANCHORS]
X_pub = df_pub[ALL_FEATURES].values
X_pub_best = X_pub[:, [i_oa, i_sm]]
pred_pub = reg.predict(X_pub_best)
y_pub = df_pub["lb"].values
print(f"{'Anchor':<25s} | predicted | actual | error")
for n, p, a in zip(PUBLIC_ANCHORS, pred_pub, y_pub):
    print(f"{n:<25s} | {p:.4f}    | {a:.4f}  | {p-a:+.4f}")

# But — re-attribute public anchors:
# - safar1_perch_logits and mtoshidesu_perch are Perch v2 raw → public LB ~0.91
# - needless090_oof_base / oof_prior are PART of pipeline, not the final LB
# - These don't have a clean LB attribution

print(f"\nRe-attributed public anchors:")
print(f"{'Anchor':<25s} | feature LB tier | predicted | reasonable?")
re_attr = {
    "safar1_perch_logits": 0.910,  # Perch v2 starter tier
    "mtoshidesu_perch": 0.910,
    "needless090_oof_base": 0.92,  # could be base before stacker
    "needless090_oof_prior": 0.92,
    "koushikrudra_oof_base": 0.91,
}
for n, p in zip(PUBLIC_ANCHORS, pred_pub):
    re_lb = re_attr.get(n, "?")
    print(f"{n:<25s} | {re_lb} (re-attr)| {p:.4f}    | err={p-re_lb if re_lb != '?' else '?':+.4f}" if re_lb != "?" else f"{n}: {p:.4f}, no LB")

# Save best metric
with open(f"{ETT}/best_metric.txt", "w") as f:
    f.write(f"BEST METRIC (LOO RMSE 0.0057, ρ +0.986)\n")
    f.write(f"=" * 60 + "\n\n")
    f.write(f"LB ≈ {reg.intercept_:.6f}\n")
    f.write(f"     + {reg.coef_[0]:.6f} * overall_auc\n")
    f.write(f"     + {reg.coef_[1]:.6f} * site_mean\n")
    f.write(f"\nWhere:\n")
    f.write(f"  overall_auc = macro-AUC on labeled OOF (sklearn roc_auc_score, per-class avg)\n")
    f.write(f"  site_mean = per-site macro-AUC weighted by row count\n")
    f.write(f"\nFit on 6 anchors:\n")
    for n, p, a in zip(OUR, pred_in, y):
        f.write(f"  {n}: in-sample {p:.4f}, actual {a:.4f}, err {p-a:+.4f}\n")
print(f"\nSaved: {ETT}/best_metric.txt")
