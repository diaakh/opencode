"""Experiment 26: Find the BEST LB-correlation metric via LOO CV.

Use 6 OUR-pipeline anchors (consistent inference):
- exp019, V73, Bruce, slot6_recon, slot11_recon, sub1_v3

Test all 1-feature, 2-feature, 3-feature linear combinations.
Evaluate by LOO RMSE + LOO Spearman rank correlation.

The BEST metric maximizes rank correlation while minimizing LOO RMSE.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut
from scipy.stats import spearmanr, pearsonr

ETT = "/home/user/opencode/birdclef-2026/analysis/entropy_tta"
feat_df = pd.read_csv(f"{ETT}/anchor_features.csv", index_col=0)

# Filter to OUR-pipeline anchors only
OUR_ANCHORS = ["exp019", "V73", "Bruce", "slot6_recon", "slot11_recon", "sub1_v3"]
df_our = feat_df.loc[OUR_ANCHORS].copy()
print(f"OUR-pipeline anchors ({len(df_our)}):")
print(df_our[["overall_auc", "site_mean", "site_std", "gap", "lb"]].to_string())

ALL_FEATURES = [c for c in df_our.columns if c != "lb"]
print(f"\nAll features ({len(ALL_FEATURES)}): {ALL_FEATURES}")

y = df_our["lb"].values
X_all = df_our[ALL_FEATURES].values

# Strategy 1: per-feature univariate correlation
print("\n" + "="*80)
print("UNIVARIATE FEATURE CORRELATIONS with LB")
print("="*80)
print(f"{'Feature':<25s} | Pearson r | Spearman ρ")
print("-"*50)
uni_results = []
for i, feat in enumerate(ALL_FEATURES):
    x = X_all[:, i]
    if np.std(x) < 1e-6: continue
    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    uni_results.append((feat, r, rho))
    print(f"{feat:<25s} | {r:+.3f}    | {rho:+.3f}")

# Sort by Spearman magnitude
uni_results.sort(key=lambda x: -abs(x[2]))
print("\nTop 5 by Spearman |ρ|:")
for f, r, rho in uni_results[:5]:
    print(f"  {f}: ρ={rho:+.3f}, r={r:+.3f}")

# Strategy 2: LOO CV for 1, 2, 3 feature combinations
def loo_evaluate(feature_idx, X, y):
    """Return LOO RMSE and Spearman ρ of predictions vs actual."""
    X_sub = X[:, feature_idx]
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for tr, te in loo.split(X_sub):
        reg = LinearRegression()
        try:
            reg.fit(X_sub[tr], y[tr])
            preds[te] = reg.predict(X_sub[te])
        except:
            preds[te] = y[tr].mean()
    rmse = np.sqrt(((preds - y)**2).mean())
    try:
        rho, _ = spearmanr(preds, y)
    except: rho = 0
    return rmse, rho, preds

print("\n" + "="*80)
print("LOO-CV: 1-FEATURE MODELS")
print("="*80)
print(f"{'Feature':<25s} | LOO RMSE | LOO Spearman ρ")
results_1f = []
for i, feat in enumerate(ALL_FEATURES):
    if np.std(X_all[:, i]) < 1e-6: continue
    rmse, rho, _ = loo_evaluate([i], X_all, y)
    results_1f.append((feat, rmse, rho))
results_1f.sort(key=lambda x: x[1])
for feat, rmse, rho in results_1f[:5]:
    print(f"{feat:<25s} | {rmse:.4f}   | {rho:+.3f}")

print("\n" + "="*80)
print("LOO-CV: 2-FEATURE MODELS (top 10)")
print("="*80)
results_2f = []
for i, j in combinations(range(len(ALL_FEATURES)), 2):
    if np.std(X_all[:, i]) < 1e-6 or np.std(X_all[:, j]) < 1e-6: continue
    rmse, rho, preds = loo_evaluate([i, j], X_all, y)
    results_2f.append((ALL_FEATURES[i], ALL_FEATURES[j], rmse, rho, preds))
results_2f.sort(key=lambda x: x[2])  # by RMSE
print(f"{'Feature 1':<22s} {'Feature 2':<22s} | LOO RMSE | LOO ρ")
for f1, f2, rmse, rho, _ in results_2f[:10]:
    print(f"{f1:<22s} {f2:<22s} | {rmse:.4f}   | {rho:+.3f}")

# Best 2-feature model
print("\nBest 2-feature LOO predictions:")
best_2 = results_2f[0]
f1, f2, rmse, rho, preds = best_2
print(f"Features: {f1}, {f2}, RMSE={rmse:.4f}, ρ={rho:+.3f}")
for n, p, a in zip(OUR_ANCHORS, preds, y):
    print(f"  {n}: predicted {p:.4f}, actual {a:.4f}, error {p-a:+.4f}")

# Strategy 3: Ridge regression with all features
print("\n" + "="*80)
print("RIDGE REGRESSION (all features, LOO CV)")
print("="*80)
from sklearn.preprocessing import StandardScaler
for alpha in [0.01, 0.1, 1.0, 10.0]:
    preds = np.zeros_like(y)
    loo = LeaveOneOut()
    for tr, te in loo.split(X_all):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[tr])
        X_te = scaler.transform(X_all[te])
        reg = Ridge(alpha=alpha)
        reg.fit(X_tr, y[tr])
        preds[te] = reg.predict(X_te)
    rmse = np.sqrt(((preds - y)**2).mean())
    rho, _ = spearmanr(preds, y)
    print(f"  alpha={alpha}: LOO RMSE={rmse:.4f}, ρ={rho:+.3f}")

# Strategy 4: Random Forest with feature importance
print("\n" + "="*80)
print("RANDOM FOREST (LOO, feature importance)")
print("="*80)
from sklearn.ensemble import RandomForestRegressor
preds = np.zeros_like(y)
for tr, te in LeaveOneOut().split(X_all):
    rf = RandomForestRegressor(n_estimators=20, max_depth=2, random_state=42)
    rf.fit(X_all[tr], y[tr])
    preds[te] = rf.predict(X_all[te])
rmse = np.sqrt(((preds - y)**2).mean())
rho, _ = spearmanr(preds, y)
print(f"RF: LOO RMSE={rmse:.4f}, ρ={rho:+.3f}")
# train on all to get importance
rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_all, y)
importances = sorted(zip(ALL_FEATURES, rf.feature_importances_), key=lambda x: -x[1])
print(f"\nTop 5 features by RF importance:")
for f, imp in importances[:5]:
    print(f"  {f}: {imp:.3f}")
