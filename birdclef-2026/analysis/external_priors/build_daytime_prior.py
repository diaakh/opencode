"""Build a daytime hour prior usable as a fallback for hours 11-16
where train_soundscapes is empty.

Strategy:
  1. Load iNat per-species hour counts (global + Pantanal-bbox).
  2. For each species, smooth the per-species hour distribution toward
     a class-level prior (Aves / Amphibia / Insecta / Mammalia).
  3. Combine global + Pantanal: Pantanal weighted higher (local signal)
     when n>=10 obs, else fall back to global.
  4. Renormalize per species → P(hour | species).
  5. Marginalize the other direction: P(species | hour) ∝ P(hour|species)
     × P(species) where P(species) comes from labeled-set positive rate
     (or uniform if unknown).
  6. Save as a CSV indexed by hour×species, matching the format of
     pseudo_hour_priors.csv so it drops in.

Output: external_hour_prior.csv (24 rows × 234 class columns)
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/user/opencode/birdclef-2026/analysis/external_priors"

# Load taxonomy
tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
label_to_class = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))

# Load iNat counts (after refetch)
g = pd.read_csv(f"{OUT}/inat_hour_global_counts.csv", index_col="hour")
p = pd.read_csv(f"{OUT}/inat_hour_pantanal_counts.csv", index_col="hour")
print(f"Global counts: {g.shape}, total={g.values.sum():.0f}")
print(f"Pantanal counts: {p.shape}, total={p.values.sum():.0f}")


# ---------- Build class-level priors (smoothing target) ----------
# For each (class_name, hour), aggregate counts across species
class_g = {}
class_p = {}
for cls in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
    cls_labels = [l for l in g.columns if label_to_class.get(l) == cls]
    if cls_labels:
        class_g[cls] = g[cls_labels].sum(axis=1)
        class_p[cls] = p[cls_labels].sum(axis=1)
class_g_df = pd.DataFrame(class_g)
class_p_df = pd.DataFrame(class_p)

# Normalize class priors
class_g_norm = class_g_df.div(class_g_df.sum(axis=0).replace(0, 1), axis=1)
class_p_norm = class_p_df.div(class_p_df.sum(axis=0).replace(0, 1), axis=1)
print(f"\nClass-level priors (global, sum-to-1 per class):")
print(class_g_norm.round(3))


# ---------- Per-species: smoothed hour prior ----------
def smooth_species(label, n_min=10):
    """Return (24,) array: P(hour | species), smoothed toward class prior."""
    cls = label_to_class.get(label)
    if cls is None:
        return np.full(24, 1.0 / 24)
    g_counts = g[label].values if label in g.columns else np.zeros(24)
    p_counts = p[label].values if label in p.columns else np.zeros(24)
    n_g = g_counts.sum()
    n_p = p_counts.sum()
    cls_g = class_g_norm[cls].values if cls in class_g_norm.columns else np.full(24, 1.0 / 24)
    cls_p = class_p_norm[cls].values if cls in class_p_norm.columns else cls_g

    # Combine global + Pantanal with adaptive weighting
    # Pantanal weight = n_p / (n_p + 5), global weight = (1-pantanal_weight)
    if n_p + n_g == 0:
        # No data — use class prior (preference for Pantanal)
        return cls_p if cls_p.sum() > 0 else cls_g
    if n_p >= 10:
        # Enough Pantanal data
        sp_hour = (p_counts / max(n_p, 1)) * 0.7 + (g_counts / max(n_g, 1)) * 0.3
    elif n_g >= 10:
        sp_hour = g_counts / n_g
    else:
        sp_hour = cls_p if cls_p.sum() > 0 else cls_g

    # Smooth toward class prior (Bayesian shrinkage, alpha=5)
    alpha = 5.0
    n_eff = n_p + n_g
    shrink_target = cls_p if n_p > 0 else cls_g
    smoothed = (n_eff * sp_hour + alpha * shrink_target) / (n_eff + alpha)
    # Renormalize
    s = smoothed.sum()
    if s == 0:
        return cls_g if cls_g.sum() > 0 else np.full(24, 1.0 / 24)
    return smoothed / s


# Build per-species per-hour
species_hour = pd.DataFrame(index=range(24), columns=classes, dtype=float)
species_hour.index.name = "hour"
for label in classes:
    species_hour[label] = smooth_species(label)

species_hour.to_csv(f"{OUT}/external_hour_per_species.csv")
print(f"\nSaved {OUT}/external_hour_per_species.csv")
print(f"Per-species hour priors (5 sample classes, 5 sample hours):")
sample_cls = ["chacha1", "whtdov", "517063", "65380", "47158son25"]
sample_cls = [c for c in sample_cls if c in species_hour.columns]
print(species_hour.loc[[0, 6, 12, 18, 22], sample_cls].round(3))


# ---------- Marginalize: P(species | hour) ∝ P(hour | species) × P(species) ----------
# P(species) prior: use uniform (1/234) for now; we don't have a marginal estimate per species
# that's leakage-safe. Later could use labeled-set positive rates.
N_CLS = len(classes)
hour_to_species_marg = species_hour.copy()
# For each row (hour), normalize across species
hour_to_species_marg = hour_to_species_marg.div(hour_to_species_marg.sum(axis=1).replace(0, 1), axis=0)
hour_to_species_marg.to_csv(f"{OUT}/external_hour_prior.csv")
print(f"Saved {OUT}/external_hour_prior.csv (drop-in replacement for pseudo_hour_priors)")


# ---------- Visualize ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: class-level hour distribution (global)
ax = axes[0, 0]
for cls, col in class_g_norm.items():
    if col.sum() > 0:
        ax.plot(col.index, col.values, marker="o", label=f"{cls} (n_species={sum(1 for l in g.columns if label_to_class.get(l)==cls)})")
ax.set_title("Class-level iNat hour-of-day (global obs)")
ax.set_xlabel("Hour of day")
ax.set_ylabel("P(hour | class)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 2))

# Top-right: class-level (Pantanal)
ax = axes[0, 1]
for cls, col in class_p_norm.items():
    if col.sum() > 0:
        ax.plot(col.index, col.values, marker="o", label=cls)
ax.set_title("Class-level iNat hour-of-day (Pantanal bbox)")
ax.set_xlabel("Hour of day")
ax.set_ylabel("P(hour | class)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 2))

# Bottom-left: comparing pseudo prior vs external prior at hour 12 (daytime gap)
ax = axes[1, 0]
try:
    pseudo = pd.read_csv("/home/user/opencode/birdclef-2026/meta_analysis/pseudo_hour_priors.csv").set_index("hour")
    common_cls = [c for c in pseudo.columns if c in hour_to_species_marg.columns]
    pseudo_h12 = pseudo.loc[12, common_cls].values if 12 in pseudo.index else np.zeros(len(common_cls))
    extern_h12 = hour_to_species_marg.loc[12, common_cls].values
    pseudo_h22 = pseudo.loc[22, common_cls].values
    extern_h22 = hour_to_species_marg.loc[22, common_cls].values
    n = min(50, len(common_cls))
    idx = np.argsort(extern_h12 + extern_h22)[::-1][:n]
    ax.barh(range(n), pseudo_h12[idx], label="pseudo h=12 (=0, dead)", color="grey")
    ax.barh(range(n), extern_h12[idx], left=0, alpha=0.6, label="external h=12", color="#3a7")
    ax.set_yticks(range(n))
    ax.set_yticklabels([common_cls[i] for i in idx], fontsize=6)
    ax.set_xlabel("P(species | hour=12)")
    ax.set_title("Daytime (h=12) prior: pseudo (zero, dead) vs external (iNat)")
    ax.legend()
except Exception as e:
    ax.text(0.5, 0.5, f"comparison plot failed: {e}", ha="center")

# Bottom-right: per-species coverage
ax = axes[1, 1]
n_obs = g.sum(axis=0).values
covered = (n_obs >= 10).sum()
ax.hist(np.log10(np.maximum(n_obs, 1)), bins=30)
ax.axvline(1.0, ls=":", color="r", label=f">=10 obs ({covered} species)")
ax.set_xlabel("log10(global obs count per species)")
ax.set_ylabel("# species")
ax.set_title("iNat coverage per species")
ax.legend()

plt.suptitle("External hour-of-day prior from iNat — for filling the hours-11-16 gap")
plt.tight_layout()
plt.savefig(f"{OUT}/external_prior_summary.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}/external_prior_summary.png")


# ---------- Summary stats ----------
print(f"\n=== Summary ===")
print(f"Species with iNat hour data (global ≥10 obs): {(g.sum(axis=0) >= 10).sum()}/234")
print(f"Species with iNat hour data (Pantanal ≥10 obs): {(p.sum(axis=0) >= 10).sum()}/234")
print(f"Class-level priors built for: {[c for c in class_g_norm.columns if class_g_norm[c].sum() > 0]}")
print(f"\nDaytime hours (11-16) species distribution sums per hour from external prior:")
for h in [11, 12, 13, 14, 15, 16]:
    row_sum = hour_to_species_marg.loc[h].sum() if h in hour_to_species_marg.index else 0
    print(f"  h={h:2d}: sum={row_sum:.3f}, top-5 species:")
    if h in hour_to_species_marg.index:
        top5 = hour_to_species_marg.loc[h].nlargest(5)
        for c, v in top5.items():
            cn = tax[tax["primary_label"] == c]["common_name"].iloc[0] if (tax["primary_label"] == c).any() else "?"
            print(f"    {c:>12s} ({cn[:30]:>30s}): {v:.4f}")
