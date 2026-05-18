"""Build a per-species hour-of-day prior from OUR predictions on the
127k unlabeled windows. Same shape as the iNat external prior, but
sourced from our own data + models.

For each species:
  - Find unlabeled windows where Bruce/Perch predict above a threshold
    (we trust them as positive observations)
  - Count hour-of-day occurrences
  - Smooth toward class-level baseline
  - Normalize → P(hour | species)

Output:
  internal_hour_per_species_from_unlabeled.csv  (24 × 234)
  internal_hour_prior.csv                       (drop-in for pseudo_hour_priors)

Notes:
  - Only covers hours 0-10 + 17-23 (where unlabeled has audio).
    Hours 11-16 remain 0 — combine with external iNat prior for those.
  - The "observations" are our own predictions, so there's a feedback
    risk if used for self-training. For inference-time priors it's
    fine — the prior just tells us "when this species calls in OUR
    deployment context".
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/user/opencode/birdclef-2026/analysis/external_priors"

# Load Bruce + Perch predictions on the 127k unlabeled windows
data = np.load(
    "/home/user/opencode/birdclef-2026/analysis/context_label/bruce_on_unlabeled.npz",
    allow_pickle=True,
)
P_bruce = data["P_bruce"]  # (127k, 234), 0-1
P_perch = data["perch_soft"]  # (127k, 234), 0-1
hours = data["hour_utc"].astype(int)
sites = data["site"].astype(str)
classes = list(data["classes"])
N_PSEUDO, N_CLS = P_bruce.shape

# Load taxonomy for class membership
tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
label_to_class = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))

# ---------- For each species, find high-confidence positive windows ----------
# Threshold: we want windows where Bruce strongly thinks this species is present.
# Bruce was trained on labels; threshold 0.85 mirrors the v2 pseudo-label tier.
THRESH = 0.85
NEAR_ZERO_THRESH = 0.05  # exclude windows where Bruce is near-zero for ALL species (silent)

# Count (species, hour) co-occurrences from high-conf predictions
hour_counts = np.zeros((24, N_CLS), dtype=np.int64)
total_per_class = np.zeros(N_CLS, dtype=np.int64)

for c_idx in range(N_CLS):
    col = P_bruce[:, c_idx]
    mask = (~np.isnan(col)) & (col >= THRESH)
    if mask.sum() == 0:
        continue
    h_arr = hours[mask]
    # Count per hour
    for h in range(24):
        hour_counts[h, c_idx] = (h_arr == h).sum()
    total_per_class[c_idx] = mask.sum()

print(f"Total high-confidence (Bruce>={THRESH}) windows: {total_per_class.sum()}")
print(f"Species with ≥10 high-conf windows: {(total_per_class >= 10).sum()}/{N_CLS}")
print(f"Species with 0 high-conf windows: {(total_per_class == 0).sum()}/{N_CLS}")
print(f"Top-10 species by # high-conf windows:")
top_idx = np.argsort(total_per_class)[::-1][:10]
for i in top_idx:
    cname = tax[tax["primary_label"] == classes[i]]["common_name"].iloc[0] if (tax["primary_label"] == classes[i]).any() else "?"
    print(f"  {classes[i]:>12s} ({cname[:30]:>30s}): {total_per_class[i]} windows")

# ---------- Class-level priors (smoothing target) ----------
class_h = {}
for cls in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
    cls_idxs = [i for i, l in enumerate(classes) if label_to_class.get(l) == cls]
    if cls_idxs:
        class_h[cls] = hour_counts[:, cls_idxs].sum(axis=1)
class_h_df = pd.DataFrame(class_h)
class_h_norm = class_h_df.div(class_h_df.sum(axis=0).replace(0, 1), axis=1)
print(f"\nClass-level hour distribution (from our high-conf predictions):")
print(class_h_norm.round(3))


# ---------- Per-species smoothed prior ----------
def smooth_species(c_idx, n_min=10, alpha=8):
    """Return (24,) P(hour | species), smoothed toward class baseline."""
    cls = label_to_class.get(classes[c_idx])
    counts = hour_counts[:, c_idx].astype(float)
    n = total_per_class[c_idx]
    cls_prior = class_h_norm[cls].values if cls in class_h_norm.columns else np.full(24, 1.0 / 24)
    if n == 0:
        return cls_prior
    sp_prior = counts / max(n, 1)
    # Bayesian shrinkage toward class prior
    smoothed = (n * sp_prior + alpha * cls_prior) / (n + alpha)
    s = smoothed.sum()
    return smoothed / (s if s > 0 else 1)


per_species_hour = pd.DataFrame(index=range(24), columns=classes, dtype=float)
per_species_hour.index.name = "hour"
for c_idx in range(N_CLS):
    per_species_hour[classes[c_idx]] = smooth_species(c_idx)
per_species_hour.to_csv(f"{OUT}/internal_hour_per_species_from_unlabeled.csv")
print(f"\nSaved {OUT}/internal_hour_per_species_from_unlabeled.csv")


# ---------- Marginal P(species | hour) — drop-in for pseudo_hour_priors ----------
hour_to_species = per_species_hour.copy()
hour_to_species = hour_to_species.div(hour_to_species.sum(axis=1).replace(0, 1), axis=0)
hour_to_species.to_csv(f"{OUT}/internal_hour_prior.csv")
print(f"Saved {OUT}/internal_hour_prior.csv (drop-in replacement for pseudo_hour_priors.csv)")


# ---------- Compare to pseudo + iNat priors ----------
print("\n=== Comparing the three hour priors ===")
pseudo = pd.read_csv(
    "/home/user/opencode/birdclef-2026/meta_analysis/pseudo_hour_priors.csv"
).set_index("hour")
external = pd.read_csv(
    f"{OUT}/external_hour_prior.csv"
).set_index("hour")

common = sorted(set(pseudo.columns) & set(external.columns) & set(hour_to_species.columns))
print(f"  shared species across all 3 priors: {len(common)}")

print("\n  Per-hour total prior mass (sums across all 234 classes):")
print(f"  {'hour':>4s} {'pseudo':>10s} {'iNat_ext':>10s} {'unlab_int':>10s}")
for h in range(24):
    p = pseudo.loc[h, common].sum() if h in pseudo.index else 0
    e = external.loc[h, common].sum() if h in external.index else 0
    u = hour_to_species.loc[h, common].sum() if h in hour_to_species.index else 0
    print(f"  {h:>4d} {p:>10.4f} {e:>10.4f} {u:>10.4f}")


# ---------- Plot comparison: 3 priors stacked ----------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Top-left: class-level patterns from our unlabeled
ax = axes[0, 0]
for cls, col in class_h_norm.items():
    if col.sum() > 0:
        ax.plot(col.index, col.values, marker="o", label=cls)
ax.set_title("Hour-of-day (P|class) from OUR high-conf predictions on 127k unlabeled")
ax.set_xlabel("Hour of day")
ax.set_ylabel("P(hour | class)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 2))

# Top-right: 3-way prior comparison at hour=2 (night-heavy) vs hour=12 (daytime)
ax = axes[0, 1]
sample = ["517063", "65380", "47158son25", "chacha1", "whtdov", "47158son07"]
sample = [s for s in sample if s in pseudo.columns and s in external.columns and s in hour_to_species.columns]
n = len(sample)
x = np.arange(n)
w = 0.25
for i, hr in enumerate([2, 12]):
    p = pseudo.loc[hr, sample].values if hr in pseudo.index else np.zeros(n)
    e = external.loc[hr, sample].values if hr in external.index else np.zeros(n)
    u = hour_to_species.loc[hr, sample].values if hr in hour_to_species.index else np.zeros(n)
    color_base = "tab:blue" if hr == 2 else "tab:red"
    offset = -1.2 if hr == 2 else 0
    ax.bar(x + offset * w, p, w, label=f"pseudo h={hr}", color=color_base, alpha=0.4)
    ax.bar(x + (offset + 1) * w, e, w, label=f"iNat h={hr}", color=color_base, alpha=0.7)
    ax.bar(x + (offset + 2) * w, u, w, label=f"unlab_int h={hr}", color=color_base, alpha=1.0)
ax.set_xticks(x)
ax.set_xticklabels(sample, rotation=30, ha="right")
ax.set_ylabel("P(species | hour)")
ax.set_title("Sample species: 3 priors at h=2 (night, blue) vs h=12 (daytime, red)")
ax.legend(fontsize=7)

# Bottom-left: how many high-conf windows per class (log scale)
ax = axes[1, 0]
ax.hist(np.log10(np.maximum(total_per_class, 1)), bins=40)
ax.set_xlabel("log10(# high-conf Bruce predictions on unlabeled)")
ax.set_ylabel("# species")
ax.set_title(f"Self-prior coverage: {(total_per_class>=10).sum()}/{N_CLS} species with ≥10 high-conf preds")
ax.axvline(1.0, ls=":", color="r", alpha=0.5)

# Bottom-right: per-hour mass (3-way)
ax = axes[1, 1]
hrs = list(range(24))
p_sum = [pseudo.loc[h, common].sum() if h in pseudo.index else 0 for h in hrs]
e_sum = [external.loc[h, common].sum() if h in external.index else 0 for h in hrs]
u_sum = [hour_to_species.loc[h, common].sum() if h in hour_to_species.index else 0 for h in hrs]
ax.plot(hrs, p_sum, marker="o", label="pseudo (from train_soundscapes)")
ax.plot(hrs, e_sum, marker="s", label="iNat external")
ax.plot(hrs, u_sum, marker="^", label="our unlabeled predictions")
ax.set_xlabel("Hour")
ax.set_ylabel("Σ P(species | hour) over 234 classes")
ax.set_title("Per-hour total prior mass (should sum to 1 per row in a well-normalized prior)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 2))

plt.tight_layout()
plt.savefig(f"{OUT}/internal_vs_external.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"\nSaved {OUT}/internal_vs_external.png")


# ---------- Build combined recipe: pick best prior per (hour, species) ----------
# Strategy:
#   For hours where pseudo has data: use pseudo (highest density)
#   For hours 11-16: fall back to external (iNat)
#   The internal prior from unlabeled is interesting but mostly REDUNDANT with
#   pseudo (since pseudo cache is also computed from unlabeled audio via Perch).
HOURS_WITH_PSEUDO = [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22,23]

combined = pd.DataFrame(index=range(24), columns=classes, dtype=float)
combined.index.name = "hour"
for h in range(24):
    if h in HOURS_WITH_PSEUDO and h in pseudo.index:
        # Use pseudo where it has data; for classes pseudo doesn't cover, use external
        for cls in classes:
            v_pseudo = pseudo.loc[h, cls] if cls in pseudo.columns else 0
            v_extern = external.loc[h, cls] if cls in external.columns else 0
            combined.loc[h, cls] = v_pseudo if v_pseudo > 0 else v_extern
    elif h in external.index:
        combined.loc[h] = external.loc[h]
    else:
        combined.loc[h] = 1.0 / N_CLS

# Re-normalize per-row
combined = combined.div(combined.sum(axis=1).replace(0, 1), axis=0)
combined.to_csv(f"{OUT}/combined_hour_prior.csv")
print(f"Saved {OUT}/combined_hour_prior.csv (use this in submission)")

# Summary of combined per-hour total mass
print("\nCombined per-hour mass (should be 1.0 per row):")
print(combined.sum(axis=1).round(4).to_string())
