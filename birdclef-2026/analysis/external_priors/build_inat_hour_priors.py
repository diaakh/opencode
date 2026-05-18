"""Build external hour-of-day priors for all 234 BC2026 species from iNat API.

For each species, query iNat's /observations endpoint, extract
`observed_on_details.hour` from each observation, aggregate counts by hour.

Output:
  inat_hour_global.csv     — global obs (more data, less calibrated)
  inat_hour_pantanal.csv   — Pantanal bbox obs (fewer obs, locally calibrated)

These are LEAKAGE-SAFE for our model audit (external to our train data)
and can be used as a fallback prior for hours 11-16 where train_soundscapes
has zero coverage.
"""
import os, time, json
import pandas as pd
import numpy as np
import requests
from collections import Counter
from pathlib import Path

OUT = "/home/user/opencode/birdclef-2026/analysis/external_priors"
os.makedirs(OUT, exist_ok=True)

# Load BC2026 taxonomy
tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
print(f"Species: {len(tax)} (Aves={(tax['class_name']=='Aves').sum()}, "
      f"Amphibia={(tax['class_name']=='Amphibia').sum()}, "
      f"Insecta={(tax['class_name']=='Insecta').sum()}, "
      f"Mammalia={(tax['class_name']=='Mammalia').sum()}, "
      f"Reptilia={(tax['class_name']=='Reptilia').sum()})")

INAT_URL = "https://api.inaturalist.org/v1/observations"
PANTANAL = dict(nelat=-16.5, nelng=-55.9, swlat=-21.6, swlng=-57.6)

SESS = requests.Session()
SESS.headers["User-Agent"] = "BC2026-research (anon-research)"


def fetch_hours(taxon_id, max_pages=5, per_page=200, geo_bbox=None, sleep=0.8):
    """Pull observations and return Counter of hours + total obs found."""
    hours = Counter()
    total_total = None
    n_with_hour = 0
    for page in range(1, max_pages + 1):
        params = {
            "taxon_id": taxon_id,
            "per_page": per_page,
            "page": page,
            "order_by": "observed_on",
            "order": "desc",
        }
        if geo_bbox:
            params.update(geo_bbox)
        try:
            r = SESS.get(INAT_URL, params=params, timeout=30)
            if r.status_code != 200:
                return hours, total_total, n_with_hour, f"http_{r.status_code}"
            data = r.json()
            if total_total is None:
                total_total = data.get("total_results", 0)
            results = data.get("results", [])
            if not results:
                break
            for obs in results:
                details = obs.get("observed_on_details") or {}
                h = details.get("hour")
                if h is not None and 0 <= h <= 23:
                    hours[h] += 1
                    n_with_hour += 1
            if len(results) < per_page:
                break
        except Exception as e:
            return hours, total_total, n_with_hour, f"err:{e!s}"
        time.sleep(sleep)
    return hours, total_total, n_with_hour, "ok"


# Build for ALL species, both global and Pantanal
print("\n=== Fetching iNat hour distributions ===")
results = {"global": {}, "pantanal": {}}
diagnostics = []

for i, row in tax.iterrows():
    taxon_id = int(row["inat_taxon_id"])
    label = row["primary_label"]
    common = row["common_name"]
    cls = row["class_name"]

    t0 = time.time()
    # Global
    hours_g, total_g, n_h_g, status_g = fetch_hours(taxon_id, max_pages=3, sleep=0.5)
    # Pantanal bbox
    hours_p, total_p, n_h_p, status_p = fetch_hours(taxon_id, max_pages=2, geo_bbox=PANTANAL, sleep=0.5)

    results["global"][label] = dict(hours_g)
    results["pantanal"][label] = dict(hours_p)
    diagnostics.append({
        "primary_label": label,
        "inat_taxon_id": taxon_id,
        "class_name": cls,
        "common_name": common,
        "n_global_total": total_g,
        "n_global_pulled": n_h_g,
        "global_status": status_g,
        "n_pantanal_total": total_p,
        "n_pantanal_pulled": n_h_p,
        "pantanal_status": status_p,
        "elapsed_s": round(time.time() - t0, 1),
    })

    if i < 5 or i % 30 == 0 or i == len(tax) - 1:
        print(f"  [{i+1:3d}/{len(tax)}] {label:>10s} ({cls:9s}) iNat={taxon_id:>10d} "
              f"global={n_h_g:5d}/{total_g or 0} "
              f"pantanal={n_h_p:5d}/{total_p or 0} "
              f"dt={time.time()-t0:.1f}s")


# ---------- Convert to (24, n_species) DataFrames ----------
def to_df(d):
    df = pd.DataFrame(index=range(24), columns=list(d.keys()), dtype=float).fillna(0)
    for label, hours in d.items():
        for h, c in hours.items():
            df.loc[h, label] = c
    df.index.name = "hour"
    return df


df_global = to_df(results["global"])
df_pantanal = to_df(results["pantanal"])
print(f"\nglobal df: {df_global.shape}, total obs counted: {df_global.values.sum():.0f}")
print(f"pantanal df: {df_pantanal.shape}, total obs counted: {df_pantanal.values.sum():.0f}")

# Normalize per-species: P(hour | species)
df_global_norm = df_global.div(df_global.sum(axis=0).replace(0, 1), axis=1)
df_pantanal_norm = df_pantanal.div(df_pantanal.sum(axis=0).replace(0, 1), axis=1)

# Save
df_global.to_csv(f"{OUT}/inat_hour_global_counts.csv")
df_global_norm.to_csv(f"{OUT}/inat_hour_global_normalized.csv")
df_pantanal.to_csv(f"{OUT}/inat_hour_pantanal_counts.csv")
df_pantanal_norm.to_csv(f"{OUT}/inat_hour_pantanal_normalized.csv")
pd.DataFrame(diagnostics).to_csv(f"{OUT}/diagnostics.csv", index=False)
print(f"\nSaved 5 files to {OUT}")


# ---------- Quick visualization ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Plot 1: global heatmap, hour × class_name
class_to_label = dict(zip(tax["primary_label"], tax["class_name"]))
class_order = ["Aves", "Mammalia", "Reptilia", "Amphibia", "Insecta"]

ax = axes[0]
for cls in class_order:
    cls_labels = [l for l in df_global.columns if class_to_label.get(l) == cls]
    if cls_labels:
        cls_hours = df_global_norm[cls_labels].mean(axis=1)
        ax.plot(cls_hours.index, cls_hours.values, marker="o", label=f"{cls} (n={len(cls_labels)})")
ax.set_xlabel("Hour of day (local time per observation)")
ax.set_ylabel("Mean P(hour | class species)")
ax.set_title("Global iNat: hour-of-day distribution by class\n(species mean within class)")
ax.legend(fontsize=9)
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)

ax = axes[1]
for cls in class_order:
    cls_labels = [l for l in df_pantanal.columns if class_to_label.get(l) == cls]
    if cls_labels:
        cls_hours = df_pantanal_norm[cls_labels].mean(axis=1)
        if cls_hours.sum() > 0:
            ax.plot(cls_hours.index, cls_hours.values, marker="o", label=f"{cls} (n={len(cls_labels)})")
ax.set_xlabel("Hour of day (local time)")
ax.set_ylabel("Mean P(hour | class species)")
ax.set_title("Pantanal-bbox iNat: hour-of-day distribution by class")
ax.legend(fontsize=9)
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/inat_hour_by_class.png", dpi=110, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}/inat_hour_by_class.png")


# Coverage summary
diag = pd.DataFrame(diagnostics)
print(f"\n=== Coverage summary ===")
print(f"Species with ≥10 global obs: {(diag['n_global_pulled'] >= 10).sum()}/234")
print(f"Species with ≥10 Pantanal obs: {(diag['n_pantanal_pulled'] >= 10).sum()}/234")
print(f"Species with ≥1 Pantanal obs: {(diag['n_pantanal_pulled'] >= 1).sum()}/234")
print(f"Species with 0 global obs: {(diag['n_global_pulled'] == 0).sum()}/234")
print(f"Mean global obs per species: {diag['n_global_pulled'].mean():.1f}")
print(f"Mean Pantanal obs per species: {diag['n_pantanal_pulled'].mean():.1f}")
