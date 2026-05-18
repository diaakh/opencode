"""For species that returned 0 observations with taxon_id, try resolving by
scientific name first, then re-querying with the correct ID.
"""
import os, time, json
import pandas as pd
import numpy as np
import requests
from collections import Counter

OUT = "/home/user/opencode/birdclef-2026/analysis/external_priors"

tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
diag = pd.read_csv(f"{OUT}/diagnostics.csv")
zero_species = diag[diag["n_global_pulled"] == 0]
print(f"Zero-obs species: {len(zero_species)}")

SESS = requests.Session()
SESS.headers["User-Agent"] = "BC2026-research (anon-research)"

TAXA_URL = "https://api.inaturalist.org/v1/taxa"
OBS_URL = "https://api.inaturalist.org/v1/observations"


def resolve_by_name(scientific_name):
    """Search iNat for a taxon by scientific name; return iNat ID + observations_count if found."""
    r = SESS.get(TAXA_URL, params={"q": scientific_name, "rank": "species", "per_page": 5}, timeout=20)
    if r.status_code != 200:
        return None, None, f"http_{r.status_code}"
    data = r.json()
    if not data.get("results"):
        return None, None, "no_results"
    # Take first match that's exact name match
    for res in data["results"]:
        if res.get("name", "").lower() == scientific_name.lower():
            return res["id"], res.get("observations_count", 0), "exact"
    # Fallback: first result
    res = data["results"][0]
    return res["id"], res.get("observations_count", 0), f"fuzzy:{res.get('name')}"


def fetch_hours_by_id(taxon_id, max_pages=3, per_page=200, geo_bbox=None, sleep=0.4):
    hours = Counter()
    for page in range(1, max_pages + 1):
        params = {"taxon_id": taxon_id, "per_page": per_page, "page": page,
                  "order_by": "observed_on", "order": "desc"}
        if geo_bbox:
            params.update(geo_bbox)
        try:
            r = SESS.get(OBS_URL, params=params, timeout=30)
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            for obs in results:
                details = obs.get("observed_on_details") or {}
                h = details.get("hour")
                if h is not None:
                    hours[h] += 1
            if len(results) < per_page:
                break
        except Exception:
            return hours, "err"
        time.sleep(sleep)
    return hours, "ok"


PANTANAL = dict(nelat=-16.5, nelng=-55.9, swlat=-21.6, swlng=-57.6)

# Load existing data so we can append
g_old = pd.read_csv(f"{OUT}/inat_hour_global_counts.csv", index_col="hour")
p_old = pd.read_csv(f"{OUT}/inat_hour_pantanal_counts.csv", index_col="hour")

results = {"global": {}, "pantanal": {}}
resolved = []

for i, row in zero_species.iterrows():
    sci = row.get("common_name") if pd.isna(row.get("common_name")) else None
    sci = None
    # Look up scientific_name from main taxonomy table
    tax_row = tax[tax["primary_label"] == row["primary_label"]]
    if len(tax_row) == 0:
        continue
    sci_name = tax_row["scientific_name"].iloc[0]
    label = row["primary_label"]

    new_id, obs_count, status = resolve_by_name(sci_name)
    if new_id is None or obs_count == 0:
        resolved.append({"label": label, "sci_name": sci_name, "new_id": new_id, "obs_count": obs_count, "status": status})
        time.sleep(0.4)
        continue

    hours_g, _ = fetch_hours_by_id(new_id, max_pages=2, per_page=200)
    hours_p, _ = fetch_hours_by_id(new_id, max_pages=1, geo_bbox=PANTANAL)
    n_g = sum(hours_g.values())
    n_p = sum(hours_p.values())
    results["global"][label] = dict(hours_g)
    results["pantanal"][label] = dict(hours_p)
    resolved.append({"label": label, "sci_name": sci_name, "new_id": new_id,
                     "obs_count": obs_count, "n_global": n_g, "n_pantanal": n_p, "status": status})
    if len(results["global"]) % 10 == 0:
        print(f"  [{len(results['global']):3d}] {label:>10s} {sci_name:>30s} new_id={new_id} "
              f"obs_count={obs_count} fetched g={n_g} p={n_p} ({status})")

print(f"\nResolved by scientific name: {len(results['global'])}")
pd.DataFrame(resolved).to_csv(f"{OUT}/resolved_diagnostics.csv", index=False)


# Update counts CSVs by merging
def update_df(old_df, new_results):
    new_df = old_df.copy()
    for label, hours in new_results.items():
        for h in range(24):
            new_df.loc[h, label] = hours.get(h, 0)
    return new_df


g_new = update_df(g_old, results["global"])
p_new = update_df(p_old, results["pantanal"])

# Renormalize
g_norm = g_new.div(g_new.sum(axis=0).replace(0, 1), axis=1)
p_norm = p_new.div(p_new.sum(axis=0).replace(0, 1), axis=1)

g_new.to_csv(f"{OUT}/inat_hour_global_counts.csv")
g_norm.to_csv(f"{OUT}/inat_hour_global_normalized.csv")
p_new.to_csv(f"{OUT}/inat_hour_pantanal_counts.csv")
p_norm.to_csv(f"{OUT}/inat_hour_pantanal_normalized.csv")

# Re-run coverage summary
def cov(df, n_min):
    return (df.sum(axis=0) >= n_min).sum()
print(f"\nCoverage after refetch:")
print(f"  Species with ≥10 global obs: {cov(g_new, 10)}/234")
print(f"  Species with ≥10 Pantanal obs: {cov(p_new, 10)}/234")
print(f"  Species with ≥1 global obs: {cov(g_new, 1)}/234")
print(f"  Total global obs counted: {g_new.values.sum():.0f}")
print(f"  Total Pantanal obs counted: {p_new.values.sum():.0f}")
