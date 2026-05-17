"""Deep EDA addendum — focuses on subtleties relevant to the submission code.

Outputs into analysis/plots/ (prefix 'd' for deep) and analysis/stats/.
"""
from __future__ import annotations
import os, json, re, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import soundfile as sf

DATA  = Path("/home/user/opencode/birdclef-2026/data")
PLOTS = Path("/home/user/opencode/birdclef-2026/analysis/plots")
STATS = Path("/home/user/opencode/birdclef-2026/analysis/stats")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

CLASS_COLORS = {"Aves":"#2c7fb8","Amphibia":"#7fcdbb","Insecta":"#e6b800",
                "Mammalia":"#d95f0e","Reptilia":"#cc4c02"}

train    = pd.read_csv(DATA / "train.csv")
taxonomy = pd.read_csv(DATA / "taxonomy.csv")
ss_lbls  = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)

train["primary_label"]    = train["primary_label"].astype(str)
taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
ss_lbls["primary_label"]  = ss_lbls["primary_label"].astype(str)

L2C = dict(zip(taxonomy["primary_label"], taxonomy["class_name"]))
L2S = dict(zip(taxonomy["primary_label"], taxonomy["scientific_name"]))
L2N = dict(zip(taxonomy["primary_label"], taxonomy["common_name"]))

PANTANAL_BOX = dict(lat=(-21.6, -16.5), lon=(-57.6, -55.9))

# Soundscape filename parser
SS_PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
def parse_ss(name):
    m = SS_PAT.match(os.path.basename(str(name)))
    if not m: return None
    return dict(idx=int(m.group(1)), site=m.group(2),
                date=pd.to_datetime(m.group(3), format="%Y%m%d"),
                hour=int(m.group(4)[:2]))

# ---------- D1: per-class clip count percentiles + low-resource counts ----------
counts = train["primary_label"].value_counts()
print("\n=== D1: percentiles of clips/species ===")
for cls, grp in train.groupby(train["primary_label"].map(L2C)):
    c = grp["primary_label"].value_counts()
    print(f"  {cls:9s}: n_species={len(c):3d}  min={c.min():4d}  p25={int(np.percentile(c, 25)):4d}  "
          f"median={int(np.median(c)):4d}  p75={int(np.percentile(c, 75)):4d}  max={c.max():4d}")

# Plot D1: ECDF of clip counts per class
fig, ax = plt.subplots(figsize=(9, 5))
for cls, grp in train.groupby(train["primary_label"].map(L2C)):
    c = sorted(grp["primary_label"].value_counts().values)
    ax.plot(c, np.arange(1, len(c)+1)/len(c), label=f"{cls} (n={len(c)})",
            color=CLASS_COLORS.get(cls, "gray"), lw=2)
ax.set_xscale("log"); ax.set_xlabel("# train_audio clips per species (log)")
ax.set_ylabel("ECDF — fraction of species with ≤ x clips")
ax.set_title("Per-class clip count ECDF — non-bird taxa are extremely data-starved")
ax.axvline(20, color="red", ls=":", lw=1, label="20-clip rule of thumb")
ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "d01_clips_ecdf.png"); plt.close(fig)

# How many species have < threshold clips?
print("\n=== D1b: low-resource counts ===")
for t in [1, 5, 10, 20, 30, 50, 100]:
    n_le = (counts <= t).sum()
    print(f"  species with ≤ {t} clips: {n_le} (of 206 with any clips, 234 total)")

# ---------- D2: co-occurrence matrix in train_soundscapes_labels.csv ----------
print("\n=== D2: species co-occurrence in labeled soundscape segments ===")
labels_per_seg = ss_lbls["primary_label"].str.split(";")
species_in_labels = sorted({sp for row in labels_per_seg for sp in row})
print(f"  species ever labeled: {len(species_in_labels)}")
sp2idx = {s: i for i, s in enumerate(species_in_labels)}
n = len(species_in_labels)
mat = np.zeros((n, n), dtype=int)
for row in labels_per_seg:
    idxs = [sp2idx[s] for s in row]
    for i in idxs:
        for j in idxs:
            mat[i, j] += 1

# Save co-occurrence matrix
co_df = pd.DataFrame(mat, index=species_in_labels, columns=species_in_labels)
co_df.to_csv(STATS / "soundscape_cooccurrence.csv")

# Save top pairs
diag = np.diag(mat).copy()
mat_off = mat.copy(); np.fill_diagonal(mat_off, 0)
top_pairs = []
for i in range(n):
    for j in range(i+1, n):
        if mat_off[i, j] > 0:
            top_pairs.append({
                "species_a": species_in_labels[i],
                "species_b": species_in_labels[j],
                "class_a": L2C.get(species_in_labels[i], "?"),
                "class_b": L2C.get(species_in_labels[j], "?"),
                "co_segments": int(mat_off[i, j]),
                "min_count":    int(min(diag[i], diag[j])),
                "jaccard":      mat_off[i, j] / (diag[i] + diag[j] - mat_off[i, j]),
            })
pairs_df = pd.DataFrame(top_pairs).sort_values("co_segments", ascending=False)
pairs_df.to_csv(STATS / "soundscape_top_cooccurrences.csv", index=False)
print(f"  unique species pairs co-occurring: {len(pairs_df)}")
print(f"  top 10 by raw co-occurrence:")
print(pairs_df.head(10).to_string(index=False))

# Plot D2: heatmap of co-occurrence, ordered by class then frequency
order = sorted(species_in_labels, key=lambda s: (L2C.get(s, "Z"), -diag[sp2idx[s]]))
order_idx = [sp2idx[s] for s in order]
mat_o = mat[np.ix_(order_idx, order_idx)]
# Normalize: P(B | A) — given A is present in a segment, prob B is too
diag_o = np.diag(mat_o).astype(float)
diag_safe = np.where(diag_o > 0, diag_o, 1)
pmat = mat_o / diag_safe[:, None]

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(pmat, aspect="auto", cmap="magma", vmin=0, vmax=1)
ax.set_title("P(species B present | species A present) — labeled 5s segments  "
             f"({len(species_in_labels)} species)")
ax.set_xlabel("species B (column)"); ax.set_ylabel("species A (row)")
# Draw class boundaries
cls_order = [L2C.get(s, "?") for s in order]
ticks_cls = []
cur = None
for i, c in enumerate(cls_order):
    if c != cur:
        ax.axhline(i-0.5, color="cyan", lw=0.5)
        ax.axvline(i-0.5, color="cyan", lw=0.5)
        ticks_cls.append((i, c))
        cur = c
ax.set_yticks([t[0] for t in ticks_cls]); ax.set_yticklabels([t[1] for t in ticks_cls])
ax.set_xticks([t[0] for t in ticks_cls]); ax.set_xticklabels([t[1] for t in ticks_cls], rotation=0)
fig.colorbar(im, ax=ax, label="P(B|A)")
fig.tight_layout(); fig.savefig(PLOTS / "d02_cooccurrence_heatmap.png"); plt.close(fig)

# ---------- D3: insect sonotype labeling pattern ----------
print("\n=== D3: insect sonotype labeling ===")
sono_labels = {s for s in species_in_labels if s.startswith("47158son")}
print(f"  sonotypes seen in labels: {len(sono_labels)}/25")
sono_counts = {s: diag[sp2idx[s]] for s in sorted(sono_labels)}
for s, c in sorted(sono_counts.items()):
    print(f"  {s}: {c} segments")

# Co-occurrence of sonotypes
fig, ax = plt.subplots(figsize=(8, 7))
sono_idx = [sp2idx[s] for s in sorted(sono_labels)]
sono_mat = mat[np.ix_(sono_idx, sono_idx)]
sono_diag = np.diag(sono_mat).astype(float)
sono_pmat = sono_mat / np.where(sono_diag > 0, sono_diag, 1)[:, None]
im = ax.imshow(sono_pmat, cmap="magma", vmin=0, vmax=1)
labels = [s.replace("47158son", "son") for s in sorted(sono_labels)]
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
ax.set_title("Insect sonotype co-occurrence  P(B|A) — many sonotypes form chorus clusters")
fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(PLOTS / "d03_sonotype_cooccurrence.png"); plt.close(fig)

# ---------- D4: geographic spread per species ----------
print("\n=== D4: geographic spread per species ===")
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))  # km

geo = train.dropna(subset=["latitude", "longitude"])
panta_center = ((-21.6 + -16.5)/2, (-57.6 + -55.9)/2)

species_geo = []
for sp, g in geo.groupby("primary_label"):
    if len(g) < 2: continue
    lat_med = g["latitude"].median()
    lon_med = g["longitude"].median()
    # Distance from Pantanal centroid
    d_pan = haversine(lat_med, lon_med, panta_center[0], panta_center[1])
    # Geographic spread: 95th vs 5th percentile latitude/longitude range
    lat_range = g["latitude"].quantile(.95) - g["latitude"].quantile(.05)
    lon_range = g["longitude"].quantile(.95) - g["longitude"].quantile(.05)
    n_in_pan = ((g["latitude"].between(*PANTANAL_BOX["lat"])) &
                (g["longitude"].between(*PANTANAL_BOX["lon"]))).sum()
    species_geo.append({
        "primary_label": sp,
        "class_name":    L2C.get(sp, "?"),
        "n_clips":       len(g),
        "lat_med":       lat_med, "lon_med": lon_med,
        "lat_range":     lat_range, "lon_range": lon_range,
        "dist_pantanal_km": d_pan,
        "n_in_pantanal_box": int(n_in_pan),
        "frac_in_pantanal":  float(n_in_pan / len(g)),
    })
sp_geo_df = pd.DataFrame(species_geo)
sp_geo_df.to_csv(STATS / "species_geographic_spread.csv", index=False)

# What fraction of species have ANY clips in Pantanal box?
n_any_pan = (sp_geo_df["n_in_pantanal_box"] > 0).sum()
n_total = len(sp_geo_df)
print(f"  species with ≥1 clip inside Pantanal box: {n_any_pan} / {n_total} ({n_any_pan/n_total*100:.1f}%)")
print(f"  species with ≥10 clips inside Pantanal:    {(sp_geo_df['n_in_pantanal_box']>=10).sum()} / {n_total}")
print(f"  species with ≥50% of clips in Pantanal:    {(sp_geo_df['frac_in_pantanal']>=0.5).sum()} / {n_total}")

# Plot D4: histogram of % of clips in Pantanal per species
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(sp_geo_df["frac_in_pantanal"]*100, bins=np.linspace(0, 100, 21), color="#2c7fb8", edgecolor="white")
ax.set_xlabel("% of species' clips that are inside the Pantanal box")
ax.set_ylabel("# species")
ax.set_yscale("log")
ax.axvline(0, color="red", ls=":", lw=1)
ax.set_title(f"Per-species coverage of the Pantanal box  ({n_any_pan}/{n_total} species have any Pantanal clip)")
fig.tight_layout(); fig.savefig(PLOTS / "d04_species_pantanal_coverage.png"); plt.close(fig)

# ---------- D5: total audio duration per class (via file sizes as proxy) ----------
print("\n=== D5: total audio duration per class (file-size proxy) ===")
# Use the sampled duration data + file sizes to estimate bytes-per-second, then apply to all files.
ta_dir = DATA / "train_audio"
all_files = list(ta_dir.rglob("*.ogg"))
print(f"  scanning {len(all_files)} files for size...")
file_info = []
for fp in all_files:
    label = fp.parent.name
    file_info.append({"primary_label": label, "size_bytes": fp.stat().st_size,
                      "class_name": L2C.get(label, "?")})
fi = pd.DataFrame(file_info)
class_bytes = fi.groupby("class_name")["size_bytes"].sum() / 1e9
print("  class GB:", dict(class_bytes))

# From our duration sample, est bytes/second
dur_sample = pd.read_csv(STATS / "audio_duration_sample.csv")
dur_sample = dur_sample.dropna(subset=["duration_s"])
bps = (dur_sample["size_bytes"] / dur_sample["duration_s"]).median()  # bytes/second median
print(f"  estimated bytes/sec (median): {bps:.0f}")
fi["est_duration_s"] = fi["size_bytes"] / bps
class_hours = fi.groupby("class_name")["est_duration_s"].sum() / 3600
print(f"  est total hours per class:")
for k, v in sorted(class_hours.items(), key=lambda x: -x[1]):
    print(f"    {k:9s}: {v:8.1f} h")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(class_hours.sort_values(ascending=False).index,
       class_hours.sort_values(ascending=False).values,
       color=[CLASS_COLORS[c] for c in class_hours.sort_values(ascending=False).index])
ax.set_yscale("log"); ax.set_ylabel("estimated hours of audio")
ax.set_title(f"Total train_audio hours per class (~{class_hours.sum():.0f}h total)")
for x, h in zip(class_hours.sort_values(ascending=False).index, class_hours.sort_values(ascending=False).values):
    ax.text(x, h, f"{h:.0f}h", ha="center", va="bottom", fontsize=9)
fig.tight_layout(); fig.savefig(PLOTS / "d05_audio_hours_per_class.png"); plt.close(fig)

# ---------- D6: soundscape labeling — site/hour coverage ----------
print("\n=== D6: where the labeled soundscapes come from ===")
labeled_files = ss_lbls["filename"].unique().tolist()
parsed = [parse_ss(f) for f in labeled_files]
parsed = [p for p in parsed if p]
lab_sites = Counter(p["site"] for p in parsed)
lab_hours = Counter(p["hour"] for p in parsed)
lab_dates = Counter(p["date"] for p in parsed)
print(f"  labeled files cover {len(lab_sites)} sites: {dict(lab_sites)}")
print(f"  hour distribution: {sorted(lab_hours.items())}")
print(f"  date range: {min(p['date'] for p in parsed).date()} → {max(p['date'] for p in parsed).date()}")
print(f"  unique dates: {len(lab_dates)}")

# All soundscape files for comparison
all_ss_files = [p.name for p in (DATA / "train_soundscapes").glob("*.ogg")]
all_parsed = [parse_ss(f) for f in all_ss_files]
all_parsed = [p for p in all_parsed if p]
all_sites = Counter(p["site"] for p in all_parsed)
all_hours = Counter(p["hour"] for p in all_parsed)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sites_sorted = sorted(set(all_sites.keys()))
axes[0].bar([s for s in sites_sorted], [all_sites.get(s, 0) for s in sites_sorted],
            color="#cccccc", label="all soundscapes")
axes[0].bar([s for s in sites_sorted], [lab_sites.get(s, 0) for s in sites_sorted],
            color="#d62728", label="labeled subset")
axes[0].set_ylabel("# files"); axes[0].set_title("Sites: labeled vs all soundscape files")
axes[0].legend()

axes[1].bar(range(24), [all_hours.get(h, 0) for h in range(24)],
            color="#cccccc", label="all soundscapes")
axes[1].bar(range(24), [lab_hours.get(h, 0) for h in range(24)],
            color="#d62728", label="labeled subset")
axes[1].set_xlabel("hour (UTC)"); axes[1].set_ylabel("# files"); axes[1].set_title("Hour-of-day coverage: labeled vs all")
axes[1].set_xticks(range(0, 24, 2)); axes[1].legend()
fig.tight_layout(); fig.savefig(PLOTS / "d06_labeled_vs_all_sites_hours.png"); plt.close(fig)

# ---------- D7: secondary labels in train.csv ----------
print("\n=== D7: secondary labels in train.csv ===")
def parse_sec(x):
    if pd.isna(x): return []
    s = str(x)
    if s in ("[]", ""): return []
    # could be "[label1, label2]" or "label1;label2"
    s = s.strip("[]").replace("'", "").replace('"', "")
    parts = [p.strip() for p in re.split("[,;]", s) if p.strip()]
    return parts

train["sec_list"] = train["secondary_labels"].map(parse_sec)
n_with_sec = (train["sec_list"].map(len) > 0).sum()
print(f"  rows with any secondary labels: {n_with_sec}/{len(train)} ({n_with_sec/len(train)*100:.1f}%)")
all_sec = [s for ss in train["sec_list"] for s in ss]
sec_counts = Counter(all_sec)
print(f"  unique secondary labels: {len(sec_counts)}")
print(f"  top 10:")
for sp, n in sec_counts.most_common(10):
    print(f"    {sp:12s} ({L2N.get(sp, '?')[:30]:30s}): {n}")

# How many of the 28 missing species appear as secondary labels?
missing28 = sorted(set(taxonomy["primary_label"]) - set(train["primary_label"]))
missing_in_sec = {m: sec_counts.get(m, 0) for m in missing28}
n_sec_covered = sum(1 for v in missing_in_sec.values() if v > 0)
print(f"\n  of 28 missing species: {n_sec_covered} appear in secondary_labels")
for m, n in sorted(missing_in_sec.items(), key=lambda x: -x[1])[:15]:
    if n > 0: print(f"    {m}: {n} secondary mentions")

# ---------- D8: train.csv class -> rating quality ----------
print("\n=== D8: rating distribution per class (XC clips only) ===")
xc = train[train["collection"] == "XC"].copy()
xc["class_name"] = xc["primary_label"].map(L2C)
fig, ax = plt.subplots(figsize=(9, 4.5))
classes = ["Aves", "Amphibia", "Mammalia"]  # XC-relevant classes
data = [xc[xc["class_name"] == c]["rating"].values for c in classes]
parts = ax.violinplot(data, showmedians=True)
ax.set_xticks(range(1, len(classes)+1)); ax.set_xticklabels(classes)
ax.set_ylabel("rating (0 = unrated, 1–5 quality)")
ax.set_title("XC rating distribution per class (rating>0 means human-judged)")
for pc, cls in zip(parts['bodies'], classes):
    pc.set_facecolor(CLASS_COLORS[cls]); pc.set_alpha(0.7)
fig.tight_layout(); fig.savefig(PLOTS / "d08_rating_per_class_xc.png"); plt.close(fig)
for c in classes:
    sub = xc[xc["class_name"] == c]["rating"]
    print(f"  {c}: n={len(sub)} mean={sub.mean():.2f} %rated={(sub>0).mean()*100:.1f}%")

# ---------- D9: train_soundscapes activity by hour ----------
ss_df = pd.DataFrame(all_parsed)
print("\n=== D9: train_soundscape activity per site/hour ===")
# In labeled segments only, count species mentions per hour
lab_sp_per_seg = ss_lbls.copy()
lab_sp_per_seg["fname"] = lab_sp_per_seg["filename"].map(lambda f: os.path.basename(str(f)))
lab_sp_per_seg["hour"] = lab_sp_per_seg["fname"].map(lambda n: int(SS_PAT.match(n).group(4)[:2]) if SS_PAT.match(n) else -1)
lab_sp_per_seg["n_species"] = lab_sp_per_seg["primary_label"].str.split(";").str.len()
hr_card = lab_sp_per_seg.groupby("hour")["n_species"].agg(["count", "mean"])
print(hr_card)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(hr_card.index, hr_card["mean"], color="#2c7fb8")
ax.set_xlabel("hour (UTC)"); ax.set_ylabel("mean # species per labeled 5s segment")
ax.set_title("Acoustic richness per hour (in labeled segments)")
ax.set_xticks(range(0, 24))
fig.tight_layout(); fig.savefig(PLOTS / "d09_hour_richness.png"); plt.close(fig)

# ---------- D10: prediction-window count (sample submission) ----------
print("\n=== D10: sample_submission characteristics ===")
sub = pd.read_csv(DATA / "sample_submission.csv")
n_rows = len(sub)
unique_files = sub["row_id"].str.rsplit("_", n=1).str[0].nunique()
print(f"  rows: {n_rows}, unique files: {unique_files}, rows/file: {n_rows/unique_files:.1f}")
# Expected: 12 windows per 1-min file

# ---------- summary json append ----------
summary = json.load(open(STATS / "summary.json"))
summary["deep"] = {
    "low_resource_species": {
        f"le_{t}_clips": int((counts <= t).sum()) for t in [1, 5, 10, 20, 30, 50, 100]
    },
    "species_geographic": {
        "species_with_any_pantanal_clip": int(n_any_pan),
        "species_with_ge10_pantanal_clips": int((sp_geo_df["n_in_pantanal_box"]>=10).sum()),
        "species_with_ge50pct_in_pantanal": int((sp_geo_df["frac_in_pantanal"]>=0.5).sum()),
    },
    "audio_hours_per_class": {k: float(v) for k, v in class_hours.items()},
    "labeled_soundscape_sites_covered": len(lab_sites),
    "labeled_soundscape_hour_uniques": len(lab_hours),
    "rows_with_secondary_labels":     int(n_with_sec),
    "missing_species_appearing_in_secondary": n_sec_covered,
    "soundscape_cooccurrence_pairs": int(len(pairs_df)),
    "submission_rows":          n_rows,
    "submission_unique_files":  int(unique_files),
    "submission_rows_per_file": float(n_rows / unique_files),
}
json.dump(summary, open(STATS / "summary.json", "w"), indent=2)
print("\nDeep analysis done — updated summary.json")
