"""BirdCLEF 2026 — comprehensive EDA from local data.

Outputs:
  analysis/stats/*.csv  — derived tables
  analysis/plots/*.png  — figures
  analysis/SUMMARY.md   — written by analyze.py
"""
from __future__ import annotations
import os, sys, json, math, csv, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

DATA = Path("/home/user/opencode/birdclef-2026/data")
OUT  = Path("/home/user/opencode/birdclef-2026/analysis")
PLOTS = OUT / "plots"
STATS = OUT / "stats"
PLOTS.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CLASS_COLORS = {
    "Aves":     "#2c7fb8",
    "Amphibia": "#7fcdbb",
    "Insecta":  "#e6b800",
    "Mammalia": "#d95f0e",
    "Reptilia": "#cc4c02",
}

# ---------- load metadata ----------
train      = pd.read_csv(DATA / "train.csv")
taxonomy   = pd.read_csv(DATA / "taxonomy.csv")
ss_labels_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv")
# Important quirk: every row is duplicated exactly once — dedupe before analysis.
ss_labels = ss_labels_raw.drop_duplicates().reset_index(drop=True)
DUP_RATIO = 1 - len(ss_labels) / len(ss_labels_raw)
print(f"train_soundscapes_labels: {len(ss_labels_raw)} raw rows -> {len(ss_labels)} unique "
      f"({DUP_RATIO*100:.0f}% duplicates)")
sample_sub = pd.read_csv(DATA / "sample_submission.csv", nrows=5)

print(f"train.csv: {train.shape}")
print(f"taxonomy.csv: {taxonomy.shape}")
print(f"train_soundscapes_labels.csv: {ss_labels.shape}")
print(f"sample_submission.csv: {sample_sub.shape} (just header inspection)")

# Numeric primary_label codes are read as ints; cast everything to str.
train["primary_label"]    = train["primary_label"].astype(str)
taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)

label_to_class = dict(zip(taxonomy["primary_label"], taxonomy["class_name"]))
label_to_sci   = dict(zip(taxonomy["primary_label"], taxonomy["scientific_name"]))
label_to_com   = dict(zip(taxonomy["primary_label"], taxonomy["common_name"]))

# ---------- (1) class distribution ----------
counts = train["primary_label"].value_counts().rename_axis("primary_label").reset_index(name="n_clips")
counts["class_name"] = counts["primary_label"].map(label_to_class)
counts.to_csv(STATS / "clips_per_species.csv", index=False)

all_species = set(taxonomy["primary_label"])
in_train    = set(counts["primary_label"])
missing_species = sorted(all_species - in_train)
miss_df = pd.DataFrame({
    "primary_label": missing_species,
    "class_name":    [label_to_class[m] for m in missing_species],
    "scientific":    [label_to_sci.get(m, "") for m in missing_species],
    "common":        [label_to_com.get(m, "") for m in missing_species],
})
miss_df.to_csv(STATS / "species_missing_from_train_audio.csv", index=False)
print(f"Species missing from train_audio: {len(missing_species)}")

# Class summary
class_summary = (
    train.groupby(train["primary_label"].map(label_to_class))
         .size().rename("clips_in_train").to_frame()
         .join(taxonomy.groupby("class_name").size().rename("n_species"))
)
class_summary["mean_clips_per_species"] = class_summary["clips_in_train"] / class_summary["n_species"]
class_summary.to_csv(STATS / "class_summary.csv")
print(class_summary)

# Plot 1: clips per class (taxonomic group)
fig, ax = plt.subplots(figsize=(8, 4.5))
class_summary_sorted = class_summary.sort_values("clips_in_train", ascending=False)
bars = ax.bar(class_summary_sorted.index, class_summary_sorted["clips_in_train"],
              color=[CLASS_COLORS[c] for c in class_summary_sorted.index])
ax.set_yscale("log")
ax.set_ylabel("# training clips (log)")
ax.set_title("Training clips per taxonomic class")
for b, n_species in zip(bars, class_summary_sorted["n_species"]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height(),
            f"{int(b.get_height()):,}\n({int(n_species)} sp.)",
            ha="center", va="bottom", fontsize=9)
fig.tight_layout(); fig.savefig(PLOTS / "01_clips_per_class.png"); plt.close(fig)

# Plot 2: clips per species, ranked (with class color)
ordered = counts.sort_values("n_clips", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(range(len(ordered)), ordered["n_clips"],
       color=[CLASS_COLORS[c] for c in ordered["class_name"]],
       width=1.0, linewidth=0)
ax.set_xlabel("Species rank (sorted by clip count)")
ax.set_ylabel("# training clips")
ax.set_title(f"Per-species clip counts in train_audio ({len(ordered)} species; "
             f"{len(missing_species)} more species have ZERO train_audio clips)")
ax.set_xlim(-1, len(ordered))
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in CLASS_COLORS.items()
                   if k in set(ordered["class_name"])],
          loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(PLOTS / "02_clips_per_species_ranked.png"); plt.close(fig)

# Plot 3: histogram of clip counts (class imbalance shape)
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.logspace(0, np.log10(counts["n_clips"].max() + 1), 30)
ax.hist(counts["n_clips"], bins=bins, color="#2c7fb8", edgecolor="white")
ax.set_xscale("log")
ax.set_xlabel("# clips per species (log)")
ax.set_ylabel("# species")
ax.axvline(counts["n_clips"].median(), color="red", ls="--", lw=1,
           label=f"median = {int(counts['n_clips'].median())}")
ax.axvline(counts["n_clips"].mean(), color="orange", ls="--", lw=1,
           label=f"mean = {counts['n_clips'].mean():.0f}")
ax.legend()
ax.set_title("Distribution of clips per species — heavy long-tail / many under-sampled classes")
fig.tight_layout(); fig.savefig(PLOTS / "03_clips_per_species_histogram.png"); plt.close(fig)

# ---------- (2) collection split ----------
coll = train.groupby(["class_name", "collection"]).size().unstack(fill_value=0)
coll.to_csv(STATS / "collection_split.csv")
print(coll)

fig, ax = plt.subplots(figsize=(7, 4))
coll.plot(kind="bar", stacked=True, ax=ax,
          color=["#2c7fb8", "#fdbb84"])
ax.set_ylabel("# training clips")
ax.set_title("Collection source (xeno-canto vs iNaturalist) by class")
ax.set_yscale("log")
ax.set_xticklabels(coll.index, rotation=0)
fig.tight_layout(); fig.savefig(PLOTS / "04_collection_split.png"); plt.close(fig)

# ---------- (3) rating distribution ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
xc = train[train["collection"] == "XC"]["rating"]
inat = train[train["collection"] == "iNat"]["rating"]
axes[0].hist(xc, bins=np.arange(0, 6, 0.5), color="#2c7fb8", edgecolor="white")
axes[0].set_title(f"Xeno-canto ratings  (n={len(xc):,})")
axes[0].set_xlabel("rating (0 = unrated, 1–5 quality)")
axes[0].set_ylabel("# clips")
axes[1].hist(inat, bins=np.arange(0, 6, 0.5), color="#fdbb84", edgecolor="white")
axes[1].set_title(f"iNaturalist ratings  (n={len(inat):,})  — all 0 (unrated by design)")
axes[1].set_xlabel("rating")
fig.tight_layout(); fig.savefig(PLOTS / "05_rating_distribution.png"); plt.close(fig)

# ---------- (4) geographic distribution ----------
geo = train.dropna(subset=["latitude", "longitude"]).copy()
print(f"Clips with geo: {len(geo):,}/{len(train):,}")
fig, ax = plt.subplots(figsize=(9, 6.5))
for cls, sub in geo.groupby("class_name"):
    ax.scatter(sub["longitude"], sub["latitude"], s=3, alpha=0.35,
               color=CLASS_COLORS.get(cls, "gray"), label=f"{cls} ({len(sub):,})")
# Pantanal bounding box (from recording_location.txt)
import matplotlib.patches as mpatches
ax.add_patch(mpatches.Rectangle((-57.6, -21.6), (-55.9)-(-57.6), (-16.5)-(-21.6),
                                fill=False, edgecolor="red", lw=2, label="Pantanal (test region)"))
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Where the training audio was recorded — Pantanal box in red")
ax.legend(loc="lower left", fontsize=8, markerscale=2)
fig.tight_layout(); fig.savefig(PLOTS / "06_geo_world.png"); plt.close(fig)

# Zoomed to Pantanal + South America
fig, ax = plt.subplots(figsize=(8, 7))
sa = geo[(geo["latitude"].between(-35, 12)) & (geo["longitude"].between(-85, -33))]
for cls, sub in sa.groupby("class_name"):
    ax.scatter(sub["longitude"], sub["latitude"], s=8, alpha=0.4,
               color=CLASS_COLORS.get(cls, "gray"), label=f"{cls} ({len(sub):,})")
ax.add_patch(mpatches.Rectangle((-57.6, -21.6), (-55.9)-(-57.6), (-16.5)-(-21.6),
                                fill=False, edgecolor="red", lw=2.5))
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("South-America zoom: training clip locations vs Pantanal (red box)")
ax.legend(loc="upper right", fontsize=8, markerscale=1.5)
fig.tight_layout(); fig.savefig(PLOTS / "07_geo_south_america.png"); plt.close(fig)

# How many clips are inside the Pantanal box?
in_box = geo[(geo["latitude"].between(-21.6, -16.5)) &
             (geo["longitude"].between(-57.6, -55.9))]
print(f"Clips inside Pantanal box: {len(in_box):,} / {len(geo):,} ({len(in_box)/len(geo)*100:.2f}%)")

# ---------- (5) audio duration via soundfile (sample) ----------
print("Sampling audio durations...")
random.seed(0)
ta_files = []
ta_dir = DATA / "train_audio"
species_dirs = sorted([d for d in ta_dir.iterdir() if d.is_dir()])
print(f"train_audio species dirs: {len(species_dirs)}")
# Sample 1500 files across species for speed
per_sp = max(1, 1500 // len(species_dirs))
for d in species_dirs:
    fs = list(d.glob("*.ogg"))
    if not fs: continue
    random.shuffle(fs)
    ta_files.extend([(d.name, f) for f in fs[:per_sp]])
print(f"Sampling {len(ta_files)} train_audio files for duration")

rows = []
for label, fp in ta_files:
    try:
        info = sf.info(str(fp))
        rows.append({
            "primary_label": label,
            "filename": str(fp.relative_to(DATA)),
            "duration_s": info.frames / info.samplerate,
            "samplerate": info.samplerate,
            "channels":   info.channels,
            "size_bytes": fp.stat().st_size,
            "class_name": label_to_class.get(label, "?"),
            "source":     "train_audio",
        })
    except Exception as e:
        rows.append({"primary_label": label, "filename": str(fp.relative_to(DATA)),
                     "error": str(e), "source": "train_audio"})

dur_df = pd.DataFrame(rows)
dur_df.to_csv(STATS / "audio_duration_sample.csv", index=False)
print(f"train_audio durations: median={dur_df['duration_s'].median():.1f}s "
      f"mean={dur_df['duration_s'].mean():.1f}s "
      f"min={dur_df['duration_s'].min():.1f}s "
      f"max={dur_df['duration_s'].max():.1f}s "
      f"p99={dur_df['duration_s'].quantile(.99):.1f}s")
print(f"sample rates: {dur_df['samplerate'].value_counts().to_dict()}")
print(f"channels: {dur_df['channels'].value_counts().to_dict()}")

fig, ax = plt.subplots(figsize=(9, 4.5))
bins = np.logspace(np.log10(0.5), np.log10(dur_df["duration_s"].max() + 1), 60)
ax.hist(dur_df["duration_s"].dropna(), bins=bins, color="#2c7fb8", edgecolor="white")
ax.set_xscale("log")
ax.set_xlabel("clip duration (seconds, log)")
ax.set_ylabel("# clips")
med = dur_df["duration_s"].median()
ax.axvline(med, color="red", ls="--", label=f"median = {med:.1f}s")
ax.axvline(5, color="orange", ls=":", label="test segment = 5s")
ax.set_title(f"train_audio clip duration  (sample of {len(dur_df):,}; all 32 kHz mono)")
ax.legend()
fig.tight_layout(); fig.savefig(PLOTS / "08_audio_duration_hist.png"); plt.close(fig)

# ---------- (6) train_soundscapes coverage ----------
ss_dir = DATA / "train_soundscapes"
ss_files = sorted([p.name for p in ss_dir.glob("*.ogg")])
print(f"train_soundscapes: {len(ss_files)} files")

# filename: BC2026_Train_<idx>_<site>_<YYYYMMDD>_<HHMMSS>.ogg
import re
PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
ss_rows = []
for name in ss_files:
    m = PAT.match(name)
    if not m: continue
    idx, site, date, tm = m.groups()
    ss_rows.append({
        "filename": f"train_soundscapes/{name}",
        "idx": int(idx), "site": site,
        "date": pd.to_datetime(date, format="%Y%m%d"),
        "time": pd.to_datetime(tm, format="%H%M%S").time(),
        "hour": int(tm[:2]),
    })
ss_df = pd.DataFrame(ss_rows)
ss_df.to_csv(STATS / "train_soundscapes_inventory.csv", index=False)
print(f"sites: {ss_df['site'].nunique()} | dates: {ss_df['date'].nunique()}")
print(ss_df["site"].value_counts())
print(f"date range: {ss_df['date'].min().date()} → {ss_df['date'].max().date()}")

# Plot 9: soundscape recordings per site
fig, ax = plt.subplots(figsize=(9, 4.5))
site_counts = ss_df["site"].value_counts().sort_index()
ax.bar(site_counts.index, site_counts.values, color="#2c7fb8")
ax.set_ylabel("# 1-minute soundscape recordings")
ax.set_xlabel("recording site")
ax.set_title(f"train_soundscapes per site (total {len(ss_df):,})")
fig.tight_layout(); fig.savefig(PLOTS / "09_soundscapes_per_site.png"); plt.close(fig)

# Plot 10: hour-of-day vs site
piv = pd.crosstab(ss_df["site"], ss_df["hour"])
piv = piv.reindex(columns=range(24), fill_value=0)
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
ax.set_xticks(range(24)); ax.set_xticklabels(range(24))
ax.set_xlabel("hour of day (UTC)"); ax.set_ylabel("site")
ax.set_title("train_soundscapes time-of-day coverage by site (count of 1-min clips)")
fig.colorbar(im, ax=ax, label="# clips")
fig.tight_layout(); fig.savefig(PLOTS / "10_soundscapes_hour_site.png"); plt.close(fig)

# Plot 11: temporal coverage (date timeline)
fig, ax = plt.subplots(figsize=(11, 4.5))
for site, g in ss_df.groupby("site"):
    dates_per_day = g.groupby("date").size()
    ax.scatter(dates_per_day.index, [site]*len(dates_per_day),
               s=dates_per_day.values*2, alpha=0.7)
ax.set_xlabel("date"); ax.set_ylabel("site")
ax.set_title("train_soundscapes — recordings over time per site (marker size = clips that day)")
fig.tight_layout(); fig.savefig(PLOTS / "11_soundscapes_timeline.png"); plt.close(fig)

# ---------- (7) train_soundscapes_labels analysis ----------
ss_labels["filename"] = ss_labels["filename"].astype(str)
ss_labels["primary_label"] = ss_labels["primary_label"].astype(str)

print(f"train_soundscapes_labels.csv: {len(ss_labels):,} segment rows; "
      f"covering {ss_labels['filename'].nunique()} soundscape files")

# Each row may have multiple species (semicolon-separated)
all_labeled_species = []
for s in ss_labels["primary_label"]:
    all_labeled_species.extend(str(s).split(";"))
species_seg_count = Counter(all_labeled_species)
print(f"Unique species mentioned in labeled segments: {len(species_seg_count)}")

# species present in soundscape labels but missing from train_audio
ss_unique = set(species_seg_count.keys())
missing_yet_labeled = sorted(ss_unique - in_train)
print(f"Species ONLY known via soundscape labels (not in train_audio): {len(missing_yet_labeled)}")
print(missing_yet_labeled[:30])

ss_species_df = pd.DataFrame({
    "primary_label": list(species_seg_count.keys()),
    "n_labeled_segments": list(species_seg_count.values()),
})
ss_species_df["class_name"] = ss_species_df["primary_label"].map(label_to_class)
ss_species_df["in_train_audio"] = ss_species_df["primary_label"].isin(in_train)
ss_species_df = ss_species_df.sort_values("n_labeled_segments", ascending=False)
ss_species_df.to_csv(STATS / "ss_labels_per_species.csv", index=False)

# Plot 12: count of labeled segments per species  (in_train vs only-soundscape)
fig, ax = plt.subplots(figsize=(11, 5))
ss_species_df["color"] = np.where(ss_species_df["in_train_audio"], "#2c7fb8", "#d62728")
ax.bar(range(len(ss_species_df)), ss_species_df["n_labeled_segments"],
       color=ss_species_df["color"], width=1.0)
ax.set_yscale("log")
ax.set_xlabel("species (sorted by # labeled 5s segments)")
ax.set_ylabel("# labeled segments (log)")
ax.set_title(f"Labeled 5s segments per species in train_soundscapes_labels "
             f"({len(ss_species_df)} species; red = species missing from train_audio)")
ax.legend(handles=[
    mpatches.Patch(color="#2c7fb8", label="Has train_audio examples"),
    mpatches.Patch(color="#d62728", label="ONLY in soundscape labels"),
])
fig.tight_layout(); fig.savefig(PLOTS / "12_soundscape_labels_per_species.png"); plt.close(fig)

# Multi-label cardinality
seg_card = ss_labels["primary_label"].apply(lambda s: len(str(s).split(";")))
print(f"segment label cardinality: mean={seg_card.mean():.2f}, "
      f"max={seg_card.max()}, % multi-label={ (seg_card > 1).mean()*100:.1f}%")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(seg_card, bins=range(1, seg_card.max() + 2), color="#2c7fb8", edgecolor="white",
        align="left")
ax.set_xticks(range(1, seg_card.max() + 1))
ax.set_xlabel("# species labeled in segment")
ax.set_ylabel("# segments")
ax.set_title(f"Multi-label cardinality per 5s segment "
             f"({(seg_card > 1).mean()*100:.1f}% are multi-label)")
fig.tight_layout(); fig.savefig(PLOTS / "13_label_cardinality.png"); plt.close(fig)

# Per-file label coverage / density
ss_labels["start_s"] = pd.to_timedelta(ss_labels["start"]).dt.total_seconds().astype(int)
labels_per_file = ss_labels.groupby("filename").agg(
    n_segments=("start_s", "count"),
    min_start=("start_s", "min"),
    max_start=("start_s", "max"),
).reset_index()
labels_per_file.to_csv(STATS / "ss_labels_per_file.csv", index=False)
print(f"Labeled soundscape files: {len(labels_per_file)}")
print(f"  segments/file: median={labels_per_file['n_segments'].median()}, "
      f"min={labels_per_file['n_segments'].min()}, max={labels_per_file['n_segments'].max()}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(labels_per_file["n_segments"], bins=20, color="#2c7fb8", edgecolor="white")
ax.set_xlabel("# labeled 5s segments per soundscape file")
ax.set_ylabel("# files")
ax.set_title(f"How densely each labeled soundscape file is annotated  "
             f"({len(labels_per_file)} files; 1 file = 12 segments if fully labeled)")
fig.tight_layout(); fig.savefig(PLOTS / "14_segments_per_labeled_file.png"); plt.close(fig)

# ---------- (8) sample submission sanity ----------
sample_cols = list(sample_sub.columns)
print(f"sample_submission columns: {len(sample_cols)} (=1 row_id + {len(sample_cols)-1} species)")
expected = set(taxonomy["primary_label"])
sub_species = set(sample_cols[1:])
print(f"  submission species match taxonomy: {sub_species == expected}")
print(f"  missing from submission: {expected - sub_species}")
print(f"  extra in submission:    {sub_species - expected}")

# ---------- emit json stats ----------
summary = {
    "totals": {
        "train_csv_rows": int(len(train)),
        "taxonomy_classes": int(len(taxonomy)),
        "submission_classes": int(len(sample_cols) - 1),
        "train_audio_dirs": int(len(species_dirs)),
        "train_soundscape_files": int(len(ss_files)),
        "labeled_segments": int(len(ss_labels)),
        "labeled_soundscape_files": int(ss_labels['filename'].nunique()),
    },
    "class_distribution": {
        "in_taxonomy": dict(Counter(taxonomy["class_name"])),
        "clips_in_train_audio": {k: int(v) for k, v in class_summary["clips_in_train"].items()},
    },
    "rare_species": {
        "n_missing_from_train_audio": len(missing_species),
        "n_only_in_soundscape_labels": len(missing_yet_labeled),
        "min_clips_in_train": int(counts["n_clips"].min()),
        "median_clips_in_train": int(counts["n_clips"].median()),
    },
    "audio_duration_sample": {
        "n_sampled": int(len(dur_df)),
        "median_s": float(dur_df["duration_s"].median()),
        "mean_s": float(dur_df["duration_s"].mean()),
        "p99_s": float(dur_df["duration_s"].quantile(.99)),
        "min_s": float(dur_df["duration_s"].min()),
        "max_s": float(dur_df["duration_s"].max()),
        "sample_rates": {int(k): int(v) for k, v in dur_df["samplerate"].value_counts().items()},
        "channels":     {int(k): int(v) for k, v in dur_df["channels"].value_counts().items()},
    },
    "geo": {
        "with_coords": int(len(geo)),
        "in_pantanal_box": int(len(in_box)),
        "pct_in_pantanal_box": float(len(in_box) / len(geo) * 100),
    },
    "soundscapes": {
        "sites":  sorted(ss_df["site"].unique().tolist()),
        "n_sites": int(ss_df["site"].nunique()),
        "date_min": str(ss_df["date"].min().date()),
        "date_max": str(ss_df["date"].max().date()),
        "n_files": int(len(ss_df)),
    },
    "multi_label": {
        "labeled_segments": int(len(ss_labels)),
        "raw_rows_in_csv": int(len(ss_labels_raw)),
        "duplicate_ratio_pct": float(DUP_RATIO * 100),
        "mean_species_per_segment": float(seg_card.mean()),
        "pct_multi_label": float((seg_card > 1).mean() * 100),
        "max_species_in_one_segment": int(seg_card.max()),
    },
}
with open(STATS / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Done. Wrote summary.json")
