"""Round 4 — recorder fingerprinting, temporal patterns, extreme-clipping inspection."""
import os, sys, re, time, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor

DATA = Path("/home/user/opencode/birdclef-2026/data")
OUT  = Path("/home/user/opencode/birdclef-2026/analysis/forensic")

PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
ss_lbl = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
labeled = set(ss_lbl["filename"].unique())

# Load D16 stats
df_full = pd.read_csv(OUT / "d16_full_clipping_stats.csv.gz")
print(f"Loaded {len(df_full)} files from D16 stats")
print(f"Clipping summary: heavy={int((df_full.clip_pct > 1).sum())}, "
      f"moderate={int((df_full.clip_pct > 0.1).sum())}")

# ---------- R4a: Clipping by date ----------
def section(t): print("\n" + "="*60); print(t); print("="*60); sys.stdout.flush()
section("R4a: Clipping by date — did recorders go bad on specific days?")
df_full["dt"] = pd.to_datetime(df_full["date"], format="%Y%m%d")
df_full["year"] = df_full["dt"].dt.year
df_full["month"] = df_full["dt"].dt.month
df_full["hour"] = df_full["time"].astype(str).str.zfill(6).str[:2].astype(int)

# Per-year clipping rate
print("\nPer-year clipping rate:")
for yr, sub in df_full.groupby("year"):
    print(f"  {int(yr)}: n={len(sub)} heavy={int((sub.clip_pct > 1).sum())} "
          f"({(sub.clip_pct > 1).mean()*100:.1f}%)")

# S01 specifically — clipping rate by month/year
s01 = df_full[df_full.site == "S01"]
print("\nS01 clipping by year-month (top 15 worst months):")
s01["yearmonth"] = s01["dt"].dt.strftime("%Y-%m")
ym_clip = s01.groupby("yearmonth").agg(
    n=("idx", "count"),
    heavy=("clip_pct", lambda s: (s > 1).sum()),
    mean_clip=("clip_pct", "mean")).sort_values("mean_clip", ascending=False)
print(ym_clip.head(15).to_string())

# S13
print("\nS13 clipping by year-month (top 15 worst):")
s13 = df_full[df_full.site == "S13"]
s13["yearmonth"] = s13["dt"].dt.strftime("%Y-%m")
ym13 = s13.groupby("yearmonth").agg(
    n=("idx", "count"), heavy=("clip_pct", lambda s: (s > 1).sum()),
    mean_clip=("clip_pct", "mean")).sort_values("mean_clip", ascending=False)
print(ym13.head(15).to_string())

# ---------- R4b: Clipping by HOUR ----------
section("R4b: Clipping by HOUR of day")
print("Hour-by-hour clipping rate (across all sites):")
for hr in sorted(df_full.hour.unique()):
    sub = df_full[df_full.hour == hr]
    print(f"  {hr:02d}: n={len(sub):4d} avg_clip={sub.clip_pct.mean():.3f}% "
          f"heavy={int((sub.clip_pct>1).sum())} ({(sub.clip_pct>1).mean()*100:.1f}%)")

# ---------- R4c: Recorder fingerprint clustering (KMeans on noise floor) ----------
section("R4c: Recorder hardware fingerprint clustering (noise floor)")
def noise_spec(rec, n_bins=64):
    try:
        x, sr = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        if x.size < 16384: return None
        n_fft = 4096; hop = 4096
        nframes = (x.size - n_fft) // hop + 1
        if nframes < 8: return None
        frames = np.stack([x[i*hop:i*hop+n_fft] * np.hanning(n_fft) for i in range(nframes)])
        spec = np.abs(np.fft.rfft(frames, axis=1))**2
        energy = spec.sum(axis=1)
        quiet_idx = np.argsort(energy)[:max(1, nframes // 10)]
        quiet_spec = spec[quiet_idx].mean(axis=0)
        bins = np.linspace(0, len(quiet_spec), n_bins+1, dtype=int)
        binned = np.array([quiet_spec[bins[i]:bins[i+1]].mean() for i in range(n_bins)])
        binned = np.log10(binned + 1e-12)
        return {"name": rec["path"].name, "site": rec["site"], "labeled": rec["labeled"],
                "fp": binned}
    except: return None

# Sample stratified — all labeled + 30 per site
all_recs = []
for fp in (DATA / "train_soundscapes").glob("*.ogg"):
    m = PAT.match(fp.name)
    if m:
        all_recs.append({"path": fp, "site": m.group(2), "labeled": fp.name in labeled})
import random; random.seed(0)
samp = [r for r in all_recs if r["labeled"]]
by_site = defaultdict(list)
for r in all_recs:
    if not r["labeled"]:
        by_site[r["site"]].append(r)
for s, recs in by_site.items():
    random.shuffle(recs)
    samp.extend(recs[:30])
print(f"sampling {len(samp)} files for fingerprint clustering")

t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    fps = [r for r in ex.map(noise_spec, samp, chunksize=16) if r is not None]
print(f"done in {time.time()-t0:.0f}s")

fp_arr = np.stack([r["fp"] for r in fps])
sites = np.array([r["site"] for r in fps])
labeled_arr = np.array([r["labeled"] for r in fps])

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
fp_scaled = scaler.fit_transform(fp_arr)
for k in [4, 6, 8, 10]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(fp_scaled)
    labels_k = km.labels_
    print(f"\nK={k}:")
    for cl in range(k):
        members = labels_k == cl
        n = int(members.sum())
        n_lab = int(labeled_arr[members].sum())
        sites_in = Counter(sites[members])
        # Compute mean distance to cluster center
        center_dist = np.linalg.norm(fp_scaled[members] - km.cluster_centers_[cl], axis=1)
        print(f"  cl {cl}: n={n:3d} n_lab={n_lab:2d}  sites: {dict(sites_in.most_common(6))}  meandist={center_dist.mean():.2f}")

# Save fingerprints
out_rows = [{"name": r["name"], "site": r["site"], "labeled": r["labeled"]} for r in fps]
fp_df = pd.DataFrame(out_rows)
for i in range(fp_arr.shape[1]):
    fp_df[f"fp_bin{i:02d}"] = fp_arr[:, i]
fp_df.to_csv(OUT / "r4_noise_fingerprints.csv.gz", index=False, compression="gzip")

# ---------- R4d: HF cutoff frequency analysis ----------
section("R4d: HF cutoff per site (low-pass / anti-aliasing signature)")
def hf_cutoff(rec):
    try:
        x, sr = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        if x.size < 32768: return None
        n_fft = 32768; hop = 16384
        nframes = min(8, (x.size - n_fft) // hop + 1)
        if nframes < 2: return None
        frames = np.stack([x[i*hop:i*hop+n_fft] * np.hanning(n_fft) for i in range(nframes)])
        spec = np.abs(np.fft.rfft(frames, axis=1))**2
        avg_spec = spec.mean(axis=0)
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        log_avg = np.log10(avg_spec + 1e-12)
        peak = log_avg.max()
        # Find -20 dB cutoff
        below_20 = freqs[log_avg < peak - 2]
        cutoff = float(below_20[0]) if below_20.size else float(freqs[-1])
        return {"site": rec["site"], "labeled": rec["labeled"], "cutoff_20db_hz": cutoff}
    except: return None

t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    hf = [r for r in ex.map(hf_cutoff, samp, chunksize=16) if r is not None]
print(f"done in {time.time()-t0:.0f}s")
hf_df = pd.DataFrame(hf)
print("\nPer-site -20dB cutoff (rounded to nearest 1000 Hz):")
for s, sub in hf_df.groupby("site"):
    cutoffs = sub.cutoff_20db_hz.values
    # Round to find modal values
    modal = pd.cut(cutoffs, bins=[0, 4000, 6000, 8000, 10000, 12000, 14000, 16000])
    print(f"  {s} (n={len(sub)}): mean={cutoffs.mean():.0f} median={np.median(cutoffs):.0f} "
          f"buckets: {dict(modal.value_counts())}")

# ---------- R4e: All-files extreme-clipping pattern ----------
section("R4e: All 104 extreme-clipping files breakdown")
extreme = df_full[df_full.clip_pct > 20]
print(f"Total extreme files: {len(extreme)}")
print(f"  All at site: {extreme.site.value_counts().to_dict()}")
print(f"  Dates: {extreme.date.nunique()} unique")
print(f"  Date range: {extreme.date.min()} - {extreme.date.max()}")
print(f"  Hours: {sorted(extreme.hour.unique())}")
# Continuous date streak
extreme_dates = sorted(extreme.date.unique())
print(f"\nExtreme-clipping date list ({len(extreme_dates)} dates):")
for d in extreme_dates:
    n = (extreme.date == d).sum()
    print(f"  {d}: {n} files")

# Is this a specific recorder failure period?
extreme["dt"] = pd.to_datetime(extreme["date"], format="%Y%m%d")
streak_starts = []
prev = None
for d in sorted(extreme.dt.unique()):
    if prev is None or (d - prev).days > 7:
        streak_starts.append(d)
    prev = d
print(f"\nStreaks (gap >7 days): {len(streak_starts)}")
for s in streak_starts: print(f"  start: {s.date()}")

# ---------- R4f: Are there gaps in recording timeline at S01? ----------
section("R4f: S01 recording timeline — gaps suggesting recorder downtime")
s01 = df_full[df_full.site == "S01"].copy()
s01["dt"] = pd.to_datetime(s01["date"], format="%Y%m%d")
date_counts = s01.groupby(s01.dt.dt.date).size()
print(f"S01 dates with recordings: {len(date_counts)}")
print(f"  date range: {date_counts.index.min()} - {date_counts.index.max()}")
# Gaps
sorted_dates = sorted(date_counts.index)
gaps = []
for i in range(1, len(sorted_dates)):
    gap_days = (sorted_dates[i] - sorted_dates[i-1]).days
    if gap_days > 30:
        gaps.append((sorted_dates[i-1], sorted_dates[i], gap_days))
print(f"S01 month-plus gaps: {len(gaps)}")
for g in gaps[:10]: print(f"  {g[0]} → {g[1]} ({g[2]} days)")

# ---------- R4g: train.csv author lat/lon for Pantanal recordists ----------
section("R4g: Top Pantanal-box recordists — do any match site coordinates?")
tr = pd.read_csv(DATA / "train.csv")
pan = tr[(tr.latitude.between(-21.6, -16.5)) & (tr.longitude.between(-57.6, -55.9))].copy()
# Top 5 lat/lon clusters with author/recordist info
top_locs = pan.groupby(["latitude", "longitude"]).agg(
    n=("filename", "count"),
    authors=("author", lambda s: s.value_counts().head(3).to_dict()),
    species=("primary_label", lambda s: s.value_counts().head(3).to_dict()),
).sort_values("n", ascending=False).head(10)
print(top_locs.to_string())

print("\n[forensic_round4 done]")
