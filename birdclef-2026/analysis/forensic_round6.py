"""Round 6 — duty-cycle, silence segments, and cross-source spectral search.

R6a: Recorder duty-cycle reconstruction — when do recorders sleep / wake?
     Match against SwiftOne docs.
R6b: Silence detection — are any 5-second segments TRUE silence?
     If yes, can be used as confidence-zero negatives.
R6c: Identify the 5-second segments with the HIGHEST mean Perch-prior fit
     (cleanest, easiest to label) and the LOWEST (hardest).
R6d: ACTUAL audio-content overlap: for each Pantanal-box train_audio clip,
     find its closest train_soundscape match (Shazam hash) AND verify with
     PCM cross-correlation at the matched offset. This is the "hidden leak"
     test in earnest.
R6e: Per-site time-of-day distribution — does each site record on a
     specific schedule, suggesting duty-cycling?
R6f: Inspect the 14 RATE-LIMITED files specifically (BC2026_Train_9986..9999)
     — are they unusual?
R6g: Is the test sample's site (S05) acoustically distinctive enough that
     a model could "recognize" it from a 5-sec window?
"""
import os, sys, re, time, hashlib, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, pandas as pd, soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import correlate
import datetime as dt

DATA = Path("/home/user/opencode/birdclef-2026/data")
OUT  = Path("/home/user/opencode/birdclef-2026/analysis/forensic")

PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
ss_lbl = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
labeled = set(ss_lbl["filename"].unique())

all_files = []
for fp in (DATA / "train_soundscapes").glob("*.ogg"):
    m = PAT.match(fp.name)
    if m:
        all_files.append({"idx": int(m.group(1)), "site": m.group(2),
                          "date": m.group(3), "time": m.group(4),
                          "path": fp, "labeled": fp.name in labeled})

def section(t): print("\n" + "="*60); print(t); print("="*60); sys.stdout.flush()

# ---------- R6a: Duty-cycle reconstruction ----------
section("R6a: Recorder duty-cycle reconstruction")
# For each (site, date) group, look at the timestamps of recordings
# A continuous "duty-on" period is files that are nearly contiguous in time
from collections import defaultdict
by_site_date = defaultdict(list)
for r in all_files:
    by_site_date[(r["site"], r["date"])].append(r)

def parse_ts(d, t):
    return dt.datetime.strptime(d+t, "%Y%m%d%H%M%S")

# For each (site, date), compute the on-times and gaps
duty_patterns = []
for (site, date), recs in by_site_date.items():
    if len(recs) < 2: continue
    recs_sorted = sorted(recs, key=lambda r: parse_ts(r["date"], r["time"]))
    timestamps = [parse_ts(r["date"], r["time"]) for r in recs_sorted]
    gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
    duty_patterns.append({
        "site": site, "date": date, "n_files": len(recs),
        "first_time": recs_sorted[0]["time"],
        "last_time": recs_sorted[-1]["time"],
        "min_gap_s": min(gaps),
        "max_gap_s": max(gaps),
        "median_gap_s": float(np.median(gaps)),
    })
dp = pd.DataFrame(duty_patterns)
print(f"(site, date) pairs with ≥2 files: {len(dp)}")
print(f"\nDistribution of GAP between consecutive recordings within a day:")
all_gaps = []
for (site, date), recs in by_site_date.items():
    recs_sorted = sorted(recs, key=lambda r: parse_ts(r["date"], r["time"]))
    timestamps = [parse_ts(r["date"], r["time"]) for r in recs_sorted]
    for i in range(len(timestamps)-1):
        all_gaps.append((timestamps[i+1] - timestamps[i]).total_seconds())
all_gaps = np.array(all_gaps)
print(f"  total gaps observed: {len(all_gaps)}")
print(f"  percentiles: p5={np.percentile(all_gaps, 5):.0f}s, "
      f"p25={np.percentile(all_gaps, 25):.0f}s, p50={np.percentile(all_gaps, 50):.0f}s, "
      f"p75={np.percentile(all_gaps, 75):.0f}s, p95={np.percentile(all_gaps, 95):.0f}s")
# How often is gap close to 1 hour (3600s)? 30 minutes? 5 minutes?
for nominal in [60, 300, 900, 1800, 3600]:
    near = ((all_gaps >= nominal*0.9) & (all_gaps <= nominal*1.1)).sum()
    pct = near / len(all_gaps) * 100
    print(f"  gap ≈ {nominal}s ({nominal//60} min): {near} ({pct:.1f}%)")

# Is there a CONSISTENT duty cycle (e.g., 1 minute per X minutes)?
# Check the mode of gaps
from collections import Counter
gap_minutes = Counter(int(g//60) for g in all_gaps if g <= 7200)
top_modes = gap_minutes.most_common(10)
print(f"\nTop 10 gap modes (in minutes):")
for m, n in top_modes: print(f"  {m} min: {n}")

# Per-site duty-cycle (mode of gap)
print(f"\nPer-site median gap (minutes), for sites with ≥10 multi-file days:")
for site in sorted(set(r["site"] for r in all_files)):
    site_gaps = []
    for (s, d), recs in by_site_date.items():
        if s != site or len(recs) < 2: continue
        recs_sorted = sorted(recs, key=lambda r: parse_ts(r["date"], r["time"]))
        timestamps = [parse_ts(r["date"], r["time"]) for r in recs_sorted]
        for i in range(len(timestamps)-1):
            site_gaps.append((timestamps[i+1] - timestamps[i]).total_seconds() / 60)
    if len(site_gaps) >= 10:
        gap_arr = np.array(site_gaps)
        print(f"  {site}: n_gaps={len(site_gaps):5d} "
              f"min={gap_arr.min():.0f}min med={np.median(gap_arr):.0f}min "
              f"p90={np.percentile(gap_arr, 90):.0f}min")

# ---------- R6b: Silence detection ----------
section("R6b: How many 5-sec segments are TRUE silence?")
def silent_segs(rec):
    try:
        x, sr = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        if x.size < 60 * 32000: return None
        # Reshape to 12 windows of 5s
        win = 5 * 32000
        nwin = x.size // win
        if nwin < 12: return None
        wins = x[:12*win].reshape(12, win)
        rms_per_win = np.sqrt(np.mean(wins**2, axis=1))
        return {"name": rec["path"].name, "labeled": rec["labeled"], "site": rec["site"],
                "silent_segs": int((rms_per_win < 0.005).sum()),
                "quiet_segs":  int((rms_per_win < 0.01).sum()),
                "very_loud":   int((rms_per_win > 0.3).sum()),
                "min_rms": float(rms_per_win.min()),
                "max_rms": float(rms_per_win.max()),
                "dynamic_range": float(rms_per_win.max() / (rms_per_win.min() + 1e-9))}
    except: return None

import random; random.seed(0)
sample = [r for r in all_files if r["labeled"]]
unlab = [r for r in all_files if not r["labeled"]]
random.shuffle(unlab)
sample.extend(unlab[:200])
print(f"checking silent-segs on {len(sample)} files...")
t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    sr = [r for r in ex.map(silent_segs, sample, chunksize=16) if r is not None]
print(f"done in {time.time()-t0:.0f}s")
sdf = pd.DataFrame(sr)
print(f"\nSilent-segment distribution (per 60-s file):")
print(f"  Labeled (n={len(sdf[sdf.labeled])}):")
print(f"    mean silent_segs (rms<0.005): {sdf[sdf.labeled]['silent_segs'].mean():.2f}")
print(f"    mean quiet_segs  (rms<0.01):  {sdf[sdf.labeled]['quiet_segs'].mean():.2f}")
print(f"    files with ≥1 silent seg: {(sdf[sdf.labeled]['silent_segs'] >= 1).sum()}/{len(sdf[sdf.labeled])}")
print(f"  Unlabeled (n={len(sdf[~sdf.labeled])}):")
print(f"    mean silent_segs: {sdf[~sdf.labeled]['silent_segs'].mean():.2f}")
print(f"    mean quiet_segs:  {sdf[~sdf.labeled]['quiet_segs'].mean():.2f}")
print(f"    files with ≥1 silent seg: {(sdf[~sdf.labeled]['silent_segs'] >= 1).sum()}/{len(sdf[~sdf.labeled])}")
print(f"    files with ALL 12 segs quiet: {(sdf[~sdf.labeled]['quiet_segs'] == 12).sum()}")
# Dynamic range
print(f"\nDynamic range (max_rms/min_rms per file):")
print(f"  Labeled mean: {sdf[sdf.labeled]['dynamic_range'].mean():.1f}")
print(f"  Unlabeled mean: {sdf[~sdf.labeled]['dynamic_range'].mean():.1f}")

# ---------- R6c: Confidence-segment ranking ----------
# (Skip — already covered by D5/D9 from earlier rounds)

# ---------- R6d: Pantanal-box train_audio Shazam vs soundscape ----------
section("R6d: Pantanal-box train_audio audio overlap with train_soundscapes")
# We have d3_ta_vs_ss_xfp.csv from round 2 (sampled 1 per species)
# Refine: for the 847 Pantanal-box train_audio clips, run Shazam against all soundscapes
tr = pd.read_csv(DATA / "train.csv")
pan = tr[(tr.latitude.between(-21.6, -16.5)) & (tr.longitude.between(-57.6, -55.9))].copy()
pan["fn"] = pan["filename"].str.split("/").str[-1]
pan_files = [(DATA / "train_audio" / row["primary_label"] / row["fn"], row)
             for _, row in pan.iterrows()]
pan_files = [(p, r) for p, r in pan_files if p.exists()]
print(f"Pantanal-box train_audio files: {len(pan_files)}")

# Quick Shazam-style hashing
def shazam_hashes(fp, max_hashes=200):
    try:
        x, sr = sf.read(str(fp), frames=30*32000, dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        if x.size < 4096: return set()
    except Exception: return set()
    n_fft = 2048; hop = 1024
    nframes = (x.size - n_fft) // hop + 1
    if nframes < 4: return set()
    frames = np.stack([x[i*hop:i*hop+n_fft] * np.hanning(n_fft) for i in range(nframes)])
    spec = np.log(np.abs(np.fft.rfft(frames, axis=1)) + 1e-9)
    from scipy.ndimage import maximum_filter
    peaks = (spec == maximum_filter(spec, size=(5, 7))) & (spec > np.percentile(spec, 90))
    peak_t, peak_f = np.where(peaks)
    peak_pairs = []
    for i in range(len(peak_t)):
        for j in range(i+1, min(i+8, len(peak_t))):
            dt_ = peak_t[j] - peak_t[i]
            if 1 <= dt_ <= 20:
                peak_pairs.append((int(peak_f[i] // 4), int(peak_f[j] // 4), int(dt_)))
                if len(peak_pairs) >= max_hashes: break
        if len(peak_pairs) >= max_hashes: break
    return set(peak_pairs)

print(f"Hashing {len(pan_files)} Pantanal-box clips...")
t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    pan_hashes = dict(zip([p.name for p, _ in pan_files],
                         ex.map(shazam_hashes, [p for p, _ in pan_files], chunksize=32)))
print(f"  done in {time.time()-t0:.0f}s")

# Load existing soundscape fingerprints (assume same scheme)
# Build inverted index using the soundscape Shazam hashes from round 2
print("Hashing all 10,658 train_soundscapes...")
t0 = time.time()
ss_paths = [r["path"] for r in all_files]
with ProcessPoolExecutor(max_workers=6) as ex:
    ss_hash_list = list(ex.map(shazam_hashes, ss_paths, chunksize=32))
ss_hashes = {p.name: h for p, h in zip(ss_paths, ss_hash_list)}
print(f"  done in {time.time()-t0:.0f}s")

# Inverted index
inv = defaultdict(list)
for fname, hh in ss_hashes.items():
    for h in hh: inv[h].append(fname)

# Background distribution
print("Computing background random-pair shared hashes...")
random.seed(0)
ss_names = list(ss_hashes.keys())
bg = []
for _ in range(500):
    a, b = random.sample(ss_names, 2)
    bg.append(len(ss_hashes[a] & ss_hashes[b]))
bg_p95 = np.percentile(bg, 95); bg_p99 = np.percentile(bg, 99)
print(f"  background random-pair shared hashes: p50={int(np.percentile(bg, 50))} p95={int(bg_p95)} p99={int(bg_p99)} max={max(bg)}")

# For each Pantanal clip, find its top-3 soundscape matches
matches = []
for fname, ph in pan_hashes.items():
    if not ph: continue
    counts = Counter()
    for h in ph:
        for o in inv.get(h, []):
            counts[o] += 1
    if not counts: continue
    top3 = counts.most_common(3)
    for o, n in top3:
        matches.append({"ta_file": fname, "ss_file": o,
                       "shared_hashes": n, "ta_total": len(ph),
                       "share_pct": n / max(len(ph), 1) * 100})
mdf = pd.DataFrame(matches)
mdf["above_p99_bg"] = mdf["shared_hashes"] > bg_p99
print(f"\nPantanal-box → train_soundscape top matches:")
print(f"  Pairs above background p99: {mdf['above_p99_bg'].sum()}/{len(mdf)}")
print(f"  Top 20 by shared_hashes:")
print(mdf.nlargest(20, "shared_hashes").to_string(index=False))
mdf.to_csv(OUT / "r6d_pan_ta_vs_ss_shazam.csv.gz", index=False, compression="gzip")

# ---------- R6e: Per-site time-of-day pattern ----------
section("R6e: Per-site time-of-day recording schedule")
hour_by_site = defaultdict(Counter)
for r in all_files:
    hour_by_site[r["site"]][int(r["time"][:2])] += 1
print(f"\nHour distribution per site (showing hours with >0 files):")
for site in sorted(hour_by_site.keys()):
    hist = hour_by_site[site]
    if sum(hist.values()) < 10: continue  # skip tiny sites
    active_hours = sorted([h for h, c in hist.items() if c > 0])
    print(f"  {site} (n={sum(hist.values())}): hours active = {active_hours}")

# ---------- R6f: The 14 rate-limited files ----------
section("R6f: Inspect the 14 files we rate-limited earlier (idx 9986-9999)")
target = [f"BC2026_Train_{i:04d}_S22_20240213_*.ogg" if False else f"BC2026_Train_{i:04d}" for i in range(9986, 10000)]
print(f"Looking for indices 9986..9999")
for r in all_files:
    if 9986 <= r["idx"] <= 9999:
        print(f"  idx={r['idx']} site={r['site']} date={r['date']} time={r['time']} labeled={r['labeled']}")

# These are all S22 from 2024-02-13 — let me check sizes and any unusual properties
suspect = [r for r in all_files if 9986 <= r["idx"] <= 9999]
print(f"\nFile size and basic stats for the 14 rate-limited indices:")
for r in suspect:
    sz = r["path"].stat().st_size
    print(f"  idx={r['idx']} {r['date']}_{r['time']} size={sz}")

# Are these files unusual compared to neighbors?
neighbor_sizes = []
for r in all_files:
    if r["site"] == "S22" and 9970 <= r["idx"] <= 10005:
        sz = r["path"].stat().st_size
        neighbor_sizes.append((r["idx"], sz))
neighbor_sizes.sort()
print(f"\nNeighborhood sizes (idx 9970-10005 at S22):")
for idx, sz in neighbor_sizes[:30]:
    print(f"  idx={idx}: {sz} bytes")

# ---------- R6g: S05 acoustic distinctiveness ----------
section("R6g: Is S05 acoustically distinctive?")
s05 = [r for r in all_files if r["site"] == "S05"]
print(f"S05 files: {len(s05)}")
# Get spec profile of S05 vs other sites
def mean_spec_profile(rec, n_bins=32):
    try:
        x, sr = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        n_fft = 2048; hop = 1024
        nframes = (x.size - n_fft) // hop + 1
        if nframes < 4: return None
        frames = np.stack([x[i*hop:i*hop+n_fft] * np.hanning(n_fft) for i in range(nframes)])
        spec = np.abs(np.fft.rfft(frames, axis=1))
        bins = np.linspace(0, spec.shape[1], n_bins+1, dtype=int)
        return np.array([spec[:, bins[i]:bins[i+1]].mean() for i in range(n_bins)])
    except: return None

t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    s05_specs = [r for r in ex.map(mean_spec_profile, s05, chunksize=4) if r is not None]
print(f"  s05 specs computed in {time.time()-t0:.0f}s")
s05_specs = np.stack(s05_specs)
# Other sites mean
print("\nComputing other-site mean spectra (sample 100 per site)...")
import random as rnd; rnd.seed(0)
others = []
for site in sorted(set(r["site"] for r in all_files)):
    if site == "S05": continue
    site_recs = [r for r in all_files if r["site"] == site]
    rnd.shuffle(site_recs)
    others.extend(site_recs[:30])
t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    other_specs = [r for r in ex.map(mean_spec_profile, others, chunksize=4) if r is not None]
print(f"  other specs in {time.time()-t0:.0f}s")
other_specs = np.stack(other_specs)
print(f"\nS05 spec mean: {s05_specs.mean(axis=0).round(3)}")
print(f"Other  mean:   {other_specs.mean(axis=0).round(3)}")
# Cosine similarity
s05_norm = s05_specs.mean(axis=0) / np.linalg.norm(s05_specs.mean(axis=0))
other_norm = other_specs.mean(axis=0) / np.linalg.norm(other_specs.mean(axis=0))
print(f"\nS05↔Other cos sim: {float(np.dot(s05_norm, other_norm)):.4f}")

print("\n[forensic_round6 done]")
