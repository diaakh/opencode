"""Round 3 forensic — find hardware/recorder fingerprints and full-corpus stats.

D16: Full clipping audit — all 10,658 train_soundscapes (not just sample).
D17: Per-file noise floor spectrum → cluster into recorder/hardware groups.
D18: High-frequency cutoff per file (low-pass filter signature).
D19: 50 / 60 Hz mains hum detection.
D20: DC-offset distribution per site.
D21: Bonus: train_audio Pantanal-box clips with matching dates → train_soundscape sessions.
D22: Test-soundscape filename pattern from sample_submission row_id.
D23: SPECIFIC look at the S05 sample-test row: does S05 audio look like test-time data?
D24: For ALL 41 same-(site,date,time) collisions, are they MULTI-MIC recordings?
"""
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

all_files = []
for fp in (DATA / "train_soundscapes").glob("*.ogg"):
    m = PAT.match(fp.name)
    if m:
        all_files.append({"idx": int(m.group(1)), "site": m.group(2),
                          "date": m.group(3), "time": m.group(4),
                          "path": fp, "labeled": fp.name in labeled})

def section(t): print("\n" + "="*60); print(t); print("="*60); sys.stdout.flush()

# ---------- D16: FULL clipping audit ----------
section("D16: Full clipping audit (all 10,658 train_soundscapes)")
def full_stats(rec):
    try:
        x, _ = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        return {
            "name": rec["path"].name, "site": rec["site"], "date": rec["date"],
            "time": rec["time"], "idx": rec["idx"], "labeled": rec["labeled"],
            "rms": float(np.sqrt(np.mean(x*x))),
            "peak": float(np.abs(x).max()),
            "dc_offset": float(x.mean()),
            "n_clipped_99": int((np.abs(x) > 0.99).sum()),
            "n_samples": int(x.size),
        }
    except Exception:
        return None

print(f"computing full stats for {len(all_files)} files...")
t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    stats = [s for s in ex.map(full_stats, all_files, chunksize=32) if s is not None]
df_full = pd.DataFrame(stats)
df_full["clip_pct"] = df_full["n_clipped_99"] / df_full["n_samples"] * 100
df_full.to_csv(OUT / "d16_full_clipping_stats.csv.gz", index=False, compression="gzip")
print(f"done in {time.time()-t0:.0f}s")
print(f"Total files: {len(df_full)}")
print(f"Heavy clipping (>1% saturated): {(df_full.clip_pct > 1).sum()} files")
print(f"Moderate clipping (>0.1%): {(df_full.clip_pct > 0.1).sum()} files")
print(f"  Per-site heavy-clipping count:")
for site, sub in df_full.groupby("site"):
    heavy = (sub.clip_pct > 1).sum()
    moderate = (sub.clip_pct > 0.1).sum()
    print(f"    {site}: {heavy:4d}/{len(sub):4d} heavy ({heavy/len(sub)*100:.1f}%), "
          f"{moderate:4d} moderate, avg_clip%={sub.clip_pct.mean():.2f}, avg_RMS={sub.rms.mean():.3f}")

# Distribution of clipping
import numpy as np
print(f"\nClipping % distribution (all files):")
for q in [50, 75, 90, 95, 99, 99.5, 99.9]:
    print(f"  p{q}: {np.percentile(df_full.clip_pct, q):.3f}%")
print(f"\nFiles with extreme clipping (>20% saturated):")
extreme = df_full[df_full.clip_pct > 20].sort_values("clip_pct", ascending=False)
print(f"  count: {len(extreme)}")
print(extreme[["idx","site","date","time","clip_pct","rms"]].head(15).to_string(index=False))

# ---------- D17: Noise floor spectrum clustering ----------
section("D17: Noise-floor spectrum per file — recorder/hardware fingerprint")
def noise_floor_spectrum(rec, n_bins=32):
    """Spectrum of the QUIETEST frames in a file (recorder self-noise)."""
    try:
        x, _ = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        if x.size < 16384: return None
        n_fft = 4096; hop = 4096
        nframes = (x.size - n_fft) // hop + 1
        if nframes < 4: return None
        frames = np.stack([x[i*hop:i*hop+n_fft] * np.hanning(n_fft) for i in range(nframes)])
        spec = np.abs(np.fft.rfft(frames, axis=1))**2
        # Frame energy
        energy = spec.sum(axis=1)
        # Take 5 quietest frames
        quiet_idx = np.argsort(energy)[:max(1, nframes // 10)]
        quiet_spec = spec[quiet_idx].mean(axis=0)
        # 32 bins
        bins = np.linspace(0, len(quiet_spec), n_bins+1, dtype=int)
        binned = np.array([quiet_spec[bins[i]:bins[i+1]].mean() for i in range(n_bins)])
        binned = np.log10(binned + 1e-12)
        return {"name": rec["path"].name, "site": rec["site"], "labeled": rec["labeled"],
                "noise_spectrum": binned}
    except Exception:
        return None

# Sample: 30 labeled + 200 unlabeled stratified per site
import random; random.seed(0)
sample_recs = list(labeled)
unlab_by_site = defaultdict(list)
for r in all_files:
    if not r["labeled"]:
        unlab_by_site[r["site"]].append(r)
unlab_sample = []
for s, recs in unlab_by_site.items():
    random.shuffle(recs)
    unlab_sample.extend(recs[:10])  # 10 per site
print(f"computing noise spectrum for {len(sample_recs)} labeled + {len(unlab_sample)} unlabeled")

lab_recs = [r for r in all_files if r["labeled"]]
all_sample = lab_recs + unlab_sample
t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    noise_results = [r for r in ex.map(noise_floor_spectrum, all_sample, chunksize=16) if r is not None]
print(f"done in {time.time()-t0:.0f}s")
noise_arr = np.stack([r["noise_spectrum"] for r in noise_results])
sites_arr = np.array([r["site"] for r in noise_results])
labeled_arr = np.array([r["labeled"] for r in noise_results])

# Cluster
from sklearn.cluster import KMeans
print("\nKMeans clustering of noise-floor spectra:")
for k in [4, 8]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(noise_arr)
    labels_k = km.labels_
    print(f"\n  K={k} clusters:")
    for cl in range(k):
        members = (labels_k == cl)
        sites_in_cluster = Counter(sites_arr[members])
        n_lab_in_cluster = labeled_arr[members].sum()
        print(f"    cluster {cl}: n={members.sum()}, n_lab={n_lab_in_cluster}, "
              f"top sites: {dict(sites_in_cluster.most_common(5))}")

# ---------- D18: High-frequency cutoff per file ----------
section("D18: HF cutoff (find low-pass corner per file)")
def hf_cutoff(rec):
    try:
        x, sr = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        if x.size < 16384: return None
        n_fft = 16384; hop = 8192
        nframes = min(8, (x.size - n_fft) // hop + 1)
        if nframes < 2: return None
        frames = np.stack([x[i*hop:i*hop+n_fft] * np.hanning(n_fft) for i in range(nframes)])
        spec = np.abs(np.fft.rfft(frames, axis=1))**2
        avg_spec = spec.mean(axis=0)
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        # Smooth in log
        log_avg = np.log10(avg_spec + 1e-12)
        # Find where energy drops below -2 of max
        peak = log_avg.max()
        below = freqs[log_avg < peak - 2]
        cutoff_3db = freqs[log_avg < peak - 0.3][0] if (log_avg < peak - 0.3).any() else freqs[-1]
        cutoff_20db = freqs[log_avg < peak - 2][0] if (log_avg < peak - 2).any() else freqs[-1]
        # Energy above 14 kHz
        e_14k = avg_spec[freqs >= 14000].mean()
        e_8k = avg_spec[(freqs >= 8000) & (freqs < 14000)].mean()
        return {"name": rec["path"].name, "site": rec["site"], "labeled": rec["labeled"],
                "cutoff_3db_hz": float(cutoff_3db),
                "cutoff_20db_hz": float(cutoff_20db),
                "e_14k_over_8k": float(e_14k / (e_8k + 1e-12))}
    except: return None

t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    hf_results = [r for r in ex.map(hf_cutoff, all_sample, chunksize=16) if r is not None]
print(f"done in {time.time()-t0:.0f}s")
hf_df = pd.DataFrame(hf_results)
print(f"\nPer-site cutoff stats (-3dB and -20dB corner):")
for s, sub in hf_df.groupby("site"):
    lab = sub[sub.labeled]; unl = sub[~sub.labeled]
    print(f"  {s}: n_lab={len(lab):2d} n_unl={len(unl):2d}  "
          f"all: cutoff_3dB={sub.cutoff_3db_hz.mean():.0f}Hz, "
          f"cutoff_20dB={sub.cutoff_20db_hz.mean():.0f}Hz, "
          f"e_14k/e_8k={sub.e_14k_over_8k.mean():.4f}")

# Distinct cutoff "modes"?
print("\nHistogram of -3dB cutoff (Hz):")
bins = [0, 4000, 6000, 8000, 10000, 12000, 14000, 16000]
for i in range(len(bins)-1):
    n = ((hf_df.cutoff_3db_hz >= bins[i]) & (hf_df.cutoff_3db_hz < bins[i+1])).sum()
    print(f"  [{bins[i]:5d}, {bins[i+1]:5d}): {n}")

# ---------- D19: Mains hum (50 / 60 Hz) ----------
section("D19: Mains hum detection (50 / 60 Hz)")
def hum_check(rec):
    try:
        x, sr = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        if x.size < 32000: return None
        n_fft = 32768  # bin resolution ~ 0.98 Hz
        if x.size < n_fft: return None
        # Use whole file
        spec = np.abs(np.fft.rfft(x[:n_fft] * np.hanning(n_fft)))**2
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        # Power at 50 Hz, 60 Hz, and harmonics
        def peak_at(target_hz, width_hz=2.0):
            mask = (freqs >= target_hz - width_hz) & (freqs <= target_hz + width_hz)
            band = spec[mask]
            return float(band.max()) if band.size else 0.0
        # Compare to noise floor at 100-200 Hz baseline
        nf = spec[(freqs >= 100) & (freqs <= 200)].mean()
        return {"name": rec["path"].name, "site": rec["site"], "labeled": rec["labeled"],
                "p50_over_floor": peak_at(50.0) / (nf + 1e-12),
                "p60_over_floor": peak_at(60.0) / (nf + 1e-12),
                "p100_over_floor": peak_at(100.0) / (nf + 1e-12),
                "p120_over_floor": peak_at(120.0) / (nf + 1e-12)}
    except: return None

t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    hum_results = [r for r in ex.map(hum_check, all_sample, chunksize=16) if r is not None]
print(f"done in {time.time()-t0:.0f}s")
hum_df = pd.DataFrame(hum_results)
print(f"Per-site mains-hum signature (50/60Hz peak relative to 100-200Hz noise floor):")
for s, sub in hum_df.groupby("site"):
    print(f"  {s}: p50={sub.p50_over_floor.mean():.2f}x, p60={sub.p60_over_floor.mean():.2f}x, "
          f"p100={sub.p100_over_floor.mean():.2f}x, p120={sub.p120_over_floor.mean():.2f}x")
# Files with strong hum
print(f"\nFiles with strong 50Hz hum (>5x noise floor): "
      f"{(hum_df.p50_over_floor > 5).sum()}/{len(hum_df)}")

# ---------- D21: train_audio Pantanal clips vs train_soundscape date overlaps ----------
section("D21: train_audio Pantanal-box ↔ train_soundscape date matching")
tr = pd.read_csv(DATA / "train.csv")
# Inside Pantanal box per recording_location.txt
pan = tr[(tr.latitude.between(-21.6, -16.5)) & (tr.longitude.between(-57.6, -55.9))].copy()
print(f"train_audio inside Pantanal box: {len(pan)}")

# Check if iNat IDs ranges suggest specific upload dates
# Big iNat IDs (>1M) = recent (2021+)
pan["fn"] = pan["filename"].str.split("/").str[-1]
inat_pan = pan[pan.collection == "iNat"].copy()
inat_pan["iNat_id"] = inat_pan["fn"].str.extract(r"iNat(\d+)\.ogg").astype(float)
print(f"  Pantanal-box iNat clips: {len(inat_pan)}")
print(f"    iNat ID range: {int(inat_pan.iNat_id.min()) if len(inat_pan) else 0} - "
      f"{int(inat_pan.iNat_id.max()) if len(inat_pan) else 0}")

# What are the most common authors in Pantanal AND when do their iNat IDs suggest they uploaded?
top_auth = pan["author"].value_counts().head(10)
print(f"  Top 10 authors in Pantanal-box clips: {dict(top_auth)}")

# CROSS-CHECK: are train_soundscape dates (2021-2025) within range that some Pantanal recordings were uploaded?
ss_dates = sorted(set(r["date"] for r in all_files))
print(f"  train_soundscape date range: {min(ss_dates)} - {max(ss_dates)}")

# ---------- D22 / D23: Test sample row deep dive ----------
section("D22: Test-row format analysis from sample_submission")
sub = pd.read_csv(DATA / "sample_submission.csv")
print(f"sample_submission rows: {len(sub)}")
print(sub.head().to_string())

# What does recording_location.txt say?
print("\nrecording_location.txt:")
print((DATA / "recording_location.txt").read_text())

# What about the test_soundscapes/readme.txt?
print("\ntest_soundscapes/readme.txt:")
print((DATA / "test_soundscapes" / "readme.txt").read_text())

# S05 4 files content: what species are detected?
print("\nS05 train_soundscapes detail (idx 84-92):")
s05 = [r for r in all_files if r["site"] == "S05"]
s05.sort(key=lambda r: r["idx"])
for r in s05:
    print(f"  idx={r['idx']} {r['date']} {r['time']}")

# ---------- D24: Hand-inspect each of the 41 same-(site,date,time) collisions ----------
section("D24: All 41 same-(site,date,time) collisions — multi-mic or single-source?")
# Group all_files by (site, date, time)
by_key = defaultdict(list)
for r in all_files:
    by_key[(r["site"], r["date"], r["time"])].append(r)
colls = [(k, v) for k, v in by_key.items() if len(v) > 1]
print(f"unique (site, date, time) with ≥2 files: {len(colls)}")

# For each, list members + idx delta
for k, members in sorted(colls):
    idxs = sorted(r["idx"] for r in members)
    delta = max(idxs) - min(idxs)
    n_lab = sum(1 for r in members if r["labeled"])
    print(f"  {k}: n={len(members)} idxs={idxs} delta={delta} n_lab={n_lab}")

print("\n[forensic_round3 done]")
