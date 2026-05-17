"""Round 5 — angles nobody has explored.

R5a: train_audio center-of-energy — where is the actual call vs file start/middle/end?
R5b: iNat URL timestamp parsing — when did organizers download iNat data?
R5c: Recordist's lat/lon precision — can we map XC recordists to PAM site locations?
R5d: Are the 66 labeled files BirdNET-filtered? Run BirdNET-style energy detection on
     a sample of unlabeled and see if labeled files have higher event density.
R5e: secondary_labels — what's the actual cardinality distribution and clustering?
R5f: train_audio "first 5s vs middle 5s" energy comparison.
R5g: Specific check — what percentage of train_audio clips are <5 sec, between 5-10s, 10-30s, 30+?
"""
import os, sys, re, time, json, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor

DATA = Path("/home/user/opencode/birdclef-2026/data")
OUT  = Path("/home/user/opencode/birdclef-2026/analysis/forensic")

def section(t): print("\n" + "="*60); print(t); print("="*60); sys.stdout.flush()

# ---------- R5a: train_audio center-of-energy ----------
section("R5a: train_audio energy localization (where is the actual call?)")
def energy_localization(fp):
    """Find where in the clip the energy is concentrated."""
    try:
        x, sr = sf.read(str(fp), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        if x.size < sr: return None
        # Per-second RMS
        win = sr
        n_win = x.size // win
        if n_win < 1: return None
        rms_per_sec = np.array([np.sqrt(np.mean(x[i*win:(i+1)*win]**2)) for i in range(n_win)])
        peak_sec = int(np.argmax(rms_per_sec))
        # Energy center of mass
        weights = rms_per_sec + 1e-9
        com_sec = float(np.sum(weights * np.arange(n_win)) / weights.sum())
        return {
            "name": fp.name,
            "duration_s": float(x.size / sr),
            "peak_sec": peak_sec,
            "com_sec": com_sec,
            "peak_frac": peak_sec / max(n_win - 1, 1),
            "com_frac": com_sec / max(n_win - 1, 1),
            "first5_rms": float(np.sqrt(np.mean(x[:5*sr]**2))) if x.size >= 5*sr else float(np.sqrt(np.mean(x*x))),
            "middle5_rms": (
                float(np.sqrt(np.mean(x[(x.size//2-2*sr+sr//2):(x.size//2+2*sr+sr//2)]**2)))
                if x.size >= 5*sr else float(np.sqrt(np.mean(x*x)))
            ),
        }
    except Exception:
        return None

# Sample stratified across train_audio
random.seed(0)
ta_files = []
for d in (DATA / "train_audio").iterdir():
    if d.is_dir():
        files = list(d.glob("*.ogg"))
        random.shuffle(files)
        ta_files.extend(files[:10])  # 10 per class
print(f"analyzing energy localization on {len(ta_files)} train_audio files...")
t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    results = [r for r in ex.map(energy_localization, ta_files, chunksize=32) if r is not None]
print(f"done in {time.time()-t0:.0f}s")
edf = pd.DataFrame(results)
edf.to_csv(OUT / "r5a_energy_localization.csv.gz", index=False, compression="gzip")
print(f"\nPeak-second location distribution:")
print(edf["peak_frac"].describe())
print(f"\nCenter-of-mass location distribution:")
print(edf["com_frac"].describe())
print(f"\nFiles where the loudest second is in the FIRST QUARTER of the clip: "
      f"{(edf['peak_frac'] < 0.25).sum()} ({(edf['peak_frac'] < 0.25).mean()*100:.1f}%)")
print(f"Files where loudest second is in MIDDLE half: "
      f"{((edf['peak_frac'] >= 0.25) & (edf['peak_frac'] <= 0.75)).sum()} "
      f"({((edf['peak_frac'] >= 0.25) & (edf['peak_frac'] <= 0.75)).mean()*100:.1f}%)")
print(f"Files where loudest second is in LAST QUARTER: "
      f"{(edf['peak_frac'] > 0.75).sum()} ({(edf['peak_frac'] > 0.75).mean()*100:.1f}%)")
print(f"\nFirst-5s RMS vs Middle-5s RMS:")
print(f"  Mean first5  RMS: {edf['first5_rms'].mean():.4f}")
print(f"  Mean middle5 RMS: {edf['middle5_rms'].mean():.4f}")
# Ratio per clip
edf["middle_over_first"] = edf["middle5_rms"] / (edf["first5_rms"] + 1e-9)
print(f"  middle/first ratio: median={edf['middle_over_first'].median():.3f} mean={edf['middle_over_first'].mean():.3f}")
print(f"  clips where middle is louder than first: {(edf['middle_over_first'] > 1).sum()}/{len(edf)} "
      f"({(edf['middle_over_first'] > 1).mean()*100:.1f}%)")

# Per-duration-bucket: where is the call?
edf["dur_bucket"] = pd.cut(edf["duration_s"], bins=[0, 5, 10, 30, 60, 1000])
print(f"\nPeak fraction by duration bucket:")
print(edf.groupby("dur_bucket", observed=False)["peak_frac"].describe()[["mean","50%","count"]])

# ---------- R5b: iNat URL timestamp parsing ----------
section("R5b: iNat URL timestamps — when did organizers process them?")
tr = pd.read_csv(DATA / "train.csv")
inat = tr[tr["url"].str.contains("inaturalist", na=False)].copy()
print(f"iNat URLs: {len(inat)}")
inat["ts"] = inat["url"].str.extract(r"\?(\d+)$")[0].astype(float)
inat["dt"] = pd.to_datetime(inat["ts"], unit="s")
print(f"iNat URL timestamp range:")
print(f"  min: {inat['dt'].min()}")
print(f"  max: {inat['dt'].max()}")
print(f"\nTimestamp histogram by month:")
month_counts = inat["dt"].dt.to_period("M").value_counts().sort_index()
for m, n in month_counts.items():
    bar = "█" * (n // 200)
    print(f"  {m}: {n:5d}  {bar}")

# ---------- R5c: Recordist Pantanal location clustering ----------
section("R5c: Recordist lat/lon — XC clip locations that might match PAM sites")
pan = tr[(tr.latitude.between(-21.6, -16.5)) & (tr.longitude.between(-57.6, -55.9))]
print(f"Pantanal-box train_audio: {len(pan)}")
# Group by exact lat/lon, count
locs = pan.groupby(["latitude", "longitude"]).agg(
    n=("filename", "count"),
    authors=("author", lambda s: s.value_counts().head(2).to_dict()),
    species_n=("primary_label", "nunique"),
).reset_index().sort_values("n", ascending=False)
print(f"Distinct (lat, lon) clusters with ≥5 clips: {(locs.n >= 5).sum()}")
print(f"Top 20:")
print(locs.head(20).to_string(index=False))

# ---------- R5d: Energy density in labeled vs unlabeled soundscape ----------
section("R5d: Acoustic event density — are labeled files chosen for high call density?")
ss_lbl = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
labeled = set(ss_lbl["filename"].unique())

PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
all_ss = []
for fp in (DATA / "train_soundscapes").glob("*.ogg"):
    m = PAT.match(fp.name)
    if m: all_ss.append({"path": fp, "labeled": fp.name in labeled, "site": m.group(2)})

def event_density(rec):
    """Detect transient events via onset detection (simple energy onsets)."""
    try:
        x, sr = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        if x.size < sr: return None
        # Frame energy at 50ms hop
        hop = sr // 20  # 50 ms
        win = sr // 10  # 100 ms
        nframes = (x.size - win) // hop + 1
        if nframes < 5: return None
        rms = np.array([np.sqrt(np.mean(x[i*hop:i*hop+win]**2)) for i in range(nframes)])
        # Onset detection: count times where rms[i] > 2 * rms[i-1]
        rises = (rms[1:] > 2 * rms[:-1] + 0.005).sum()
        return {"name": rec["path"].name, "labeled": rec["labeled"], "site": rec["site"],
                "n_onsets": int(rises),
                "rms_max": float(rms.max()),
                "rms_p10": float(np.percentile(rms, 10)),
                "snr_proxy": float(rms.max() / (np.percentile(rms, 10) + 1e-9))}
    except Exception:
        return None

# Sample 66 labeled + 200 random unlabeled
unlab = [r for r in all_ss if not r["labeled"]]
random.shuffle(unlab)
sample = [r for r in all_ss if r["labeled"]] + unlab[:200]
t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    er = [r for r in ex.map(event_density, sample, chunksize=16) if r is not None]
print(f"done in {time.time()-t0:.0f}s")
e_df = pd.DataFrame(er)
print(f"\nEvent-onset count comparison:")
for k in ["n_onsets", "rms_max", "rms_p10", "snr_proxy"]:
    lab = e_df[e_df.labeled][k].mean()
    unl = e_df[~e_df.labeled][k].mean()
    print(f"  {k:14s}: labeled mean={lab:.3f}, unlabeled mean={unl:.3f}, ratio={unl/lab if lab>0 else 0:.2f}")

# ---------- R5e: secondary_labels deep ----------
section("R5e: secondary_labels per row cardinality and clustering")
import ast
def parse_sec(x):
    if pd.isna(x): return []
    s = str(x).strip()
    if s in ("[]", "", "nan"): return []
    try:
        if s.startswith("["): return [t.strip().strip("'\"") for t in ast.literal_eval(s)]
    except: pass
    return [t.strip() for t in s.split(";") if t.strip()]

tr["sec_list"] = tr["secondary_labels"].apply(parse_sec)
tr["sec_n"] = tr["sec_list"].apply(len)
print(f"sec_n histogram: {dict(tr['sec_n'].value_counts().sort_index())}")
# Which primary labels have the most secondary mentions?
sec_co = Counter()
for _, row in tr.iterrows():
    primary = str(row["primary_label"])
    for s in row["sec_list"]:
        sec_co[(primary, s)] += 1
print(f"\nTop 20 (primary, secondary) co-occurrence pairs:")
for (p, s), n in sec_co.most_common(20):
    print(f"  {p:14s} → {s:14s}: {n}")

# Are some species ALWAYS as secondary (never primary)? Check
all_primary = set(tr["primary_label"].astype(str))
all_secondary = set(s for l in tr["sec_list"] for s in l)
sec_only = all_secondary - all_primary
print(f"\nSecondary labels that are NEVER primary: {len(sec_only)}")
if sec_only:
    print(f"  examples: {list(sec_only)[:10]}")
prim_only = all_primary - all_secondary
print(f"Primary labels that NEVER appear as secondary: {len(prim_only)}")

# ---------- R5g: train_audio duration buckets ----------
section("R5g: train_audio duration distribution (re-run on FULL set)")
def get_dur(fp):
    try:
        return {"name": fp.name, "label": fp.parent.name,
                "frames": sf.info(str(fp)).frames}
    except: return None

print(f"Computing duration for ALL {sum(1 for d in (DATA/'train_audio').iterdir() if d.is_dir())} species...")
ta_all = list((DATA / "train_audio").rglob("*.ogg"))
print(f"  total files: {len(ta_all)}")
t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    dr = [r for r in ex.map(get_dur, ta_all, chunksize=128) if r is not None]
print(f"  done in {time.time()-t0:.0f}s")
ddf = pd.DataFrame(dr)
ddf["dur_s"] = ddf["frames"] / 32000
ddf.to_csv(OUT / "r5g_full_durations.csv.gz", index=False, compression="gzip")
print(f"\nDuration buckets:")
buckets = [(0, 5), (5, 10), (10, 30), (30, 60), (60, 300), (300, 9999)]
for lo, hi in buckets:
    n = ((ddf.dur_s >= lo) & (ddf.dur_s < hi)).sum()
    print(f"  [{lo:4d}, {hi:4d}) sec: {n:5d} ({n/len(ddf)*100:.1f}%)")
print(f"\nFull stats: min={ddf.dur_s.min():.2f} median={ddf.dur_s.median():.2f} "
      f"mean={ddf.dur_s.mean():.2f} max={ddf.dur_s.max():.2f}")

# Per-class duration totals (hours)
ddf["class"] = ddf["label"]
per_class = ddf.groupby("class")["dur_s"].agg(["sum", "count"])
per_class["hours"] = per_class["sum"] / 3600
print(f"\nTotal hours: {per_class['hours'].sum():.1f}")
print(f"Top 10 species by total hours of train_audio:")
print(per_class.nlargest(10, "hours").to_string())
print(f"\nBottom 10 species by total hours:")
print(per_class[per_class["count"] >= 1].nsmallest(10, "hours").to_string())

print("\n[forensic_round5 done]")
