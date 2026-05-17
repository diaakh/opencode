"""Deeper forensic — beyond the first pass. Targets:
  D1: PCM cross-correlation for ALL 41 same-(site,date,time) collisions
  D2: Shazam-style audio fingerprinting across all train_soundscapes
  D3: Cross-fingerprint train_audio vs train_soundscapes
  D4: Sample-submission test-date overlap with train_soundscapes
  D5: Energy / silence patterns vs labeled status
  D6: Per-class lat/lon precision and uniqueness
  D7: iNat ID range vs file-creation epoch
  D8: Perch-label leak — train.csv ↔ Perch known species
  D9: Are labeled segments OVER-CONCENTRATED on rare frequencies?
  D10: Spectral signature similarity at per-second resolution
"""
import os, sys, re, time, hashlib, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import correlate
from concurrent.futures import ProcessPoolExecutor

DATA = Path("/home/user/opencode/birdclef-2026/data")
OUT  = Path("/home/user/opencode/birdclef-2026/analysis/forensic")
OUT.mkdir(parents=True, exist_ok=True)

def section(t):
    print("\n" + "="*60); print(t); print("="*60); sys.stdout.flush()

PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

# Build (site,date,time)->files index
section("Building (site,date,time) index")
by_key = defaultdict(list)
all_files = {}
for fp in sorted((DATA / "train_soundscapes").glob("*.ogg")):
    m = PAT.match(fp.name)
    if m:
        idx, site, date, t = m.groups()
        rec = {"idx": int(idx), "site": site, "date": date, "time": t, "path": fp}
        by_key[(site, date, t)].append(rec)
        all_files[fp.name] = rec
ss_lbl = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
labeled = set(ss_lbl["filename"].unique())
print(f"unique (site, date, time) keys: {len(by_key)}; collisions: {sum(1 for v in by_key.values() if len(v)>1)}")

# ---------- D1: PCM cross-corr for ALL 41 same-(site,date,time) groups ----------
section("D1: PCM cross-correlation for ALL same-(site,date,time) collisions")
def pcm_xcorr(a_path, b_path, decim=8):
    a, sr = sf.read(a_path, dtype="float32", always_2d=False)
    b, _  = sf.read(b_path, dtype="float32", always_2d=False)
    if a.ndim > 1: a = a.mean(axis=1)
    if b.ndim > 1: b = b.mean(axis=1)
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    aa = a[::decim] - a[::decim].mean()
    bb = b[::decim] - b[::decim].mean()
    if aa.std() < 1e-9 or bb.std() < 1e-9:
        return {"peak_offset_s": 0.0, "peak_norm": 0.0, "n0": n}
    c = correlate(bb, aa, mode='full')
    peak_idx = int(np.argmax(np.abs(c)))
    peak_offset = (peak_idx - (len(aa)-1)) * decim
    peak_val = c[peak_idx] / (aa.std()*bb.std()*len(aa))
    return {"peak_offset_s": peak_offset/32000.0, "peak_norm": float(peak_val), "n0": n}

rows = []
for key, group in by_key.items():
    if len(group) < 2: continue
    site, date, t = key
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            a = group[i]; b = group[j]
            xc = pcm_xcorr(a["path"], b["path"])
            rows.append({
                "site": site, "date": date, "time": t,
                "idx_a": a["idx"], "idx_b": b["idx"],
                "a_labeled": a["path"].name in labeled,
                "b_labeled": b["path"].name in labeled,
                "peak_offset_s": xc["peak_offset_s"],
                "peak_norm_corr": xc["peak_norm"],
            })
d1 = pd.DataFrame(rows)
d1.to_csv(OUT / "d1_pcm_xcorr_collisions.csv", index=False)
print(f"pairs evaluated: {len(d1)}")
print(f"abs(peak_norm) summary: median={d1['peak_norm_corr'].abs().median():.3f}, "
      f"max={d1['peak_norm_corr'].abs().max():.3f}")
print(f"\nPairs with |peak_corr| > 0.5 (likely truly same audio):")
print(d1[d1['peak_norm_corr'].abs() > 0.5].sort_values('peak_norm_corr', key=lambda s: s.abs(), ascending=False).head(20).to_string(index=False))
print(f"\nLabeled↔unlabeled pairs only:")
mixed = d1[d1["a_labeled"] != d1["b_labeled"]]
print(f"  n={len(mixed)}, |peak_corr| max={mixed['peak_norm_corr'].abs().max():.3f}")
print(mixed.sort_values('peak_norm_corr', key=lambda s: s.abs(), ascending=False).head(10).to_string(index=False))

# ---------- D2: Shazam-style fingerprinting of train_soundscapes ----------
# Find spectrogram peak constellation pairs, hash, look for duplicates ANYWHERE in dataset
section("D2: Shazam-style constellation fingerprinting of train_soundscapes")

def shazam_hashes(fp, max_hashes=300):
    """Extract spectral peak pairs as hashes. Each hash is (f1, f2, dt) tuple."""
    try:
        x, sr = sf.read(str(fp), frames=30*32000, dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        if x.size < 4096: return set()
    except Exception:
        return set()
    n_fft = 2048; hop = 1024
    nframes = (x.size - n_fft) // hop + 1
    if nframes < 4: return set()
    frames = np.stack([x[i*hop:i*hop+n_fft] * np.hanning(n_fft) for i in range(nframes)])
    spec = np.log(np.abs(np.fft.rfft(frames, axis=1)) + 1e-9)
    # Find local peaks: stronger than 3x3 neighborhood
    from scipy.ndimage import maximum_filter
    peaks = (spec == maximum_filter(spec, size=(5, 7))) & (spec > np.percentile(spec, 90))
    peak_t, peak_f = np.where(peaks)
    # Limit number of peaks per frame
    peak_pairs = []
    for i in range(len(peak_t)):
        for j in range(i+1, min(i+8, len(peak_t))):
            dt = peak_t[j] - peak_t[i]
            if 1 <= dt <= 20:
                key = (int(peak_f[i] // 4), int(peak_f[j] // 4), int(dt))
                peak_pairs.append(key)
                if len(peak_pairs) >= max_hashes:
                    break
        if len(peak_pairs) >= max_hashes:
            break
    return set(peak_pairs)

# Run on all train_soundscapes
ss_paths = list((DATA / "train_soundscapes").glob("*.ogg"))
print(f"fingerprinting {len(ss_paths)} files (parallel)...")
t0 = time.time()
def _hash(fp):
    return fp.name, shazam_hashes(fp, max_hashes=200)
with ProcessPoolExecutor(max_workers=6) as ex:
    ss_hash_map = dict(ex.map(_hash, ss_paths, chunksize=16))
print(f"hashes computed in {time.time()-t0:.0f}s")

# Build inverted index hash -> [files]
inverted = defaultdict(list)
for fname, hashes in ss_hash_map.items():
    for h in hashes:
        inverted[h].append(fname)

# For each labeled file, find unlabeled files with most shared hashes
section("Labeled→unlabeled shared-hash matches")
match_rows = []
labeled_names = sorted(labeled)
for lab_name in labeled_names:
    lab_hashes = ss_hash_map.get(lab_name, set())
    if not lab_hashes: continue
    other_counts = Counter()
    for h in lab_hashes:
        for o in inverted.get(h, []):
            if o != lab_name:
                other_counts[o] += 1
    # Top 3 matches
    top = other_counts.most_common(3)
    for other, n_shared in top:
        if other in labeled: continue
        match_rows.append({
            "labeled": lab_name, "unlabeled": other,
            "n_shared_hashes": n_shared,
            "lab_total_hashes": len(lab_hashes),
            "share_pct": n_shared / max(len(lab_hashes), 1) * 100,
        })
mdf = pd.DataFrame(match_rows)
mdf.to_csv(OUT / "d2_shazam_lab_to_unlab.csv", index=False)
print(f"top labeled→unlabeled hash-overlap pairs (n={len(mdf)} retained):")
print(mdf.sort_values("n_shared_hashes", ascending=False).head(20).to_string(index=False))

# What's the BACKGROUND noise level of shared hashes? Compare to random pair
import random; random.seed(0)
random_pairs = [(random.choice(ss_paths).name, random.choice(ss_paths).name) for _ in range(500)]
bg_counts = []
for a, b in random_pairs:
    sa = ss_hash_map.get(a, set()); sb = ss_hash_map.get(b, set())
    if sa and sb:
        bg_counts.append(len(sa & sb))
print(f"\nbackground (random-pair) shared hash count: "
      f"median={np.median(bg_counts):.0f}, p95={np.percentile(bg_counts, 95):.0f}, max={max(bg_counts)}")

# ---------- D3: Cross-fingerprint train_audio vs train_soundscapes ----------
section("D3: Cross-fingerprint train_audio vs train_soundscapes")
# Pick subset of train_audio (1 per class to keep fast)
ta_paths = []
for d in sorted((DATA / "train_audio").iterdir()):
    if d.is_dir():
        files = list(d.glob("*.ogg"))
        if files: ta_paths.append((d.name, files[0]))
print(f"sampling 1 train_audio per species: {len(ta_paths)} files")
t0 = time.time()
ta_hash_map = {}
for cls, fp in ta_paths:
    ta_hash_map[fp.name] = shazam_hashes(fp, max_hashes=200)
print(f"done in {time.time()-t0:.0f}s")

# Compare to inverted soundscape index
ta_cross = []
for cls_fname, fp in ta_paths:
    ta_h = ta_hash_map.get(fp.name, set())
    if not ta_h: continue
    matches = Counter()
    for h in ta_h:
        for o in inverted.get(h, []):
            matches[o] += 1
    if matches:
        top, n = matches.most_common(1)[0]
        share = n / max(len(ta_h), 1) * 100
        ta_cross.append({"train_audio": fp.name, "class": cls_fname, "best_match_ss": top,
                         "n_shared": n, "ta_total": len(ta_h), "share_pct": share})
xdf = pd.DataFrame(ta_cross)
xdf.to_csv(OUT / "d3_ta_vs_ss_xfp.csv", index=False)
if len(xdf):
    print(f"top train_audio↔train_soundscapes hash overlaps:")
    print(xdf.sort_values("share_pct", ascending=False).head(15).to_string(index=False))
    print(f"\nbackground (random-pair) shared hash count (re-using bg): median={np.median(bg_counts):.0f}")

# ---------- D4: Sample_submission date overlap with train_soundscapes ----------
section("D4: Sample submission test-file date and site → overlap with train")
sub = pd.read_csv(DATA / "sample_submission.csv")
# row_id pattern: BC2026_Test_0001_S05_20250227_010002_5
example_meta = re.match(r"BC2026_Test_(\d+)_(S\d+)_(\d{8})_(\d{6})_(\d+)", sub["row_id"].iloc[0])
if example_meta:
    test_idx, test_site, test_date, test_time, test_end = example_meta.groups()
    print(f"Sample test row 1: idx={test_idx} site={test_site} date={test_date} time={test_time}")
    # Any train_soundscape file at this exact site+date?
    train_at_site_date = [v for v in all_files.values()
                          if v["site"] == test_site and v["date"] == test_date]
    print(f"train_soundscapes at SAME site={test_site}, date={test_date}: {len(train_at_site_date)}")
    if train_at_site_date:
        for v in train_at_site_date:
            print(f"  idx={v['idx']} time={v['time']} labeled={v['path'].name in labeled}")
    # Any train_soundscape at SAME site any date
    train_at_site = [v for v in all_files.values() if v["site"] == test_site]
    dates_at_site = Counter(v["date"] for v in train_at_site)
    print(f"\nAll dates at site {test_site} in train_soundscapes: {dict(dates_at_site)}")
    # Total file count at this site
    print(f"Total files at site {test_site}: {len(train_at_site)}")
    # Any labeled files at this site?
    print(f"Labeled files at site {test_site}: {sum(1 for v in train_at_site if v['path'].name in labeled)}")

# ---------- D5: Energy / silence patterns ----------
section("D5: Energy / silence vs labeled status")
import random; random.seed(0)
sample_labeled = list(labeled)[:30]
sample_unlabeled = random.sample(list(set(all_files.keys()) - labeled), 100)

def file_energy(fp):
    try:
        x, sr = sf.read(str(fp), frames=60*32000, dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        # 5-sec window RMS
        w = 5*32000
        rms_per_win = []
        for i in range(0, min(len(x), 12*w), w):
            seg = x[i:i+w]
            if seg.size > 0:
                rms_per_win.append(float(np.sqrt(np.mean(seg*seg))))
        return rms_per_win
    except:
        return None

lab_rms = []
unl_rms = []
for f in sample_labeled:
    e = file_energy(all_files[f]["path"])
    if e: lab_rms.append(e)
for f in sample_unlabeled:
    e = file_energy(all_files[f]["path"])
    if e: unl_rms.append(e)

print(f"sampled {len(lab_rms)} labeled / {len(unl_rms)} unlabeled")
print(f"per-window RMS — labeled: mean={np.mean([np.mean(r) for r in lab_rms]):.4f}, "
      f"variance across windows = {np.mean([np.std(r) for r in lab_rms]):.4f}")
print(f"per-window RMS — unlabel: mean={np.mean([np.mean(r) for r in unl_rms]):.4f}, "
      f"variance across windows = {np.mean([np.std(r) for r in unl_rms]):.4f}")

# ---------- D6: Per-class lat/lon dispersion ----------
section("D6: Each class's geographic spread")
tr = pd.read_csv(DATA / "train.csv")
tr["primary_label"] = tr["primary_label"].astype(str)
# For each species, how big is the lat/lon convex range?
sp_geo = tr.groupby("primary_label").agg(
    n=("filename","count"),
    lat_min=("latitude","min"), lat_max=("latitude","max"),
    lon_min=("longitude","min"), lon_max=("longitude","max"),
    unique_locs=("latitude", lambda s: s.nunique()),
).reset_index()
sp_geo["lat_span_deg"] = sp_geo["lat_max"] - sp_geo["lat_min"]
sp_geo["lon_span_deg"] = sp_geo["lon_max"] - sp_geo["lon_min"]
# Distance proxy: max lat-lon span in km (1 deg ≈ 111 km)
sp_geo["span_km_approx"] = (sp_geo["lat_span_deg"]**2 + sp_geo["lon_span_deg"]**2)**0.5 * 111
sp_geo.to_csv(OUT / "d6_species_geographic_spread.csv", index=False)
print(f"species with lat/lon range < 10km (essentially one location): "
      f"{(sp_geo['span_km_approx'] < 10).sum()}/{len(sp_geo)}")
print(f"species with lat/lon range < 100km: {(sp_geo['span_km_approx'] < 100).sum()}")
print("Most geographically-bound species (smallest span, n_clips>=5):")
print(sp_geo[sp_geo['n'] >= 5].nsmallest(10, "span_km_approx").to_string(index=False))

# ---------- D7: iNat ID range analysis ----------
section("D7: iNat ID range vs upload era")
inat_rows = tr[tr["collection"] == "iNat"]
inat_rows = inat_rows.copy()
inat_rows["iNat_id"] = inat_rows["filename"].str.extract(r"iNat(\d+)\.ogg")[0].astype(float)
print(f"iNat ID range: {int(inat_rows.iNat_id.min())} - {int(inat_rows.iNat_id.max())}")
# Group by 100k buckets, count
inat_rows["bucket"] = (inat_rows.iNat_id // 100000).astype(int) * 100000
buckets = inat_rows["bucket"].value_counts().sort_index()
print(f"iNat ID histogram (buckets of 100,000):")
for b, n in buckets.items():
    if n > 100:
        print(f"  {b}+: {n}")
# Per-class iNat ID statistics
print(f"\nSpecies with highest iNat IDs (most recently uploaded):")
sp_inat = inat_rows.groupby("primary_label")["iNat_id"].agg(["min","max","mean","count"])
print(sp_inat.nlargest(10, "mean").to_string())

# ---------- D8: Perch label space hint via train.csv ----------
section("D8: train.csv inat_taxon_id field — Perch knows what?")
print(tr["inat_taxon_id"].value_counts().head(20))
# Are any of the 28 missing classes' iNat IDs in train.csv? They should NOT be (since they're missing)
tax = pd.read_csv(DATA / "taxonomy.csv")
tax["primary_label"] = tax["primary_label"].astype(str)
missing28 = set(tax["primary_label"]) - set(tr["primary_label"])
missing_inat_ids = set(tax[tax["primary_label"].isin(missing28)]["inat_taxon_id"])
print(f"iNat IDs of the 28 missing classes: {sorted(missing_inat_ids)}")
print(f"  (47158 = Insecta — entire class; others are specific frog species)")

# ---------- D9: Frequency-band coverage of labeled segments ----------
section("D9: Frequency-band coverage / quiet-bin analysis")
# Take labeled segments, look at their mel-spectrogram concentration
# For each labeled segment, what frequency band dominates?
sample_labeled = list(labeled)[:20]
sample_unlabeled = random.sample(list(set(all_files.keys()) - labeled), 20)

def avg_mel(fp_path):
    try:
        x, _ = sf.read(str(fp_path), frames=60*32000, dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        n_fft = 2048; hop = 1024
        nframes = (x.size - n_fft) // hop + 1
        if nframes < 4: return None
        frames = np.stack([x[i*hop:i*hop+n_fft] * np.hanning(n_fft) for i in range(nframes)])
        spec = np.abs(np.fft.rfft(frames, axis=1))
        # 32 mel-ish bands
        bins = np.linspace(0, spec.shape[1], 33, dtype=int)
        mel = np.stack([spec[:, b:bins[i+1]].mean(axis=1) for i, b in enumerate(bins[:-1])], axis=1)
        return mel.mean(axis=0)
    except:
        return None

lab_specs = [avg_mel(all_files[f]["path"]) for f in sample_labeled]
unl_specs = [avg_mel(all_files[f]["path"]) for f in sample_unlabeled]
lab_specs = np.stack([s for s in lab_specs if s is not None])
unl_specs = np.stack([s for s in unl_specs if s is not None])
print(f"labeled spec mean (32 bands): {lab_specs.mean(axis=0).round(4)}")
print(f"unlabel spec mean (32 bands): {unl_specs.mean(axis=0).round(4)}")
# Most distinctive band
diff = lab_specs.mean(axis=0) - unl_specs.mean(axis=0)
print(f"max-difference band: {diff.argmax()} (val={diff.max():.4f}); min-difference: {diff.argmin()} (val={diff.min():.4f})")

# ---------- D10: secondary_labels — are they auto-generated? ----------
section("D10: secondary_labels analysis")
import ast
def parse_sec(x):
    if pd.isna(x): return []
    s = str(x).strip()
    if s in ("[]", "", "nan"): return []
    try:
        if s.startswith("["):
            return [t.strip().strip("'\"") for t in ast.literal_eval(s)]
    except: pass
    return [t.strip() for t in s.split(";") if t.strip()]

tr["sec_list"] = tr["secondary_labels"].apply(parse_sec)
sec_lens = tr["sec_list"].apply(len)
print(f"rows with sec labels: {(sec_lens > 0).sum()}")
print(f"max sec labels per row: {sec_lens.max()}")
# Are sec labels a STRICT subset of taxonomy?
all_sec = set(s for ss in tr["sec_list"] for s in ss)
tax_set = set(tax["primary_label"].astype(str))
outside_tax = all_sec - tax_set
print(f"sec labels OUTSIDE taxonomy: {len(outside_tax)}")
if outside_tax:
    print(f"  examples: {list(outside_tax)[:5]}")
# Are there sec labels matching the 28 missing classes?
in_missing = all_sec & missing28
print(f"sec labels in missing-28: {len(in_missing)}")

print("\n[forensic_deep done]")
