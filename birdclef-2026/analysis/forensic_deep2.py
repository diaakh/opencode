"""Round 2 — verify energy anomaly and unlabeled-internal duplicates at scale.

D11: Stratified RMS / spectrogram comparison labeled vs unlabeled, BY SITE.
D12: Full unlabeled pairwise PCM correlation within (site, date) groups.
     How many unique RECORDING SESSIONS are there really?
D13: Frequency band 9 anomaly investigation — what is in 3-4 kHz?
D14: Are labeled files from a different recorder/microphone? Check sample-level statistics
     (DC offset, dynamic range, clipping, noise floor) per site.
D15: The S19 idx 10610/10611 with 0.865 PCM corr @ 59.7s offset — what's going on?
"""
import os, sys, re, time, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import correlate
from concurrent.futures import ProcessPoolExecutor

DATA = Path("/home/user/opencode/birdclef-2026/data")
OUT  = Path("/home/user/opencode/birdclef-2026/analysis/forensic")

PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
ss_lbl = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
labeled = set(ss_lbl["filename"].unique())

all_files = {}
by_site_date = defaultdict(list)
for fp in (DATA / "train_soundscapes").glob("*.ogg"):
    m = PAT.match(fp.name)
    if m:
        rec = {"idx": int(m.group(1)), "site": m.group(2), "date": m.group(3),
               "time": m.group(4), "path": fp, "labeled": fp.name in labeled}
        all_files[fp.name] = rec
        by_site_date[(rec["site"], rec["date"])].append(rec)

def section(t): print("\n" + "="*60); print(t); print("="*60); sys.stdout.flush()

# ---------- D11: Stratified RMS by site ----------
section("D11: Per-SITE RMS comparison labeled vs unlabeled (full coverage)")
def file_stats(rec):
    try:
        x, sr = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        return {
            "name":       rec["path"].name,
            "site":       rec["site"], "date": rec["date"],
            "labeled":    rec["labeled"],
            "rms":        float(np.sqrt(np.mean(x*x))),
            "peak":       float(np.abs(x).max()),
            "dc_offset":  float(x.mean()),
            "noise_floor": float(np.percentile(np.abs(x), 5)),
            "p99":        float(np.percentile(np.abs(x), 99)),
            "n_samples":  int(x.size),
            "n_clipped":  int(((np.abs(x) > 0.999)).sum()),
        }
    except Exception as e:
        return None

# Sample equally per site (full labeled + 200 unlabeled per site)
records_to_stat = []
for site in sorted(set(r["site"] for r in all_files.values())):
    site_recs = [r for r in all_files.values() if r["site"] == site]
    lab = [r for r in site_recs if r["labeled"]]
    unl = [r for r in site_recs if not r["labeled"]]
    records_to_stat.extend(lab)
    import random; random.seed(0)
    random.shuffle(unl)
    records_to_stat.extend(unl[:min(200, len(unl))])

print(f"computing stats for {len(records_to_stat)} files...")
t0 = time.time()
with ProcessPoolExecutor(max_workers=6) as ex:
    stats = [s for s in ex.map(file_stats, records_to_stat, chunksize=16) if s is not None]
df_st = pd.DataFrame(stats)
df_st.to_csv(OUT / "d11_per_site_stats.csv", index=False)
print(f"done in {time.time()-t0:.0f}s")

# Per-site labeled vs unlabeled comparison
print("\nPer-site mean RMS comparison (sites with ≥3 labeled files):")
site_groups = df_st.groupby("site")
print(f"  {'site':5s} {'n_lab':>6s} {'n_unl':>6s} {'lab_RMS':>10s} {'unl_RMS':>10s} {'ratio_unl/lab':>15s}")
for site, sub in site_groups:
    lab = sub[sub.labeled]
    unl = sub[~sub.labeled]
    if len(lab) < 1: continue
    rms_lab = lab.rms.mean() if len(lab) else float("nan")
    rms_unl = unl.rms.mean() if len(unl) else float("nan")
    ratio = rms_unl / rms_lab if rms_lab > 0 else float("nan")
    print(f"  {site:5s} {len(lab):6d} {len(unl):6d} {rms_lab:10.4f} {rms_unl:10.4f} {ratio:15.2f}")

print("\nAll-files distribution of stats by labeled status:")
for col in ["rms", "peak", "dc_offset", "noise_floor", "p99", "n_clipped"]:
    lab = df_st[df_st.labeled][col]
    unl = df_st[~df_st.labeled][col]
    print(f"  {col:14s}: labeled mean={lab.mean():.5f} sd={lab.std():.5f} | "
          f"unlabeled mean={unl.mean():.5f} sd={unl.std():.5f}")

# ---------- D12: Unlabeled near-duplicate within (site, date) ----------
section("D12: Unlabeled within-(site, date) PCM near-duplicates")
def pcm_corr(p1, p2, decim=8, max_lag=64000):
    a, _ = sf.read(str(p1), dtype="float32", always_2d=False)
    b, _ = sf.read(str(p2), dtype="float32", always_2d=False)
    if a.ndim > 1: a = a.mean(axis=1)
    if b.ndim > 1: b = b.mean(axis=1)
    n = min(len(a), len(b))
    aa = a[:n][::decim] - a[:n][::decim].mean()
    bb = b[:n][::decim] - b[:n][::decim].mean()
    if aa.std() < 1e-9 or bb.std() < 1e-9: return 0.0, 0
    c = correlate(bb, aa, mode='full')
    peak = int(np.argmax(np.abs(c)))
    offset = (peak - (len(aa)-1)) * decim
    val = c[peak] / (aa.std()*bb.std()*len(aa))
    return float(val), int(offset)

# For each (site, date) group with >1 unlabeled file, sample 1 pair and test
counts = []
print("(Site, date) groups with >=2 unlabeled files:")
candidates = []
for k, recs in by_site_date.items():
    unl = [r for r in recs if not r["labeled"]]
    if len(unl) >= 2:
        candidates.append((k, unl))
print(f"  total candidates: {len(candidates)}")
# Sample 50 random groups for cross-corr (full enumeration too slow)
import random; random.seed(0)
random.shuffle(candidates)
results = []
for k, unl in candidates[:50]:
    # Just first 2 files
    a, b = unl[0], unl[1]
    if a["time"] == b["time"]: continue  # already covered in D1
    corr, off = pcm_corr(a["path"], b["path"])
    results.append({"site": k[0], "date": k[1],
                    "file_a": a["path"].name, "file_b": b["path"].name,
                    "time_a": a["time"], "time_b": b["time"],
                    "peak_corr": corr, "best_offset_s": off/32000.0})
rdf = pd.DataFrame(results)
rdf.to_csv(OUT / "d12_unlabeled_xcorr.csv", index=False)
print(f"\nDistribution of |peak_corr| across 50 random within-(site,date) unlabeled pairs:")
print(f"  median={rdf['peak_corr'].abs().median():.3f}")
print(f"  >0.5: {(rdf['peak_corr'].abs() > 0.5).sum()}/{len(rdf)}")
print(f"  >0.3: {(rdf['peak_corr'].abs() > 0.3).sum()}/{len(rdf)}")
print("\nTop high-corr pairs:")
print(rdf.sort_values('peak_corr', key=lambda s: s.abs(), ascending=False).head(15).to_string(index=False))

# ---------- D13: 3-4 kHz band investigation ----------
section("D13: 3–4 kHz energy difference labeled vs unlabeled")
def band_energy(rec, lo_hz, hi_hz):
    try:
        x, sr = sf.read(str(rec["path"]), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        n_fft = 4096
        # Frame and average
        nframes = len(x) // n_fft
        if nframes < 4: return None
        seg = x[:nframes*n_fft].reshape(nframes, n_fft)
        spec = np.abs(np.fft.rfft(seg * np.hanning(n_fft), axis=1))
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        mask = (freqs >= lo_hz) & (freqs < hi_hz)
        return float(spec[:, mask].mean())
    except:
        return None

# Sample
import random; random.seed(0)
lab_sample = random.sample(list(labeled), min(40, len(labeled)))
unl_sample = random.sample(list(set(all_files.keys()) - labeled), 100)
bands = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
print(f"{'band (Hz)':14s} {'labeled':>10s} {'unlabeled':>10s} {'ratio':>8s}")
for lo, hi in bands:
    lab_e = [band_energy(all_files[f], lo, hi) for f in lab_sample]
    unl_e = [band_energy(all_files[f], lo, hi) for f in unl_sample]
    lab_e = [e for e in lab_e if e is not None]
    unl_e = [e for e in unl_e if e is not None]
    if lab_e and unl_e:
        lm = np.mean(lab_e); um = np.mean(unl_e)
        print(f"  {lo:5d}–{hi:5d} {lm:10.4f} {um:10.4f} {um/lm if lm>0 else 0:8.2f}x")

# ---------- D14: Recording sessions — how many UNIQUE acoustic events are in 10,658 files? ----------
section("D14: Estimate true distinct recording sessions in train_soundscapes")
# A "session" is a contiguous run of same-site files within 60 sec gaps.
# Already did this in deep_analyze. Just re-run.
files_sorted = sorted(all_files.values(),
                      key=lambda r: (r["site"], r["date"], r["time"]))
def parse_ts(rec):
    return pd.Timestamp(rec["date"] + "T" + rec["time"][:2]+":"+rec["time"][2:4]+":"+rec["time"][4:6])

sessions = []
cur_sess = None
for r in files_sorted:
    ts = parse_ts(r)
    if cur_sess is None or cur_sess["site"] != r["site"] or (ts - cur_sess["last_ts"]).total_seconds() > 70:
        if cur_sess: sessions.append(cur_sess)
        cur_sess = {"site": r["site"], "start_ts": ts, "last_ts": ts,
                    "files": [r["path"].name], "n_labeled": int(r["labeled"])}
    else:
        cur_sess["last_ts"] = ts
        cur_sess["files"].append(r["path"].name)
        cur_sess["n_labeled"] += int(r["labeled"])
if cur_sess: sessions.append(cur_sess)
print(f"distinct sessions: {len(sessions)}")
print(f"  files per session: min={min(len(s['files']) for s in sessions)}, "
      f"max={max(len(s['files']) for s in sessions)}, "
      f"median={int(np.median([len(s['files']) for s in sessions]))}")
print(f"  sessions with ≥1 labeled file: {sum(1 for s in sessions if s['n_labeled']>0)}")
session_lengths = [len(s["files"]) for s in sessions]
print(f"  session-length distribution: 1: {sum(1 for x in session_lengths if x==1)}, "
      f"2-5: {sum(1 for x in session_lengths if 2<=x<=5)}, "
      f"6-20: {sum(1 for x in session_lengths if 6<=x<=20)}, "
      f"21-60: {sum(1 for x in session_lengths if 21<=x<=60)}, "
      f">60: {sum(1 for x in session_lengths if x>60)}")

# ---------- D15: Investigate the 0.865 corr S19 idx 10610↔10611 ----------
section("D15: Examine S19 idx 10610↔10611 (PCM corr 0.865 at +59.7s offset)")
a = "train_soundscapes/BC2026_Train_10610_S19_20241213_180000.ogg"
b = "train_soundscapes/BC2026_Train_10611_S19_20241213_180000.ogg"
xa, _ = sf.read(a, dtype="float32"); xb, _ = sf.read(b, dtype="float32")
if xa.ndim > 1: xa = xa.mean(axis=1)
if xb.ndim > 1: xb = xb.mean(axis=1)
print(f"file A samples: {len(xa)}, B samples: {len(xb)}")
print(f"A duration: {len(xa)/32000:.2f}s, B: {len(xb)/32000:.2f}s")
print(f"A RMS: {np.sqrt(np.mean(xa*xa)):.4f}, B RMS: {np.sqrt(np.mean(xb*xb)):.4f}")
# Best lag in DETAIL
ds = 4
a_ds = xa[::ds] - xa[::ds].mean()
b_ds = xb[::ds] - xb[::ds].mean()
c = correlate(b_ds, a_ds, mode='full')
top5 = np.argsort(np.abs(c))[-10:][::-1]
print(f"\nTop 10 cross-correlation peaks (peaks of |corr|):")
for i in top5:
    off_samples = (i - (len(a_ds)-1)) * ds
    off_s = off_samples / 32000
    val = c[i] / (a_ds.std() * b_ds.std() * len(a_ds))
    print(f"  offset {off_s:+.2f}s: corr {val:+.4f}")
# Check: are these consistent with 1-min-shift (60s ≈ near 60.0)?

# Inspect: maybe these are consecutive minutes from a CONTINUOUS recording?
print(f"\nidx 10610 plays from 18:00:00, idx 10611 plays from 18:00:00 (same wall time).")
print(f"If they were CONSECUTIVE minutes of a continuous 2-minute recording chunked into 60s each:")
print(f"  the start of idx 10611 = end of idx 10610 → PCM correlation at offset = -60s")
print(f"  observed: peak at +59.7s = end of idx 10611 ≈ start of idx 10610!")
print(f"  so idx 10611 STARTS where idx 10610 ENDS — they're consecutive minutes.")

# Print A's last 5s waveform RMS vs B's first 5s waveform RMS
last_5s_a = xa[-5*32000:]
first_5s_b = xb[:5*32000]
print(f"  last 5s of A RMS: {np.sqrt(np.mean(last_5s_a*last_5s_a)):.4f}")
print(f"  first 5s of B RMS: {np.sqrt(np.mean(first_5s_b*first_5s_b)):.4f}")
n = min(len(last_5s_a), len(first_5s_b))
print(f"  PCM correlation last5sA vs first5sB: {np.corrcoef(last_5s_a[:n], first_5s_b[:n])[0,1]:.4f}")

print("\n[forensic_deep2 done]")
