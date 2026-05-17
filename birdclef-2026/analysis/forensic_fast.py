"""Targeted forensic — just the high-value checks. Output streamed to FINDINGS_FAST.md."""
import os, sys, re, time, json, hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, pandas as pd, soundfile as sf
from concurrent.futures import ProcessPoolExecutor

DATA = Path("/home/user/opencode/birdclef-2026/data")
OUT  = Path("/home/user/opencode/birdclef-2026/analysis/forensic"); OUT.mkdir(parents=True, exist_ok=True)

def section(title):
    print("\n" + "="*60); print(title); print("="*60); sys.stdout.flush()

# ------- 1) Hash first 4s of every TRAIN_SOUNDSCAPE -------
section("1) Compute first-4s spectral fingerprints for all train_soundscapes")
ss_dir = DATA / "train_soundscapes"
files = sorted(ss_dir.glob("*.ogg"))
print(f"Files: {len(files)}")
t0 = time.time()

def fingerprint(fp):
    try:
        x, sr = sf.read(str(fp), frames=4*32000, dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
        # 16-band Haar fingerprint, 32 frames
        n_fft = 2048; hop = 1024
        nframes = min(32, (len(x) - n_fft) // hop + 1)
        if nframes < 4: return (str(fp.relative_to(DATA)), None, x.size)
        frames = np.stack([x[i*hop:i*hop+n_fft] * np.hanning(n_fft) for i in range(nframes)])
        spec = np.abs(np.fft.rfft(frames, axis=1))**2
        bins = np.linspace(0, spec.shape[1], 17, dtype=int)
        mel = np.stack([spec[:, b:bins[i+1]].sum(axis=1) for i, b in enumerate(bins[:-1])], axis=1)
        mel = np.log(mel + 1e-9)
        sig = (mel[:, 1:] > mel[:, :-1]).astype(np.uint8)
        h = hashlib.md5(sig.tobytes()).hexdigest()
        return (str(fp.relative_to(DATA)), h, x.size)
    except Exception:
        return (str(fp.relative_to(DATA)), "ERR", 0)

with ProcessPoolExecutor(max_workers=4) as ex:
    rows = list(ex.map(fingerprint, files, chunksize=32))
df = pd.DataFrame(rows, columns=["path","fp","n_samples"])
df.to_csv(OUT / "fp_train_soundscapes.csv.gz", index=False, compression="gzip")
print(f"Done in {time.time()-t0:.0f}s; collisions: {len(df) - df['fp'].nunique()}")
sys.stdout.flush()

# Same fingerprint = same first 4s of audio content
groups = df.groupby("fp").filter(lambda g: len(g) > 1)
print(f"Files in fingerprint collision: {len(groups)} ({groups['fp'].nunique()} groups)")
# Top duplicates
top_groups = df.groupby("fp").size().sort_values(ascending=False)
print(f"top 10 group sizes: {top_groups.head(10).tolist()}")

# ------- 2) Cross-check fingerprint vs labeled set -------
section("2) Do LABELED files share a fingerprint with UNLABELED files?")
ss_lbl = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
labeled = set(ss_lbl["filename"].unique())
df["name"] = df["path"].str.rsplit("/", n=1).str[-1]
df["is_labeled"] = df["name"].isin(labeled)
print(f"labeled with fp: {df['is_labeled'].sum()}, total: {len(df)}")
labeled_fps = set(df[df.is_labeled]["fp"])
labeled_fps.discard("ERR")
overlap_unlab = df[~df["is_labeled"] & df["fp"].isin(labeled_fps)]
print(f"UNLABELED files sharing first-4s fingerprint with a LABELED file: {len(overlap_unlab)}")
if len(overlap_unlab):
    print("Examples:")
    for fp, sub in overlap_unlab.groupby("fp"):
        lab_match = df[(df["fp"]==fp) & (df["is_labeled"])]["name"].tolist()
        print(f"  fp={fp[:8]}  unlabeled={sub['name'].tolist()[:3]}  matches labeled={lab_match}")
        if len(overlap_unlab) > 20: break

# ------- 3) Same fingerprint between train_audio and train_soundscapes? -------
section("3) Does any TRAIN_AUDIO clip have the same first-4s fp as a train_soundscape?")
ta_files = list((DATA / "train_audio").rglob("*.ogg"))
print(f"Hashing {len(ta_files)} train_audio files...")
t0 = time.time()
with ProcessPoolExecutor(max_workers=4) as ex:
    ta_rows = list(ex.map(fingerprint, ta_files, chunksize=128))
ta_df = pd.DataFrame(ta_rows, columns=["path","fp","n_samples"])
ta_df.to_csv(OUT / "fp_train_audio.csv.gz", index=False, compression="gzip")
print(f"Done in {time.time()-t0:.0f}s")
ss_fp_set = set(df["fp"]); ss_fp_set.discard("ERR")
cross = ta_df[ta_df["fp"].isin(ss_fp_set) & (ta_df["fp"] != "ERR")]
print(f"train_audio files whose first-4s fp matches a train_soundscape: {len(cross)}")
if len(cross):
    cross["match_ss"] = cross["fp"].map(lambda f: df[df.fp==f]["name"].head(3).tolist())
    print(cross.head(20).to_string(index=False))

# ------- 4) Check sample_submission for hidden info -------
section("4) sample_submission deep look")
sub = pd.read_csv(DATA / "sample_submission.csv")
print(f"shape: {sub.shape}, columns ({len(sub.columns)}):")
print(f"  first 5 cols: {list(sub.columns[:5])}")
print(f"  row_ids: {sub.row_id.tolist()}")
# Column-order vs taxonomy order
tax = pd.read_csv(DATA / "taxonomy.csv")
tax_order = tax["primary_label"].astype(str).tolist()
sub_order = [c for c in sub.columns if c != "row_id"]
print(f"\ncolumn order == taxonomy.primary_label order? {tax_order == sub_order}")
print(f"taxonomy order first 8: {tax_order[:8]}")
print(f"submission order first 8: {sub_order[:8]}")

# ------- 5) Check the 41 (site,date,time) collisions exhaustively -------
section("5) All exact (site, date, time) collisions in train_soundscapes")
PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
by_key = defaultdict(list)
for fp in files:
    m = PAT.match(fp.name)
    if m:
        by_key[(m.group(2), m.group(3), m.group(4))].append((int(m.group(1)), fp.name))
coll = [(k, v) for k, v in by_key.items() if len(v) > 1]
print(f"unique (site, date, time) keys with >1 file: {len(coll)}")
for k, v in coll[:20]:
    print(f"  {k}: {sorted(v)[:5]}")
# Are these all at the same MINUTE — i.e. multiple recorders OR same file repackaged?
# Show fingerprint relationships
coll_fps = []
for k, v in coll:
    fps = []
    for idx, name in v:
        row = df[df.name == name]
        if len(row):
            fps.append((idx, name, row["fp"].iloc[0]))
    unique_fps = set(fp for _, _, fp in fps if fp != "ERR")
    coll_fps.append({"site": k[0], "date": k[1], "time": k[2],
                     "n_files": len(v), "n_unique_fps": len(unique_fps),
                     "indices": sorted(idx for idx,_,_ in fps)})
ck = pd.DataFrame(coll_fps)
ck.to_csv(OUT / "site_date_time_collisions.csv", index=False)
print(f"\nCollisions with all files having SAME fingerprint: {(ck.n_unique_fps == 1).sum()}/{len(ck)}")
print(f"  (same fp = byte/spectral identical first 4s)")

print("\n[forensic_fast done]")
