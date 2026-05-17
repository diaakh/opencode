"""Windowed fingerprint analysis: find audio overlap between labeled and unlabeled
files even when they're at different time offsets within the same 60-s clip."""
import os, re, time, hashlib, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, soundfile as sf
from concurrent.futures import ProcessPoolExecutor

DATA = Path("/home/user/opencode/birdclef-2026/data")
OUT  = Path("/home/user/opencode/birdclef-2026/analysis/forensic"); OUT.mkdir(parents=True, exist_ok=True)

ss_dir = DATA / "train_soundscapes"
labeled = set(pd.read_csv(DATA / "train_soundscapes_labels.csv")
              .drop_duplicates()["filename"].unique())
print(f"labeled: {len(labeled)}")

# Index files by (site, date) so we only compare within-day
PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
by_site_date = defaultdict(list)
for fp in ss_dir.glob("*.ogg"):
    m = PAT.match(fp.name)
    if m:
        by_site_date[(m.group(2), m.group(3))].append({
            "idx": int(m.group(1)), "site": m.group(2), "date": m.group(3),
            "time": m.group(4), "path": fp, "labeled": fp.name in labeled})

# Statistics on within-day file counts
counts = [len(v) for v in by_site_date.values()]
print(f"(site, date) groups: {len(by_site_date)}; files per group: "
      f"median={np.median(counts):.0f}, max={max(counts)}, mean={np.mean(counts):.1f}")
# Groups with labeled file
labeled_groups = {k: v for k, v in by_site_date.items() if any(f["labeled"] for f in v)}
print(f"groups containing a labeled file: {len(labeled_groups)}")

# For each labeled file, compute 6 windowed fingerprints (offsets 0,10,20,30,40,50 sec)
# Then compare against same-(site,date) unlabeled files at all their offsets.

def windowed_fps(fp_path):
    """Return list of (offset_s, fp_hash) for 6 non-overlapping 10-s windows of the 60s clip."""
    try:
        x, sr = sf.read(str(fp_path), dtype="float32", always_2d=False)
        if x.ndim > 1: x = x.mean(axis=1)
    except Exception:
        return []
    out = []
    win_samples = 10 * 32000   # 10-sec windows
    n_fft = 2048; hop = 1024
    for off_s in range(0, 51, 10):  # 0,10,20,30,40,50 sec → 6 windows
        seg = x[off_s*32000 : off_s*32000 + win_samples]
        if seg.size < n_fft:
            continue
        # STFT on this 10-sec window
        nframes = (seg.size - n_fft) // hop + 1
        frames = np.stack([seg[i*hop:i*hop+n_fft] * np.hanning(n_fft) for i in range(nframes)])
        spec = np.abs(np.fft.rfft(frames, axis=1))**2
        bins = np.linspace(0, spec.shape[1], 17, dtype=int)
        mel = np.stack([spec[:, b:bins[i+1]].sum(axis=1) for i, b in enumerate(bins[:-1])], axis=1)
        mel = np.log(mel + 1e-9)
        sig = (mel[:, 1:] > mel[:, :-1]).astype(np.uint8)
        h = hashlib.md5(sig.tobytes()).hexdigest()
        out.append((off_s, h, mel.mean(axis=0)))
    return out

# Compute for all files within groups that have at least one labeled file
files_to_hash = []
for k, group in labeled_groups.items():
    files_to_hash.extend(group)
print(f"files to hash (windowed): {len(files_to_hash)}")

t0 = time.time()
def _process(rec):
    return rec["path"].name, windowed_fps(rec["path"])

with ProcessPoolExecutor(max_workers=6) as ex:
    results = dict(ex.map(_process, files_to_hash, chunksize=8))
print(f"hashing done in {time.time()-t0:.0f}s")

# For each labeled file in each group, search for windowed fp matches
# within the SAME (site, date) group's unlabeled files.
matches = []
for k, group in labeled_groups.items():
    lab = [f for f in group if f["labeled"]]
    unlab = [f for f in group if not f["labeled"]]
    if not lab or not unlab:
        continue
    for lf in lab:
        lab_fps = results.get(lf["path"].name, [])
        if not lab_fps: continue
        # Build mel-vec map for labeled file
        for uf in unlab:
            unlab_fps = results.get(uf["path"].name, [])
            if not unlab_fps: continue
            # Exact hash matches
            lab_hashes = {h for _, h, _ in lab_fps}
            shared_hashes = lab_hashes & {h for _, h, _ in unlab_fps}
            # Mel-vector cosine similarity per offset pair (look for near-matches)
            best_sim = -1; best_pair = None
            for lo, lh, lv in lab_fps:
                for uo, uh, uv in unlab_fps:
                    den = (np.linalg.norm(lv) * np.linalg.norm(uv) + 1e-9)
                    sim = float(np.dot(lv, uv) / den)
                    if sim > best_sim:
                        best_sim = sim; best_pair = (lo, uo)
            matches.append({"labeled": lf["path"].name, "unlabeled": uf["path"].name,
                            "site": k[0], "date": k[1],
                            "labeled_time": lf["time"], "unlabeled_time": uf["time"],
                            "best_cosine_sim": best_sim,
                            "best_offset_pair": str(best_pair),
                            "exact_hash_matches": len(shared_hashes)})

df = pd.DataFrame(matches)
df.to_csv(OUT / "labeled_unlabeled_overlap.csv", index=False)
print(f"\nlabeled↔unlabeled comparisons in same-(site,date) group: {len(df)}")
if len(df):
    hi = df[df["best_cosine_sim"] > 0.95]
    print(f"  pairs with cosine sim > 0.95: {len(hi)} of {len(df)}")
    print(f"  cosine sim percentiles: p50={df.best_cosine_sim.median():.3f} "
          f"p90={df.best_cosine_sim.quantile(.9):.3f} max={df.best_cosine_sim.max():.3f}")
    print("\nTop-20 best matches (likely audio overlap):")
    print(df.nlargest(20, "best_cosine_sim").to_string(index=False))
    print(f"\nLabeled files with ≥1 unlabeled near-twin (cos>0.9):")
    near = df[df.best_cosine_sim > 0.9].groupby("labeled").size().reset_index(name="n_twins")
    print(near.to_string(index=False))

# Also compare labeled vs labeled (within the same group) → are labeled files ALSO redundant?
print("\n=== Labeled-vs-labeled within-group similarity ===")
ll_matches = []
for k, group in labeled_groups.items():
    lab = [f for f in group if f["labeled"]]
    if len(lab) < 2: continue
    for i in range(len(lab)):
        for j in range(i+1, len(lab)):
            lab_fps_i = results.get(lab[i]["path"].name, [])
            lab_fps_j = results.get(lab[j]["path"].name, [])
            if not lab_fps_i or not lab_fps_j: continue
            best_sim = -1
            for lo, _, lv in lab_fps_i:
                for uo, _, uv in lab_fps_j:
                    sim = float(np.dot(lv, uv) / (np.linalg.norm(lv)*np.linalg.norm(uv)+1e-9))
                    if sim > best_sim: best_sim = sim
            ll_matches.append({"a": lab[i]["path"].name, "b": lab[j]["path"].name,
                               "best_sim": best_sim})
ll = pd.DataFrame(ll_matches)
if len(ll):
    print(f"  labeled-labeled pairs: {len(ll)}")
    print(f"  pairs with cos>0.9: {(ll.best_sim>0.9).sum()}")
    print(ll.head(15).to_string(index=False))
print("\n[forensic_overlap done]")
