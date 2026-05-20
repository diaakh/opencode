"""Check if we (or public kernels) have predictions on more than labeled rows.

train_soundscapes_labels.csv has 739 rows from 66 files.
But train_soundscapes directory might have MANY MORE files (the unlabeled rest).
Public kernels might have cached predictions on the full set.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Check public kernel cache sizes — 708 rows is what we saw
public_caches = [
    "/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_arrays.npz",
    "/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_meta.parquet",
    "/tmp/pub_outputs/itshyao_birdclef-2026-s106-eos5-0949-safealign2/perch_cache/full_oof_meta_features.npz",
    "/tmp/pub_outputs/itshyao_birdclef-2026-s106-eos5-0949-safealign2/perch_cache/full_perch_arrays.npz",
]

# Inspect meta in detail
m = pd.read_parquet("/tmp/pub_outputs/safar1_lb-score-0-948/cache/perch_meta.parquet")
print(f"safar1 meta: {m.shape}")
print(f"unique files: {m['filename'].nunique()}")
print(f"sites: {m['site'].value_counts().to_dict()}")
print(f"hours: {sorted(m['hour_utc'].unique())}")
print(f"\nrows per file distribution: {m.groupby('filename').size().describe()}")
print(f"\nfirst 10 unique files:")
print(m['filename'].drop_duplicates().head(10).tolist())

# Compare to our labels
ex = np.load("/home/user/opencode/birdclef-2026/analysis/entropy_tta/exp019_aligned.npz", allow_pickle=True)
row_fn = ex["row_filename"]
labels_unique_fn = set(str(f) for f in row_fn)
print(f"\nOur labels unique files: {len(labels_unique_fn)}")
print(f"Their meta unique files: {m['filename'].nunique()}")

# Files in their cache but NOT in our labels
their_files = set(m['filename'])
unique_to_them = their_files - labels_unique_fn
print(f"Files in their cache but NOT in our labels: {len(unique_to_them)}")
print(f"Files in our labels but NOT in their cache: {len(labels_unique_fn - their_files)}")

# Other public caches sizes
print("\n--- Check other public NPZ sizes ---")
import glob
for f in sorted(glob.glob("/tmp/pub_outputs/**/perch_arrays.npz", recursive=True))[:5]:
    d = np.load(f)
    s = d["scores"].shape if "scores" in d.keys() else "?"
    print(f"  {f}: {s}")
for f in sorted(glob.glob("/tmp/pub_outputs/**/full_oof_meta_features.npz", recursive=True))[:5]:
    d = np.load(f)
    s = d["oof_base"].shape if "oof_base" in d.keys() else "?"
    print(f"  {f}: {s}")
# Check for any FULL set caches (bigger than 708)
print("\n--- LOOKING FOR LARGE CACHES ---")
all_npz = list(glob.glob("/tmp/pub_outputs/**/*.npz", recursive=True))
print(f"All NPZ files: {len(all_npz)}")
for f in all_npz:
    try:
        d = np.load(f)
        for k in d.keys():
            v = d[k]
            if hasattr(v, 'shape') and len(v.shape) == 2 and v.shape[0] > 1000:
                print(f"  BIG: {f}: {k}={v.shape}")
    except: pass
