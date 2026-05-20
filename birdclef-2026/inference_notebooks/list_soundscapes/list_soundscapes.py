"""Quick listing of train_soundscapes directory."""
import os
from pathlib import Path
import pandas as pd

for cand in ["/kaggle/input/birdclef-2026/train_soundscapes",
             "/kaggle/input/competitions/birdclef-2026/train_soundscapes"]:
    p = Path(cand)
    if p.exists():
        files = sorted(p.glob("*.ogg"))
        print(f"Found dir: {p}")
        print(f"Total .ogg files: {len(files)}")
        print(f"First 5: {[f.name for f in files[:5]]}")
        print(f"Last 5: {[f.name for f in files[-5:]]}")
        # Check labels coverage
        for lp in ["/kaggle/input/birdclef-2026/train_soundscapes_labels.csv",
                   "/kaggle/input/competitions/birdclef-2026/train_soundscapes_labels.csv"]:
            if Path(lp).exists():
                labels = pd.read_csv(lp)
                print(f"Labels: {len(labels)} rows, {labels['filename'].nunique()} unique files")
                # Files in directory NOT in labels
                label_fns = set(labels['filename'])
                dir_fns = set(f.name for f in files)
                unlabeled_files = dir_fns - label_fns
                print(f"  Files in dir not in labels: {len(unlabeled_files)}")
                if unlabeled_files:
                    print(f"  Sample unlabeled: {list(unlabeled_files)[:5]}")
                break
        break
    else:
        print(f"NOT FOUND: {p}")

# Also check for unlabeled_soundscapes
for cand in ["/kaggle/input/birdclef-2026/unlabeled_soundscapes",
             "/kaggle/input/birdclef-2026/extra",
             "/kaggle/input/birdclef-2026/additional_audio"]:
    if Path(cand).exists():
        n = len(list(Path(cand).glob("*")))
        print(f"FOUND extra: {cand}, files: {n}")
