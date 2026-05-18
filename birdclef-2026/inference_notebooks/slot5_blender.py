"""SLOT 5 BLENDER — combine two LB submission CSVs via rank-norm + weighted blend.

USAGE:
  python slot5_blender.py exp019_submission.csv sub_v8_submission.csv slot5_final.csv 0.7 0.3

Or run in a Kaggle kernel that has both as input datasets.

The math:
  1. For each class column, rank-normalize across all rows (per CSV separately)
  2. Weighted average of ranks: w1 * R1 + w2 * R2
  3. Write final submission CSV with rank-normalized scores

Recommended weights (May 19 plan):
  - exp019: 0.70  (LB 0.949 anchor)
  - sub_v8: 0.30  (orthogonal Bruce-derived signal)

Expected LB: 0.95-0.96 (if sub_v8 transfers ≥ 0.85)
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def rank_norm(arr):
    """Rank-normalize each column across rows, returns ranks in [0, 1]."""
    R = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[1]):
        col = arr[:, c]
        R[:, c] = rankdata(col, method="average") / len(col)
    return R

def blend(csv1, csv2, out_csv, w1=0.7, w2=0.3):
    print(f"Loading {csv1}...")
    df1 = pd.read_csv(csv1)
    print(f"  shape: {df1.shape}")
    print(f"Loading {csv2}...")
    df2 = pd.read_csv(csv2)
    print(f"  shape: {df2.shape}")
    
    # Identify row_id column and class columns
    id_col = "row_id"
    assert id_col in df1.columns and id_col in df2.columns, "Both CSVs must have row_id"
    
    # Align by row_id
    if not df1[id_col].equals(df2[id_col]):
        print("  row_id orders differ — aligning df2 to df1...")
        df2 = df2.set_index(id_col).reindex(df1[id_col]).reset_index()
    
    class_cols = [c for c in df1.columns if c != id_col]
    assert all(c in df2.columns for c in class_cols), "Class columns mismatch"
    
    # Extract matrices
    M1 = df1[class_cols].to_numpy(dtype=np.float32)
    M2 = df2[class_cols].to_numpy(dtype=np.float32)
    print(f"  M1 range: [{M1.min():.4f}, {M1.max():.4f}]")
    print(f"  M2 range: [{M2.min():.4f}, {M2.max():.4f}]")
    
    # Rank-normalize each
    print("Rank-normalizing per class...")
    R1 = rank_norm(M1)
    R2 = rank_norm(M2)
    
    # Weighted blend
    print(f"Blending w1={w1} (csv1) + w2={w2} (csv2)...")
    R_final = w1 * R1 + w2 * R2
    
    # Output as ranks (in [0, 1]) — Kaggle accepts rank-space submissions
    out = df1[[id_col]].copy()
    for i, c in enumerate(class_cols):
        out[c] = R_final[:, i]
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(f"  Final range: [{R_final.min():.4f}, {R_final.max():.4f}]")
    return out

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: slot5_blender.py csv1 csv2 out_csv [w1=0.7] [w2=0.3]")
        sys.exit(1)
    w1 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
    w2 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
    blend(sys.argv[1], sys.argv[2], sys.argv[3], w1, w2)
