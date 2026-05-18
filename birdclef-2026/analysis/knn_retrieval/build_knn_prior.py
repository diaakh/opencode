"""Build a production-ready KNN retrieval prior over the labeled + v4 pseudo
embedding database. For each test window, find top-K cosine-similar Perch
embeddings from this database and return weighted labels.

This is a leakage-safe inference-time prior because:
  - The database includes only labels we trust (labeled hard + v4 pseudo-soft)
  - At LB time, we query test embeddings against this static index
  - No retraining, no GPU, ~ms per query with sklearn NearestNeighbors

Saves: knn_index.pkl (sklearn NN + emb + labels) for use in submission kernel.
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

REPO = "/home/user/opencode/birdclef-2026"
OUT = f"{REPO}/analysis/knn_retrieval"
os.makedirs(OUT, exist_ok=True)

# Labeled
br = np.load("/tmp/bruce_out/labeled_oof_perch_bruce.npz", allow_pickle=True)
emb_lab = br["embeddings"].astype(np.float32)
Y_lab = br["Y"].astype(np.float32)  # hard labels
file_lab = br["row_filename"]
hour_lab = br["row_hour"].astype(int)
site_lab = br["row_site"].astype(str)
print(f"Labeled embeddings: {emb_lab.shape}")

# v4 pseudo embeddings
emb_unlab = np.load("/tmp/pseudo_cache/pseudo_emb.npy", mmap_mode="r")
unlab = np.load(f"{REPO}/analysis/context_label/bruce_on_unlabeled.npz", allow_pickle=True)
row_id_unlab = unlab["row_id"]
file_unlab = unlab["filename"]
hour_unlab = unlab["hour_utc"].astype(int)
site_unlab = unlab["site"].astype(str)
rid_to_idx = {rid: i for i, rid in enumerate(row_id_unlab)}

tax = pd.read_csv("/tmp/bc26_verify/taxonomy.csv")
classes = tax["primary_label"].astype(str).tolist()
cls_idx = {c: i for i, c in enumerate(classes)}
C = len(classes)

v4 = pd.read_parquet(f"{REPO}/analysis/context_label/pseudo_labels_v4.parquet")
print(f"v4 pseudo-labels: {len(v4)} rows")

# Build aggregated pseudo Y matrix
pseudo_idx_list = []
pseudo_labels_list = []
pseudo_meta = []
for rid, group in v4.groupby("row_id"):
    if rid not in rid_to_idx:
        continue
    i = rid_to_idx[rid]
    labels = np.zeros(C, dtype=np.float32)
    for _, r in group.iterrows():
        labels[cls_idx[r["class"]]] = r["bruce_shifted_v4"]
    pseudo_idx_list.append(i)
    pseudo_labels_list.append(labels)
    pseudo_meta.append({
        "row_id": rid,
        "filename": file_unlab[i],
        "hour": int(hour_unlab[i]),
        "site": str(site_unlab[i]),
    })

emb_pseudo = emb_unlab[pseudo_idx_list].astype(np.float32)
Y_pseudo = np.array(pseudo_labels_list, dtype=np.float32)
print(f"Pseudo embeddings: {emb_pseudo.shape}, Y_pseudo: {Y_pseudo.shape}")

# Combined database: labeled + pseudo
emb_db = np.concatenate([emb_lab, emb_pseudo], axis=0)
Y_db = np.concatenate([Y_lab, Y_pseudo], axis=0)
file_db = np.concatenate([file_lab, [m["filename"] for m in pseudo_meta]])
hour_db = np.concatenate([hour_lab, [m["hour"] for m in pseudo_meta]])
site_db = np.concatenate([site_lab.astype(str), [m["site"] for m in pseudo_meta]])
print(f"Combined db: {emb_db.shape}")

# Normalize for cosine sim
emb_db_n = normalize(emb_db, axis=1)
print(f"L2-normalized db ready")

# Build sklearn NN index
K = 20
nn = NearestNeighbors(n_neighbors=K, metric="cosine", n_jobs=-1)
nn.fit(emb_db_n)
print(f"Fitted KNN index with K={K}, n={len(emb_db_n)}")

# Save for inference-time use
import pickle
with open(f"{OUT}/knn_index.pkl", "wb") as f:
    pickle.dump({
        "emb_db_n": emb_db_n,
        "Y_db": Y_db,
        "file_db": file_db,
        "hour_db": hour_db,
        "site_db": site_db,
        "classes": np.array(classes),
        "K": K,
    }, f, protocol=4)
print(f"Saved {OUT}/knn_index.pkl ({os.path.getsize(f'{OUT}/knn_index.pkl')/1e6:.1f} MB)")


# ---------- Self-validation: KNN on labeled (held-out file) ----------
print("\n=== Self-validation: KNN on labeled with leave-one-file-out ===")
from sklearn.model_selection import GroupKFold
file_idx_lab = pd.Categorical(file_lab).codes
gkf = GroupKFold(n_splits=5)
N_LAB = len(emb_lab)
emb_lab_n = normalize(emb_lab, axis=1)

knn_pred_oof = np.zeros((N_LAB, C), dtype=np.float32)

for fold, (tr, va) in enumerate(gkf.split(emb_lab, Y_lab, groups=file_idx_lab)):
    val_files = set(file_lab[va])
    # Build DB = labeled-tr + pseudo (excluding val-files)
    pseudo_keep_mask = np.array([m["filename"] not in val_files for m in pseudo_meta])
    db_emb_n = np.concatenate([emb_lab_n[tr], normalize(emb_pseudo[pseudo_keep_mask], axis=1)], axis=0)
    db_Y = np.concatenate([Y_lab[tr], Y_pseudo[pseudo_keep_mask]], axis=0)

    # Query val
    query = emb_lab_n[va]
    # Cosine sim = dot product since normalized
    sims = query @ db_emb_n.T  # (n_va, n_db)
    top_k_idx = np.argpartition(-sims, K, axis=1)[:, :K]
    for i in range(len(va)):
        idx = top_k_idx[i]
        s = sims[i, idx]
        s = np.maximum(s, 0)
        s_sum = s.sum() if s.sum() > 0 else 1.0
        w = s / s_sum
        knn_pred_oof[va[i]] = (w[:, None] * db_Y[idx]).sum(axis=0)
    print(f"  Fold {fold+1}/5: db={db_emb_n.shape[0]} val={len(va)}")

# Eval
from sklearn.metrics import roc_auc_score
def macro_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0 or y[:, c].sum() == len(y):
            continue
        col = p[:, c]
        mask = ~np.isnan(col)
        if mask.sum() < 5 or y[mask, c].sum() == 0 or y[mask, c].sum() == mask.sum():
            continue
        try: aucs.append(roc_auc_score(y[mask, c], col[mask]))
        except: pass
    return float(np.mean(aucs)) if aucs else float("nan")

print(f"\nKNN-pseudo-augmented (K={K}) macro-AUC: {macro_auc(Y_lab, knn_pred_oof):.4f}")

# Now blend with Bruce
P_bruce = br["P_bruce"]
def rank_norm(P):
    from scipy.stats import rankdata
    R = np.zeros_like(P, dtype=np.float32)
    for c in range(P.shape[1]):
        col = P[:, c]
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            R[valid, c] = rankdata(col[valid]) / valid.sum()
            R[~valid, c] = 0.5
    return R
R_bruce = rank_norm(P_bruce)
R_knn = rank_norm(knn_pred_oof)
print(f"Bruce + KNN-aug blend macro-AUC: {macro_auc(Y_lab, (R_bruce + R_knn) / 2):.4f}")
print(f"0.5*B + 0.5*K: {macro_auc(Y_lab, 0.5*R_bruce + 0.5*R_knn):.4f}")
print(f"0.6*B + 0.4*K: {macro_auc(Y_lab, 0.6*R_bruce + 0.4*R_knn):.4f}")
print(f"0.7*B + 0.3*K: {macro_auc(Y_lab, 0.7*R_bruce + 0.3*R_knn):.4f}")

np.savez_compressed(f"{OUT}/knn_oof_labeled.npz", P_knn=knn_pred_oof, Y=Y_lab)
print(f"\nSaved {OUT}/knn_oof_labeled.npz")
