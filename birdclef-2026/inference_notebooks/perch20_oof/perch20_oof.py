"""Perch 2.0 — use venv to isolate TF 2.20 install from Kaggle's stale deps."""
import os, sys, subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Create venv with isolated TF 2.20+
print("Creating venv...")
subprocess.check_call([sys.executable, "-m", "venv", "/tmp/p20_venv", "--system-site-packages"])
venv_py = "/tmp/p20_venv/bin/python"
subprocess.check_call([venv_py, "-m", "pip", "install", "-q", "--upgrade",
    "--force-reinstall", "--no-deps", "tensorflow-cpu>=2.20", "keras>=3.5", "absl-py>=2.0"])
subprocess.check_call([venv_py, "-m", "pip", "install", "-q",
    "ml-dtypes", "numpy", "protobuf", "wrapt", "gast", "termcolor", "h5py", "opt-einsum",
    "flatbuffers", "google-pasta", "libclang", "tensorboard", "namex", "optree", "rich",
    "soundfile", "pandas", "scikit-learn"])
print("Venv ready")

# Now spawn a subprocess running in venv that does the actual work
script = '''
import os, sys, glob, re, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
import tensorflow as tf
print(f"TF version: {tf.__version__}")

PERCH20_ROOT = None
for cand in glob.glob("/kaggle/input/**/saved_model.pb", recursive=True):
    p = Path(cand).parent
    if "cpu" in str(p):
        PERCH20_ROOT = p; break
    if PERCH20_ROOT is None: PERCH20_ROOT = p
print(f"Perch 2.0: {PERCH20_ROOT}")
model = tf.saved_model.load(str(PERCH20_ROOT))
sig = model.signatures["serving_default"]
print(f"Outputs: {list(sig.structured_outputs.keys())}")
input_key = list(sig.structured_input_signature[1].keys())[0]

# Test
test = tf.constant(np.zeros((1, 160000), dtype=np.float32))
out = sig(**{input_key: test})
print("Forward works!")
for k, v in out.items():
    print(f"  {k}: {v.shape}")

# Load BC2026
COMP = Path("/kaggle/input/competitions/birdclef-2026")
if not COMP.exists(): COMP = Path("/kaggle/input/birdclef-2026")
labels = pd.read_csv(COMP / "train_soundscapes_labels.csv").drop_duplicates()
labels = labels.sort_values(["filename", "start"]).reset_index(drop=True)
tax = pd.read_csv(COMP / "taxonomy.csv")
bc_classes = tax["primary_label"].astype(str).tolist()
N = len(labels); C = len(bc_classes)
cls_idx = {c: i for i, c in enumerate(bc_classes)}
Y = np.zeros((N, C), dtype=np.float32)
for i, row in labels.iterrows():
    for c in str(row["primary_label"]).split(";"):
        c = c.strip()
        if c in cls_idx: Y[i, cls_idx[c]] = 1.0

# Load Perch labels
perch_sci = None
for p in PERCH20_ROOT.rglob("labels.csv"): perch_sci = pd.read_csv(p); break
perch_ebird = None
for p in PERCH20_ROOT.rglob("perch_v2_ebird_classes.csv"): perch_ebird = pd.read_csv(p); break

sci_to_idx = {str(s).strip().lower(): i for i, s in enumerate(perch_sci.iloc[:, 0])} if perch_sci is not None else {}
ebird_to_idx = {str(s).strip().lower(): i for i, s in enumerate(perch_ebird.iloc[:, 0])} if perch_ebird is not None else {}

bc_to_perch = {}
for bi, bc_lbl in enumerate(bc_classes):
    sci = str(tax.iloc[bi].get("scientific_name", "")).strip().lower()
    if sci in sci_to_idx: bc_to_perch[bi] = sci_to_idx[sci]; continue
    if str(bc_lbl).lower() in ebird_to_idx: bc_to_perch[bi] = ebird_to_idx[str(bc_lbl).lower()]; continue
print(f"BC2026 → Perch mapping: {len(bc_to_perch)}/{C}")

SR = 32000
WIN_SAMP = 5 * SR
ts_dir = COMP / "train_soundscapes"
file_to_idx = {}
for i, f in enumerate(labels["filename"]):
    file_to_idx.setdefault(f, []).append(i)
P_p20 = np.zeros((N, C), dtype=np.float32)
emb_p20 = np.zeros((N, 1536), dtype=np.float32)
print(f"Processing {len(file_to_idx)} files...")
t0 = time.time(); nd = 0
for f, idx_list in file_to_idx.items():
    fpath = ts_dir / f
    if not fpath.exists():
        for ext in [".ogg", ".wav", ".flac"]:
            if (ts_dir / (f + ext)).exists(): fpath = ts_dir / (f + ext); break
    try:
        audio, _ = sf.read(str(fpath))
        if audio.ndim > 1: audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
    except: continue
    wins = []
    for row_i in idx_list:
        st = labels.iloc[row_i]["start"]
        if isinstance(st, str) and ":" in st:
            h, m, s = st.split(":"); ss = int(h)*3600+int(m)*60+int(s)
        else: ss = int(float(st))
        s_samp = ss * SR
        clip = np.zeros(WIN_SAMP, dtype=np.float32)
        avail = max(0, len(audio) - s_samp)
        if avail >= WIN_SAMP: clip = audio[s_samp:s_samp+WIN_SAMP]
        elif avail > 0: clip[:avail] = audio[s_samp:s_samp+avail]
        wins.append(clip)
    if not wins: continue
    batch = tf.constant(np.stack(wins).astype(np.float32))
    out = sig(**{input_key: batch})
    logits = out["label"].numpy()
    embs = out["embedding"].numpy()
    probs = 1.0/(1.0+np.exp(-logits))
    for j, row_i in enumerate(idx_list):
        emb_p20[row_i] = embs[j]
        for bi, pi in bc_to_perch.items():
            P_p20[row_i, bi] = probs[j, pi]
    nd += 1
    if nd % 10 == 0:
        el = time.time()-t0
        print(f"  [{nd}/{len(file_to_idx)}] {el:.0f}s rate={nd/el:.2f}/s")
print(f"Total: {time.time()-t0:.0f}s")
np.savez("/kaggle/working/perch20_labeled_oof.npz",
    P_perch20=P_p20, emb=emb_p20, Y=Y,
    classes=np.array(bc_classes),
    row_filename=labels["filename"].to_numpy())
print("Saved.")
from sklearn.metrics import roc_auc_score
aucs = []
for c in range(C):
    if Y[:, c].sum() < 2 or Y[:, c].sum() == N: continue
    if P_p20[:, c].max() == 0: continue
    try: aucs.append(roc_auc_score(Y[:, c], P_p20[:, c]))
    except: pass
print(f"Perch 2.0 macro-AUC: {np.mean(aucs):.4f} ({len(aucs)} mapped)")
'''
with open("/tmp/p20_run.py", "w") as f: f.write(script)
subprocess.check_call([venv_py, "/tmp/p20_run.py"])
