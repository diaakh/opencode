"""Forensic deep-dive on BirdCLEF 2026 data — iterates through many hypotheses
and documents each one in analysis/forensic/FINDINGS.md.

Run as: python3 forensic.py
"""
from __future__ import annotations
import os, json, re, hashlib, struct, gzip, time
from pathlib import Path
from collections import Counter, defaultdict
from itertools import islice
import numpy as np
import pandas as pd
import soundfile as sf

DATA = Path("/home/user/opencode/birdclef-2026/data")
OUT  = Path("/home/user/opencode/birdclef-2026/analysis/forensic")
OUT.mkdir(parents=True, exist_ok=True)

FIND = []   # collected findings; each entry: (heading, body)
def log(heading: str, body: str = ""):
    FIND.append((heading, body))
    print(f"\n==== {heading} ====")
    if body: print(body)

# ----------- H1: full audio metadata audit (sample rate, ch, duration, codec) -----------
def H1_file_audit():
    t0 = time.time()
    rows = []
    n_total = 0
    for src_dir, src in [(DATA / "train_audio", "train_audio"),
                         (DATA / "train_soundscapes", "train_soundscapes")]:
        for fp in src_dir.rglob("*.ogg"):
            n_total += 1
            try:
                info = sf.info(str(fp))
                rows.append({
                    "src":         src,
                    "path":        str(fp.relative_to(DATA)),
                    "label":       fp.parent.name if src == "train_audio" else "",
                    "samplerate":  info.samplerate,
                    "channels":    info.channels,
                    "frames":      info.frames,
                    "duration_s":  info.frames / info.samplerate if info.samplerate else 0,
                    "format":      info.format,
                    "subtype":     info.subtype,
                    "size_bytes":  fp.stat().st_size,
                })
            except Exception as e:
                rows.append({"src": src, "path": str(fp.relative_to(DATA)),
                             "error": str(e)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "01_file_audit.csv", index=False)
    elapsed = time.time() - t0
    sr = df["samplerate"].value_counts(dropna=False).to_dict()
    ch = df["channels"].value_counts(dropna=False).to_dict()
    fmt = df["format"].value_counts(dropna=False).to_dict()
    sub = df["subtype"].value_counts(dropna=False).to_dict()
    err = int(df["error"].notna().sum()) if "error" in df.columns else 0
    body = (f"Files scanned: {n_total} in {elapsed:.1f}s\n"
            f"Sample rates: {sr}\n"
            f"Channels: {ch}\n"
            f"Format: {fmt}\n"
            f"Subtype: {sub}\n"
            f"Errors decoding: {err}\n")
    # Are there any odd values?
    if len(sr) > 1: body += "** Multiple sample rates → potential anomaly\n"
    if len(ch) > 1: body += "** Multiple channel counts → potential anomaly\n"
    log("H1 — File-level audit (samplerate, channels, codec, duration)", body)
    return df

# ----------- H2: Vorbis comment extraction ---------------
def H2_vorbis():
    """OGG Vorbis files store metadata in 'comment' packets. Read raw."""
    rows = []
    for src_dir, src in [(DATA / "train_audio", "train_audio"),
                         (DATA / "train_soundscapes", "train_soundscapes")]:
        files = list(src_dir.rglob("*.ogg"))
        # sample up to 200 from each
        import random; random.seed(0); random.shuffle(files)
        for fp in files[:200]:
            try:
                with open(fp, "rb") as f:
                    head = f.read(65536)
                # locate Vorbis comment header (packet type 0x03 0x76 0x6f 0x72 0x62 0x69 0x73)
                idx = head.find(b"\x03vorbis")
                if idx < 0:
                    continue
                p = idx + 7
                vend_len = struct.unpack_from("<I", head, p)[0]; p += 4
                vendor = head[p:p+vend_len].decode("utf-8", errors="replace"); p += vend_len
                if p + 4 > len(head): continue
                n_comments = struct.unpack_from("<I", head, p)[0]; p += 4
                comments = []
                for _ in range(min(n_comments, 64)):
                    if p + 4 > len(head): break
                    L = struct.unpack_from("<I", head, p)[0]; p += 4
                    if p + L > len(head): break
                    c = head[p:p+L].decode("utf-8", errors="replace"); p += L
                    comments.append(c)
                rows.append({"src": src, "path": str(fp.relative_to(DATA)),
                             "vendor": vendor, "n_comments": n_comments,
                             "comments": "|".join(comments)})
            except Exception as e:
                rows.append({"src": src, "path": str(fp.relative_to(DATA)), "error": str(e)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "02_vorbis_comments_sample.csv", index=False)
    vendors = df["vendor"].value_counts().to_dict() if "vendor" in df.columns else {}
    # Find any comments that look interesting
    interesting = []
    for _, r in df.iterrows():
        c = str(r.get("comments", ""))
        if c and c != "nan":
            for tag in ["LOCATION", "DATE", "TIME", "ARTIST", "ENCODER", "PERFORMER",
                        "ALBUM", "TITLE", "ALBUMARTIST", "COMMENT", "GENRE", "TRACK",
                        "RECORDIST", "SITE", "DEVICE", "COORDINATE"]:
                if tag.lower() in c.lower():
                    interesting.append((r["src"], r["path"], tag, c[:200]))
                    break
    body = f"Sampled {len(df)} files for Vorbis metadata.\n"
    body += f"Encoder vendors: {vendors}\n"
    body += f"Files with interesting tags: {len(interesting)}\n"
    if interesting:
        body += "Examples:\n"
        for ex in interesting[:5]:
            body += f"  {ex[0]:18s} {ex[1][:60]:60s} {ex[2]:12s} {ex[3]}\n"
    else:
        body += "** No populated Vorbis comments found in sample → likely stripped on resampling.\n"
    log("H2 — Vorbis comment metadata", body)
    return df

# ----------- H3: SHA256 hashing of full file bytes ---------------
def H3_byte_hashing(audit_df):
    """Two files with the same SHA256 are byte-identical. Quick check for redundancy."""
    rows = []
    # full scan for all files
    for src_dir, src in [(DATA / "train_audio", "train_audio"),
                         (DATA / "train_soundscapes", "train_soundscapes")]:
        for fp in src_dir.rglob("*.ogg"):
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            rows.append({"src": src, "path": str(fp.relative_to(DATA)),
                         "size": fp.stat().st_size, "sha256": h.hexdigest()})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "03_sha256.csv.gz", index=False, compression="gzip")
    dup_groups = df.groupby("sha256").filter(lambda g: len(g) > 1)
    n_dups = len(dup_groups)
    body = (f"Files hashed: {len(df)}\n"
            f"Byte-identical duplicates: {n_dups} rows in "
            f"{dup_groups['sha256'].nunique() if n_dups else 0} groups.\n")
    if n_dups > 0:
        body += "Example duplicate groups:\n"
        for sha, grp in list(dup_groups.groupby("sha256"))[:5]:
            body += f"  {sha[:12]}: {' | '.join(grp['path'].tolist())}\n"
    log("H3 — Byte-identical (SHA256) duplicate audio files", body)
    return df

# ----------- H4: Audio CONTENT hashing -- find decoded-audio duplicates ---------------
def H4_audio_content_hashing(audit_df):
    """Hash a normalized PCM signature so we catch re-encoded duplicates with same content."""
    # Sample heuristically: for each train_audio file, hash first 4 s of decoded audio (32 kHz mono)
    rows = []
    import random; random.seed(0)
    # train_audio: full coverage but trim to first 4 s
    files_t = list((DATA / "train_audio").rglob("*.ogg"))
    print(f"H4: hashing first 4s of {len(files_t)} train_audio files")
    for fp in files_t:
        try:
            x, sr = sf.read(str(fp), frames=4 * 32000, always_2d=False)
            if x.ndim > 1: x = x.mean(axis=1)
            if x.size < 100: continue
            # Quantize PCM coarsely so tiny float drift doesn't bust the hash
            q = np.int16(np.clip(x[:128000], -1.0, 1.0) * 32767)
            h = hashlib.md5(q.tobytes()).hexdigest()
            rows.append({"src": "train_audio", "path": str(fp.relative_to(DATA)),
                         "label": fp.parent.name, "first4s_md5": h, "samples": q.size})
        except Exception as e:
            pass
    # train_soundscapes: also include
    files_s = list((DATA / "train_soundscapes").glob("*.ogg"))
    print(f"H4: hashing first 4s of {len(files_s)} train_soundscape files")
    for fp in files_s:
        try:
            x, sr = sf.read(str(fp), frames=4 * 32000, always_2d=False)
            if x.ndim > 1: x = x.mean(axis=1)
            q = np.int16(np.clip(x[:128000], -1.0, 1.0) * 32767)
            h = hashlib.md5(q.tobytes()).hexdigest()
            rows.append({"src": "train_soundscapes", "path": str(fp.relative_to(DATA)),
                         "label": "", "first4s_md5": h, "samples": q.size})
        except Exception:
            pass
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "04_first4s_md5.csv.gz", index=False, compression="gzip")
    dup_groups = df.groupby("first4s_md5").filter(lambda g: len(g) > 1)
    body = (f"Hashes computed: {len(df)}\n"
            f"first-4s-identical groups: {dup_groups['first4s_md5'].nunique()}, "
            f"total members: {len(dup_groups)}\n")
    if len(dup_groups):
        # Cross-source duplicates (train_audio ↔ train_soundscapes)
        cross = dup_groups.groupby("first4s_md5").filter(
            lambda g: g["src"].nunique() > 1)
        body += f"** Cross-source duplicates (train_audio ↔ train_soundscapes): "
        body += f"{cross['first4s_md5'].nunique()} groups, {len(cross)} rows\n"
        if len(cross):
            body += "Examples:\n"
            for sha, grp in list(cross.groupby("first4s_md5"))[:5]:
                body += "  " + sha[:12] + "\n"
                for _, r in grp.iterrows():
                    body += f"     {r['src']:18s} {r['path']}\n"
    log("H4 — Audio CONTENT hash (first 4 s of decoded PCM)", body)
    return df

# ----------- H5: 30-byte signature of last 4 s — catches partial re-uses ----------
def H5_audio_signature(audit_df):
    """Same idea as H4 but using a robust spectral signature for near-duplicates."""
    # Use a 32-band mel "fingerprint" from first 4 s — bucketize, hash
    rows = []
    import random; random.seed(0)
    files = []
    for src_dir, src in [(DATA / "train_audio", "train_audio"),
                         (DATA / "train_soundscapes", "train_soundscapes")]:
        files.extend([(src, fp) for fp in src_dir.rglob("*.ogg")])
    print(f"H5: spectral fingerprinting {len(files)} files (slow — ~1 ms/file)")
    n_fft = 1024; hop = 512
    for src, fp in files:
        try:
            x, sr = sf.read(str(fp), frames=4 * 32000, always_2d=False)
            if x.ndim > 1: x = x.mean(axis=1)
            if x.size < n_fft: continue
            # STFT power spectrum, log
            from numpy.fft import rfft
            # Frame the signal
            n_frames = (x.size - n_fft) // hop + 1
            if n_frames < 4: continue
            frames = np.stack([x[i*hop:i*hop+n_fft] * np.hanning(n_fft)
                              for i in range(min(n_frames, 32))])
            spec = np.abs(rfft(frames, axis=1))**2
            # Bin to 32 mel-ish bands (linear approx)
            bins = np.linspace(0, spec.shape[1], 33, dtype=int)
            mel = np.stack([spec[:, b:bins[i+1]].sum(axis=1) for i, b in enumerate(bins[:-1])], axis=1)
            mel = np.log(mel + 1e-9)
            # Sign-pattern fingerprint (Haar-style)
            sig = (mel[:, 1:] > mel[:, :-1]).astype(np.uint8)
            # Pack to bytes & hash
            h = hashlib.md5(sig.tobytes()).hexdigest()
            rows.append({"src": src, "path": str(fp.relative_to(DATA)),
                         "label": fp.parent.name if src == "train_audio" else "",
                         "spec_sig_md5": h})
        except Exception:
            pass
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "05_spec_fingerprint.csv.gz", index=False, compression="gzip")
    dup = df.groupby("spec_sig_md5").filter(lambda g: len(g) > 1)
    cross = dup.groupby("spec_sig_md5").filter(lambda g: g["src"].nunique() > 1)
    body = (f"Spectral fingerprints: {len(df)}\n"
            f"Within-source matches: {dup['spec_sig_md5'].nunique()} groups\n"
            f"Cross-source matches (train_audio ↔ train_soundscapes): "
            f"{cross['spec_sig_md5'].nunique()} groups\n")
    if len(cross):
        body += "Cross-source examples:\n"
        for sig, grp in list(cross.groupby("spec_sig_md5"))[:5]:
            for _, r in grp.iterrows():
                body += f"  {r['src']:18s} {r['path']}\n"
            body += "  ---\n"
    log("H5 — Spectral fingerprint (32-band sign-pattern hash)", body)
    return df

# ----------- H6: Inspect the LABELED soundscapes carefully ---------------
def H6_labeled_soundscape_inspect():
    ss = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    raw_rows = len(ss)
    ss_dedup = ss.drop_duplicates().reset_index(drop=True)
    # Are duplicates exact byte-identical rows? Or labelers reaching the same labels?
    # Check whether each (filename, start, end) appears exactly twice
    counts = ss.groupby(["filename","start","end"]).size().value_counts().to_dict()
    # Sort: per-key count
    body = (f"raw rows: {raw_rows}, deduped: {len(ss_dedup)} ({(1-len(ss_dedup)/raw_rows)*100:.0f}% dup)\n"
            f"counts per (filename,start,end): {counts}\n")
    # Check if duplicates ALWAYS have identical primary_label or differ
    grp = ss.groupby(["filename","start","end"])
    label_consistency = grp["primary_label"].nunique().value_counts().to_dict()
    body += f"unique labels per segment key: {label_consistency}\n"
    if label_consistency.get(1, 0) == grp.ngroups:
        body += ("** Every duplicate has the SAME primary_label → not two independent annotators,\n"
                 "   just the CSV containing each row twice (export-side bug). The labels themselves\n"
                 "   are still single-source.\n")
    # Save unique-label distribution
    body += f"\nlabeled segments per file:\n"
    spf = ss_dedup.groupby("filename").size()
    body += f"  min={spf.min()} median={int(spf.median())} max={spf.max()}\n"
    body += f"  files with <12 segments (partial labeling): {(spf < 12).sum()}/{len(spf)}\n"
    # Save which file pattern this is
    files_lab = ss_dedup["filename"].unique()
    PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
    parsed = [PAT.match(f).groups() for f in files_lab if PAT.match(f)]
    sites = Counter(p[1] for p in parsed)
    body += f"\nsites in labeled set: {dict(sites)}\n"
    # Are there sub-strings in primary_label that look auto-generated?
    raw_lab = ss_dedup["primary_label"].astype(str)
    body += f"\nhas only ';'-separated tokens: {raw_lab.str.contains('^[A-Za-z0-9son;]*$', regex=True).all()}\n"
    body += f"any whitespace: {raw_lab.str.contains(' ').any()}\n"
    log("H6 — Labeled soundscape file & duplicate audit", body)
    return ss_dedup

# ----------- H7: Cross-cuts: are LABELED soundscapes also in unlabeled set? ----------
def H7_labeled_in_full():
    ss_dedup = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    labeled_files = set(ss_dedup["filename"].unique())
    all_files = set([p.name for p in (DATA / "train_soundscapes").glob("*.ogg")])
    overlap = labeled_files & all_files
    missing_from_disk = labeled_files - all_files
    body = (f"labeled filenames referenced: {len(labeled_files)}\n"
            f"actually present on disk: {len(overlap)}\n"
            f"missing from disk: {len(missing_from_disk)} (any → broken reference)\n")
    if missing_from_disk:
        body += "Examples missing: " + ", ".join(list(missing_from_disk)[:5]) + "\n"
    log("H7 — Labeled-file references vs disk inventory", body)

# ----------- H8: Recording session detection (consecutive timestamps) ----------
def H8_recording_sessions():
    PAT = re.compile(r"BC2026_Train_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
    rows = []
    for fp in (DATA / "train_soundscapes").glob("*.ogg"):
        m = PAT.match(fp.name)
        if not m: continue
        idx, site, date, t = m.groups()
        rows.append({"file": fp.name, "idx": int(idx), "site": site,
                     "date": date,
                     "ts": pd.Timestamp(date + " " + t[:2]+":"+t[2:4]+":"+t[4:6]),
                     "hour": int(t[:2]),
                     })
    df = pd.DataFrame(rows).sort_values(["site","ts"]).reset_index(drop=True)
    df["site_prev_ts"] = df.groupby("site")["ts"].shift(1)
    df["gap_s"] = (df["ts"] - df["site_prev_ts"]).dt.total_seconds()
    # Same-site, gap <= 70 sec → consecutive
    df["consecutive"] = (df["gap_s"] <= 70) & df["gap_s"].notna()
    body = f"train_soundscape files parsed: {len(df)}\n"
    body += f"consecutive-pair count (same site, ≤70s gap): {df['consecutive'].sum()}\n"
    # Histogram of gap_s
    smaller = df["gap_s"].dropna()
    body += f"gap distribution (s): min={smaller.min():.0f} median={smaller.median():.0f} "
    body += f"p90={smaller.quantile(.9):.0f} p99={smaller.quantile(.99):.0f} max={smaller.max():.0f}\n"
    # Are there a lot of 60-second gaps (back-to-back)? That would mean continuous recording
    counts_60 = ((smaller >= 55) & (smaller <= 65)).sum()
    body += f"gaps of 55–65s (continuous recording): {counts_60}\n"
    # Session-length histogram
    df["session_id"] = (~df["consecutive"]).cumsum()
    sess = df.groupby(["site", "session_id"]).agg(n=("file","count"),
                                                    start=("ts","min"), end=("ts","max"))
    body += f"\nsessions detected: {len(sess)}\n"
    body += f"  session length (files): min={sess['n'].min()} median={int(sess['n'].median())} "
    body += f"max={sess['n'].max()} p95={int(sess['n'].quantile(.95))}\n"
    # Long sessions are interesting (a single uninterrupted recording chopped into 60-s segments)
    big = sess[sess["n"] >= 20].sort_values("n", ascending=False)
    body += f"sessions of ≥20 consecutive 60-s files: {len(big)}\n"
    if len(big):
        body += big.head(10).to_string() + "\n"
    df.to_csv(OUT / "08_session_index.csv", index=False)
    log("H8 — Recording-session reconstruction (consecutive 60-s soundscapes)", body)
    return df, sess

# ----------- H9: Recording-session × labeled-set ----------
def H9_session_label_relation(sess_df, idx_df):
    """Are the LABELED soundscapes from session START, MIDDLE, or scattered?"""
    ss = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    labeled = set(ss["filename"].unique())
    sess_df["session_pos"] = sess_df.groupby(["site", "session_id"]).cumcount()
    sess_df["session_len"] = sess_df.groupby(["site", "session_id"])["file"].transform("count")
    sess_df["session_frac"] = sess_df["session_pos"] / sess_df["session_len"].clip(lower=1)
    sess_df["is_labeled"] = sess_df["file"].isin(labeled)
    # Compare distribution of session_frac for labeled vs unlabeled
    lab = sess_df[sess_df["is_labeled"]]
    unl = sess_df[~sess_df["is_labeled"]]
    body = (f"labeled files: {len(lab)}  unlabeled files: {len(unl)}\n"
            f"session size of labeled  files: mean={lab['session_len'].mean():.1f} "
            f"median={int(lab['session_len'].median())}\n"
            f"session size of unlabeled files: mean={unl['session_len'].mean():.1f} "
            f"median={int(unl['session_len'].median())}\n"
            f"labeled-file session position: mean={lab['session_frac'].mean():.2f} "
            f"(0=start of session, 1=end)\n"
            )
    # Are labeled files all early in their sessions?
    log("H9 — Labeled files vs recording-session structure", body)
    return sess_df

# ----------- H10: train.csv full column audit ----------
def H10_train_csv_audit():
    tr = pd.read_csv(DATA / "train.csv")
    body = f"shape: {tr.shape}\ncolumns: {list(tr.columns)}\n\n"
    for c in tr.columns:
        try:
            n_unique = tr[c].nunique(dropna=True)
            null = tr[c].isna().sum()
            sample = tr[c].dropna().unique()[:5].tolist()
            body += f"  {c:20s} unique={n_unique:6d} null={null:6d} sample={sample}\n"
        except Exception as e:
            body += f"  {c}: <err {e}>\n"
    # license distribution
    if "license" in tr.columns:
        lic = tr["license"].value_counts().head(10).to_dict()
        body += f"\nlicense top 10: {lic}\n"
    # type field
    if "type" in tr.columns:
        tt = tr["type"].astype(str).value_counts().head(20).to_dict()
        body += f"\ntype field top 20: {tt}\n"
    # author top
    if "author" in tr.columns:
        auth = tr["author"].astype(str).value_counts()
        body += f"\nauthor: {len(auth)} unique  top 10: {auth.head(10).to_dict()}\n"
    # URL structure
    if "url" in tr.columns:
        urls = tr["url"].dropna().astype(str)
        body += f"\nURL hosts: {urls.str.extract(r'https?://([^/]+)')[0].value_counts().head(5).to_dict()}\n"
    log("H10 — train.csv full column audit", body)

# ----------- H11: 28 missing classes — full taxonomic detective ----------
def H11_missing_classes():
    tr = pd.read_csv(DATA / "train.csv")
    tax = pd.read_csv(DATA / "taxonomy.csv")
    tr["primary_label"] = tr["primary_label"].astype(str)
    tax["primary_label"] = tax["primary_label"].astype(str)
    missing = set(tax["primary_label"]) - set(tr["primary_label"])
    miss_df = tax[tax["primary_label"].isin(missing)]
    body = f"missing classes: {len(missing)}\n"
    body += miss_df.to_string(index=False) + "\n\n"
    # Are all sonotypes from same iNat taxon_id?
    sonos = [m for m in missing if "son" in m]
    non_sonos = [m for m in missing if "son" not in m]
    body += f"sonotypes ({len(sonos)}): {sorted(sonos)}\n"
    body += f"non-sonotype missing ({len(non_sonos)}): {sorted(non_sonos)}\n"
    # What's iNat taxon 47158?
    body += "\nlooking up '47158' parent: in taxonomy.csv, all '47158sonXX' rows share inat_taxon_id 47158?\n"
    rows = tax[tax["primary_label"].astype(str).str.contains("47158")]
    body += f"all 47158-prefixed rows:\n{rows.head(30).to_string(index=False)}\n"
    log("H11 — 28 missing-from-train_audio classes", body)

# ----------- H12: Filename ID patterns — XC vs iNat ID structure ----------
def H12_filename_ids():
    tr = pd.read_csv(DATA / "train.csv")
    # Sample filenames per species/collection
    by_coll = tr["filename"].str.extract(r"([^/]+\.ogg)$")
    fn = tr["filename"].astype(str).str.split("/").str[-1]
    tr["fn"] = fn
    # iNat prefix
    inat = tr[tr["collection"] == "iNat"]["fn"]
    xc = tr[tr["collection"] == "XC"]["fn"]
    body = f"iNat filename samples:\n  {inat.head(5).tolist()}\n\n"
    body += f"XC filename samples:\n  {xc.head(5).tolist()}\n\n"
    inat_ids = inat.str.extract(r"iNat(\d+)\.ogg")[0].astype("Int64")
    xc_ids   = xc.str.extract(r"XC(\d+)\.ogg")[0].astype("Int64")
    body += f"iNat ID range: {inat_ids.min()} - {inat_ids.max()}, n={inat_ids.notna().sum()}\n"
    body += f"XC ID range:   {xc_ids.min()} - {xc_ids.max()}, n={xc_ids.notna().sum()}\n"
    # Are there filename collisions between iNat and XC (same numeric)?
    inat_set = set(inat_ids.dropna().astype(int))
    xc_set   = set(xc_ids.dropna().astype(int))
    coll = inat_set & xc_set
    body += f"numeric-ID collisions iNat ∩ XC: {len(coll)} (cosmetic — these are different collections)\n"
    # Per-collection per-species: ratio of within-species ID range
    log("H12 — Filename ID structure", body)

# ----------- H13: Sample-submission ROW analysis ----------
def H13_sample_submission():
    sub = pd.read_csv(DATA / "sample_submission.csv")
    body = f"sample_submission shape: {sub.shape}\nrows:\n{sub.head().to_string()}\n"
    body += f"\nall values appear constant? {(sub.drop(columns=['row_id']).nunique(axis=0)<=1).all()}\n"
    log("H13 — Sample submission inspection", body)

# ----------- H14: secondary_labels structure  ----------
def H14_secondary_labels():
    tr = pd.read_csv(DATA / "train.csv")
    sec_raw = tr["secondary_labels"].astype(str)
    body = f"secondary_labels examples (5):\n"
    for s in sec_raw.head(5).tolist():
        body += f"  {s!r}\n"
    # Check format: is it Python-repr list ('[..., ...]')?
    is_list_repr = sec_raw.str.startswith("[")
    body += f"\nrows starting with '[': {int(is_list_repr.sum())}/{len(sec_raw)}\n"
    # How many list-like strings parse cleanly?
    import ast
    parsed = sec_raw.head(100).apply(lambda x:
        ast.literal_eval(x) if (x.startswith('[') and x.endswith(']')) else None)
    body += f"parse OK first 100: {int(parsed.notna().sum())}/100\n"
    log("H14 — secondary_labels parsing", body)

# ----------- H15: Final write findings ----------
def write_findings():
    out = OUT / "FINDINGS.md"
    with open(out, "w") as f:
        f.write("# Forensic findings — BirdCLEF 2026\n\n")
        f.write("Generated by `forensic.py`. Each section is one hypothesis.\n\n")
        f.write("---\n\n")
        for h, b in FIND:
            f.write(f"## {h}\n\n")
            if b:
                f.write("```\n")
                f.write(b)
                f.write("\n```\n")
            f.write("\n")
    print(f"\nWrote {out}")

if __name__ == "__main__":
    audit = H1_file_audit()
    H2_vorbis()
    h3 = H3_byte_hashing(audit)
    h4 = H4_audio_content_hashing(audit)
    h5 = H5_audio_signature(audit)
    H6_labeled_soundscape_inspect()
    H7_labeled_in_full()
    sess_df, sess_summary = H8_recording_sessions()
    H9_session_label_relation(sess_df, sess_summary)
    H10_train_csv_audit()
    H11_missing_classes()
    H12_filename_ids()
    H13_sample_submission()
    H14_secondary_labels()
    write_findings()
