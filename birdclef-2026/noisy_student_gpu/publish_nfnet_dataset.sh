#!/usr/bin/env bash
# Pull the completed nfnet kernel's output and publish it as the artifact dataset.
# Usage: bash publish_nfnet_dataset.sh
set -euo pipefail
export KAGGLE_USERNAME=adkasd KAGGLE_KEY=$(cat ~/.kaggle/access_token)
KSLUG="adkasd/bc26-ns-student-nfnet-train"
DDIR="/home/user/opencode/birdclef-2026/noisy_student_gpu/nfnet_dataset"

mkdir -p "$DDIR/out"
echo "Pulling kernel output from $KSLUG ..."
kaggle kernels output "$KSLUG" -p "$DDIR/out"
echo "=== pulled files ==="
find "$DDIR/out" -type f -printf '%p  %s bytes\n'

# Stage just the artifact + preds into the dataset root (next to dataset-metadata.json)
find "$DDIR/out" -name 'ns_student_nfnet_fold1_fp16.pt' -exec cp {} "$DDIR/" \;
find "$DDIR/out" -name 'ns_student_nfnet_soundscape_preds.parquet' -exec cp {} "$DDIR/" \;
echo "=== dataset dir staged ==="
ls -la "$DDIR"

# Create (first time) or version the dataset
cd "$DDIR"
if kaggle datasets status adkasd/bc26-ns-student-nfnet 2>/dev/null | grep -qi ready; then
  kaggle datasets version -p . -m "nfnet noisy-student student + soundscape preds" --dir-mode skip
else
  kaggle datasets create -p . --dir-mode skip
fi
