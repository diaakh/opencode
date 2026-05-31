#!/usr/bin/env bash
# Run AFTER a kernel completes. Pulls fp16 ckpt + train preds and publishes dataset.
# Usage: ./publish_artifacts.sh b0   |   ./publish_artifacts.sh nfnet
set -e
export KAGGLE_USERNAME=adkasd KAGGLE_KEY=$(cat ~/.kaggle/access_token)
RUN="$1"
case "$RUN" in
  b0)    KSLUG=adkasd/bc26-focal-b0-v1;    DIR=/home/user/opencode/birdclef-2026/focal_sed/b0/artifact ;;
  nfnet) KSLUG=adkasd/bc26-focal-nfnet-v1; DIR=/home/user/opencode/birdclef-2026/focal_sed/nfnet/artifact ;;
  *) echo "arg must be b0|nfnet"; exit 1 ;;
esac
kaggle kernels output "$KSLUG" -p "$DIR"
# keep only the publishable artifacts (fp16 ckpt, train preds, index, labels)
find "$DIR" -maxdepth 1 -type f ! -name "dataset-metadata.json" \
  ! -name "sed_${RUN}_fp16.pt" ! -name "sed_${RUN}_trainpreds.npy" \
  ! -name "sed_${RUN}_predindex.csv" ! -name "sed_${RUN}_labels.json" -delete || true
ls -la "$DIR"
if kaggle datasets create -p "$DIR" -r zip 2>&1 | grep -q "already exists"; then
  kaggle datasets version -p "$DIR" -m "update" -r zip
fi
