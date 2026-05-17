#!/bin/bash
# Watch the 5 kernels and submit each to the competition upon successful completion.
# Logs to /tmp/kpush/monitor.log so we can check progress later.
set -uo pipefail

LOG=/tmp/kpush/monitor.log
mkdir -p /tmp/kpush
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ')  monitor starting" >> "$LOG"

SLUGS=(
  "birdclef-2026-sub1-hour3"
  "birdclef-2026-sub2-bruce-standalone"
  "birdclef-2026-sub3-hour2-alias-blind"
  "birdclef-2026-sub4-hour3-alias"
  "birdclef-2026-sub5-hour3-calib-alias"
)
MESSAGES=(
  "sub1: exp019 + hour_prior w=3.0 (OOF 0.9586)"
  "sub2: standalone Bruce+hour_prior (OOF 0.9586)"
  "sub3: exp019 + hour_prior w=2.0 + alias + site_blind"
  "sub4: exp019 + hour_prior w=3.0 + sonotype_alias"
  "sub5: exp019 + hour_prior w=3.0 + perch_calib + alias"
)

declare -A DONE
declare -A SUBMITTED

iter=0
while true; do
  iter=$((iter+1))
  remaining=0
  for i in "${!SLUGS[@]}"; do
    slug="${SLUGS[$i]}"
    if [[ -n "${DONE[$slug]:-}" ]]; then
      continue
    fi
    status=$(kaggle kernels status adkasd/$slug 2>&1 | head -1 | sed -E 's/.*status "([^"]+)".*/\1/')
    echo "$(date -u +'%H:%M:%S')  iter=$iter  $slug  -> $status" >> "$LOG"
    case "$status" in
      *RUNNING*|*QUEUED*)
        remaining=$((remaining+1))
        ;;
      *COMPLETE*|*COMPLETED*)
        DONE[$slug]="complete"
        echo "$(date -u +'%H:%M:%S')  $slug COMPLETED — submitting..." >> "$LOG"
        # Determine latest version (sub2 is on v2, others on v1)
        ver=1
        if [[ "$slug" == "birdclef-2026-sub2-bruce-standalone" ]]; then ver=2; fi
        result=$(kaggle competitions submit birdclef-2026 -k adkasd/$slug -v $ver -f submission.csv -m "${MESSAGES[$i]}" 2>&1)
        echo "$result" >> "$LOG"
        SUBMITTED[$slug]="yes"
        ;;
      *ERROR*|*FAIL*|*CANCELLED*)
        DONE[$slug]="failed"
        echo "$(date -u +'%H:%M:%S')  $slug FAILED — skipping submit" >> "$LOG"
        ;;
      *)
        echo "$(date -u +'%H:%M:%S')  $slug unknown status: $status" >> "$LOG"
        remaining=$((remaining+1))
        ;;
    esac
  done
  if [[ $remaining -eq 0 ]]; then
    echo "$(date -u +'%H:%M:%S')  ALL DONE — completed=${#DONE[@]} submitted=${#SUBMITTED[@]}" >> "$LOG"
    break
  fi
  sleep 120
done
