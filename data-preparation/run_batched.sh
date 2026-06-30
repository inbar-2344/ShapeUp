#!/bin/bash
# Process a range of Objaverse uids in fixed-size batches, restarting the Python
# process between batches so resident memory is released back to the OS.
#
# Usage: ./run_batched.sh START END LOG [SCRIPT]
#   START  first uid index (inclusive)
#   END    last uid index (exclusive)
#   LOG    file to append stdout/stderr to
#   SCRIPT preprocessing script to run (default: objaverse_preprocess.py)
#
# Example: ./run_batched.sh 0 1000 render_0_1000.log
set -euo pipefail

BATCH_SIZE=${BATCH_SIZE:-200}
START=$1
END=$2
LOG=$3
SCRIPT=${4:-objaverse_preprocess.py}
PYTHON=${PYTHON:-python}

for ((i=START; i<END; i+=BATCH_SIZE)); do
    batch_end=$((i + BATCH_SIZE))
    if [ $batch_end -gt $END ]; then batch_end=$END; fi
    echo "[$(date)] Processing $i to $batch_end" >> "$LOG"
    "$PYTHON" "$SCRIPT" --start_ind "$i" --end_ind "$batch_end" >> "$LOG" 2>&1
done
