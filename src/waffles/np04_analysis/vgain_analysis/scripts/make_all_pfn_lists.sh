#!/usr/bin/env bash
# Generate PFN list text files for a list of runs using fetch_rucio_replicas.py
# Each run gets up to 5 attempts, each capped at 20 s.
# If successful → move txt file + remove run from list.
# If all retries fail → leave run in list.

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <runs_list.txt> <output_dir>"
    exit 1
fi

RUNLIST="$1"
OUTDIR="$2"
MAXFILES=10              # number of files to fetch
TIMEOUT_SECS=20          # per-attempt timeout
MAX_RETRIES=5            # ⏳ attempts per run
SCRIPT_DIR="/home/ecristal/software/DUNE/waffles/scripts"

mkdir -p "$OUTDIR"

# Normalize line endings just once
TMP_RUNLIST="$(mktemp)"
tr -d '\r' < "$RUNLIST" > "$TMP_RUNLIST"
mv "$TMP_RUNLIST" "$RUNLIST"

TOTAL=$(grep -cvE '^\s*#|^\s*$' "$RUNLIST")
echo "Starting processing of $TOTAL runs..."

i=0

while IFS= read -r RUN; do
    RUN="${RUN%%[$'\r']}"               # strip stray CR
    [[ -z "$RUN" || "$RUN" =~ ^# ]] && continue

    ((i++))
    PCT=$(( i * 100 / TOTAL ))
    echo "[${i}/${TOTAL}] (${PCT}%) >>> Fetching replicas for run $RUN..."

    success=0
    for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
        echo "   Attempt ${attempt}/${MAX_RETRIES} ..."
        if timeout "$TIMEOUT_SECS" \
                python "$SCRIPT_DIR/fetch_rucio_replicas.py" \
                        --runs "$RUN" \
                        --max-files "$MAXFILES"; then
            success=1
            break
        else
            echo "     ⚠️  attempt ${attempt} failed or timed out (${TIMEOUT_SECS}s)"
            sleep 1   # brief pause between retries
        fi
    done

    if (( success )); then
        # ✅ Move the generated file
        if [[ -f "$PWD/0${RUN}.txt" ]]; then
            mv "$PWD/0${RUN}.txt" "$OUTDIR/"
        elif [[ -f "$PWD/${RUN}.txt" ]]; then
            mv "$PWD/${RUN}.txt" "$OUTDIR/"
        else
            echo "   ⚠️  WARNING: fetcher succeeded but no PFN file found for run $RUN"
        fi

        # ✅ Remove processed run from list
        sed -i "/^${RUN}$/d" "$RUNLIST"

    else
        echo "   ❌ Skipped run $RUN after ${MAX_RETRIES} failed attempts"
        # (run stays in $RUNLIST for later re-try if you wish)
    fi

done < "$RUNLIST"

echo "All done. Any runs that failed all ${MAX_RETRIES} attempts remain in $RUNLIST"

