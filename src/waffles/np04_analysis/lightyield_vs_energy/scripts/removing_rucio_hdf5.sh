#!/bin/bash

# Check if run number is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_number>"
    exit 1
fi

run_number=$1

rucio_folder="/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW/hd-protodune"
hd5f_folder="/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW/run0${run_number}"

for f in "$rucio_folder"/np04hd_raw_run*.hdf5; do
    filename=$(basename "$f")
    prefix="${filename%.hdf5}"

    # Look for any matching file in hd5f folder
    match=$(ls "$hd5f_folder"/processed_"${prefix}"*_structured.hdf5 2>/dev/null | head -n 1)

    if [[ -f "$match" ]]; then
        echo "➡️  Found: $filename"
        echo "   Deleting $f ..."
        rm -f "$f" && echo "🗑️ Deleted: $filename"
    else
        echo "⏭️  Skipped: $filename (no match found in $hd5f_folder)"
    fi
done
