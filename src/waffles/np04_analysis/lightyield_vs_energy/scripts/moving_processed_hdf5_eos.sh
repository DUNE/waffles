#!/bin/bash

# Check if run number is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_number>"
    exit 1
fi

run_number=$1

# cartella di origine (dove hai i file .hdf5 locali)
hdf5_folder="/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW/run0${run_number}"

# cartella di destinazione su EOS
eos_folder="/eos/user/a/anbalbon/reading_beamrun_NEW/run0${run_number}/processed_hdf5"

# Verifica accesso a EOS
if ! cd "$eos_folder" 2>/dev/null; then
    echo "❌ Error: cannot access $eos_folder"
    exit 1
fi
cd - >/dev/null

echo "✅ Access to EOS folder confirmed: $eos_folder"

# Crea la cartella se non esiste
mkdir -p "$eos_folder"

# Sposta i file processed uno per uno usando rsync
for f in "${hdf5_folder}"/processed_np04hd_raw_run0${run_number}_*.hdf5_structured.hdf5; do
    if [ -f "$f" ]; then
        echo "➡️  Copying $f to EOS..."
        rsync -ah --progress "$f" "$eos_folder/" && rm -f "$f"
    fi
done

echo "✅ Done!"
