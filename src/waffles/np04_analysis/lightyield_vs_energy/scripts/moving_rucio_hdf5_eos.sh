#!/bin/bash

rucio_folder="/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW/hd-protodune"
hd5f_folder="/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW/run027343"
eos_folder="/eos/user/a/anbalbon/reading_beamrun_NEW/run027343"

# Controllo accesso
if ! cd "$eos_folder" 2>/dev/null; then
    echo "Errore: permesso negato per accedere a $eos_folder"
    exit 1
fi
cd - >/dev/null

for f in "$rucio_folder"/np04hd_raw_run*.hdf5; do
    filename=$(basename "$f")
    prefix="${filename%.hdf5}"

    # Cerca qualsiasi file corrispondente nel folder hd5f
    match=$(ls "$hd5f_folder"/processed_"${prefix}"*_structured.hdf5 2>/dev/null | head -n 1)

    if [[ -f "$match" ]]; then
        echo "➡️  Trovato: $filename"
        echo "   Copia in corso verso $eos_folder ..."
        rsync -ah --progress "$f" "$eos_folder/" && rm -f "$f"
        echo "✅ Copiato: $filename"
    else
        echo "⏭️  Saltato: $filename (nessun corrispondente in $hd5f_folder)"
    fi
done
