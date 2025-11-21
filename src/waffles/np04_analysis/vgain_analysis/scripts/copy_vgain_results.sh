#!/usr/bin/env bash
set -euo pipefail

# === CONFIGURE THESE TWO PATHS ===
SRC_BASE="/home/ecristal/software/DUNE/waffles/src/waffles/np04_analysis/vgain_analysis/output/vgain_scan"
DST_DIR="/home/ecristal/software/DUNE/waffles/src/waffles/np04_analysis/vgain_analysis/output/vgain_scan_compiled_results/run_1"

mkdir -p "$DST_DIR"

# Lists from your description
vgains=(931 1064 1197 1330 1463 1596 1729 1862 1995 2128 2394 2527 2660 2793 2926 3192)
pdes=(4 45 5)
apas=(1 2 3 4)

for vgain in "${vgains[@]}"; do
  for pde in "${pdes[@]}"; do
    pde_dir="0.${pde}"           # matches 0.<PDE> directory, e.g. 0.45
    for apa in "${apas[@]}"; do
      src="${SRC_BASE}/vgain_run_${vgain}/${pde_dir}/apa_${apa}/single_channel_results_filters.csv"
      dst="${DST_DIR}/vgain_${vgain}_0P${pde}_apa_${apa}_result.csv"

      if [[ -f "$src" ]]; then
        echo "Copying: $src -> $dst"
        cp "$src" "$dst"
      else
        echo "WARNING: file not found: $src" >&2
      fi
    done
  done
done
