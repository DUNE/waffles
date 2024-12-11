#!/bin/bash

# Remember to kinit your_fnal_username@FNAL.GOV 
# before running this script
# RUCIO_ACCOUNT=your_fnal_username
export RUCIO_ACCOUNT=your_fnal_username

# Either vd-coldbox, hd-protodune...
export DETECTOR=vd-coldbox

# List the runs you want
runs=(
33619
33620
)

source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup python v3_9_15
setup rucio
setup kx509
rucio whoami


subdir=${1:-""}

RUCIOOUT=$( mktemp )
echo "RUCIO OUT: ${RUCIOOUT}"
for run in ${runs[*]}; do
  dataset="./files_location_cb${subdir}/0${run}.txt"

  if [ -f $dataset ]; then
    if [ -s $dataset ]; then
      echo "${run} already there.."
      continue
    fi
  fi
  echo "Getting run $run"
  rucio list-file-replicas --pfns ${DETECTOR}:${DETECTOR}_${run} | grep -v "tpwriter" > $RUCIOOUT
  replica=$( grep eos $RUCIOOUT | grep eospublic )
  if [[ -z $replica ]]; then
    replica=$( grep xroot $RUCIOOUT )
  fi
  if [[ -z $replica ]]; then
    replica=$( grep golias $RUCIOOUT )
  fi
  if [[ -z $replica ]]; then
    replica=$( grep sdcc $RUCIOOUT )
  fi
  if [[ -z $replica ]]; then
    echo "No file replica for ${run}"
    cat $RUCIOOUT
    continue
  fi
  echo "$replica" >> $dataset
done

echo "Done..."



