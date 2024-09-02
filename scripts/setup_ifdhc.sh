#!/bin/bash
export UPS_OVERRIDE="-H Linux64bit+3.10-2.17"
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
kx509
export ROLE=Analysis
voms-proxy-init -rfc -noregen -voms=dune:/dune/Role=$ROLE -valid 120:00
setup ifdhc
ifdh cp /pnfs/fnal.gov/usr/dune/tape_backed/dunepro//hd-protodune/raw/2024/detector/cosmics/None/00/02/63/65/np04hd_raw_run026365_0001_dataflow0_datawriter_0_20240522T135540.hdf5 .