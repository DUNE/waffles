#!/bin/bash
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup python v3_9_15

setup rucio 
setup kx509

kdestroy
read -p "Enter your @FNAL.GOV username: " username
echo "Please enter your password: "
read -s password
echo "${password}" | kinit ${username}@FNAL.GOV
kx509

export RUCIO_ACCOUNT=${username}
rucio whoami