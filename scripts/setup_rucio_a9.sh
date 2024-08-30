#!/bin/bash
# Source the setup script for the environment
source /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh

# Load the necessary packages
spack load r-m-dd-config experiment=dune
spack load kx509

# Prompt the user for their username and password
read -p "Enter your @FNAL.GOV username: " username
echo "Please enter your password: "
read -s password

# Authenticate with kinit and obtain a Kerberos ticket
echo "${password}" | kinit ${username}@FNAL.GOV

# Obtain a certificate from the Kerberos ticket
kx509

# Set the Rucio account environment variable
export RUCIO_ACCOUNT=${username}

# Check the Rucio identity
rucio whoami

# Ensure the script exits cleanly
exit 0