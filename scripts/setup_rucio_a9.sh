#!/bin/bash
# set -e  # Exit if any command fails

# Load the required environment
source /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh

# Load necessary packages
spack load r-m-dd-config experiment=dune
spack load kx509

# Prompt for FNAL credentials
read -p "Enter your @FNAL.GOV username: " username
read -s -p "Please enter your password: " password
echo ""

# Authenticate with Kerberos
echo "${password}" | kinit "${username}@FNAL.GOV"
unset password  # Remove password from memory for security

# Obtain Kerberos-based certificate
kx509

# Set Rucio account
export RUCIO_ACCOUNT="${username}"

# Verify Rucio authentication
rucio whoami

# Initialize VOMS proxy
voms-proxy-init -rfc -noregen -voms=dune:/dune/Role=Analysis -valid 120:00

# Set UPS override
export UPS_OVERRIDE="-H Linux64bit+3.10-2.17"

# Load IFDH for file handling
setup ifdhc

echo -e "\033[92mEnvironment setup complete. You are now authenticated for Rucio.\033[0m"