#!/bin/bash

# RUN: sh scripts/run_display.sh
# OUTPUT: display app in your browser by following the generated link

#Script must be run from MAIN folder or paths will be messed up
# echo -e "\e[31mWARNING: Only run this script from MAIN directory!\e[0m"
# # Ask if sure to continue
# read -p "Are you sure you want to continue? (y/n) " -n 1 -r
# echo
# # If the user did not answer with y, exit the script
# if [[ ! $REPLY =~ ^[Yy]$ ]]
# then
#     exit 1
# fi

# pip install .

python3 - <<EOF
from np04_display import Display
Display().run_app()
EOF