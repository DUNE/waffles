#!/bin/bash

# Define the default verbosity
verbose=true

# Parse the arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)
            verbose=true
            shift
            ;;
        -nv|--no-verbose)
            verbose=false
            shift
            ;;
        *)
            echo "Use: $0 [-v|--verbose] [-nv|--no-verbose]"
            exit 1
            ;;
    esac
done

# Get the path to the directory from which this script is being executed 
current_dir=$(pwd)

# Ask for confirmation
echo "The minimal analysis structure will be created in $current_dir, continue? (yes/no): "
read answer

# Convert the answer to lower case
lower_case_answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')

# Check whether the script can proceed
if [[ "$lower_case_answer" == "y" || "$lower_case_answer" == "yes" ]]; then
    # Create the minimal analysis structure
    mkdir -p "$current_dir/configs"
    mkdir -p "$current_dir/output"
    mkdir -p "$current_dir/data"
    mkdir -p "$current_dir/scripts"
    touch "$current_dir/steering.yml"
    touch "$current_dir/utils.py"
    touch "$current_dir/params.yml"
    touch "$current_dir/imports.py"
    touch "$current_dir/Analysis1.py"

    # Show the message only if running with verbosity
    if [[ "$verbose" == "true" ]]; then
    	echo "The minimal analysis structure has been created in $current_dir"
    fi
else
    echo "The action was canceled."
fi