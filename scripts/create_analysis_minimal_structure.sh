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

    if [ -e "$current_dir/steering.yml" ]; then
        echo "Warning: $current_dir/steering.yml already exists"
    else
	    printf "1:\n  name: \"Analysis1\"\n  parameters_file: \"params.yml\"\n  overwriting_parameters: \"\"" > $current_dir/steering.yml
    fi

    if [ -e "$current_dir/params.yml" ]; then
        echo "Warning: $current_dir/params.yml already exists"
    else
	    printf "input_path: \"\"\noutput_path: \"output/\"" > $current_dir/params.yml
    fi

    if [ -e "$current_dir/imports.py" ]; then
        echo "Warning: $current_dir/imports.py already exists"
    else
	    printf "from pydantic import Field\nfrom waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams" > $current_dir/imports.py
    fi

    touch "$current_dir/utils.py"

    if [ -e "$current_dir/Analysis1.py" ]; then
        echo "Warning: $current_dir/Analysis1.py already exists"
    else
        if [ -e "$current_dir/../example_analysis/Analysis1.py" ]; then
            cp $current_dir/../example_analysis/Analysis1.py $current_dir/Analysis1.py
        else
            touch "$current_dir/Analysis1.py"
            echo "Warning: The example analysis $current_dir/../example_analysis/Analysis1.py was not found. An empty $current_dir/Analysis1.py file was created."
        fi
    fi

    # Show the message only if running with verbosity
    if [[ "$verbose" == "true" ]]; then
    	echo "The minimal analysis structure has been created in $current_dir"
    fi
else
    echo "The action was canceled."
fi
