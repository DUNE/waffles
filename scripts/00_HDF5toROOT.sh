#!/bin/bash

echo -e "\e[36m[INFO] Welcome to the script decoding the HDF5 files using the CPP TOOLS [make sure you have compiled the decoder]! \e[0m"
echo -e "\e[36m\n To execute the script just run: sh 00_HDF5toROOT.sh and optionally the run number as first argument (separated by commas: 02ABC,02XYZ)"
echo -e " You can run the script without any flag and the required arquments will be asked"
echo -e " Enjoy! :) \n \e[0m"

# The confirmation message need to be run with $ bash setup.sh (this lines are to allow $ sh setup.sh too)
if [ ! "$BASH_VERSION" ] ; then
    exec /bin/bash "$0" "$@"
fi

waffles_path=$(realpath ${BASH_SOURCE[0]})
waffles_path=$(echo $waffles_path | sed 's|\(.*\)/.*|\1|')

# Check if the arguments are provided and if not ask for them
if [ -n "$1" ];then
    run_number=$1
    else
        read -p "Please provide a run(s) number(s) to be analysed, separated by commas :) " run_number
fi

read -p "Do you want to run the DUMP an output root file in eos (1) or just a DUPLICATION check (2)? (1/2) " script_mode
declare -A mode_script_map
mode_script_map[1]="HDF5toROOT_decoder"
mode_script_map[2]="HDF5LIBS_duplications"

run_string=$(IFS=,; echo "${run_number[*]}")
echo -e "\e[31mWARNING: You are about to run the CPP ${mode_script_map[$script_mode]} for runs: ["${run_string[@]}"] !\e[0m"
read -p "Are you sure you want to continue? (y/n) " -n 1 -r
# If the user did not answer with y, exit the script
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi


# Loop over the list of runs
run_string=(${run_string//,/ })
for run in ${run_string[@]}
do
    run=$(printf "%06d" $run)
    # run get_rucio script with the run number to get the paths
    echo -e "\e[35m\nGetting the paths for the run(s) $run... \n \e[0m"

    rucio_path=$(cat /eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/$run.txt)
    if [ -z "$rucio_path" ]; then
        python get_rucio.py --runs $run
        rucio_path=$(cat /eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/$run.txt)
    fi

    echo -e "\e[32m --> Raw .hdf5 file found at $rucio_path \e[0m"
    
    if [ ${mode_script_map[$script_mode]} == "HDF5toROOT_decoder" ]; then
        cd /eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root
        # if run_run_number folder does not exist create it
        if [ ! -d "run_$run" ]; then
            mkdir run_$run
        fi

        cd run_$run
        echo -e "\e[35m\n... Running HDF5toROOT_decoder ...\n \e[0m"
    fi

    ${mode_script_map[$script_mode]} $rucio_path
    cd $waffles_path
done