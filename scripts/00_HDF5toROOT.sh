#!/bin/bash

echo -e "\e[36m[INFO] Welcome to the script decoding the HDF5 files using the CPP TOOLS [make sure you have compiled the decoder]! \e[0m"
echo -e "\e[36m\n To execute the script just run: sh 00_HDF5toROOT.sh."
echo -e " Optionally [1st argument] the run number (separated by commas: 02ABC,02XYZ), [2nd,3rd] first and number of hdf5 files to process (default 0, -1 --> ALL) and [4th] path to save the root files (default /eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/)."
echo -e " You can run the script without any flag and the required arquments will be asked."
echo -e " Example: sh 00_HDF5toROOT.sh 27644,27645 3 5 --> will process 5 files starting from the 4rd (index 3) file of the run 27644 and 27645 and save the output in the default path. \n "
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

if [ -n "$2" ];then
    ini_nfile=$2
    else
        ini_nfile=0
        echo -e "Processing from the first rucio_path"
fi

if [ -n "$3" ];then
    end_nfile=$3
    else
        end_nfile=-1
        echo -e "Processing until the last rucio_path"
fi

if [ -n "$4" ];then
    save_path=$4
    else
        save_path="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/"
        echo -e "Saving the output files in the default /eos/ path"
fi


read -p "Do you want to run the DUMP an output root file in eos (1) or just a DUPLICATION check (2)? (1/2) " script_mode
declare -A mode_script_map
mode_script_map[1]="HDF5toROOT_decoder"
mode_script_map[2]="HDF5LIBS_duplications"

run_string=$(IFS=,; echo "${run_number[*]}")
echo -e "\e[31mWARNING: You are about to run the CPP ${mode_script_map[$script_mode]} for runs: ["${run_string[@]}"] from rucio_path #$ini_nfile and n=$end_nfile!\e[0m"
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

    rucio_paths=$(cat /eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/$run.txt)
    if [ -z "$rucio_paths" ]; then
        python get_rucio.py --runs $run
        rucio_paths=$(cat /eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/$run.txt)
    fi

    echo -e "\e[32m --> Raw .hdf5 file found at $rucio_paths \e[0m"
    
    if [ ${mode_script_map[$script_mode]} == "HDF5toROOT_decoder" ]; then
        cd ${save_path}
        if [ ! -d "run_$run" ]; then  #if run_run_number folder does not exist create it
            mkdir run_$run
            chmod 777 run_$run
        fi

        cd run_$run
        echo -e "\e[35m\n... Running HDF5toROOT_decoder ...\n \e[0m"
    fi

    # Convert rucio_paths to an array
    rucio_paths_array=($rucio_paths)
    if [ $end_nfile -eq -1 ]; then
        end_nfile=16 #HARDCODED 16 files per run (TEMPORAL)
        # end_nfile=${#rucio_paths_array[@]} #if the user wants to process all the files
    fi

    # Get the subset of rucio_paths between indices $2 and $3
    rucio_paths_subset=${rucio_paths_array[@]:$ini_nfile:$end_nfile}
    for rucio_path in $rucio_paths_subset #check if there are several lines in the txt and run a loop over them
    do
        ${mode_script_map[$script_mode]} $rucio_path 
        
        new_name=$(basename "$rucio_path")
        new_name=${new_name#np04hd_raw_}
        new_name=${new_name%.hdf5}
        echo -e "\e[32m --> The output file is $new_name.root\e[0m"
        
        # Get the most recently created file in the current directory
        file=$(ls -t | head -n1)
        # Rename the file
        mv "$file" "$new_name.root"
        
    done
    cd $waffles_path

done
