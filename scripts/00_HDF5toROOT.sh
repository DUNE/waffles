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
    if [ ${#run} -ne 6 ]; then
        run=$(printf "%06d" $run)
    fi  
    echo -e "\e[35m\nGetting the paths for the run(s) $run... \n \e[0m"
    rucio_paths=$(cat /eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/$run.txt)

    # run get_rucio script with the run number to get the paths if the file does not exist
    if [ -z "$rucio_paths" ]; then
        python get_rucio.py --runs $run
        rucio_paths=$(cat /eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/$run.txt)
    fi

    echo -e "\e[32m --> Raw .hdf5 file found at $rucio_paths \e[0m"
    
    # Check if the output directory exists and create it if it does not
    cd ${save_path}
    if [ ! -d "run_$run" ]; then  # if run_run_number folder does not exist create it
        mkdir run_$run            # Create a directory for each run
        chmod 777 run_$run        # Give permissions to the directory (to be removed if stable root files are produced)
    fi

    cd run_$run
    echo -e "\e[35m\n... Running ${mode_script_map[$script_mode]} ...\n \e[0m"

    # Convert rucio_paths to an array
    rucio_paths_array=($rucio_paths)
    if [ $end_nfile -eq -1 ]; then
        end_nfile=16 # HARDCODED 16 files per run (TEMPORAL)
        # end_nfile=${#rucio_paths_array[@]} #if the user wants to process all the files
    fi

    # Get the subset of rucio_paths between indices $2 and $3
    rucio_paths_subset=${rucio_paths_array[@]:$ini_nfile:$end_nfile}
    for rucio_path in $rucio_paths_subset #check if there are several lines in the txt and run a loop over them
    do
        if [ ${mode_script_map[$script_mode]} == "HDF5toROOT_decoder" ]; then
                new_name=$(basename "$rucio_path") # Get the name of the file
                new_name=${new_name#np04hd_raw_}   # Remove the prefix
                new_name=${new_name%.hdf5}         # Remove the suffix

                #Check if the output file already exists in the folder and ask if want to overwrite
                if [ -f "$new_name.root" ]; then
                    echo -e "\e[31m\nWARNING: The output file $new_name.root already exists in the folder!\e[0m"
                    read -p "Do you want to overwrite it? (y/n) " -n 1 -r
                    if [[ ! $REPLY =~ ^[Yy]$ ]]
                    then
                        continue
                    fi
                fi
                
                ${mode_script_map[$script_mode]} $rucio_path 
            
                file=$(ls -t | head -n1)    # Get the most recently created file in the current directory
                mv "$file" "$new_name.root" # Rename the file (same as the hdf5 file and python script output)
                echo -e "\e[32m --> The output file is $new_name.root\n\e[0m"
            else
                ${mode_script_map[$script_mode]} $rucio_path >> run${run}_duplications.txt
        fi

    done

    # Check if the mode is HDF5toROOT_decoder and merge the root files if necessary
    if [ ${mode_script_map[$script_mode]} == "HDF5toROOT_decoder" ]; then
        declare -A groups
        files=$(ls .) # Get the files in the directory

        # Loop over the files
        for file in $files
        do
            prefix=$(echo $file | cut -d'_' -f2) # Get the prefix of the file name
            groups[$prefix]+="$file " # Add the file to the appropriate group
        done

        # Loop over the groups
        for prefix in "${!groups[@]}"
        do
            
            group=(${groups[$prefix]}) # Get the group
            common_name=${group[0]} # Get the common name from the first file in the group
            # Replace the dataflow number in the common name with 0-3
            common_name=$(echo $common_name | awk -F'dataflow' '{print $1 "dataflow0-3_" substr($0, index($0,$2)+2)}')

            # Count the number of files in the group that are smaller than 200MB
            small_files=0
            for file in ${group[@]}
            do
                size=$(du -m "${file}" | cut -f1)
                if [ ! -z "$size" ] && [ $size -lt 200 ]; then
                    small_files=$((small_files + 1))
                fi
            done

            #If more than 1 file in the group smaller than 200MB, merge them:
            if [ $small_files -gt 1 ]; then
                echo -e "\n\nMerging group ${prefix}... into ${common_name}"
                echo -e "\nFILES TO BE REMOVED ${group[@]}"

                read -p "Do you want to merge the files in the group ${prefix}? (y/n) " -n 1 -r
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    hadd -k ${common_name} ${group[@]} # Run the hadd command
                    rm ${group[@]} # Remove the files in the group
                fi
                
                else
                    echo -e "\nNo small files so NO merging root files ${prefix}"
            fi
        done
    fi

    cd $waffles_path

done