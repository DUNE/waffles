#!/bin/bash
##########################################################################################################################
## Script to find path of protodune files. It will be used to access the data acquired by the DAQ from any lxplus machine.
## Usage: source get_protodunedhd_files.sh <localgrid> <where> <run>
# Example  source get_protodunedhd_files.sh local cern 25107 --> will return a list with the path of the files in /eos
## Created by: Jairo Rodriguez (jairorod@fnal.gov) + Laura Pérez-Molina (laura.perez@cern.ch)
##########################################################################################################################

os_name=$(grep "^NAME" /etc/os-release | cut -d '=' -f 2 | tr -d '"')
version_id=$(grep VERSION_ID /etc/os-release | cut -d '=' -f 2 | tr -d '"')
current_dir=$(pwd)

# Check if the user has provided the arguments and if not, ask for them
if [ -n "$1" ];then
    localgrid=$1
    else
        read -p "Enter where the files are [local/grid]: " localgrid
         while [[ $localgrid != "local" && $localgrid != "grid" ]]; do
               read -p "Invalid entry, try again. Enter the localgrid [local/grid]: " localgrid
         done
fi
if [ -n "$2" ];then
    where=$2
    else
        read -p "Enter where are you [cern/fnal]: " where
         while [[ $where != "cern" && $where != "fnal" ]]; do
               read -p "Invalid entry, try again. Enter where [cern/fnal/all]: " where
         done
fi
if [ -n "$3" ];then
    run=$3
    else
        read -p "Enter the run number [25107]: " run
         while [[ ! $run =~ ^[0-9]+$ ]]; do
               read -p "Invalid entry, try again. Enter the run number [25107]: " run
         done
fi

if [ ${#run} -ne 6 ]; then
   run0=$(printf "%06d" $run)
   else
      run0=$run
fi  
rucio_paths_file="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/${run0}.txt"
# Check if already someone have run this script and rucio paths are stored already
if [ -f ${rucio_paths_file} ]; then
   echo -e "\e[92mRucio paths already stored :) No rucio setup needed!!\e[0m"
   # Read the file and print the paths
   cat ${rucio_paths_file}

   else
   echo "You are the first one to look at the paths of run ${run0}!!"
      # Check if rucio is sourced
      if rucio whoami; then
         echo -e "\e[92mRucio loaded successfully!!\e[0m"
         else
            echo -e "\e[31mConfiguring rucio in ${os_name}: ${version_id}\e[0m"
            if [[ ($os_name == "CentOS Linux" || $os_name == "Scientific Linux") && $version_id == 7* ]]; then
               wget https://authentication.fnal.gov/krb5conf/SL7/krb5.conf
               export KRB5_CONFIG="$current_dir/krb5.conf"
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
            fi
            if [[ $os_name == "Red Hat Enterprise Linux" && $version_id == 9* ]]; then
               wget https://authentication.fnal.gov/krb5conf/SL7/krb5.conf
               export KRB5_CONFIG="$current_dir/krb5.conf"
               source /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh
               spack load r-m-dd-config experiment=dune
               spack load kx509
               kdestroy
               read -p "Enter your @FNAL.GOV username: " username
               echo "Please enter your password: "
               read -s password
               echo "${password}" | kinit ${username}@FNAL.GOV
               kx509

               export RUCIO_ACCOUNT=${username}
               rucio whoami
            fi
      fi
      
      replicas=$( rucio list-file-replicas --pfns hd-protodune:hd-protodune_${run} )
      for line in $replicas; do
         if [[ $line == *$where* ]]; then
            case $where in
            cern)
               if [[ $line ==  *"experiment/neutplatform"* ]];then
               case $localgrid in
               local)
               foo="/eos"${line//*'//eos'/} # remove everything before //eos
               echo $foo | tee -a $HOME/${run0}.txt
               ;;
               grid)
               echo $line | tee -a $HOME/${run0}.txt
               ;;
               esac
               fi
            ;;
            fnal)
               case $localgrid in
               local)
               foo="/pnfs"${line//*'/pnfs'/} 
               # fbb=${foo//'dunepro/'/'dunepro'} # remove everything before /dunepro
               echo $foo | tee -a $HOME/${run0}.txt
               ;;
               grid)
               echo $line | tee -a $HOME/${run0}.txt
               ;;
               esac
            ;;
            esac
         fi
      done
fi

rm -f $current_dir/krb5.conf*
kdestroy