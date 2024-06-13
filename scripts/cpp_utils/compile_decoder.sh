#!/bin/bash
##########################################################################################################################
## Script to compile our C++ decoder (HDF5 to ROOT files) in ProtoDUNE-HD.
## This script need to be run just the first time you want to use the decoder.
## Requirements: You need to be inside the daq_environment to run this script.
## Usage: sh compile_decoder.sh
## Created by: Jairo Rodriguez (jairorod@fnal.gov) + Laura Pérez-Molina (laura.perez@cern.ch)
##########################################################################################################################

echo -e "\e[36m[INFO] Welcome to our tool for compiling the decoder"
echo -e "\e[36m To execute the script just run: sh compile_decoder.sh"
echo -e " This process is needed just the first time you want to run this tool "
echo -e " Enjoy! :) \n \e[0m"

waffles_path=$(realpath ${BASH_SOURCE[0]})
waffles_path=$(echo $waffles_path | sed 's|\(.*\)/.*|\1|')

echo -e "\e[35mCloning hdf5libs repo in your env sourcecode folder: \n \e[0m"
git clone https://github.com/DUNE-DAQ/hdf5libs.git $DBT_AREA_ROOT/sourcecode/hdf5libs

echo -e "\e[35m Compiling hdf5libs (1): \n \e[0m"
cd $DBT_AREA_ROOT
dbt-build --clean


echo -e "\e[35mCopying the decoder and CMakeLists.txt in the correct folders... \n \e[0m"
cp $waffles_path/HDF5toROOT_decoder.cpp $DBT_AREA_ROOT/sourcecode/hdf5libs/test/apps/
cp $waffles_path/HDF5LIBS_duplications.cpp $DBT_AREA_ROOT/sourcecode/hdf5libs/test/apps/
cp $waffles_path/CMakeLists.txt $DBT_AREA_ROOT/sourcecode/hdf5libs/

echo -e "\e[35mCompiling the added decoder (2): \n \e[0m"
cd $DBT_AREA_ROOT
dbt-build --clean

echo -e "\e[32m\nFor using the decoder just type HDF5LIBS_DumptoROOT_filtered <input_file_name> <channel_map_file>\e[0m"
