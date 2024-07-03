# 🤖 **SCRIPTS** 

<!-- Missing to add expected outputs -->

## 00_HDF5toROOT

We have developed two decoders (`C++` and `Python`) which output is a `root` file with the same structure. 

The common input for both decoders is the `rucio_path` of the HDF5 file to be decoded. For that reason, we have developed a script to get them (`get_rucio.py`) which can be used standalone but is already integrated in the decoders. For using this script you need valid `FNAL` credentials. Have a look at the [RUCIO DOCS](https://github.com/DUNE/data-mgmt-ops/wiki/Using-Rucio-to-find-Protodune-files-at-CERN/) for more details of this process.

After running them check `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/run_YOUR_RUN_NUMBER/` for the output `root` files.

To use the decoders can follow the next instructions:

### **C++**

1. Compile the scripts (just the first time)

```bash
cd scripts/cpp_utils # Go to the C++ scripts folder
./compile_decoder.sh # Run the script to compile the C++ scripts (will clone the HDF5 library and compile it together with the decoder)
```

2. Run the decoder

```bash
cd ..               # Go back to the scripts folder
sh 00_HDF5toROOT.sh # Run the bash script to manage the C++ macros

# You can also give the run number as an argument (sh 00_HDF5toROOT.sh 27632)
```

During the execution of the `bash` script you will be asked for the run number to be processed:

<p style="color: cyan;">[INFO] Welcome to the script decoding the HDF5 files using the CPP TOOLS [make sure you have compiled the decoder]!</p>
<p style="color: cyan;">To execute the script just run: sh 00_HDF5toROOT.sh and optionally the run number as first argument (separated by commas: 02ABC,02XYZ)
You can run the script without any flag and the required arguments will be asked
Enjoy! :)</p>

The user should input the run number (`27632` or a list as `26755,26756`) and choose the running mode (`1=decoder`, `2=duplications check`):

* <p>Please provide a run(s) number(s) to be analysed, separated by commas :)</p> 

* <p>Do you want to run the DUMP an output root file in eos (1) or just a DUPLICATION check (2)? (1/2)</p>

### **Python**

1. Run the decoder

```bash
python 00_HDF5toROOT.py # Run the Python script to decode the HDF5 files

# You can also give the run number as an argument (python 00_HDF5toROOT.py --runs 27632)
```

In order to use this decoder you need to have `ROOT` installed in your virtual enviroment. If you don't have it, you can install it by running:

**_UNDER TESTING_**

```bash
# INSIDE THE VIRTUAL ENVIRONMENT (you have source env.sh before)

git clone --branch latest-stable --depth=1 https://github.com/root-project/root.git root_src

mkdir root_build root_install && cd root_build

cmake -DCMAKE_INSTALL_PREFIX=../root_install -Ddataframe=OFF ../root_src

cmake --build . -- install -j1
```

If some error appears during the installation related with Roofit, just disable it by adding the command ``-Droofit=OFF`` on the cmake.

After that, everytime you login, you need to source ROOT, or you can edit ``env.sh`` and add this command:

``source ../root_install/bin/thisroot.sh``


## 01_ANA...
