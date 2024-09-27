# 🤖 **SCRIPTS** 

<!-- Missing to add expected outputs -->

## Obtain file paths

Before starting with the analysis you may need to get the paths of the HDF5 files you want to use. To get this `rucio_path` you can find some tools listed below:
* `get_protodunehd_files.sh`: this script return the filepaths of the input run if `rucio` is setup correctly (if not it tries and setup it per run).
* `setup_rucio_*.sh`: scripts to setup the `rucio` environment variables in your terminal. After that you can produce all the paths you need without re-authenticating.
* `get_rucio.py`: script that wraps the needed tools to save the paths of the HDF5 files you want to decode and move it to the shared `eos` folder. This script will ask for the run number and the number of files you want to get. The output will be a list of paths to the HDF5 files. Moreover, this scripts handles the runs that have already been moved to tape and are not available in the `eos` folder. In this case, the script will ask for the tape location `root://filepath/`.

This is summarized in the following steps:

#### Get path for a single run
```bash
python get_rucio.py # Optional argument --runs 27632
```

#### Get path for several runs (without re-authenticating)
```bash
source setup_rucio_a9.sh
python get_rucio.py # Optional argument --runs 27632,27633
```

For using these scripts you need valid `FNAL` credentials. Have a look at the [RUCIO DOCS](https://github.com/DUNE/data-mgmt-ops/wiki/Using-Rucio-to-find-Protodune-files-at-CERN/) for more details of this process.

## 00_HDF5toROOT

We have developed two decoders (`C++` and `Python`) which output is a `root` file with the same structure. After running them check `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/run_YOUR_RUN_NUMBER/` for the output `root` files.

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

<p style="color: teal;">[INFO] Welcome to the script decoding the HDF5 files using the CPP TOOLS [make sure you have compiled the decoder]!. To execute the script just run: sh 00_HDF5toROOT.sh. Optionally: </p>
<p style="color: teal;">* [1st argument] the run number (separated by commas: 02ABC,02XYZ), </p>
<p style="color: teal;">* [2nd,3rd] first and number of hdf5 files to process (default 0, -1 --> ALL) and </p>
<p style="color: teal;">* [4th] path to save the root files (default /eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/). </p>
<p style="color: teal;">You can run the script without any flag and the required arguments will be asked </p>
<p style="color: teal;">Example: sh 00_HDF5toROOT.sh 27644,27645 3 5 --> will process 5 files starting from the 4rd (index 3) file of the run 27644 and 27645 and save the output in the default path </p>
<p style="color: teal;">Enjoy! :)</p>

The user should input the run number (`27632` or a list as `26755,26756`) and choose the running mode (`1=decoder`, `2=duplications check`):

* <p>Please provide a run(s) number(s) to be analysed, separated by commas :)</p> 

* <p>Do you want to run the DUMP an output root file in eos (1) or just a DUPLICATION check (2)? (1/2)</p>

The outputs will be saved in the default eos path `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root/run_XXXX` or in the path provided by the user.

```{tip} 
**CPP UTILS**

Once you have compiled the cpp macros (`HDF5toROOT_decoder` and `HDF5LIBS_duplications`) you can use them from the command line from wherever you are. The bash script `00_HDF5toROOT.sh` is just a wrapper to manage the input arguments and the output paths. Therefore if you have an output `.hdf5` file not included in rucio you can decode it by running: `HDF5toROOT_decoder /path/to/your/file.hdf5`

```


### **Python**

1. Run the decoder

```bash
python 00_HDF5toROOT.py # Run the Python script to decode the HDF5 files

# You can also give the run number as an argument (python 00_HDF5toROOT.py --runs 27632)
```

In order to use this decoder you need to have `ROOT` installed in your virtual environment. If you don't have it, you can install it by running:

**_UNDER TESTING_**

```bash
# INSIDE THE VIRTUAL ENVIRONMENT (you have source env.sh before)

git clone --branch latest-stable --depth=1 https://github.com/root-project/root.git root_src

mkdir root_build root_install && cd root_build

cmake -DCMAKE_INSTALL_PREFIX=../root_install -Ddataframe=OFF ../root_src

cmake --build . -- install -j1
```

If some error appears during the installation related with Roofit, just disable it by adding the command ``-Droofit=OFF`` on the cmake.

After that, every time you log in, you need to source ROOT, or you can edit ``env.sh`` and add this command:

``source ../root_install/bin/thisroot.sh``


## 01_Process

This script is used to convert the raw `.hdf5` files into `waffles` classes and save it locally in `.pkl` format. In order to run this script make sure you have a `waffles/data` folder to store the output files. In summary what this script does is:
```python
import waffles.input.raw_hdf5_reader as reader

rucio_files = f"/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/028602.txt"
allfilepath = reader.get_filepaths_from_rucio(rucio_filepath)
waveformset = reader.WaveformSet_from_hdf5_files( filepaths[:int(file_lim)], read_full_streaming_data = False)
with open(f"data/{run}_full_wfset_raw.pkl", "wb") as f:
            pickle.dump(wfset, f)
```
Some decorators are included to require the user inputs, such as the run number and the number of files to process. This allows the analysers to have a common starting point and avoid the need to change the code every time they want to process a new run.

## 02_RawAna

Following the same structure as the previous script, this one reads the `WaveformSet` objects stored in the `.pkl` files from `01_Process` and performs some basic analysis. The goal of this script is to provide the user quick visualization of the BasicAnalysis output channel by channel. Make sure you have a `waffles/config/run_number.json` folder and file with the input parameters for the analysis. A template for these configuration files could be:

```python
{
    "standard": {                                       #analysis_label
                    "base_lim": [0, 70, 900, 1000],     #limits for the baseline computation
                    "int_ll": 50,                       #lower limit for the integral computation
                    "int_ul": 130,                      #upper limit for the integral computation
                    "amp_ll": 0,                        #lower limit for the amplitude computation
                    "amp_ul": 1000                      #upper limit for the amplitude computation
                },
    "avanced": { ... }
}

```

After filtering the full `WaveformSet` object, a plot will be displayed using `plot_ChannelWsGrid` from the `waffles.plotting` module. This plot will show 10 `Waveform` objects together with the baseline and analysis limits. Additionally, the analyser can plot the waveforms objects using the `plot_WaveformAdcs` function from the same module with a loop as follows:
```python
    for wf in filter_wfset.waveforms:
        plot_WaveformAdcs(  wf, figure = figure,
                            plot_analysis_markers = True,
                            show_baseline_limits = True, 
                            show_baseline = True,
                            show_general_integration_limits = True,
                            show_general_amplitude_limits = True,
                            show_spotted_peaks = True,
                            show_peaks_integration_limits = False)
```