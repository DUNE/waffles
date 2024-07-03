# 🔍 **INTRODUCTION**


This is a python library to process and analyze raw data from the ProtoDUNEs. The design objectives were:

* **Unify** the tools and efforts into a common framework for the PDS working group.
* Avoid over calculation as possible

🧐 **<u> OBJECTIVES </u>** 

1. Characterize the detector's response
    * SPE Calibration (Gain, persistence)
    * ...
2. Physics studies
    * Signal deconvolution
    * Physics fits
3. Electronics studies --> DAPHNE response ?


## Contributing

The idea of WAFFLES framework is to unify efforts and to develop a robust analysis tool. 
Nevertheless, it is very important to keep some common rules not to harm others work.

```{tip} 
**Good coding practises here**

    *   Create your own branch for developing code and make puntual commits with your changes. Once you want to share with the world, open a pull request and wait for two reviewers to approve the merging.

    * To include functions/methods... [COMPLETE]

```

## Current Workflow

<!-- IMAGE SUMMARISING THE WORKFLOW? -->

After running our extractors (see `scripts/00_HDF5toROOT`) a folder will be generated in `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root` with your selected run. The structure inside this `.root` files is:

```bash
├── 'metadata'/
    ├── [integer]   'endpoint'
    ├── [integer]   'threshold'
    ├── integer     'run'
    ├── integer     'nrecords'
    ├── string      'detector'
    ├── string      'date'
    ├── integer     'ticks_to_nsec'
    └── integer     'adcs_to_volts'
    ├── integer     'daq_window_size'
    └── integer     'daq_pre_trigger'
└── 'raw_waveforms'/
    ├── [integer]   'record'
    ├── [integer]   'daq_timestamp'
    ├── [[integer]] 'adcs'
    ├── [integer]   'timestamp'
    ├── [integer]   'channel'
    ├── [integer]   'baseline'
    ├── [integer]   'trigger_sample_value'
    └── [bool]      'is_fullstream'
```

This file is used to debug and check the quality of the data but in future releases of `waffles` we will load directly the `.hdf5` daq files.

The next steps are loading the `root` files (you can select the fraction of statistics you want to analyse) and start visualizing your data,


## **Getting Started - SETUP**  ⚙️

If it is your first time here you need to create a `daq_env` to be able to use all their tools:

```bash
source /cvmfs/dunedaq.opensciencegrid.org/setup_dunedaq.sh

setup_dbt latest
dbt-create -l 
dbt-create fddaq-v4.4.3-a9 <my_dir>
```

We recommend installing [VSCode](https://code.visualstudio.com/) as editor. Some useful extensions are: Remote-SSH, Jupyter, vscode-numpy-viewer, **Python Environment Manager**

### 0. Download the library by cloning it from GitHub

```bash
git clone https://github.com/DUNE/waffles.git # Clone the repository
git checkout -b <your_branch_name>            # Create a branch to develop
```

The expected folder structure of the repository should be

```bash
├── 'docs'/ # FOLDER WITH THE REPO DOCUMENTATION (THIS TEXT CAN BE IMPROVED BY YOU!)
    ├── 'examples'/
        └── '4_Examples.rst'
    ├── '1_Intro.md'
    ├── '2_Scripts.md'
    ├── '3_Libraries.rst'
    ├── 'conf.py'
    └── 'requirements.txt'

└── 'scripts'/ # FOLDER WITH THE SCRIPTS
    ├── 'cpp_utils'/ # C++raw functions and scripts (can be used in standalone mode) [Thanks Jairo!]
        ├── 'functions'/
            ├── 'channelmap.txt'
            ├── 'hdf5torootclass.h'
            └── 'wffunctions.h'
        ├── 'CMakeLists.txt'
        ├── 'compile_decoder.sh' #Script to compile c++ scripts (just 1st time) and be able to use them
        ├── 'HDF5LIBS_duplications.cpp'
        ├── 'HDF5toROOT_decoder.cpp'
        ├── 'plotsAPA.C'
        └── 'README.md'
    ├── '00_HDF5toROOT.py' # Python decoder (hdf5 to root) with multithreading
    ├── '00_HDF5toROOT.sh' # Bash script for managing CPP macros. If you already compiled (cpp_utils) them you can run this one.
    ├── 'get_protodunehd_files.sh' # Script to get rucio_paths from the hdf5 daq files
    ├── 'get_rucio.py' # RUN to make rucio_paths sincronize with /eos/ folder. You will save time and make others save time too!
    ├── 'README.md'
    └── 'setup_rucio.sh' # Standalone commands for setting up rucio once you are inside a SL7

└── 'src'/  # MAIN CODE CORE WITH ALL THE CLASSES DEFINITIONS HERE#
    ├── 'waffles'/
        ├── '__init__.py'
        ├── 'APAmap.py'
        ├── 'CalibrationHistogram.py'
        ├── 'ChannelWS.py'
        ├── 'ChannelWSGrid..py'
        ├── 'Exceptions.c.py'
        ├── 'Map.c.py'
        ├── 'UniqueChannel.py'
        └── 'Waveform.py'
        └── 'WaveformAdcs.py'
        └── 'WaveformSet.py'
        └── 'WfAna.py'
        └── 'WfAnaResult.py'
        └── 'WfPeak.py'

└── '.gitattributes'/
└── '.gitignore'/
└── '.readthedocs.yaml'/
└── '.README.md'/
└── '.setup.py'/
```

### 1. Install packages needed for the library to run

After activating the `daq_env` with `source env.sh` you can install all the requirements to run `waffles` with:

```bash
pip install -r requirements.txt
```

### 2. Make sure you have access to data to analyze

* Make sure you know how to connect and work from `@lxplus.cern.ch` machines.

* To access raw data locations you need to be able to generate a `FNAL.GOV` ticket. This is already configured in the `scripts/get_rucio.py` script which is used to generate `txt` files with the data paths and store them in `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths`

### 3. Have a look at the examples and enjoy!