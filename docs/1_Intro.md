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


## Current Workflow

<!-- IMAGE SUMMARISING THE WORKFLOW? -->

After running our extractors (`00_HDF5toROOT`) a folder will be generated in `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root` with your selected run. The structure inside this `.root` files is:

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
    ├── integer     'daq_pre_trigger'
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
dbt-create fddaq-v4.4.2-a9 <my_dir>
```

We recommend installing [VSCode](https://code.visualstudio.com/) as editor. Some useful extensions are: Remote-SSH, Jupyter, vscode-numpy-viewer, **Python Environment Manager**

### 0. Download the library by cloning it from GitHub

```bash
git clone https://github.com/DUNE/waffles.git # Clone the repository
git checkout -b <your_branch_name>            # Create a branch to develop
```

<!-- FOLDER STRUCTURE -->

### 1. Install packages needed for the library to run

After activating the `daq_env` with `source env.sh` you can install all the requirements to run `waffles` with:

```bash
pip install -r requirements.txt
```

### 2. Make sure you have access to data to analyze

* Make sure you know how to connect and work from `@lxplus.cern.ch` machines.

* To access raw data locations you need to be able to generate a `FNAL.GOV` ticket. This is already configured in the `scripts/get_rucio.py` script which is used to generate `txt` files with the data paths and store them in `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths`

### 3. Have a look at the examples and enjoy!









<!-- * GOOD CODING TIPS -->
```{tip} 
**Good coding practises here**

```