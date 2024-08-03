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

The idea of `waffles` framework is to unify efforts and to develop a robust analysis tool. 
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

The next steps are loading the `root` files (you can select the fraction of statistics you want to analyse) and start visualizing your data.


## **Getting Started - SETUP**  ⚙️

We recommend installing [VSCode](https://code.visualstudio.com/) as editor. Some useful extensions are: Remote-SSH, Jupyter, vscode-numpy-viewer, **Python Environment Manager**

If it is your first time here you need to create an environment to be able to use all their tools. Depending on the scope of your work you can create a `daq_env` (run `hdf5` file processing) or a `ana_env` (general analysis scope) environment.

### DAQ ENVIRONMENT

In this case all the dependencies from the DAQ needed to translate the information from the `.hdf5` files to `.root` files are included. (We are still working to have `ROOT` directly available in this environment from a `Python` interpreter). You don't need this environment unless you plan to work on the decoding side.

```bash
source /cvmfs/dunedaq.opensciencegrid.org/setup_dunedaq.sh

setup_dbt latest
dbt-create -l 
dbt-create fddaq-v4.4.3-a9 <my_dir>
```


### ANA ENVIRONMENT

This general environment is used to run the `waffles` library and all the tools needed to analyze the data. To create it just run in your terminal (from whatever `lxplus` machine or locally):

```bash
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```
Or use the **Python Environment Manager** VScode extension to manage your environments.

In order to have access to the `ROOT` library you need to have it sourced in your environment. Add these lines at the end of the `bin/activate` file:

```bash
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.32.02/x86_64-almalinux9.4-gcc114-opt/bin/thisroot.sh
export JUPYTER_CONFIG_DIR=$VIRTUAL_ENV
```

To deactivate the environment just run `deactivate` in your terminal.

If you are using Jupyter inside VSCode you may want the virtual enviroment to be recognized by the Kernel selector, for that follow:

```bash

#source /path/to/your/venv/bin/activate # Activate the virtual environment

# Install ipykernel in the virtual environment
pip install ipykernel 
# Add the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=your_env_name --display-name "Python_WAFFLES"
```

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
        ├── 'HDF5LIBS_duplications.cpp' # C++ script to check for duplications in the hdf5 files
        ├── 'HDF5toROOT_decoder.cpp' # C++ script to decode hdf5 files to root files
        ├── 'plotsAPA.C' # ROOT script to plot the APA map
        └── 'README.md' # Instructions to compile and run the C++ scripts
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

After activating the `env` with `source env.sh` or `source /path/to/new/virtual/environment/bin/activate` you can install all the requirements to run `waffles` with [NEED TO BE IN THE MAIN FOLDER]:

```bash
pip install -r docs/requirements.txt 
```

### 2. Make sure you have access to data to analyze

* Make sure you know how to connect and work from `@lxplus.cern.ch` machines.

* To access raw data locations you need to be able to generate a `FNAL.GOV` ticket. This is already configured in the `scripts/get_rucio.py` script which is used to generate `txt` files with the data paths and store them in `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths`

* Request access to the `np04-t0comp-users` and `np04-daq-dev` egroup on the [CERN egroups page](https://egroups.cern.ch). This also adds you to the `np-comp` Linux group.

### 3. Have a look at the examples and enjoy!