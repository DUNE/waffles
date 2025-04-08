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
    * ...
3. Electronics studies --> DAPHNE response ?
4. Fast analysis
    * Look at the data during the protoDUNE data taking
    * Plot various quantities such as full waveforms, average waveform, FFT 
5. Share your codes with the PDS group


## Contributing

The idea of `waffles` framework is to unify efforts and to develop a robust analysis tool. 
Users should become developers! if you need something that is not available, or you have an idea on how to improve and optimize a part of the framework, try to do it and commit your changes.
Join the waffles Slack channel - send a Slack message to Francesca Alemanno or Andrea Roche.
You can always ask in the Slack channel if you have any doubt. 
Your analysis can become part of the framework, and you can promote the tools you developed, if they can be useful to other people.

Nevertheless, it is very important to keep some common rules not to harm others work.

```{tip} 
**Good coding practises here**

*   Create your own branch for developing code and make puntual commits with your changes. Once you want to share with the world, open a pull request and wait for two reviewers to approve the merging

* To include functions/methods follow the analysis structure and coding conventions

* Update this documentation with your latest additions and modifications to the main branch. This will help everyone, also yourself!

```

## Current Workflow

<!-- IMAGE SUMMARISING THE WORKFLOW? -->
1. **Files location**: the `rucio` paths for the NP04 runs we want to analyse are stored in `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths`. If they are not available, you can generate these files. First, you have to move to the waffles scripts directory, then `source setup_rucio_a9.sh` and finally run the `scripts/fetch_rucio_replicas.py` script, with options `--runs run_number` `--max-files number_of_files`. 

2. **Data extraction**: the raw data is stored in `.hdf5` files. The optimal way of extracting the data is as follows:
    * Move to the waffles scripts directory
    * Modify the `config.json` file, according to your needs
    * Run `python3 06_RawHDF5Reader.py --config config.json `

In this way, you can store the `WaveformSet` in a lighter `.hdf5` file and load it whenever you want to work with it. 

3. **Analysis and Visualization**: the `WaveformSet` object can be visualized using the `waffles.plotting.plot` module.

## **Getting Started - SETUP**  ⚙️

We recommend installing [VSCode](https://code.visualstudio.com/) as editor. Some useful extensions are: Remote-SSH, Jupyter, vscode-numpy-viewer, **Python Environment Manager**

If it is your first time here you need to create an environment to be able to use all their tools. Depending on the scope of your work you can create a `daq_env` (run `hdf5` file processing) or a `ana_env` (general analysis scope) environment.

### DAQ ENVIRONMENT [**NEEDED TO READ THE HDF5!!**]

In this case all the dependencies from the DAQ needed to translate the information from the `.hdf5` files to `.root` files are included. (We are still working to have `ROOT` directly available in this environment from a `Python` interpreter). You don't need this environment unless you plan to work on the decoding side.

```bash
source /cvmfs/dunedaq.opensciencegrid.org/setup_dunedaq.sh

setup_dbt latest
dbt-create -l 
dbt-create fddaq-v4.4.7-a9 <my_dir>
```


### ANA ENVIRONMENT [OPTIONAL]

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
    ├── 'data_classes.rst'
    ├── 'input.rst'
    ├── 'np04_data.rst'
    ├── 'np04_data_classes.rst'
    ├── 'np04_utils.rst'
    ├── 'output.rst'
    ├── 'plotting.rst'
    └── 'utils.rst'

└── 'scripts'/ # FOLDER WITH THE SCRIPTS
    ├── 'cpp_utils'/ # C++raw functions and scripts (can be used in standalone mode) [Thanks Jairo!]
        ├── 'functions'/
            ├── 'channelmap.txt'
            ├── 'hdf5torootclass.h'
            └── 'wffunctions.h'
        ├── 'CMakeLists.txt'
        ├── 'compile_decoder.sh' #Script to compile c++ scripts (just 1st time) and be able to use them
        ├── 'HDF5LIBS_duplications.cpp' # C++ script to check for duplications in the hdf5 files
        ├── 'HDF5toROOT_decoder.cpp'    # C++ script to decode hdf5 files to root files
        ├── 'plotsAPA.C' # ROOT script to plot the APA map
        └── 'README.md'  # Instructions to compile and run the C++ scripts
    ├── '00_HDF5toROOT.py' # Python decoder (hdf5 to root) with multithreading
    ├── '00_HDF5toROOT.sh' # Bash script for managing CPP macros. If you already compiled (cpp_utils) them you can run this one.
    ├── 'get_protodunehd_files.sh' # Script to get rucio_paths from the hdf5 daq files
    ├── 'get_rucio.py' # RUN to make rucio_paths sincronize with /eos/ folder. You will save time and make others save time too!
    ├── 'README.md'
    └── 'setup_rucio.sh' # Standalone commands for setting up rucio once you are inside a SL7

└── 'src'/  # MAIN CODE CORE WITH ALL THE CLASSES DEFINITIONS HERE#
    ├── 'waffles'/
        ├── 'data_classes'/ # FOLDER WITH THE DATA CLASSES DEFINITIONS
            ├── 'BasicWfAna.py'
            ├── 'CalibrationHistogram.py'
            ├── 'ChannelWs.py'
            ├── 'ChannelWsGrid..py'
            ├── 'IODict.py'
            ├── 'IPDict.py'
            ├── 'Map.py'
            ├── 'ORDict.py'
            ├── 'PeakFindingWfAna.py'
            ├── 'TrackedHistogram.py'
            ├── 'Waveform.py'
            ├── 'WaveformAdcs.py'
            ├── 'WaveformSet.py'
            ├── 'WfAna.py'
            ├── 'WfAnaResult.py'
            └── 'WfPeak.py'
        ├── 'input'/ # FOLDER WITH THE INPUT UTILS
            ├── 'input_utils.py'
            ├── 'pickle_file_to_WaveformSet.py'
            ├── 'raw_hdf5_reader.py'
            └── 'raw_root_reader..py'
        ├── 'np04_analysis'/ # FOLDER WITH THE ANALYSIS UTILS
            ├── 'LED_calibration'
            └── 'np04_ana.py'
        ├── 'np04_data'/ # FOLDER WITH THE DATA UTILS
            └── 'ProtoDUNE_HD_APA_maps.py'
        ├── 'np04_data_classes'/ # FOLDER WITH THE DATA CLASSES 
            └── 'APAmap.py'
        ├── 'np04_utils'/ # FOLDER WITH NP04 UTILS
            └── 'utils.py'
        ├── 'plotting'/ # FOLDER WITH THE PLOTTING UTILS
            └── 'display'
            ├── 'plot_utils.py'
            └── 'plot.py'
        └── 'utils'/ # FOLDER WITH THE GENERAL UTILS
            ├── 'deconvolution'/ # FOLDER WITH THE DECONVOLUTION METHODS
            ├── 'fit_peaks'/     # FOLDER WITH THE FITTING PEAKS METHODS 
            ├── 'check_utils.py' 
            ├── 'filtering_utils.py' 
            ├── 'numerical_utils.py'
            ├── 'wf_maps_utils.py'
            └── 'Exceptions.py'
        └── 'test' # FOLDER WITH FILES UNDER TEST (temporary)
├── '.gitattributes'
├── '.gitignore'
├── '.readthedocs.yaml' # Configuration file for the documentation
├── 'environment.yaml'  # Environment file for the conda environment
├── '.README.md'
├── 'requirements.md' # Requirements for the evironment
└── 'setup.py'        # Setup file for the library
```

### 1. Install packages needed for the library to run

After activating the `env` with `source env.sh` or `source /path/to/new/virtual/environment/bin/activate` you can install all the requirements to run `waffles` by navigating to the repository main folder and running:

```bash
pip install -r requirements.txt
pip install .
```

If at some point you needed to re-run waffles with the changes you have introduced to the source code, you would just need to run the second command.

### 2. Make sure you have access to data to analyze

HERE WE SHOULD ADD SOME USEFUL LINKS (NP02 and NP04 run list, channels map, ... )

* Make sure you know how to connect and work from `@lxplus.cern.ch` machines.

* To access raw data locations you need to be able to generate a `FNAL.GOV` ticket. This is already configured in the `scripts/get_rucio.py` script which is used to generate `txt` files with the data paths and store them in `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths`

* Request access to the `np04-t0comp-users` and `np04-daq-dev` egroup on the [CERN egroups page](https://egroups.cern.ch). This also adds you to the `np-comp` Linux group.

### 3. Have a look at the examples and enjoy!