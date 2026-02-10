## Time Resolution
Time resolution studies as performed for ProtoDUNE-HD. This analysis module tests the PDS performance in
terms of time resolution mostly relying on LED data. We implemented a cross-checl analysis based on
cosmic rays events as well. In this case the events are the same as the ones used for `../light_yield_vs_e/`.
*NOTE:* The Time Resolution analysis makes a heavy use of `ROOT`, so you will probably have to setup
a python environment with `ROOT` installed oustide the DAQ environment.
If running on CERN machines (lxplus), you can simply add this line:
```bash
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.32.02/x86_64-almalinux9.4-gcc114-opt/bin/thisroot.sh
```
to the `env/bin/activate` file.
If running locally, you have to do something similar... like...
```bash
source /path/to/your/root/installation/bin/thisroot.sh
```
or (MacOS with Homebrew)
```bash
source /opt/homebrew/bin/thisroot.sh 
```

# Usage
The user is supposed to properly configure the analyses through the following configuration file:
```
steering.yml
params.yml
config/time_resolution_config.yml
```
and then run the following macro:
```
raw_single_channel_ana.py
plotter/single_channel_ana_plotter.py
channel_pair_ana.py
```

The first one dump the relevant information for the single channel analysis into a ROOT TTree, the second one plot the relevant
distributions for the single channel analysis, and the third one combines the information for a pair of channels
to perform a cross analysis.
The workflow is such that, after `raw_single_channel_ana.py` the following macros already know where to look for their input.

The cosmic ray analysis is performed by running the following macro:
```
raw_cosmic_ana.py
```
instead of `raw_single_channel_ana.py`. The rest of the workflow is the same as for the LED analysis.
