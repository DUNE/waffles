## Self-Trigger
##### Author: Federico Galizzi (federico.galizzi@cern.ch)
Self-Trigger studies as perfomed for ProtoDUNE-HD. This analysis module tests the PDS performance in
terms of self-triggering capabilities, in particular we evaluated the trigger efficiency and time resolution
as function of the number of photoelectrons in the event at different threshold.
*NOTE:* The Self-Trigger analysis makes a heavy use of `ROOT`, so you will probably have to setup
a python environment with `ROOT` installed outside the DAQ environment.
If running on CERN machines (lxplus), you can simply add this line:
```bash
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.32.02/x86_64-almalinux9.4-gcc114-opt/bin/thisroot.sh
```
to the `env/bin/activate` file. I think I doesn't work with Python3.10... Python3.9 is ok, I don't know
about other versions...
If running locally, you have to do something similar... like...
```bash
source /path/to/your/root/installation/bin/thisroot.sh
```
or (MacOS with Homebrew)
```bash
source /opt/homebrew/bin/thisroot.sh 
```

# Usage
Before runnng the analysis the user must create a dedicated dataset with files containing the raw data from
the "SiPM" and "SelfTrigger" channels. It is necessary that the two channels have the same number of waveforms
and that their timestamps match. To do this, you can use `./ST_conversion_configs/` scripts.
The user is supposed to properly configure the analyses through the following configuration file:
```
steering.yml
params.yml
```
Be also sure you have the necessary infos about the run you are analysing in `./configs/SelfTrigger_RunInfo.csv`
and the calibration results in `./configs/SelfTrigger_Calibration.csv`.

*STEP 1:* dump raw data in meaningful ROOT TTrees by running `./dump_raw_to_meta.py`.
*STEP 2:* run `./ana_self_trigger.py` to perform trigger efficiency studies.
*STEP 3:* run `./jitter.py` to perform time resolution studies. You can also run `./jitter_analyzer.py` to plot
the results in ROOT TGraphs.

### I would have loved to, but I never...
I would have loved to do more accurate analysis using also non-LED-illuminated channels to perform deadtim studies
etc, but I never had the time to do it. There is something already in place in `./non_illuminated.py`. Who knows,
maybe one day...
