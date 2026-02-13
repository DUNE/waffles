These templates were obtained using the following procedure:
- Selection of LED signals with large amplitude for every channel
- Removal of self-trigger events (they should not be there but it happens to have some self-triggers)
- Baseline subtraction
- Cleaning of the persistence plot by applying some cuts
- Computation of the average waveform and crosscheck by superimposing it to the persistence plot
- Normalization to the SPE amplitude taken from this file: src/waffles/np02_data/calibration_data/np02-config-v3.0.0.csv
- Plot and save