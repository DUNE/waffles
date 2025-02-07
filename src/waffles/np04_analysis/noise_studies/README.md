# Overview
To caracterize the ProtoDUNE-HD PDS noise level we dedicated few runs with SiPMs biased below
the breakdown voltage. In particular, we acquired data setting all the endpoints (DAPHNEs) at
different gains.
The macros in this folder serve to estimate a realistic power spectral density (PSD) of the noise
(or FFT) for the gain settings used in the ProtoDUNE-HD PDS runs (beam, cosmics, ...). This might be
useful to the deconvolution algorithm.

# How to run
Few .yaml files help to configure the analysis.
1. `noise_run_info.yaml`: contains the list of noise runs acquired. A disctionary stores the VGain set in DAPHNEs and the corresponding run number.
    In principle, you won't need to modify this file.
2. `user_setting.yaml`: here you can declare which runs are you interested in, where to save the outputs, whether to use full statistics or not, etc.
3. `channel_vgain.yaml`: here is where you specify the VGain set in each DAPHNE during the Physics run you want to analyze. Need to provide a VGain
    for each AFE.

Once the configuration files are set, you can run the analysis with:
```bash
python compute_FFT.py
python create_FFT_set.py
```

## compute_FFT.py

## create_FFT_set.py

# Outputs
