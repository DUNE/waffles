from input_params import *
from utils import *

if period == 'June':
    df_calibration = pd.read_csv(june_calibration_csv)
else:
    df_calibration = pd.read_csv(other_calibration_csv)

with open(waveform_pkl_file, 'rb') as f:
    wfset = pickle.load(f) 
    print('done\n')


# 1st: BASELINE
baseliner_input_parameters = IPDict({'baseline_limits': baseline_limits[apa], 'std_cut': baseliner_std_cut, 'type': baseliner_type})    
checks_kwargs = IPDict({'points_no': wfset.points_per_wf})

_ = wfset.analyse(baseline_analysis_label, WindowBaseliner, baseliner_input_parameters, checks_kwargs=checks_kwargs, overwrite=True)

# Add a dummy baseline analysis to the merged WaveformSet(we will use this for the integration stage after having - subtracted the actual baseline)
_ = wfset.analyse(null_baseline_analysis_label, StoreWfAna, {'baseline': 0.}, overwrite=True)


#Separate the WaveformSet into a grid of WaveformSets, so that each WaveformSet contains all of the waveforms which come from the same channel
grid_apa = ChannelWsGrid(APA_map[apa], wfset, compute_calib_histo=False)

# Analysis channel by channel
for endpoint in [endpoint_selected]: #grid_apa.ch_wf_sets.keys():
    for channel in [channel_selected]: #grid_apa.ch_wf_sets[endpoint].keys():
        vendor = fbk_or_hpk(endpoint, channel)
        
        # compute average baseline std
        average_baseline_std = compute_average_baseline_std( grid_apa.ch_wf_sets[endpoint][channel], baseline_analysis_label)
        # remove baseline
        grid_apa.ch_wf_sets[endpoint][channel].apply(subtract_baseline, baseline_analysis_label, show_progress=False)
        # compute average  waveform
        mean_wf = grid_apa.ch_wf_sets[endpoint][channel].compute_mean_waveform()
        # compute integration limits
        limits = get_pulse_window_limits(mean_wf.adcs, 0, deviation_from_baseline, lower_limit_correction, upper_limit_correction )
        integrator_input_parameters = IPDict({'baseline_analysis': null_baseline_analysis_label, 'inversion': True, 'int_ll': limits[0], 'int_ul': limits[1], 'amp_ll': limits[0], 'amp_ul': limits[1]})
        checks_kwargs = IPDict({'points_no': grid_apa.ch_wf_sets[endpoint][channel].points_per_wf})

        # compute interal
        _ = grid_apa.ch_wf_sets[endpoint][channel].analyse(integration_analysis_label, WindowIntegrator, integrator_input_parameters, checks_kwargs=checks_kwargs, overwrite=True)

        subset = df_calibration.loc[
            (df_calibration['batch'] == bias_info[period]['batch']) &
            (df_calibration['APA'] == apa) &
            (df_calibration['endpoint'] == endpoint) &
            (df_calibration['channel'] == channel) &
            (df_calibration[f'OV_V'] == bias_info[period][vendor]),
            'gain'
        ]

        spe_value = subset.squeeze() if len(subset) == 1 else None
        print(f"APA {apa} - endpoint {endpoint} - channel {channel}\nSPE mean amplitude: {spe_value:.2f} \nIntegration limits: {limits}")
        print(f"Integral beam wf example: {grid_apa.ch_wf_sets[endpoint][channel].waveforms[0].analyses[integration_analysis_label].result['integral']:.2f}")
        print(f"# Photoelectrons: {grid_apa.ch_wf_sets[endpoint][channel].waveforms[0].analyses[integration_analysis_label].result['integral']/spe_value:.2f}")

        plotting_overlap_wf(grid_apa.ch_wf_sets[endpoint][channel], n_wf = 50, show = True, save = False, x_min=None, x_max=None, y_min=None, y_max=None, int_ll=limits[0], int_ul=limits[1], baseline=None, output_folder = 'output', deconvolution = False, analysis_label = integration_analysis_label)

        # Problem 24/10/2025: ho bisogno dell'integrale del singolo fotoeletttrone
        