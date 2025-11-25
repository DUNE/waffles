import pickle
import plotly.graph_objects as go
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map # apa map
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid # channel grid

from utils_plots import plotting_overlap_wf
from integral_computation_function import channel_integral_computation

waveform_pkl_file = f"/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW/run027361_ALL_hdf5.pkl"
period = 'june'
apa = 2
endpoint_selected = 109
channel_selected = 35
integration_analysis_label = 'integrator'

print("\nreading pkl file...")
with open(waveform_pkl_file, 'rb') as f:
    wfset = pickle.load(f) 
    print('done\n')


grid_apa = ChannelWsGrid(APA_map[apa], wfset, compute_calib_histo=False)

for endpoint in [endpoint_selected]: #grid_apa.ch_wf_sets.keys():
    for channel in [channel_selected]: #grid_apa.ch_wf_sets[endpoint].keys():
        grid_apa.ch_wf_sets[endpoint][channel] = channel_integral_computation(grid_apa.ch_wf_sets[endpoint][channel], period = period, integration_analysis_label = integration_analysis_label)

        # print(f"APA {apa} - endpoint {endpoint} - channel {channel}\nSPE mean amplitude: {grid_apa.ch_wf_sets[endpoint][channel].waveforms[0].analyses[integration_analysis_label].result['gain']:.2f} \nIntegration limits: {grid_apa.ch_wf_sets[endpoint][channel].waveforms[0].analyses[integration_analysis_label].result['integration_limits']}")
        # print(f"Integral beam wf example: {grid_apa.ch_wf_sets[endpoint][channel].waveforms[0].analyses[integration_analysis_label].result['integral']:.2f}")
        # print(f"# Photoelectrons: {grid_apa.ch_wf_sets[endpoint][channel].waveforms[0].analyses[integration_analysis_label].result['integral_pe']}")

        # plotting_overlap_wf(grid_apa.ch_wf_sets[endpoint][channel], n_wf = 10, show = True, save = False, x_min=None, x_max=None, y_min=None, y_max=None, int_ll=grid_apa.ch_wf_sets[endpoint][channel].waveforms[0].analyses[integration_analysis_label].result['integration_limits'][0], int_ul=grid_apa.ch_wf_sets[endpoint][channel].waveforms[0].analyses[integration_analysis_label].result['integration_limits'][1], baseline=None, output_folder = 'output', deconvolution = False, analysis_label = integration_analysis_label)
