import numpy as np
from typing import Union, Dict
import warnings 
import json
import os
import click
from pathlib import Path

import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map, cat_geometry_map
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.core.utils import build_parameters_dictionary
from waffles.data_classes.IPDict import IPDict
from waffles.utils.utils import print_colored
import waffles.input_output.raw_hdf5_reader as reader
from waffles.input_output.persistence_utils import WaveformSet_to_file
from waffles.input_output.hdf5_structured import load_structured_waveformset

input_parameters = build_parameters_dictionary('params.yml')
    
def get_analysis_params(
    ):

    int_ll = input_parameters['starting_tick']

    analysis_input_parameters = IPDict(
        baseline_limits=\
            input_parameters['baseline_limits']
    )
    analysis_input_parameters['int_ll'] = int_ll
    analysis_input_parameters['int_ul'] = \
        int_ll + input_parameters['integ_window']
    analysis_input_parameters['amp_ll'] = int_ll
    analysis_input_parameters['amp_ul'] = \
        int_ll + input_parameters['integ_window']

    return analysis_input_parameters


def get_endpoints(det:str, det_id: int):

    eps=[]
    
    if   det == 'Membrane': eps = [107]
    elif det == 'Cathode':  eps = [106] 
    elif det == 'PMTs':     eps = []
            
    # Change according to the NP02 mapping
    
    return eps


def get_wfs(wfs: list,                
            ep: Union[int, list] = -1,
            ch: Union[int, list] = -1,
            nwfs: int = -1,
            tmin: int = -1,
            tmax: int = -1,
            rec: Union[int, list] = -1,
            adc_max_threshold: int = None): 


    ep = set([ep]) if isinstance(ep, int) else set(ep)
    ch = set([ch]) if isinstance(ch, int) else set(ch)
    rec = set([rec]) if isinstance(rec, int) else set(rec)

    waveforms = []
    n = 0

    for wf in wfs:
        t = np.float32(int(wf.timestamp) - int(wf.daq_window_timestamp))
        max_adc = np.max(wf.adcs)

        if ((-1 in ep or wf.endpoint in ep) and
            (-1 in ch or wf.channel in ch) and
            (-1 in rec or wf.record_number in rec) and
            ((t > tmin and t < tmax) or (tmin == -1 and tmax == -1)) and
            (adc_max_threshold is None or max_adc <= adc_max_threshold)):

            n += 1
            waveforms.append(wf)

        if nwfs != -1 and n >= nwfs:
            break

    return waveforms, WaveformSet(*waveforms)


def baseline_cut(wfs: list):
    """
    Filters waveforms whose baseline_rms is below the mean of all baseline_rms values.
    """
    
    baseline_rms = [wf.analyses['baseline_computation'].result['baseline_rms'] for wf in wfs]
    mean_rms = np.mean(baseline_rms)

    selected_wfs = [
        wf for wf in wfs
        if wf.analyses['baseline_computation'].result['baseline_rms'] < mean_rms
    ]

    return selected_wfs, WaveformSet(*selected_wfs)


def adc_cut(wfs: list, thr_adc: int):
    """
    Filters waveforms with a maximum ADC value above a given threshold.
    """
    if thr_adc == -1:
        return wfs, WaveformSet(*wfs)
    
    selected_wfs = [wf for wf in wfs if np.max(wf.adcs)-wf.analyses['baseline_computation'].result['baseline'] < thr_adc]
    
    return selected_wfs, WaveformSet(*selected_wfs)


def get_gain_snr_and_amplitude(grid: ChannelWsGrid, verbose: bool = False, run=None):
    data = {}

    for i in range(grid.ch_map.rows):
        for j in range(grid.ch_map.columns):

            endpoint = grid.ch_map.data[i][j].endpoint
            channel = grid.ch_map.data[i][j].channel

            try:
                fit_params = grid.ch_wf_sets[endpoint][channel].calib_histo.gaussian_fits_parameters
            except KeyError:
                if verbose:
                    print(f"[WARN] Endpoint {endpoint}, channel {channel} not found in waveform sets. Skipping...")
                continue

            if len(fit_params['mean']) < 2 or len(fit_params['std']) < 2 or len(fit_params['scale']) < 2:
                if verbose:
                    print(f"[WARN] Endpoint {endpoint}, channel {channel} has fewer than 2 gaussian fits. Skipping...")
                continue

            try:
                mean0, std0 = fit_params['mean'][0][0], fit_params['std'][0][0]
                mean1, std1 = fit_params['mean'][1][0], fit_params['std'][1][0]
                amplitude2, amplitude2_err = fit_params['scale'][1]

                gain = mean1 - mean0
                noise = np.sqrt(std0**2 + std1**2)
                snr = gain / noise if noise > 0 else np.nan

                if endpoint not in data:
                    data[endpoint] = {}

                data[endpoint][channel] = {
                    'gain': gain,
                    'snr': snr,
                    'amplitude2': amplitude2,
                    'amplitude2_err': amplitude2_err,
                    'mean0': mean0,
                    'std0': std0,
                    'mean1': mean1,
                    'std1': std1,
                    'run': grid.ch_wf_sets[endpoint][channel].runs,
                    'number_waveforms': len(grid.ch_wf_sets[endpoint][channel].waveforms)
                }

                if verbose:
                    print(f"[DEBUG] EP {endpoint}, CH {channel} | Gain: {gain:.2f}, SNR: {snr:.2f}, Amp2: {amplitude2:.2f}")

            except Exception as e:
                if verbose:
                    print(f"[ERROR] Failed to process endpoint {endpoint}, channel {channel}: {e}")
                continue

    return data


def find_best_snr_per_channel(all_full_data_by_interval):
    """
    Finds the best signal-to-noise ratio (S/N) per (endpoint, channel) from the given data.
    """
    best_snr_info_per_channel = {}

    for interval, full_data in all_full_data_by_interval.items():
        for ep, ch_dict in full_data.items():
            for ch, values in ch_dict.items():
                snr = values['snr']
                mean0 = values['mean0']
                
                if not np.isfinite(snr) or not (-1000 <= mean0 <= 1000):
                    continue
                
                key = (ep, ch)
                if key not in best_snr_info_per_channel or snr > best_snr_info_per_channel[key]['snr']:
                    best_snr_info_per_channel[key] = {
                        'snr': snr,
                        'interval': interval,
                        'gain': values['gain'],
                        'amplitude': values['amplitude2'],
                        'amplitude_err': values['amplitude2_err'],
                        'mean0': values['mean0'],
                        'std0': values['std0'],
                        'mean1': values['mean1'],
                        'std1': values['std1'],
                        'run': values['run'],
                        'number_waveforms': values['number_waveforms']
                        
                    }

    print("\n>>> Best S/N per (endpoint, channel) with mu0 in [-1000, 1000]:")
    for (ep, ch), info in best_snr_info_per_channel.items():
        print(f"EP {ep}, CH {ch} → Best S/N = {info['snr']:.2f} at interval {info['interval']}, "
              f"Gain: {info['gain']:.2f}, Amplitude: {info['amplitude']:.2f}, "
              f"Amplitude Err: {info['amplitude_err']:.2f}, Mean0: {info['mean0']:.2f}, "
              f"Std0: {info['std0']:.2f}, Mean1: {info['mean1']:.2f}, Std1: {info['std1']:.2f},"
              f"Run: {info['run']}, Number of processed wfs: {info['number_waveforms']}")
    
    return best_snr_info_per_channel


def select_waveforms_around_spe(best_snr_info_per_channel, wfset2):
    """
    Selects waveforms whose integrals fall within 1 sigma around the mean1 (peak) 
    for each (endpoint, channel) based on the best S/N interval.

    Parameters:
        best_snr_info_per_channel (dict): Output from find_best_snr_per_channel().
        wfset2 (WaveformSet): A WaveformSet object with `.waveforms`.

    Returns:
        dict: A dictionary of (endpoint, channel) → WaveformSet with selected waveforms.
    """
    selected_wfs3_lists = {}

    for (ep, ch), info in best_snr_info_per_channel.items():
        mu1 = info['mean1']
        sigma1 = info['std1']
        interval = info['interval']
        integral_min = mu1 - sigma1
        integral_max = mu1 + sigma1

        for wf in wfset2.waveforms:
            if wf.endpoint != ep or wf.channel != ch:
                continue

            integral_key = f"charge_histogram_{interval}"
            if integral_key not in wf.analyses or 'integral' not in wf.analyses[integral_key].result:
                continue  # skip if data is missing

            wf_integral = wf.analyses[integral_key].result['integral']
            if not np.isnan(wf_integral) and integral_min <= wf_integral <= integral_max:
                selected_wfs3_lists.setdefault((ep, ch), []).append(wf)

    waveformsets_by_channel = {}
    for (ep, ch), wf_list in selected_wfs3_lists.items():
        if wf_list:  # only if we have waveforms
            waveformsets_by_channel[(ep, ch)] = WaveformSet(*wf_list)

    return waveformsets_by_channel


def compute_average_amplitude(waveforms, interval, ch, run, output_dir=None):
    """
    Computes the average amplitude of waveforms in a given interval and channel.
    """
    ll, ul = interval 
    amps_wf = []

    for wf in waveforms:
        key = f'baseline_computation'
        if key in wf.analyses and 'baseline' in wf.analyses[key].result:
            baseline = wf.analyses[key].result['baseline']
            adcs = wf.adcs
            offset = wf.time_offset

            amp_region = adcs[ll - offset : ul + 1 - offset]
            amplitude = np.max(amp_region) - baseline
            amps_wf.append(amplitude)

    if amps_wf:
        mean_amp = np.mean(amps_wf)
        std_amp = np.std(amps_wf)

        hist_trace = go.Histogram(
            x=amps_wf,
            nbinsx=25,
            marker=dict(color='black', line=dict(color='black', width=2)),
            name='Amplitude Distribution'
        )

        fig = go.Figure(hist_trace)
        fig.update_layout(
            title=f"Histogram of Amplitudes (interval {interval})",
            xaxis_title="Amplitude",
            yaxis_title="Counts",
        )

        os.makedirs(output_dir, exist_ok=True)
        base_name = f"amplitudehist_run_{run}_ch_{ch}_interval{ll}_{ul}"
        html_path = os.path.join(output_dir, base_name + ".html")
        png_path = os.path.join(output_dir, base_name + ".png")

        print(f"\n >>> Histogram of amplitudes saved to:")

        fig.show(method='external', renderer='browser')
        
        fig.write_html(html_path)
        fig.write_image(png_path, width=800, height=600)

        return mean_amp, amps_wf
    else:
        print("No valid waveforms found.")
        return np.nan, []


def get_histogram(values: list,
                   nbins: int = 100,
                   xmin: float = None,
                   xmax: float = None,
                   line_color: str = 'black',
                   line_width: float = 2):
    if not values:  
        raise ValueError("'values' is empty, the histogram can't be computed.")
    
    # Histogram limits
    tmin = min(values)
    tmax = max(values)

    if xmin is None:
        xmin = tmin - (tmax - tmin) * 0.1
    if xmax is None:
        xmax = tmax + (tmax - tmin) * 0.1
    
    domain = [xmin, xmax]
    
    # Create the histogram
    counts, edges = np.histogram(values, bins=nbins, range=domain)
    
    histogram_trace = go.Scatter(
        x=edges[:-1],  
        y=counts,
        mode='lines',
        fill='tozeroy',
        fillcolor='black',
        line=dict(
            color=line_color,
            width=line_width,
            shape='hv'
        )
    )
    
    return histogram_trace


def get_grid(wfs: list,                
             det: str,
             det_id: list):

    if det == 'Membrane':
        grid = ChannelWsGrid(mem_geometry_map[det_id], WaveformSet(*wfs))
    elif det == 'Cathode':
        grid = ChannelWsGrid(cat_geometry_map[det_id], WaveformSet(*wfs))  
    elif det == 'PMTs':
        grid = None      
        
    return grid


def get_grid_charge(wfs: list,                
                    det: str,
                    det_id: list,
                    bins_number: int,
                    analysis_label: str):

    if   det == 'Membrane': map = mem_geometry_map[det_id]
    elif det == 'Cathode': map = cat_geometry_map[det_id]
    elif det == 'PMTs': map = None      
        
    grid = ChannelWsGrid(map, 
                        WaveformSet(*wfs),
                        compute_calib_histo=True, 
                        bins_number=bins_number,
                        domain=np.array((-10000.0, 50000.0)),
                        variable="integral",
                        analysis_label=analysis_label
                        )
        
    return grid

def plot_snr_per_channel_grid(snr_data, det, det_id, title="S/N vs integration intervals", show_figures=True):
    
    if det=='Membrane':
        if det_id == 2:
            geometry = [
                [46, 44],
                [43, 41],
                [30, 37],
                [10, 17],
            ]
        elif det_id == 1:
            geometry = [
                [47, 45],
                [40, 42],
                [ 0,  7],
                [20, 27],
            ]
    else:
        geometry = None  
            
    time_intervals = sorted(snr_data.keys())  

    # Determine the number of rows and columns of the grid
    nrows, ncols = len(geometry), len(geometry[0])

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[f"Ch {ch}" for row in geometry for ch in row],
        shared_xaxes=True, shared_yaxes=True
    )
        
    for i, row in enumerate(geometry):
        for j, ch in enumerate(row):
            snrs = []
            for interval in time_intervals:
                # Acceder a los datos de S/N para el canal 'ch' en el intervalo 'interval'
                snr = snr_data.get(interval, {}).get(107, {}).get(ch, {}).get('snr', np.nan)
                snrs.append(snr)

            if any(not np.isnan(snr) for snr in snrs):  # Verifica si hay algún valor válido
                # Añadir la traza para cada canal en el grid correspondiente
                fig.add_trace(go.Scatter(
                    x=time_intervals,
                    y=snrs,
                    mode='lines+markers',
                    name=f"Ch {ch}",
                    line=dict(shape='linear'),
                ), row=i+1, col=j+1)  # Row y col se ajustan en 1-based index
            else:
                print(f"No data for Ch {ch} in the intervals {time_intervals}")
                        
    fig.update_layout(
        title=title,
        xaxis_title="Interval",
        yaxis_title="S/N",
        width=1100,
        height=1200
    )

    fig.update_xaxes(zeroline=True, zerolinecolor='black')
    fig.update_yaxes(zeroline=True, zerolinecolor='black')
    
    # Show the figure
    if show_figures:
        fig.show(method='external', renderer='browser')


def get_det_id_name(det_id: int):

    if   det_id == 1: det_id_name='nonTCO' 
    elif det_id ==2 : det_id_name= 'TCO'      
        
    return det_id_name


# ------------ Functions to plot in a grid mode ---------------

def plot_wf(waveform_adcs: WaveformAdcs,  
            figure,
            row,
            col,
            baseline: float = None,
            offset: bool = False) -> None:
    """
    Plot a single waveform
    """
    x0 = np.arange(len(waveform_adcs.adcs), dtype=np.float32)
    y0 = waveform_adcs.adcs

    if baseline is not None:
        y0 = y0 - baseline  

    if offset:        
        dt = np.float32(np.int64(waveform_adcs.timestamp) -
                        np.int64(waveform_adcs.daq_window_timestamp))
    else:
        dt = 0

    wf_trace = go.Scatter(
        x = x0 + dt,   
        y = y0,
        mode = 'lines',
        line = dict(width=0.5, color='black')
    )

    figure.add_trace(wf_trace, row, col)


# ------------- Plot a set of waveforms ----------

def plot_wfs(channel_ws,  
             figure, 
             row, 
             col,           
             nwfs_plot: int = -1,
             xmin: int = -1,
             xmax: int = -1,
             tmin: int = -1,
             tmax: int = -1,
             offset: bool = False,
             baseline: bool = True,
             ) -> None:
    """
    Plot a list of waveforms. If baseline=True, subtract baseline from each waveform.
    Limits number of plotted waveforms with `nwfs`.
    """

    if tmin == -1 and tmax == -1:
        tmin = xmin - 1024
        tmax = xmax        

    for i, wf in enumerate(channel_ws.waveforms):
        if nwfs_plot != -1 and i >= nwfs_plot:
            break

        if baseline:
            bl = wf.analyses['baseline_computation'].result['baseline']
        else:
            bl = None

        plot_wf(wf, figure, row, col, baseline=bl, offset=offset)

    return figure


#-------------- Time offset histograms -----------

def plot_to_function(channel_ws, figure, row, col, nbins):

    # Compute the time offset
    times = [wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp for wf in channel_ws.waveforms]

    if not times:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return

    # Generaate the histogram
    histogram = get_histogram(times, nbins, line_width=0.5)
    
    # Add the histogram to the corresponding channel
    figure.add_trace(histogram, row=row, col=col)
    
    return figure


# --------------- Sigma vs timestamp  --------------

def plot_sigma_vs_ts_function(channel_ws, figure, row, col):

    timestamps = []
    sigmas = []

    # Iterate over each waveform in the channel
    for wf in channel_ws.waveforms:
        # Calculate the timestamp for the waveform
        timestamp = wf._Waveform__timestamp
        timestamps.append(timestamp)

        # Calculate the standard deviation (sigma) of the ADC values
        sigma = np.std(wf.adcs)
        sigmas.append(sigma)
    
    # Add the histogram to the corresponding channel
    figure.add_trace(go.Scatter(
        x=timestamps,
        y=sigmas,
        mode='markers',
        marker=dict(color='black', size=2.5)  
    ), row=row, col=col)
    
    return figure


# --------------- Sigma histograms  --------------
 
def plot_sigma_function(channel_ws, figure, row, col, nbins):
    
    # Compute the sigmas
    
    sigmas = [np.std(wf.adcs) for wf in channel_ws.waveforms]

    if not sigmas:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return None, None, None, None  # Return None if no data
    
        
    # Generate the histogram
    histogram = get_histogram(sigmas, nbins, line_width=0.5)
    
    # Add the histogram to the corresponding channel
    figure.add_trace(histogram, row=row, col=col)
    
    return figure


# -------------------- Mean FFT --------------------

def fft(sig, dt=16e-9):
    np.seterr(divide = 'ignore')
    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
    else:
        t = np.arange(0, sig.shape[-1]) * dt
    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[:-1]
        sig = sig[:-1]
    sigFFT = np.fft.fft(sig) / t.shape[0]
    freq = np.fft.fftfreq(t.shape[0], d=dt)
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[:firstNegInd]
    sigFFTPos = 2 * sigFFT[:firstNegInd]
    x = freqAxisPos /1e6
    y = 20*np.log10(np.abs(sigFFTPos)/2**14)
    return x,y


def plot_meanfft_function(channel_ws, figure, row, col):

    selected_wfs,_=get_wfs(channel_ws.waveforms,-1, -1,-1,-1,-1,[-1])
    
    fft_list_x = []
    fft_list_y = []

    # Compute the FFT
    for wf in selected_wfs:
        tmpx, tmpy = fft(wf.adcs) 
        fft_list_x.append(tmpx)
        fft_list_y.append(tmpy)

    # Compute the mean FFT
    freq = np.mean(fft_list_x, axis=0)
    power = np.mean(fft_list_y, axis=0)

    figure.add_trace(go.Scatter(
        x=freq,
        y=power,
        mode='lines',
        ), row=row, col=col)  
    
    return figure  


def save_waveform_hdf5(wfset, input_filepath, output_filepath):
        input_filename = Path(input_filepath).name
        output_filepath = f"{output_filepath}/{input_filename}.hdf5"

        print_colored(f"Saving waveform data to {output_filepath}...", color="DEBUG")
        try:

            print_colored(f"📦 About to save WaveformSet with {len(wfset.waveforms)} waveforms", color="DEBUG")

            WaveformSet_to_file(
                waveform_set=wfset,
                output_filepath=str(output_filepath),
                overwrite=True,
                format="hdf5",
                compression="gzip",
                compression_opts=0,
                structured=True
            )
            print_colored(f"WaveformSet saved to {output_filepath}", color="SUCCESS")

            print_colored("Going to load...")
            wfset_loaded = load_structured_waveformset(str(output_filepath))
            print_colored("Loaded, about to compare...")  # If you see this, the load worked
            print_colored(f"wfset_loaded type={type(wfset_loaded)}")
            
            return True
        except Exception as e:
            print_colored(f"Error saving output: {e}", color="ERROR")
            return False


def save_dict_to_json(data: dict, file_path: str) -> None:
    """
    Save (ep, ch) keyed data into nested JSON format: {ep: {ch: {...}}}
    
    Args:
        data (dict): Keys are tuples (ep, ch), values are dicts with analysis info.
        file_path (str): Path to save the JSON file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    nested_data = {}
    for (ep, ch), values in data.items():
        ep_str = str(ep)
        ch_str = str(ch)

        # Convert 'run' set to list if needed
        if isinstance(values.get("run"), set):
            values["run"] = list(values["run"])

        if ep_str not in nested_data:
            nested_data[ep_str] = {}
        nested_data[ep_str][ch_str] = values

    with open(file_path, 'w') as f:
        json.dump(nested_data, f, indent=4)

    print(f"\n >>> Dictionary saved to {file_path}")

