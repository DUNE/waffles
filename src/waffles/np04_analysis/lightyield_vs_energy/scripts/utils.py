
import os 
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors

from itertools import combinations

import plotly.graph_objects as go
import matplotlib.ticker as ticker
import pickle 
import click
from tqdm import tqdm
# from tqdm.notebook import tqdm
import json
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
from landaupy import langauss as lg

from scipy.odr import ODR, Model, RealData

from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from matplotlib.colors import LinearSegmentedColormap

import ast
import copy


from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.IPDict import IPDict

from waffles.Exceptions import GenerateExceptionMessage
# from waffles.input_output.hdf5_structured import load_structured_waveformset

# from waffles.utils.baseline.WindowBaseliner import WindowBaseliner
# from waffles.data_classes.ChannelWsGrid import ChannelWsGrid 
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map


# from waffles.np04_analysis.lightyield_vs_energy.scripts.MyAnaPeak_NEW import MyAnaPeak_NEW
# from waffles.np04_analysis.lightyield_vs_energy.scripts.MyAnaConvolution import MyAnaConvolution


#####################################################################

def inv_sqrt_model(B, x):
        return B[0] / np.sqrt(x)


#####################################################################

def noise_term_fit_array(beta, x):
    p0 = beta[0]   
    return p0 / x


#####################################################################
def energy_resolution_fit(x, p0, p1,p2):
    x = np.array(x)
    return np.sqrt(p0**2+(p1/np.sqrt(x))**2+(p2/x)**2)

def energy_resolution_fit_array(params, x):
    p0, p1, p2 = params
    x = np.array(x)
    return energy_resolution_fit(x, p0, p1, p2)


def make_energy_resolution_fit_p2_fixed(p2_fixed):
    def energy_resolution_fit_p2_fixed(beta, x):
        p0, p1 = beta
        x = np.array(x)
        return np.sqrt(
            p0**2 +
            (p1/np.sqrt(x))**2 +
            (p2_fixed/x)**2
        )
    return energy_resolution_fit_p2_fixed


#####################################################################

def linear(x, A, B):
    return A + B*x

def linear_array(params, x):
    a,b = params
    return linear(x,a,b)

#####################################################################

def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gaussian_array(params, x):
    mu, sigma, A = params
    return gaussian(x,mu,sigma,A)

#####################################################################

def langauss(x, mpv, eta, sigma, A):
    return A * lg.pdf(x, mpv, eta, sigma)

def find_peak(params):
    mpv, eta, sigma, A = params

    # Prevent very small sigma from breaking the optimizer
    sigma = max(sigma, 1e-3)
    search_width = max(0.1, 5 * sigma)  # Ensure minimum range

    result = minimize_scalar(
        lambda x: -langauss(x, mpv, eta, sigma, A),
        bounds=(mpv - search_width, mpv + search_width),
        method='bounded'
    )
    return result.x


def propagate_error(find_peak, params, errors, epsilon=1e-5):
    peak = find_peak(params)
    partials = []

    for i in range(len(params)):
        params_eps_plus = params.copy()
        params_eps_minus = params.copy()

        params_eps_plus[i] += epsilon
        params_eps_minus[i] -= epsilon

        f_plus = find_peak(params_eps_plus)
        f_minus = find_peak(params_eps_minus)

        derivative = (f_plus - f_minus) / (2 * epsilon)
        partials.append(derivative)

    # Now propagate the errors
    squared_terms = [(partials[i] * errors[i])**2 for i in range(len(params))]
    total_error = np.sqrt(sum(squared_terms))

    return peak, total_error

#####################################################################

channel_vendor_map = { 104: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
                                10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK"},
                        105: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
                                10: "FBK", 12: "FBK", 15: "FBK", 17: "FBK", 21: "HPK", 23: "HPK", 24: "HPK", 26: "HPK"},
                        107: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK",
                                10: "HPK", 12: "HPK", 15: "HPK", 17: "HPK"},
                        109: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
                                10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
                                20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
                                30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
                                40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
                        111: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
                                10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
                                20: "FBK", 21: "FBK", 22: "FBK", 23: "FBK", 24: "FBK", 25: "FBK", 26: "FBK", 27: "FBK",
                                30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
                                40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
                        112: {0: "HPK", 1: "HPK", 2: "HPK", 3: "HPK", 4: "HPK", 5: "HPK", 6: "HPK", 7: "HPK",
                                10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK",
                                20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
                                30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
                                40: "HPK", 42: "HPK", 45: "HPK", 47: "HPK"},
                        113: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK"}}

#####################################################################


def fbk_or_hpk(endpoint: int, channel: int):
    channel_vendor_map = {
    104: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK"},
    105: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 12: "FBK", 15: "FBK", 17: "FBK", 21: "HPK", 23: "HPK", 24: "HPK", 26: "HPK"},
    107: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK",
          10: "HPK", 12: "HPK", 15: "HPK", 17: "HPK"},
    109: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
          20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
    111: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
          20: "FBK", 21: "FBK", 22: "FBK", 23: "FBK", 24: "FBK", 25: "FBK", 26: "FBK", 27: "FBK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
    112: {0: "HPK", 1: "HPK", 2: "HPK", 3: "HPK", 4: "HPK", 5: "HPK", 6: "HPK", 7: "HPK",
          10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK",
          20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 42: "HPK", 45: "HPK", 47: "HPK"},
    113: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK"}}

    return channel_vendor_map[endpoint][channel]
    

#####################################################################    


def beam_filter(waveform: Waveform, analysis_label) -> bool:
    if len(waveform.analyses[analysis_label].result['beam_peak_index'])==1:
        return True
    else:
        return False
        

#####################################################################    


def timestamp_filter(waveform : Waveform, analysis_label : str, ts : int, ts_delta : int = 200) -> bool:
    bool_return = False
    for single_peak_absolute_time in waveform.analyses[analysis_label].result['beam_peak_absolute_time']:
        if abs(single_peak_absolute_time - ts) < ts_delta :
            bool_return = True
    return bool_return
    

#####################################################################    


def subtract_baseline_invert(
        waveform: WaveformAdcs,
        baseline_analysis_label: str,
        inversion: bool
) -> None:
    """This method overwrites the adcs method of the given
    WaveformAdcs object, by subtracting its baseline.

    waveform: WaveformAdcs
        The waveform whose adcs will be modified
    baseline_analysis_label: str
        The baseline to subtract must be available 
        under the 'baseline' key of the result of the analysis
        whose label is given by this parameter, i.e. in
        waveform.analyses[analysis_label].result['baseline']
    inversion: bool
        If True, the baseline-subtracted signal is inverted.
        If False, the baseline-subtracted signal is not inverted.
    """

    try:
        baseline = waveform.analyses[baseline_analysis_label].result['baseline']

    except KeyError:
        raise Exception(
            GenerateExceptionMessage(
                1,
                "subtract_baseline()",
                f"The given waveform does not have the analysis"
                f" '{baseline_analysis_label}' in its analyses "
                "attribute, or it does, but the 'baseline' key "
                "is not present in its result."
            )
        )
    
    if inversion:
        waveform._WaveformAdcs__set_adcs(
            baseline - waveform.adcs
        )
    else:
        waveform._WaveformAdcs__set_adcs(
            waveform.adcs - baseline
        )

    return
    

#####################################################################    


def trigger_time_searching(values, delta = 200):
    if not values:
        return []

    values = sorted(values)
    filtered = [values[0]]

    for v in values[1:]:
        if abs(v - filtered[-1]) > delta:
            filtered.append(v)

    return filtered
    

#####################################################################    


def cut_full_streaming_window(
        waveform: WaveformAdcs,
        analysis_label: str,
        amplitude_threshold: float = 0.3,
        pre_peak_window: int = 60,
        window_timeticks_length: int = 588 #check the template, to have the same length
) -> None:
    """This method overwrites the adcs method of the given
    WaveformAdcs object.

    waveform: WaveformAdcs
        The waveform whose adcs will be modified
    analysis_label: str
        Name of the analysis including the peak information
    amplitude_threshold: float
        Percentage of the peak amplitude to define the starting point for the cut 
    pre_peak_window: int
        Ticks before the peak 
    window_timeticks_length: int
        Legth of the new wf
    """

    try:
        peak_index = waveform.analyses[analysis_label].result['beam_peak_index']
        baseline = waveform.analyses[analysis_label].result['mean_baseline']
    except KeyError:
        raise Exception(
            GenerateExceptionMessage(
                1,
                "Problem finding 'beam_peak_index' or 'beam_peak_amplitude' "
            )
        )
    
    peak_index = waveform.analyses[analysis_label].result['beam_peak_index'][0]
    peak_amplitude = waveform.analyses[analysis_label].result['beam_peak_amplitude'][0]

    amplitude_threshold_dac = peak_amplitude * amplitude_threshold

    window = waveform.adcs[peak_index-pre_peak_window : peak_index]
    idx_in_window = np.abs(window - amplitude_threshold_dac).argmin()
    idx = (peak_index - pre_peak_window) + idx_in_window

    start = idx - pre_peak_window
    stop  = start + window_timeticks_length

    if start < 0:
        start = 0
        stop = window_timeticks_length

    if stop > len(waveform.adcs):
        stop = len(waveform.adcs)
        start = stop - window_timeticks_length

    if stop - start != window_timeticks_length:
        print("Skipping: cannot enforce fixed window length")

    waveform._Waveform__slice_adcs(start, stop)

    return 
    

#####################################################################    


def reading_template(directory_path):
    # Reading templates and createing a datagrame with three column: 'endpoint' 'channel' 'Template_avg' (i.e. adcs)
    template_files = os.listdir(directory_path)

    # Filter only files (not directories)
    template_files = [f for f in template_files if os.path.isfile(os.path.join(directory_path, f))]

    edp_list = []
    ch_list  = []
    adc_list = []

    for f in template_files:
        try:
            parts = f.split()
            if len(parts) < 3 or "_" not in parts[2]:
                print(f"Unexpected filename format: {f}")
                continue

            edp = parts[2].split("_")[0]
            ch = parts[2].split("_")[1].split('.')[0]

            with open(os.path.join(directory_path, f), 'rb') as file:
                temp = pickle.load(file)

            adc_list.append(temp)
            edp_list.append(edp)
            ch_list.append(ch)

        except Exception as e:
            print(f"Error loading {f}: {e}")

    df_template = pd.DataFrame({
        'endpoint' : edp_list,
        'channel' : ch_list,
        'Template_avg': adc_list})
    df_template["endpoint"] = df_template["endpoint"].astype(int)
    df_template["channel"]  = df_template["channel"].astype(int)

    return df_template
    

#####################################################################    


# Function to plot waveforms with/without peaks
def plotting_overlap_wf_PEAK_NEW(wfset, n_wf: int = 50, show : bool = True, save : bool = False, x_min=None, x_max=None, y_min=None, y_max=None, int_ll=None, int_ul=None, baseline=None, output_folder : str = 'output', analysis_label : str = 'test_peak_finding', peak_bool : bool = False, peak_beam_bool : bool = False):
    fig = go.Figure()

    if n_wf > len(wfset.waveforms):
        n_wf = len(wfset.waveforms)

    for i in range(n_wf):
        y = wfset.waveforms[i].adcs 
        x = np.arange(len(y))

            
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(width=0.5),
            showlegend=False))

        if peak_bool:
            fig.add_trace(go.Scatter(
                x=wfset.waveforms[i].analyses[analysis_label].result['peak_time'],
                y=wfset.waveforms[i].analyses[analysis_label].result['peak_amplitude'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle'),
                name='Peaks'
            ))

        
        if peak_beam_bool:
            fig.add_trace(go.Scatter(
                x=wfset.waveforms[i].analyses[analysis_label].result['beam_peak_time'],
                y=wfset.waveforms[i].analyses[analysis_label].result['beam_peak_amplitude'],
                mode='markers',
                marker=dict(color='blue', size=8, symbol='circle'),
                name='Beam peaks'
            ))

    xaxis_range = dict(range=[x_min, x_max]) if x_min is not None and x_max is not None else {}
    yaxis_range = dict(range=[y_min, y_max]) if y_min is not None and y_max is not None else {}

    fig.update_layout(
        xaxis_title="Time ticks",
        yaxis_title="Adcs",
        xaxis=xaxis_range,  
        yaxis=yaxis_range,  
        margin=dict(l=50, r=50, t=20, b=50),
        template="plotly_white",
        legend=dict(
            x=1,  
            y=1,  
            xanchor="right",
            yanchor="top",
            orientation="v", 
            bgcolor="rgba(255, 255, 255, 0.8)" ))

    if int_ll is not None:
        fig.add_shape(
            type="line",
            x0=int_ll,
            x1=int_ll,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="coral", width=2, dash="dash"),
            name=f"Lower integral limit \n(x = {int_ll})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="coral", width=2, dash="dash"),
            showlegend=True,
            name=f"Lower integral limit \n(x = {int_ll})"
        ))

    if int_ul is not None:
        fig.add_shape(
            type="line",
            x0=int_ul,
            x1=int_ul,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="chocolate", width=2, dash="dash"),
            name=f"Upper integral limit \n(x = {int_ul})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="chocolate", width=2, dash="dash"),
            showlegend=True,
            name=f"Upper integral limit \n(x = {int_ul})"
        ))

    if baseline is not None:
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=baseline,
            y1=baseline,
            xref="paper",
            yref="y",
            line=dict(color="red", width=1.5, dash="dash"),
            name=f"Baseline \n(y = {baseline})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", width=1.5, dash="dash"),
            showlegend=True,
            name=f"Baseline \n(y = {baseline})"
        ))

    if save:
        fig.write_image(f"{output_folder}/waveform_plot.png", scale=2)
    
    if show:
        fig.show()
    

#####################################################################    

   
# def reading_merge_hdf5(run_number, trigger, event_type, rucio_hdf5_dir, n_files_start, n_files_stop):
#     run_folder = os.path.join(rucio_hdf5_dir,f"run0{run_number}", trigger, event_type)
#     if not os.path.isdir(run_folder):
#         raise click.BadParameter(f"The folder {run_folder} does not exist")

#     # List all processed HDF5 files
#     all_files = sorted([f for f in os.listdir(run_folder) if f.endswith(".hdf5") and f.startswith("processed_")])

#     if not all_files:
#         print(f"No processed HDF5 files found in {run_folder}")
#         return
#     else:
#         print(f"Found {len(all_files)} files to merge in {run_folder}\n")

#         if n_files_start < 0:
#             n_files_start = 0
#         if n_files_stop > len(all_files):
#             n_files_stop = len(all_files)
#         if n_files_start > n_files_stop:
#             n_files_start = n_files_start-1
#         files_to_read = all_files[n_files_start:n_files_stop]

#         print(f"Reaging {len(files_to_read)} files, from n° {n_files_start} to {n_files_stop}\n")

#         wfset = None
#         i_index = 0
#         i_index_error = 0

#         for fname in tqdm(files_to_read, desc="Merging files", unit="file"):
#             filepath = os.path.join(run_folder, fname)
#             try:
#                 current_wfset = load_structured_waveformset(filepath)  # load HDF5 structured waveform set
#                 if i_index == 0:
#                     wfset = current_wfset
#                 else:
#                     wfset.merge(current_wfset)
#                 i_index += 1
#             except Exception as e:
#                 print(f"Error loading {filepath}: {e}")
#                 i_index_error += 1
#                 continue

#         print(f"\n# files read: {i_index}")
#         print(f"# files with errors: {i_index_error}")
#         print(f"# waveforms: {len(wfset.waveforms)}\n")

#         return wfset, files_to_read
    

#####################################################################    


def prepare_for_json(obj):
    # NaN → None
    if isinstance(obj, float) and math.isnan(obj):
        return None
    
    # numpy scalar → Python scalar
    if isinstance(obj, (np.generic,)):
        return obj.item()

    # numpy array → Python list
    if isinstance(obj, np.ndarray):
        return [prepare_for_json(x) for x in obj.tolist()]

    # list or tuple
    if isinstance(obj, (list, tuple)):
        return [prepare_for_json(x) for x in obj]

    # dict
    if isinstance(obj, dict):
        return {str(key): prepare_for_json(value) for key, value in obj.items()}

    # default: leave unchanged
    return obj


#####################################################################


def apa1_hist_distribution(df, energy, output_dir, show=True, save = True, additional_output = '', x_range = None, bins = None, bin_width = None):
    # Create a histogram for FS with the average n_pe for all triggers (you should see two distribution for E>2 GeV)

    xmin, xmax = (x_range if x_range is not None else (df['apa1_mean'].min(), df['apa1_mean'].max()))
    
    # Se è specificata la larghezza del bin, calcola il numero di bin
    if bin_width is not None:
        bins = int(np.ceil((xmax - xmin) / bin_width))
    elif bins is None:
        bins = 90 

    plt.figure(figsize=(10,5))
    plt.hist(
        df['apa1_mean'], 
        bins=bins, 
        color='tomato', 
        alpha=0.7, 
        edgecolor='black', 
        label=f'{len(df)} triggers'
    )

    if x_range is not None:
        plt.xlim(x_range[0],x_range[1])

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

    plt.xlabel(r"$\langle N_{\mathrm{PE}} \rangle$")
    plt.ylabel("Counts")
    plt.title(f"#pe distribution APA 1 at {energy} GeV")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"{output_dir}/apa1_hist_{additional_output}{energy}GeV.png", dpi=300)

    if show:
        plt.show()


#####################################################################    


def apa2_hist_distribution(df, energy, output_dir, show=True, save = True, additional_output = '', x_range = None, bins = None, bin_width = None):
    # Create a histogram for FS with the average n_pe for all triggers (you should see two distribution for E>2 GeV)

    # Se è specificata la larghezza del bin, calcola il numero di bin
    if bin_width is not None:
        bins = int(np.ceil((df['apa2_mean'].max() - df['apa2_mean'].min()) / bin_width))
        print(bins)
    elif bins is None:
        bins = 90 
    
    plt.figure(figsize=(10,5))
    plt.hist(
        df['apa2_mean'], 
        bins=bins, 
        color='dodgerblue',   # colore diverso da APA 1
        alpha=0.7,            # trasparenza
        edgecolor='black',    # bordo barre
        label=f"{len(df['apa2_mean'])} triggers"
    )

    if x_range is not None:
        plt.xlim(x_range[0],x_range[1])

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

    plt.xlabel(r"$\langle N_{\mathrm{PE}} \rangle$")
    plt.ylabel("Counts")
    plt.title(f"#pe distribution APA 2 at {energy} GeV")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"{output_dir}/apa2_hist_{additional_output}{energy}GeV.png", dpi=300)

    if show:
        plt.show()


#####################################################################    


def apa12_scatter(df, energy, output_dir, show=True, save = True, additional_output = '', apa1_threshold = None, muon_selection = False, match_apa12_tollerance = 0.05, x_range = None, y_range = None):
    # Create a scatter plot where each point corresponds to a trigger time: x=FS mean pe  y=ST mean pe
    plt.figure(figsize=(12, 4))
    plt.scatter(
        df["apa1_mean"], 
        df["apa2_mean"], 
        c='dodgerblue',   
        s=7,             
        alpha=0.7,        
        label=f"{len(df['apa1_mean'])} triggers"
    )

    if muon_selection: 
        mask_valid = (df["apa1_mean"].notna() & df["apa2_mean"].notna() & (df["apa1_mean"]) != 0 & (df["apa2_mean"] != 0))
        diff = (df.loc[mask_valid, "apa1_mean"] - df.loc[mask_valid, "apa2_mean"]).abs()
        mask_match = diff <= match_apa12_tollerance * df.loc[mask_valid, "apa1_mean"].abs()
        apa1_match = df.loc[mask_valid].loc[mask_match, "apa1_mean"]
        apa2_match = df.loc[mask_valid].loc[mask_match, "apa2_mean"]
        plt.scatter(apa1_match, apa2_match, c='orange', s=7, alpha=0.7, label=f"APA 1 - 2 match (Same #PE within {match_apa12_tollerance*100} % tollerance) - {len(apa1_match)} triggers")
        additional_output = 'muonselection_' + additional_output

    if apa1_threshold is not None:
        mask = df["apa1_mean"] < apa1_threshold
        apa1_sel = df.loc[mask, "apa1_mean"]
        apa2_sel = df.loc[mask, "apa2_mean"]
        plt.scatter(apa1_sel, apa2_sel, c='red', s=7, alpha=0.7, label=f"APA 1 #PE < {apa1_threshold} - {len(apa1_sel)} triggers")
        additional_output = 'apa1threshold_' + additional_output

    if x_range is not None:
        plt.xlim(x_range[0],x_range[1])
    if y_range is not None:
        plt.ylim(y_range[0],y_range[1])

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

    plt.xlabel(r"$\langle N_{\mathrm{PE}} \rangle$ on APA 1")
    plt.ylabel(r"$\langle N_{\mathrm{PE}} \rangle$ on APA 2")
    plt.title(f"Photoelectron distribution at {energy} GeV")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  

    if save:
        plt.savefig(os.path.join(output_dir, f"apa12_pe_distribution_{additional_output}{energy}GeV.png"), dpi=300)
    
    if show:
        plt.show()


#####################################################################    


def load_energy_dataframe(energy: int, apa12_folder: str, outpit_dir: str) -> pd.DataFrame:
    energy_folder = Path(apa12_folder) / f"{energy}GeV"

    if not energy_folder.exists():
        raise FileNotFoundError(f"Folder {energy_folder} doesn't exist")

    dataframes = []
    all_txt_lines = []

    for subfolder in sorted(energy_folder.iterdir()):
        if not subfolder.is_dir():
            continue
    
        if "_to_" not in subfolder.name:
            # print(f"⚠️ Ignored folder (no _to_): {subfolder.name}")
            continue

        csv_file = subfolder / f"photoelectron_dataframe_{energy}GeV.csv"
        txt_file = subfolder / "files_read.txt"

        # ---- CSV ----
        if not csv_file.exists():
            print(f"⚠️ Missing CSV file: {csv_file}")
            continue

        df = pd.read_csv(csv_file)

        start, _, stop = subfolder.name.partition("_to_")
        df["start"] = int(start)
        df["stop"] = int(stop)

        dataframes.append(df)

        # ---- TXT ----
        if txt_file.exists():
            with txt_file.open("r") as f:
                lines = f.readlines()
                all_txt_lines.extend(lines)
        else:
            print(f"⚠️ Missing TXT file: {txt_file}")

    if not dataframes:
        raise RuntimeError(f"No CSV file found for energy = {energy}")

    # concat dataframe
    full_df = pd.concat(dataframes, ignore_index=True)

    # scrivi file di testo unico
    output_txt = Path(outpit_dir) / "files_read_ALL.txt"
    with output_txt.open("w") as f:
        f.writelines(all_txt_lines)

    print(f"✅ Written merged txt file, {len(all_txt_lines)} files! \n")

    return full_df



#####################################################################    


def timeoffset_distribution_hist(timeoffset_list, apa, energy, output_dir, zoom_start = -100, zoom_stop = 0, show = True, save = True):
    offset_list = timeoffset_list # time_info_dic[trigger_mode]['offset_list']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  
    fig.suptitle(f'APA {apa} offset distribution - {energy} GeV', fontsize=20)

    axes[0].hist(offset_list, bins=10000, color='skyblue', edgecolor='black')
    axes[0].set_xlabel("Timeticks offset", fontsize=15)
    axes[0].set_ylabel("Counts", fontsize=15)
    axes[0].set_title(f"Complete histogram", fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend([f"Total counts = {len(offset_list)}"], fontsize = '15')

    axes[1].hist(offset_list, bins=100, color='skyblue', edgecolor='black', range=(zoom_start, zoom_stop))
    axes[1].set_xlabel("Timeticks offset", fontsize=15)
    axes[1].set_ylabel("Counts", fontsize=15)
    axes[1].set_title(f"Zoom histogram [{zoom_start} ; {zoom_stop}]", fontsize=16)
    axes[1].grid(True, alpha=0.3)

    if save:
        plt.savefig(os.path.join(output_dir, f"apa{apa}_offsethist_{energy}GeV.png"), dpi=300)
    if show:
        plt.show()

#####################################################################

def which_endpoints_in_the_APA(APA : int):
    endpoint_list = []
    for row in APA_map[APA].data: # cycle on rows
        for ch_info in row: # cycle on columns elements (i.e. channels)
            endpoint_list.append(ch_info.endpoint)
    return list(set(endpoint_list))


def which_APA_for_the_ENDPOINT(endpoint: int):
    apa_endpoints = {1: {104, 105, 107}, 2: {109}, 3: {111}, 4: {112, 113}}
    for apa, endpoints in apa_endpoints.items():
        if endpoint in endpoints:
            return apa
    return None


#####################################################################

def r2_score(y, y_fit):
    """
    Calcola il coefficiente di determinazione R².
    """
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    return r2

def r2_score_weighted(y, y_fit, yerr):
    """
    Calcola R² pesato.
    """
    w = 1 / yerr**2
    y_mean_w = np.average(y, weights=w)
    ss_res_w = np.sum(w * (y - y_fit)**2)
    ss_tot_w = np.sum(w * (y - y_mean_w)**2)
    r2_w = 1 - ss_res_w / ss_tot_w
    return r2_w

def chi2_func(y, y_fit, yerr, n_params):
    """
    Calcola chi-quadro e chi-quadro ridotto.
    """
    residuals = (y - y_fit) / yerr
    chi2_val = np.sum(residuals**2)
    ndf = len(y) - n_params
    chi2_red = chi2_val / ndf
    return chi2_val, chi2_red, ndf


#####################################################################

"""Function to round results with error, and by using scientific notation"""

def round_to_significant(value, error):
    if not isinstance(error, (int, float, np.number)) or not np.isfinite(error) or error == 0:
        return "N/A", "N/A"  # Restituisce una stringa se l'errore non è valido

    error_order = int(np.log10(error))
    significant_digits = 2  
    rounded_error = round(error, -error_order + (significant_digits - 1))
    rounded_value = round(value, -error_order + (significant_digits - 1))
    return rounded_value, rounded_error


'''
# Per avere una sola significativa nell'errore

def round_to_significant(value, error):
    if not isinstance(error, (int, float, np.number)) or not np.isfinite(error) or error == 0:
        return "N/A", "N/A"

    error = abs(error)

    # ordine di grandezza dell'errore
    error_order = int(np.floor(np.log10(error)))

    # prima cifra significativa dell'errore
    first_digit = int(error / 10**error_order)

    # regola standard: 1 cifra se >=3, altrimenti 2
    significant_digits = 1 if first_digit >= 3 else 2

    # posizione di arrotondamento
    decimals = -error_order + (significant_digits - 1)

    rounded_error = round(error, decimals)
    rounded_value = round(value, decimals)

    return rounded_value, rounded_error
'''

def to_scientific_notation(value, error):
    # Controllo se il valore non è un numero valido
    if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        return "N/A"  # Se il valore è NaN o infinito, restituisce "N/A"

    if value == 0:
        return "0"  # Caso speciale: zero non ha notazione scientifica

    exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0  # Evita log10(0)
    mantissa_value = value / 10**exponent
    mantissa_error = error / 10**exponent if error and np.isfinite(error) else 0  # Evita divisioni errate

    mantissa_value, mantissa_error = round_to_significant(mantissa_value, mantissa_error)
    
    return rf"({mantissa_value} $\pm$ {mantissa_error}) $\times 10^{exponent}$" if mantissa_value != "N/A" else "N/A"


def fmt(val, err):
        if not isinstance(err, (int, float, np.number)) or not np.isfinite(err) or err == 0:
            return "N/A"
        # usa notazione scientifica solo se necessario
        if abs(val) >= 1e3 or abs(val) <= 1e-2:
            return to_scientific_notation(val, err)
        v, e = round_to_significant(val, err)
        return f"{v} ± {e}"


#####################################################################

# Using it in channel_photoelectron_distriution_json.ipynb and adjacent_channel_study.ipynb

def load_energy_json_dict(energy: int, apa12_folder: str, output_dir: str) -> dict:
    """
    Legge tutti i JSON 'photoelectron_dic_{energy}GeV.json' nelle sottocartelle
    '{start}_to_{stop}' di {energy}GeV e li raccoglie in un unico dizionario:

        merged_dict["start_to_stop"] = contenuto_json

    Inoltre concatena tutti i files_read.txt in un unico file
    'files_read_ALL.txt' dentro output_dir.
    """

    energy_folder = Path(apa12_folder)

    if not energy_folder.exists():
        raise FileNotFoundError(f"Folder {energy_folder} doesn't exist")

    merged_dict = {}
    all_txt_lines = []

    for subfolder in sorted(energy_folder.iterdir()):
        # solo directory valide
        if not subfolder.is_dir():
            continue

        # solo cartelle tipo {start}_to_{stop}
        if "_to_" not in subfolder.name:
            continue

        json_file = subfolder / f"photoelectron_dic_{energy}GeV.json"
        txt_file = subfolder / "files_read.txt"

        # ---------- JSON ----------
        if not json_file.exists():
            print(f"⚠️ Missing JSON file: {json_file}")
            continue

        with json_file.open("r") as f:
            data = json.load(f)

        # salva il JSON sotto il nome della cartella
        merged_dict[subfolder.name] = data

        # ---------- TXT ----------
        if txt_file.exists():
            with txt_file.open("r") as f:
                # pulisce newline e mantiene ordine
                all_txt_lines.extend(line.rstrip() + "\n" for line in f)
        else:
            print(f"⚠️ Missing TXT file: {txt_file}")

    if not merged_dict:
        raise RuntimeError(f"No JSON files found for energy = {energy}")

    # ---------- SCRITTURA FILE TXT UNICO ----------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # rimuove eventuali duplicati mantenendo l'ordine
    all_txt_lines = list(dict.fromkeys(all_txt_lines))

    output_txt = output_dir / "files_read_ALL.txt"
    with output_txt.open("w") as f:
        f.writelines(all_txt_lines)

    print(f"✅ Written merged txt file: {output_txt}")
    print(f"✅ Total txt lines: {len(all_txt_lines)}")
    print(f"✅ Total JSON blocks: {len(merged_dict)}\n")

    return merged_dict


####
def adjacent_channel_info(): 
    adiacent_channels_1 = [[{'apa' : 1, 'end': 104, 'ch': 7, 'ly' : 30.44279648, 'ly_err': 0.9273210415192022}, {'apa' : 1, 'end': 104, 'ch': 5 , 'ly' : 40.37354826, 'ly_err': 1.1806092066610088}],  # 1
                        [{'apa' : 1, 'end': 104, 'ch': 5 , 'ly' : 40.37354826, 'ly_err': 1.1806092066610088}, {'apa' : 1, 'end': 104, 'ch': 2 , 'ly' : 47.12535466, 'ly_err': 1.15958222727287}],  # 1
                        [{'apa' : 1, 'end': 104, 'ch': 2 , 'ly' : 47.12535466, 'ly_err': 1.15958222727287}, {'apa' : 1, 'end': 104, 'ch': 0 , 'ly' : 41.27424546, 'ly_err': 1.0679136553355928}],  # 1
                        [{'apa' : 1, 'end': 104, 'ch': 1 , 'ly' : 38.98072494, 'ly_err': 2.1327739859549597}, {'apa' : 1, 'end': 104, 'ch': 3 , 'ly' : 45.20487447, 'ly_err': 1.1869000852324099}],  # 2
                        [{'apa' : 1, 'end': 104, 'ch': 3 , 'ly' : 45.20487447, 'ly_err': 1.1869000852324099}, {'apa' : 1, 'end': 104, 'ch': 4 , 'ly' : 58.72405404, 'ly_err': 1.2991189797899554}],  # 2
                        [{'apa' : 1, 'end': 104, 'ch': 4 , 'ly' : 58.72405404, 'ly_err': 1.2991189797899554}, {'apa' : 1, 'end': 104, 'ch': 6 , 'ly' : 48.25950909, 'ly_err': 1.8724407692495848}],  # 2
                        [{'apa' : 1, 'end': 104, 'ch': 17 , 'ly' : 92.78697251, 'ly_err': 3.7816848718547518}, {'apa' : 1, 'end': 104, 'ch': 15 , 'ly' : 109.05330274, 'ly_err': 2.449493632284315}],  # 3
                        [{'apa' : 1, 'end': 104, 'ch': 15 , 'ly' : 109.05330274, 'ly_err': 2.449493632284315}, {'apa' : 1, 'end': 104, 'ch': 12 , 'ly' : 108.89348459, 'ly_err': 3.5122386232258176}],  # 3
                        [{'apa' : 1, 'end': 104, 'ch': 12 , 'ly' : 108.89348459, 'ly_err': 3.5122386232258176}, {'apa' : 1, 'end': 104, 'ch': 10 , 'ly' : 116.83621838, 'ly_err': 5.742501993787541}],  # 3
                        [{'apa' : 1, 'end': 104, 'ch': 11 , 'ly' : 101.34080387, 'ly_err': 2.689821210147427}, {'apa' : 1, 'end': 104, 'ch': 13 , 'ly' : 118.20655528, 'ly_err': 3.1165405622418545}],  # 4
                        [{'apa' : 1, 'end': 104, 'ch': 13 , 'ly' : 118.20655528, 'ly_err': 3.1165405622418545}, {'apa' : 1, 'end': 104, 'ch': 14 , 'ly' : 103.36793441, 'ly_err': 4.340257608825766}],  # 4
                        [{'apa' : 1, 'end': 104, 'ch': 14 , 'ly' : 103.36793441, 'ly_err': 4.340257608825766}, {'apa' : 1, 'end': 104, 'ch': 16 , 'ly' : 92.98313468, 'ly_err': 5.036615910932509}],  # 4
                        [{'apa' : 1, 'end': 105, 'ch': 7 , 'ly' : 87.312015, 'ly_err': 2.317803342434923}, {'apa' : 1, 'end': 105, 'ch': 5 , 'ly' : 117.31386478, 'ly_err': 3.387779920269546}],  # 5 
                        [{'apa' : 1, 'end': 105, 'ch': 5 , 'ly' : 117.31386478, 'ly_err': 3.387779920269546}, {'apa' : 1, 'end': 105, 'ch': 2 , 'ly' : 90.97246823, 'ly_err': 3.5747908979307588}],  # 5 
                        [{'apa' : 1, 'end': 105, 'ch': 2 , 'ly' : 90.97246823, 'ly_err': 3.5747908979307588}, {'apa' : 1, 'end': 105, 'ch': 0 , 'ly' : 89.53119499, 'ly_err': 4.531103328033819}],  # 5 
                        [{'apa' : 1, 'end': 105, 'ch': 1 , 'ly' : 79.93694879, 'ly_err': 1.702966211248327}, {'apa' : 1, 'end': 105, 'ch': 3 , 'ly' : 93.11190323, 'ly_err': 2.5303282802683014}],  # 6
                        [{'apa' : 1, 'end': 105, 'ch': 3 , 'ly' : 93.11190323, 'ly_err': 2.5303282802683014}, {'apa' : 1, 'end': 105, 'ch': 4 , 'ly' : 99.69944766, 'ly_err': 4.655990694031124}],  # 6 
                        [{'apa' : 1, 'end': 105, 'ch': 4 , 'ly' : 99.69944766, 'ly_err': 4.655990694031124}, {'apa' : 1, 'end': 105, 'ch': 6 , 'ly' : 72.06117263, 'ly_err': 4.727299606198586}],  # 6 
                        [{'apa' : 1, 'end': 105, 'ch': 26 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 1, 'end': 105, 'ch': 24 , 'ly' : np.nan, 'ly_err': np.nan}],  # 7
                        [{'apa' : 1, 'end': 105, 'ch': 24 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 1, 'end': 105, 'ch': 23 , 'ly' : np.nan, 'ly_err': np.nan}],  # 7 
                        [{'apa' : 1, 'end': 105, 'ch': 23 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 1, 'end': 105, 'ch': 21 , 'ly' : np.nan, 'ly_err': np.nan}],  # 7 
                        [{'apa' : 1, 'end': 105, 'ch': 10 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 1, 'end': 105, 'ch': 12 , 'ly' : np.nan, 'ly_err': np.nan}],  # 8
                        [{'apa' : 1, 'end': 105, 'ch': 12 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 1, 'end': 105, 'ch': 15 , 'ly' : 40.7653192, 'ly_err': 1.776237936610994}],  # 8 
                        [{'apa' : 1, 'end': 105, 'ch': 15 , 'ly' : 40.7653192, 'ly_err': 1.776237936610994}, {'apa' : 1, 'end': 105, 'ch': 17 , 'ly' : 32.36634835, 'ly_err': 1.7519993830195832}],  # 8 
                        ]

    adiacent_channels_2 = [[{'apa' : 2, 'end': 109, 'ch': 27 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 25 , 'ly' : np.nan, 'ly_err': np.nan}],  # 1 
                        [{'apa' : 2, 'end': 109, 'ch': 25 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 22 , 'ly' : np.nan, 'ly_err': np.nan}],  # 1 
                        [{'apa' : 2, 'end': 109, 'ch': 22 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 20 , 'ly' : np.nan, 'ly_err': np.nan}],  # 1 
                        [{'apa' : 2, 'end': 109, 'ch': 21 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 23 , 'ly' : np.nan, 'ly_err': np.nan}],  # 2
                        [{'apa' : 2, 'end': 109, 'ch': 23 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 24 , 'ly' : np.nan, 'ly_err': np.nan}],  # 2 
                        [{'apa' : 2, 'end': 109, 'ch': 24 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 26 , 'ly' : np.nan, 'ly_err': np.nan}],  # 2
                        [{'apa' : 2, 'end': 109, 'ch': 37 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 35 , 'ly' : np.nan, 'ly_err': np.nan}],  # 3 
                        [{'apa' : 2, 'end': 109, 'ch': 35 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 32 , 'ly' : np.nan, 'ly_err': np.nan}],  # 3 
                        [{'apa' : 2, 'end': 109, 'ch': 32 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 30 , 'ly' : np.nan, 'ly_err': np.nan}],  # 3
                        [{'apa' : 2, 'end': 109, 'ch': 31 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 33 , 'ly' : np.nan, 'ly_err': np.nan}],  # 4 
                        [{'apa' : 2, 'end': 109, 'ch': 33 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 34 , 'ly' : np.nan, 'ly_err': np.nan}],  # 4 
                        [{'apa' : 2, 'end': 109, 'ch': 34 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 36 , 'ly' : np.nan, 'ly_err': np.nan}],  # 4   
                        [{'apa' : 2, 'end': 109, 'ch': 7 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 5 , 'ly' : np.nan, 'ly_err': np.nan}],  # 5 
                        [{'apa' : 2, 'end': 109, 'ch': 5 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 2 , 'ly' : np.nan, 'ly_err': np.nan}],  # 5 
                        [{'apa' : 2, 'end': 109, 'ch': 2 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 0 , 'ly' : np.nan, 'ly_err': np.nan}],  # 5 
                        [{'apa' : 2, 'end': 109, 'ch': 1 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 3 , 'ly' : np.nan, 'ly_err': np.nan}],  # 6
                        [{'apa' : 2, 'end': 109, 'ch': 3 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 4 , 'ly' : np.nan, 'ly_err': np.nan}],  # 6 
                        [{'apa' : 2, 'end': 109, 'ch': 4 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 6 , 'ly' : np.nan, 'ly_err': np.nan}],  # 6
                        [{'apa' : 2, 'end': 109, 'ch': 17 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 15 , 'ly' : np.nan, 'ly_err': np.nan}],  # 7 
                        [{'apa' : 2, 'end': 109, 'ch': 15 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 12 , 'ly' : np.nan, 'ly_err': np.nan}],  # 7 
                        [{'apa' : 2, 'end': 109, 'ch': 12 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 10 , 'ly' : np.nan, 'ly_err': np.nan}],  # 7
                        [{'apa' : 2, 'end': 109, 'ch': 11 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 13 , 'ly' : np.nan, 'ly_err': np.nan}],  # 8 
                        [{'apa' : 2, 'end': 109, 'ch': 13 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 14 , 'ly' : np.nan, 'ly_err': np.nan}],  # 8 
                        [{'apa' : 2, 'end': 109, 'ch': 14 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 16 , 'ly' : np.nan, 'ly_err': np.nan}],  # 8
                        [{'apa' : 2, 'end': 109, 'ch': 47 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 45 , 'ly' : np.nan, 'ly_err': np.nan}],  # 9 
                        [{'apa' : 2, 'end': 109, 'ch': 45 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 42 , 'ly' : np.nan, 'ly_err': np.nan}],  # 9 
                        [{'apa' : 2, 'end': 109, 'ch': 42 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 40 , 'ly' : np.nan, 'ly_err': np.nan}],  # 9 
                        [{'apa' : 2, 'end': 109, 'ch': 41 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 43 , 'ly' : np.nan, 'ly_err': np.nan}],  # 10
                        [{'apa' : 2, 'end': 109, 'ch': 43 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 44 , 'ly' : np.nan, 'ly_err': np.nan}],  # 10
                        [{'apa' : 2, 'end': 109, 'ch': 44 , 'ly' : np.nan, 'ly_err': np.nan}, {'apa' : 2, 'end': 109, 'ch': 46 , 'ly' : np.nan, 'ly_err': np.nan}],  # 10  
                        ]

    channels_excluded = [{'end': 109, 'ch': 0},
                        {'end': 109, 'ch': 2},
                        {'end': 109, 'ch': 4},
                        {'end': 109, 'ch': 6},
                        {'end': 109, 'ch': 31},
                        {'end': 109, 'ch': 34},
                        {'end': 109, 'ch': 40},                       
                        ]

    return adiacent_channels_1, adiacent_channels_2, channels_excluded

def all_channels_info_possible_pairs():
    return list(combinations(channel_list_apa1(), 2)), [], []

def channel_list_apa1():
    channel_list_info = [ 
    {'apa' : 1, 'end': 104, 'ch': 7, 'ly' : 30.44279648, 'ly_err': 0.9273210415192022}, #1
    {'apa' : 1, 'end': 104, 'ch': 5 , 'ly' : 40.37354826, 'ly_err': 1.1806092066610088}, 
    {'apa' : 1, 'end': 104, 'ch': 2 , 'ly' : 47.12535466, 'ly_err': 1.15958222727287},
    {'apa' : 1, 'end': 104, 'ch': 0 , 'ly' : 41.27424546, 'ly_err': 1.0679136553355928},
    {'apa' : 1, 'end': 104, 'ch': 1 , 'ly' : 38.98072494, 'ly_err': 2.1327739859549597}, #2
    {'apa' : 1, 'end': 104, 'ch': 3 , 'ly' : 45.20487447, 'ly_err': 1.1869000852324099},
    {'apa' : 1, 'end': 104, 'ch': 4 , 'ly' : 58.72405404, 'ly_err': 1.2991189797899554}, 
    {'apa' : 1, 'end': 104, 'ch': 6 , 'ly' : 48.25950909, 'ly_err': 1.8724407692495848},
    {'apa' : 1, 'end': 104, 'ch': 17 , 'ly' : 92.78697251, 'ly_err': 3.7816848718547518}, #3
    {'apa' : 1, 'end': 104, 'ch': 15 , 'ly' : 109.05330274, 'ly_err': 2.449493632284315}, 
    {'apa' : 1, 'end': 104, 'ch': 12 , 'ly' : 108.89348459, 'ly_err': 3.5122386232258176}, 
    {'apa' : 1, 'end': 104, 'ch': 10 , 'ly' : 116.83621838, 'ly_err': 5.742501993787541},  
    {'apa' : 1, 'end': 104, 'ch': 11 , 'ly' : 101.34080387, 'ly_err': 2.689821210147427}, #5
    {'apa' : 1, 'end': 104, 'ch': 13 , 'ly' : 118.20655528, 'ly_err': 3.1165405622418545}, 
    {'apa' : 1, 'end': 104, 'ch': 14 , 'ly' : 103.36793441, 'ly_err': 4.340257608825766}, 
    {'apa' : 1, 'end': 104, 'ch': 16 , 'ly' : 92.98313468, 'ly_err': 5.036615910932509},  
    {'apa' : 1, 'end': 105, 'ch': 7 , 'ly' : 87.312015, 'ly_err': 2.317803342434923}, #5
    {'apa' : 1, 'end': 105, 'ch': 5 , 'ly' : 117.31386478, 'ly_err': 3.387779920269546},   
    {'apa' : 1, 'end': 105, 'ch': 2 , 'ly' : 90.97246823, 'ly_err': 3.5747908979307588}, 
    {'apa' : 1, 'end': 105, 'ch': 0 , 'ly' : 89.53119499, 'ly_err': 4.531103328033819},   
    {'apa' : 1, 'end': 105, 'ch': 1 , 'ly' : 79.93694879, 'ly_err': 1.702966211248327}, #6
    {'apa' : 1, 'end': 105, 'ch': 3 , 'ly' : 93.11190323, 'ly_err': 2.5303282802683014},  
    {'apa' : 1, 'end': 105, 'ch': 4 , 'ly' : 99.69944766, 'ly_err': 4.655990694031124}, 
    {'apa' : 1, 'end': 105, 'ch': 6 , 'ly' : 72.06117263, 'ly_err': 4.727299606198586},   
    {'apa' : 1, 'end': 105, 'ch': 26 , 'ly' : 74.92851465, 'ly_err': 1.459814209023413}, #7
    {'apa' : 1, 'end': 105, 'ch': 24 , 'ly' : 84.68736824, 'ly_err': 0.9293838571366848},  
    {'apa' : 1, 'end': 105, 'ch': 23 , 'ly' : 71.26030192, 'ly_err': 3.5812201411203324}, 
    {'apa' : 1, 'end': 105, 'ch': 21 , 'ly' : 91.37828259, 'ly_err': 5.559005442664104},   
    {'apa' : 1, 'end': 105, 'ch': 10 , 'ly' : 37.10260122, 'ly_err': 0.8892024181673603}, #8
    {'apa' : 1, 'end': 105, 'ch': 12 , 'ly' : 38.22011045, 'ly_err': 1.029763484557239},  
    {'apa' : 1, 'end': 105, 'ch': 15 , 'ly' : 40.7653192, 'ly_err': 1.776237936610994}, 
    {'apa' : 1, 'end': 105, 'ch': 17 , 'ly' : 32.36634835, 'ly_err': 1.7519993830195832}
    ]

    return channel_list_info

def apa1_columns_channels():
    channel_list_info = channel_list_apa1()
    col1 = channel_list_info[0::4]  # indice 0, 4, 8, ...
    col2 = channel_list_info[1::4]  # indice 1, 5, 9, ...
    col3 = channel_list_info[2::4]  # indice 2, 6, 10, ...
    col4 = channel_list_info[3::4]  # indice 3, 7, 11, ...

    return {1: col1, 2: col2, 3: col3, 4: col4}

#####################################################################

def weighted_mean_kinetic_energy_function(beam_momentum_list, 
                                          beam_composition_filepath : str = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/np04_beam_particle_content.csv",
                                          particles_dict = {'e' : 0.92828, 'pi' : 139.6, 'p': 938.28, 'k': 493.67} # {particle : mass} in MeV/c2
                                          ):
    
    df_composition = pd.read_csv(beam_composition_filepath)
    df_composition.columns = (df_composition.columns.str.replace(r'\s*\[.*?\]', '', regex=True).str.strip().str.lower())
    df_composition['for_normalization'] = df_composition[particles_dict.keys()].sum(axis=1)

    weighted_mean_kinetic_energy_list, weighted_mean_kinetic_energy_error_list = [], []

    for momentum in beam_momentum_list:
        w_K = 0
        w_K_e = 0
        var_K = 0
        row_composition = df_composition[df_composition['momentum'] == momentum].iloc[0]

        # Weighted mean value + error
        for particle, mass in particles_dict.items():
            w_p = row_composition[particle] / row_composition['for_normalization']
            K_p = np.sqrt(momentum**2 + (mass/1000)**2) - (mass/1000)
        
            w_K += w_p * K_p
            var_K+= w_p * (K_p**2)
        
        sigma_K = np.sqrt(var_K - w_K**2)
        w_K_e = np.sqrt((sigma_K)**2 + (w_K*0.05)**2)

        weighted_mean_kinetic_energy_list.append(w_K)
        weighted_mean_kinetic_energy_error_list.append(w_K_e)

    return weighted_mean_kinetic_energy_list, weighted_mean_kinetic_energy_error_list
