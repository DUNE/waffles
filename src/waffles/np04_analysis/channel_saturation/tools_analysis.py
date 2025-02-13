import numpy as np
import click
import pickle
import sys
import os
import re
import json
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from tqdm import tqdm 
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
fig, axes = plt.subplots(2,2)

from matplotlib.backends.backend_pdf import PdfPages
pdf_file_backup = PdfPages(f"/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/output/pdf_file_backup.pdf")


import waffles

from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna

from waffles.data_classes.WaveformSet import *
from waffles.data_classes.Waveform import *

#import waffles.input.raw_root_reader as root_reader 
import waffles.input.raw_hdf5_reader as hdf5_reader

 
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map

from input.beam_run_info import *


################################################################################################################################################
# Fits and plots

def round_to_significant(value, error):
    error_order = int(np.log10(error))
    significant_digits = 2  
    rounded_error = round(error, -error_order + (significant_digits - 1))
    rounded_value = round(value, -error_order + (significant_digits - 1))
    return rounded_value, rounded_error

def to_scientific_notation(value, error):
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa_value = value / 10**exponent
    mantissa_error = error / 10**exponent
    mantissa_value, mantissa_error = round_to_significant(mantissa_value, mantissa_error)
    #return mantissa_value, mantissa_error, exponent
    return f"({mantissa_value} ± {mantissa_error}) × 10^{exponent}"



def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def linear_fit(x, a, b):
    return a * x + b

################################################################################################################################################
# Filters

# To select wf of a given run  
def run_filter(waveform : waffles.Waveform, run : int = 27361) -> bool:
    if (waveform.run_number == run):
        return True
    else:
        return False


# To select a channel 
def channel_filter(waveform : waffles.Waveform, end : int = 109, ch : int = 35) -> bool:
    if (waveform.channel == ch) and (waveform.endpoint == end) :
        return True
    else:
        return False

# To select beam events for self-trigger channels looking at the time_offset DAQ-PDS (check value!!)    
def beam_self_trigger_filter(waveform : waffles.Waveform, timeoffset_min : int = -120, timeoffset_max : int = -90) -> bool:
    daq_pds_timeoffset = waveform.timestamp - waveform.daq_window_timestamp
    if (daq_pds_timeoffset < timeoffset_max) and (daq_pds_timeoffset > timeoffset_min) :
        return True
    else:
        return False

# To select saturated events
def saturation_filter(waveform : waffles.Waveform, min_adc_saturation : int = 0) -> bool:
    if min(waveform.adcs) <= min_adc_saturation :
        return True
    else:
        return False
    
    
# To select wf in a certain ENDPOINT list 
def endpoint_list_filter(waveform : waffles.Waveform, endpoint_list) -> bool:
    if waveform.endpoint in endpoint_list:
        return True
    else:
        return False
    
################################################################################################################################################
#Other stuff

# To determine if we have full_streaming (True) or self_trigger (False) data     
def full_streaming_mode(wfset : waffles.WaveformSet, run : int = 27367) -> bool:
    if 104 in wfset.available_channels[run].keys():
        return True
    else:
        return False 

def full_self_mode(wfset : waffles.WaveformSet, run : int = 27367) -> tuple[bool, bool]:
    full_streaming =  False
    self_trigger =  False
    available_end = wfset.available_channels[run].keys()
    if (104 in available_end) or (105 in available_end) or (107 in available_end): 
        full_streaming = True
    if (109 in available_end) or (111 in available_end) or (112 in available_end) or (113 in available_end): 
        self_trigger = True
    return full_streaming, self_trigger


def trigger_string(full_streaming : bool) -> str:
    if full_streaming:
        return 'full'
    else:
        return 'self' 
    

################################################################################################################################################
# Light yield analysis

def LightYield_SelfTrigger_channel_analysis(wfset : waffles.WaveformSet, end : int = 109, ch : int = 35, run_set : dict = A, pdf_file : PdfPages = pdf_file_backup, analysis_label : str = 'standard'):
    ly_data_dic = {key: {} for key in run_set['Runs'].keys()} 
    ly_result_dic = {'x':[], 'y':[], 'e_y':[], 'slope':{}, 'intercept':{}} 
    
    x = []
    y = []
    y_err = []

    n_row_plot = int(len(wfset.runs)/2)
    
    if n_row_plot>0: 
        n_row_plot+=1
        if (len(wfset.runs) - n_row_plot) == 0:
            n_row_plot+=1
            
        
        fig, ax = plt.subplots(n_row_plot, 2, figsize=(12, 10))
        plt.suptitle(f'Endpoint {end} - Channel {ch}')
        ax = ax.flatten()
        i=0
        for energy, run in run_set['Runs'].items():
            try:
                wfset_run = wfset.from_filtered_WaveformSet(wfset, run_filter, run)
                ly_data_dic[energy] = charge_study(wfset_run, end, ch, run, energy, analysis_label, ax[i])
                ly_result_dic['x'].append(energy)
                ly_result_dic['y'].append(ly_data_dic[energy]['gaussian fit']['mean']['value'])
                ly_result_dic['e_y'].append(ly_data_dic[energy]['gaussian fit']['sigma']['value'])
                print(f"For energy {energy} GeV --> {ly_data_dic[energy]['gaussian fit']['mean']['value']:.0f} +/- {ly_data_dic[energy]['gaussian fit']['sigma']['value']:.0f}")
            except Exception as e:
                ax[i].set_title(f"Energy: {energy} GeV")
                print(f'For energy {energy} GeV --> no data')
            i+=1
            
        # Fit lineare
        x = np.array(ly_result_dic['x'])
        y = np.array(ly_result_dic['y'])
        y_err = np.array(ly_result_dic['e_y'])
        
        if len(x) > 1:
            popt, pcov = curve_fit(linear_fit, x, y, sigma=y_err, absolute_sigma=True)
            slope, intercept = popt
            slope_err, intercept_err = np.sqrt(np.diag(pcov))
            
            # Plot dati e fit
            ax[i].errorbar(x, y, yerr=y_err, fmt='o', label='Data')
            ax[i].plot(x, linear_fit(x, *popt), 'r-', label=f"Linear fit: y=a+bx\n$a = {to_scientific_notation(intercept, intercept_err)}$ \n$b = {to_scientific_notation(slope, slope_err)}$") 
            ax[i].set_xlabel('Beam energy (GeV)')
            ax[i].set_ylabel('Integrated charge')
            ax[i].set_title('Charge vs energy with linear fit')
            ax[i].legend()
            
            ly_result_dic['slope'] = {'value': slope, 'error': slope_err}
            ly_result_dic['intercept'] = {'value': intercept, 'error': intercept_err}
        else:
            ly_result_dic['slope'] = {'value' : 0, 'error' : 0}
            ly_result_dic['intercept'] = {'value' : 0, 'error' : 0}
         
        plt.tight_layout()
        pdf_file.savefig(fig)
        plt.close(fig)
        return ly_data_dic, ly_result_dic
        
    
    else:
        print('Not enought runs avilable for that channel --> skipped')
        return {},{} 


def charge_study(wfset : waffles.WaveformSet, end : int = 109, ch : int = 35, run : int = 27367, energy : float = 1, analysis_label : str = 'standard', ax : Axes = axes[0]) -> tuple[bool, bool]:  
    ly_data = {'histogram data': [], 'gaussian fit' : {}} ######### da implementare
    charges = []
    for wf in wfset.waveforms:
        charges.append(wf.analyses[analysis_label].result['integral'])
    ly_data['histogram data'] = charges
    charges = np.array(charges)
    
    try:
        bin_heights, bin_edges = np.histogram(charges, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  
        p0 = [np.mean(charges), np.std(charges), max(bin_heights)]

        popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=p0)
        perr = np.sqrt(np.diag(pcov))    
        
        # Genera i dati della curva adattata
        x_fit = np.linspace(min(charges), max(charges), 1000)
        y_fit = gaussian(x_fit, popt[0], popt[1], popt[2])

        # Crea la figura e l'istogramma
        ax.hist(charges, bins=100, density=True, alpha=0.6, color='blue', label="Data")
        ax.plot(x_fit, y_fit, color='red', lw=2, label=f"Gaussian fit: \n$\mu= {to_scientific_notation(popt[0], perr[0])}$ \n$\sigma= {to_scientific_notation(popt[1], perr[1])}$ \n$A= {to_scientific_notation(popt[2], perr[2])} $") #label=f"Fit Gaussiano\n$\mu= {popt[0]:.2f} \pm {perr[0]:.2f}$\n$\sigma={popt[1]:.2f} \pm {perr[1]:.2f}$")
        ax.set_xlabel("Integrated Charge")
        ax.set_ylabel("Density")
        ax.set_title(f"Energy: {energy} GeV")
        ax.legend(fontsize='small')
        
        ly_data['gaussian fit'] = {'mean': {'value': popt[0], 'error': perr[0]}, 'sigma': {'value': popt[1], 'error': perr[1]}, 'normalized amplitude': {'value': popt[2], 'error': perr[2]}} 
        
        return ly_data
        
    except Exception as e:
        print(f'Fit error: {e} --> skipped')
        return ly_data


################################################################################################################################################
# Searching for beam events

def searching_for_beam_events(wfset : waffles.WaveformSet, bin : int = 1000, x_min=None, x_max=None):
    timeoffset_list_DAQ = []
    for wf in wfset.waveforms: 
        timeoffset_list_DAQ.append(wf.timestamp - wf.daq_window_timestamp)
    
    # Imposta x_min e x_max se non forniti
    if x_min is None:
        x_min = min(timeoffset_list_DAQ)
    if x_max is None:
        x_max = max(timeoffset_list_DAQ)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))   
    ax.hist(timeoffset_list_DAQ, bins=bin, color='blue', edgecolor='black')
    ax.set_xlim(x_min, x_max) 
    fig.savefig('/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/output/beam_timeoffset_histogram.jpg')
    plt.close()
    
def searching_for_beam_events_interactive(wfset : waffles.WaveformSet, show : bool = False, save : bool = True, bin : int = 100000, x_min = None, x_max = None, beam_min = None, beam_max = None):
    timeoffset_list_DAQ = []
    for wf in wfset.waveforms: 
        timeoffset_list_DAQ.append(wf.timestamp - wf.daq_window_timestamp)
    
    if x_min is None:
        x_min = min(timeoffset_list_DAQ)
    if x_max is None:
        x_max = max(timeoffset_list_DAQ)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=timeoffset_list_DAQ,
        nbinsx=bin,  
        marker=dict(color='blue', line=dict(color='black', width=1))))
    
    if beam_min is not None:
        fig.add_shape(
            type="line",
            x0=beam_min,
            x1=beam_min,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            name=f"Beam Min \n(x = {beam_min})")

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", width=2, dash="dash"),
            showlegend=True,
            name=f"Beam Min \n(x = {beam_min})"))

    if beam_max is not None:
        fig.add_shape(
            type="line",
            x0=beam_max,
            x1=beam_max,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            name=f"Beam Max \n(x = {beam_max})")

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", width=2, dash="dash"),
            showlegend=True,
            name=f"Beam Max \n(x = {beam_max})"))
    
    fig.update_layout(
        title="Interactive Histogram of Time Offsets",
        xaxis_title="Time Offset",
        yaxis_title="Count",
        xaxis=dict(range=[x_min, x_max]),  
        template="plotly_white",
        bargap=0.1)
    
    if save:
        fig.write_image("/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/output/beam_timeoffset_histogram.png", scale=2)
    if show:
        fig.show()

################################################################################################################################################    
# Plot waveforms (searching for integration range)
    
def plotting_overlap_wf(wfset, n_wf: int = 50, show : bool = False, save : bool = True, x_min=None, x_max=None, y_min=None, y_max=None, int_ll=None, int_ul=None, baseline=None):
    fig = go.Figure()

    for i in range(n_wf):
        fig.add_trace(go.Scatter(
            x=np.arange(len(wfset.waveforms[i].adcs)) + wfset.waveforms[i].time_offset,
            y=wfset.waveforms[i].adcs,
            mode='lines',
            line=dict(width=0.5),
            showlegend=False))

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
        fig.write_image("/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/output/waveform_plot.png", scale=2)
    
    if show:
        fig.show()
  
        
################################################################################################################################################
# Searching in the channel map    
    
def which_endpoints_in_the_APA(APA : int):
    endpoint_list = []
    for row in APA_map[APA].data: # cycle on rows
        for ch_info in row: # cycle on columns elements (i.e. channels)
            endpoint_list.append(ch_info.endpoint)
    return list(set(endpoint_list))


def which_channels_in_the_ENDPOINT(endpoint : int):
    channel_list = []
    for APA, apa_info in APA_map.items():
        for row in apa_info.data: # cycle on rows
            for ch_info in row: # cycle on columns elements (i.e. channels)
                if ch_info.endpoint == endpoint :
                    channel_list.append(ch_info.channel)       
    return channel_list

################################################################################################################################################
# Check or click option

def validate_choice(ctx, param, value):
    value = value.lower()
    valid_choices = param.type.choices
    if value not in valid_choices:
        raise click.BadParameter(f"Invalid choice: {value}. Must be one of {valid_choices}")
    return value == "yes"

def validate_start_stop(value, start_or_stop):
    if start_or_stop == 'start':
        if value == 'min':
            return float('-inf')  
        try:
            return int(value)
        except ValueError:
            raise click.BadParameter(f"Invalid value for start index: {value}. Must be 'min' or a valid number.")
    elif start_or_stop == 'stop':
        if value == 'max':
            return float('inf')  
        try:
            return int(value)
        except ValueError:
            raise click.BadParameter(f"Invalid value for stop index: {value}. Must be 'max' or a valid number.")
        
        
def validate_set_list_all(ctx, param, value):
    valid_names = {item["Name"] for item in run_set_list}
    if value is None or value == 'all':
        return ['A', 'B']
    if value not in valid_names:
        raise click.BadParameter(f"Invalid set name: {value}. Must be one of {valid_names}")
    return value


def validate_set_list_or(ctx, param, value):
    valid_names = {item["Name"] for item in run_set_list}
    if value not in valid_names:
        raise click.BadParameter(f"Invalid set name: {value}. Must be one of {valid_names}")
    else:
        return value
    
    
def validate_full_streaming(ctx, param, value):
    value = value.lower()
    if value in ['yes', 'no']:
        return value == 'yes'
    raise click.BadParameter("Error: invalid full-streaming mode. Use 'yes' or 'no'.")

def validate_folder(ctx, param, value):
    if not os.path.isdir(value):
        raise click.BadParameter(f"Invalid folder path: {value}")
    return value


def validate_run_list(ctx, param, value):
    set_selected = ctx.params.get("set_list")
    if isinstance(set_selected, list):
        return None  
    selected_set = next((item for item in run_set_list if item["Name"] == set_selected), None)
    if selected_set is None:
        raise click.BadParameter(f"You must select a set")
    valid_runs = set(selected_set["Runs"].values())  
    if value is None:
        return None  
    try:
        run_numbers = [int(num) for num in value.split(",")]
    except ValueError:
        raise click.BadParameter("Run list must be a comma-separated list of integers.")
    if not all(run in valid_runs for run in run_numbers):
        raise click.BadParameter(f"Invalid run numbers: {run_numbers}. Must be in {valid_runs}")
    return run_numbers  

def validate_beam_selection(ctx, param, value):
    value = value.lower()
    if value == 'no':
        return False
    elif value == 'yes':
        return True
    raise click.BadParameter("Error: invalid beam selection. Use 'yes' or 'no'.")