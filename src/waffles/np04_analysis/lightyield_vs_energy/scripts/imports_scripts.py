import numpy as np
import click
import pickle
import sys
import os
import re
import json
from pathlib import Path
from tqdm import tqdm 
from datetime import datetime

from waffles.data_classes.WaveformSet import *
from waffles.data_classes.Waveform import *

import waffles.input_output.raw_hdf5_reader as hdf5_reader
 
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map

############################
with open('/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/beam_run_info.json', "r") as file:
    run_set_list = json.load(file)

############################

# Filters

# To select wf of a given run  
def run_filter(waveform : Waveform, run : int = 27361) -> bool:
    if (waveform.run_number == run):
        return True
    else:
        return False


# To select a channel 
def channel_filter(waveform : Waveform, end : int = 109, ch : int = 35) -> bool:
    if (waveform.channel == ch) and (waveform.endpoint == end) :
        return True
    else:
        return False

# To select beam events for self-trigger channels looking at the time_offset DAQ-PDS (check value!!)    
def beam_self_trigger_filter(waveform : Waveform, timeoffset_min : int = -120, timeoffset_max : int = -90) -> bool:
    daq_pds_timeoffset = waveform.timestamp - waveform.daq_window_timestamp
    if (daq_pds_timeoffset < timeoffset_max) and (daq_pds_timeoffset > timeoffset_min) :
        return True
    else:
        return False

# To select saturated events
def saturation_filter(waveform : Waveform, min_adc_saturation : int = 0) -> bool:
    if min(waveform.adcs) <= min_adc_saturation :
        return True
    else:
        return False
    
    
# To select wf in a certain ENDPOINT list 
def endpoint_list_filter(waveform : Waveform, endpoint_list) -> bool:
    if waveform.endpoint in endpoint_list:
        return True
    else:
        return False
    
################################################################################################################################################
#Other stuff

# To determine if we have full_streaming (True) or self_trigger (False) data     
def full_streaming_mode(wfset : WaveformSet, run : int = 27367) -> bool:
    if 104 in wfset.available_channels[run].keys():
        return True
    else:
        return False 

def full_self_mode(wfset : WaveformSet, run : int = 27367) -> tuple[bool, bool]:
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
    valid_names = run_set_list.keys()
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
    selected_set = run_set_list.get(set_selected)
    if selected_set is None:
        raise click.BadParameter(f"You must select a valid set. Available sets are: {', '.join(run_set_list.keys())}")
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


##########

