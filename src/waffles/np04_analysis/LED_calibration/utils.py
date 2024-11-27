import numpy as np
import pandas as pd
import os

from waffles.input.raw_root_reader import WaveformSet_from_root_files
from waffles.input.pickle_file_reader import WaveformSet_from_pickle_files
from waffles.input.raw_root_reader import WaveformSet_from_root_file
from waffles.input.pickle_file_reader import WaveformSet_from_pickle_file
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.IPDict import IPDict
from waffles.np04_utils.utils import get_channel_iterator

from waffles.np04_analysis.LED_calibration.params import *

def get_input_filepath(base_folderpath: str, batch: int, run: int, apa: int, pde: float):
    aux = 'apa_2' if apa == 2 else 'apas_34'
    return  f"{base_folderpath}/batch_{batch}/{aux}/{pde}/run_{run}_chunk_0.pkl"

def get_input_folderpath(base_folderpath: str, batch: int, apa: int, pde: float):
    aux = 'apa_2' if apa == 2 else 'apas_34'
    return  f"{base_folderpath}/batch_{batch}/{aux}/{pde}"


def get_apa_foldername(measurements_batch, apa_no):
    """This function encapsulates the non-homogeneous 
    naming convention of the APA folders depending 
    on the measurements batch.""" 

    if measurements_batch not in [1, 2, 3]:
        raise ValueError(f"Measurements batch {measurements_batch} is not valid")
    
    if apa_no not in [1, 2, 3, 4]:
        raise ValueError(f"APA number {apa_no} is not valid")
                         
    if measurements_batch == 1:
        if apa_no in [1, 2]:
            return 'apas_12'
        else:
            return 'apas_34'
        
    if measurements_batch in [2, 3]:
        if apa_no == 1:
            return 'apa_1'
        elif apa_no == 2:
            return 'apa_2'
        else:
            return 'apas_34'

def comes_from_channel(
        waveform: Waveform, 
        endpoint, 
        channels) -> bool:

    if waveform.endpoint == endpoint:
        if waveform.channel in channels:
            return True
    return False

def get_analysis_params(run: int = None, apa_no: int = None):

    int_ll = get_starting_tick(run, apa_no)

    input_parameters = IPDict(baseline_limits=get_baseline_limits(apa_no))
    input_parameters['int_ll'] = int_ll
    input_parameters['int_ul'] = int_ll + integ_window
    input_parameters['amp_ll'] = int_ll
    input_parameters['amp_ul'] = int_ll + integ_window

    return input_parameters

def get_starting_tick(run: int = None, apa_no: int = None):
    if apa_no == 1:
        return starting_tick_apa1[run]
    else:
        return starting_tick_apa234

def get_baseline_limits(apa_no: int = None):

    if apa_no == 1:
        return baseline_limits_apa1
    else:
        return baseline_limits_apa234

def read_data(input_folderpath:str='', batch: int = None, apa_no:int=None, stop_fraction: float = 1., is_folder: bool= True):

    fProcessRootNotPickles = True if batch == 1 else False

    if is_folder: 
        if fProcessRootNotPickles:
            new_wfset = WaveformSet_from_root_files(
                "pyroot",
                folderpath=input_folderpath,
                bulk_data_tree_name="raw_waveforms",
                meta_data_tree_name="metadata",
                set_offset_wrt_daq_window=True if apa_no == 1 else False,
                read_full_streaming_data=True if apa_no == 1 else False,
                truncate_wfs_to_minimum=True if apa_no == 1 else False,
                start_fraction=0.0,
                stop_fraction=stop_fraction,
                subsample=1,
            )
        else:
            new_wfset = WaveformSet_from_pickle_files(                
                folderpath=input_folderpath,
                target_extension=".pkl",
                verbose=True,
            )
    else:
        if fProcessRootNotPickles:
            new_wfset = WaveformSet_from_root_file(
                "pyroot",
                filepath=input_folderpath,
                bulk_data_tree_name="raw_waveforms",
                meta_data_tree_name="metadata",
                set_offset_wrt_daq_window=True if apa_no == 1 else False,
                read_full_streaming_data=True if apa_no == 1 else False,
                truncate_wfs_to_minimum=True if apa_no == 1 else False,
                start_fraction=0.0,
                stop_fraction=stop_fraction,
                subsample=1,
            )
        else:
            new_wfset = WaveformSet_from_pickle_file(input_folderpath)

    return new_wfset


def get_wfset_in_channels(wfset: WaveformSet, endpoint: int, channels: list):
        
    new_wfset = WaveformSet.from_filtered_WaveformSet(
        wfset,
        comes_from_channel,
        endpoint,
        channels,
    )

    return new_wfset

def get_gain_and_sn(grid_apa: ChannelWsGrid, excluded_channels: list):

    data = {}

    for i in range(grid_apa.ch_map.rows):
        for j in range(grid_apa.ch_map.columns):

            ep = grid_apa.ch_map.data[i][j].endpoint
            ch = grid_apa.ch_map.data[i][j].channel

            if ep in excluded_channels.keys():
                if ch in excluded_channels[ep]:
                    print(f"Excluding channel {ch} from endpoint {ep}...")
                    continue

            try:
                fit_params = grid_apa.ch_wf_sets[ep][ch].calib_histo.gaussian_fits_parameters
            except KeyError:
                print(f"Endpoint {ep}, channel {ch} not found in data. Continuing...")
                continue
 
            # this is to avoid a problem the first time ep is used
            try:
                aux = data[ep]
            except KeyError:
                data[ep] = {}
                aux = data[ep]

            # compute the gain
            try:
                aux_gain = fit_params['mean'][1][0] - fit_params['mean'][0][0]
            except IndexError:
                print(f"Endpoint {ep}, channel {ch} not found in data. Continuing...")
                continue
            
            # this is to avoid a problem the first time ch is used
            try:
                aux_2 = aux[ch]
            except KeyError:
                aux[ch] = {}
                aux_2 = aux[ch]

            aux_2['gain'] = aux_gain

            # compute the signal to noise ratio
            aux_2['snr'] = aux_gain/np.sqrt( fit_params['std'][0][0]**2 + fit_params['std'][1][0]**2 )

    return data

def get_nbins_for_charge_histo(pde: float = None, apa_no: int = None):

    # Number of bins for the histogram
    bins_number = 125 # [90 - 125]

    if apa_no in [2, 3, 4]:
        if pde == 0.4:
            bins_number = 125
        elif pde == 0.45:
            bins_number = 110 # [100-110]
        else:
            bins_number = 90

    return bins_number


def save_data_to_dataframe(data: list, path_to_output_file: str, self):
    
    # Warning: Settings this variable to True will save
    # changes to the output dataframe, potentially introducing
    # spurious data. Only set it to True if you are sure of what
    # you are saving.
    actually_save = True   

    # Do you want to potentially overwrite existing rows of the dataframe?
    overwrite = True

    expected_columns = {
        "APA": [],
        "endpoint": [],
        "channel": [],
        "channel_iterator": [],
        "PDE": [],
        "gain": [],
        "snr": [],
        "OV#": [],
        "HPK_OV_V": [],
        "FBK_OV_V": [],
    }

    # If the file does not exist, create it
    if not os.path.exists(path_to_output_file):
        df = pd.DataFrame(expected_columns)

        # Force column-wise types
        df['APA'] = df['APA'].astype(int)
        df['endpoint'] = df['endpoint'].astype(int)
        df['channel'] = df['channel'].astype(int)
        df['channel_iterator'] = df['channel_iterator'].astype(int)
        df['PDE'] = df['PDE'].astype(float)
        df['gain'] = df['gain'].astype(float)
        df['snr'] = df['snr'].astype(float)
        df['OV#'] = df['OV#'].astype(int)
        df['HPK_OV_V'] = df['HPK_OV_V'].astype(float)
        df['FBK_OV_V'] = df['FBK_OV_V'].astype(float)

        df.to_pickle(path_to_output_file)

    df = pd.read_pickle(path_to_output_file)

    if len(df.columns) != len(expected_columns):
        raise Exception(f"The columns of the found dataframe do not match the expected ones. Something went wrong.")

    elif not bool(np.prod(df.columns == pd.Index(data = expected_columns))):
        raise Exception(f"The columns of the found dataframe do not match the expected ones. Something went wrong.")

    else:
        for endpoint in data.keys():
            for channel in data[endpoint]:
                # Assemble the new row
                new_row = {
                    "APA":      [int(self.apa)],
                    "endpoint": [endpoint],
                    "channel":  [channel],
                    "channel_iterator":   [get_channel_iterator(self.apa, endpoint, channel)],
                    "PDE":      [self.pde],
                    "gain":     [data[endpoint][channel]["gain"]],
                    "snr":      [data[endpoint][channel]["snr"]],
                    "OV#":      [ov_no],
                    "HPK_OV_V": [hpk_ov],
                    "FBK_OV_V": [fbk_ov],
                }

                # Check if there is already an entry for the
                # given endpoint and channel for this OV
                matching_rows_indices = df[
                    (df['endpoint'] == endpoint) &       
                    (df['channel'] == channel) &
                    (df['OV#'] == ov_no)].index          

                if len(matching_rows_indices) > 1:
                    raise Exception(f"There are already more than one rows for the given endpoint ({endpoint}), channel ({channel}) and OV# ({ov_no}). Something went wrong.")

                elif len(matching_rows_indices) == 1:
                    if overwrite:

                        row_index = matching_rows_indices[0]

                        new_row = { key : new_row[key][0] for key in new_row.keys() }  

                        if actually_save:
                            df.loc[row_index, :] = new_row

                    else:
                        print(f"Skipping the entry for endpoint {endpoint}, channel {channel} and OV# {ov_no} ...")

                else: # len(matching_rows_indices) == 0
                    if actually_save:
                        df = pd.concat([df, pd.DataFrame(new_row)], axis = 0, ignore_index = True)
                        df.reset_index()
        df.to_pickle(path_to_output_file)