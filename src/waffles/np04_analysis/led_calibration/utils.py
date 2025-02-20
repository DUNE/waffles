import os
import numpy as np
import pandas as pd
import datetime

from waffles.core.utils import build_parameters_dictionary
from waffles.input.pickle_file_reader import WaveformSet_from_pickle_files
from waffles.input.pickle_file_reader import WaveformSet_from_pickle_file
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.IPDict import IPDict
from waffles.np04_utils.utils import get_channel_iterator

input_parameters = build_parameters_dictionary('params.yml')

def get_input_filepath(
        base_folderpath: str,
        run: int
    ) -> str:
    return  f"{base_folderpath}/wvf_{run}.pkl"


def comes_from_channel(
        waveform: Waveform, 
        endpoint, 
        channels
    ) -> bool:

    if waveform.endpoint == endpoint:
        if waveform.channel in channels:
            return True
    return False

def get_analysis_params(
        apa_no: int,
        run: int = None
    ):

    if apa_no == 1:
        if run is None:
            raise Exception(
                "In get_analysis_params(): A run number "
                "must be specified for APA 1"
            )
        else:
            int_ll = input_parameters['starting_tick'][1][run]
    else:
        int_ll = input_parameters['starting_tick'][apa_no]

    analysis_input_parameters = IPDict(
        baseline_limits=\
            input_parameters['baseline_limits'][apa_no]
>>>>>>> main
    )
    analysis_input_parameters['int_ll'] = int_ll
    analysis_input_parameters['int_ul'] = \
        int_ll + input_parameters['integ_window']
    analysis_input_parameters['amp_ll'] = int_ll
    analysis_input_parameters['amp_ul'] = \
        int_ll + input_parameters['integ_window']

    return analysis_input_parameters

def read_data(
        input_path: str,
        batch: int,
        apa_no: int,
        is_folder: bool = True,
        stop_fraction: float = 1.
    ):

    fProcessRootNotPickles = True if batch == 1 else False

    if is_folder: 
        if fProcessRootNotPickles:
            new_wfset = WaveformSet_from_root_files(
                "pyroot",
                folderpath=input_path,
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
                folderpath=input_path,
                target_extension=".pkl",
                verbose=True,
            )
    else:
            new_wfset = WaveformSet_from_pickle_file(input_path)

    return new_wfset

def get_gain_and_snr(
        grid_apa: ChannelWsGrid,
        excluded_channels: list
    ):

    data = {}

    for i in range(grid_apa.ch_map.rows):
        for j in range(grid_apa.ch_map.columns):

            endpoint = grid_apa.ch_map.data[i][j].endpoint
            channel = grid_apa.ch_map.data[i][j].channel

            if endpoint in excluded_channels.keys():
                if channel in excluded_channels[endpoint]:
                    print(f"    - Excluding channel {channel} from endpoint {endpoint}...")
                    continue

            try:
                fit_params = grid_apa.ch_wf_sets[endpoint][channel].calib_histo.gaussian_fits_parameters
            except KeyError:
                print(f"Endpoint {endpoint}, channel {channel} not found in data. Continuing...")
                continue
 
            # Handle a KeyError the first time we access a certain endpoint
            try:
                aux = data[endpoint]
            except KeyError:
                data[endpoint] = {}
                aux = data[endpoint]

            # compute the gain
            try:
                aux_gain = fit_params['mean'][1][0] - fit_params['mean'][0][0]
            except IndexError:
                print(f"Endpoint {endpoint}, channel {channel} not found in data. Continuing...")
                continue
            
            # this is to avoid a problem the first time ch is used
            try:
                aux_2 = aux[channel]
            except KeyError:
                aux[channel] = {}
                aux_2 = aux[channel]

            aux_2['gain'] = aux_gain

            # compute the signal to noise ratio
            aux_2['snr'] = aux_gain/np.sqrt(fit_params['std'][0][0]**2 + fit_params['std'][1][0]**2)

    return data

def get_nbins_for_charge_histo(
        pde: float,
        apa_no: int
    ):

    if apa_no in [2, 3, 4]:
        if pde == 0.4:
            bins_number = 125
        elif pde == 0.45:
            bins_number = 110 # [100-110]
        else:
            bins_number = 90
    else:
        # It is still required to
        # do this tuning for APA 1
        bins_number = 125

    return bins_number

def save_data_to_dataframe(
    Analysis1_object,
    data: list,
    path_to_output_file: str
):
    
    hpk_ov = input_parameters['hpk_ov'][Analysis1_object.pde]
    fbk_ov = input_parameters['fbk_ov'][Analysis1_object.pde]
    ov_no = input_parameters['ov_no'][Analysis1_object.pde]
    
    # Warning: Settings this variable to True will save
    # changes to the output dataframe, potentially introducing
    # spurious data. Only set it to True if you are sure of what
    # you are saving.
    actually_save = True   

    # Do you want to potentially overwrite existing rows of the dataframe?
    overwrite = False

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
        raise Exception(
            f"The columns of the found dataframe do not "
            "match the expected ones. Something went wrong.")

    elif not bool(np.prod(df.columns == pd.Index(data = expected_columns))):
        raise Exception(
            f"The columns of the found dataframe do not "
            "match the expected ones. Something went wrong.")

    else:
        for endpoint in data.keys():
            for channel in data[endpoint]:
                # Assemble the new row
                new_row = {
                    "APA": [int(Analysis1_object.apa)],
<<<<<<< HEAD
                    "APA": [int(Analysis1_object.apa)],
                    "endpoint": [endpoint],
                    "channel": [channel],
                    "channel_iterator": [get_channel_iterator(
                        Analysis1_object.apa,
                        endpoint,
                        channel
                    )],
                    "PDE": [Analysis1_object.pde],
=======
                    "APA": [int(Analysis1_object.params.apa)],
                    "endpoint": [endpoint],
                    "channel": [channel],
                    "channel_iterator": [get_channel_iterator(
                        Analysis1_object.apa,
                        endpoint,
                        channel
                    )],
                    "PDE": [Analysis1_object.params.pde],
>>>>>>> main
                    "gain": [data[endpoint][channel]["gain"]],
                    "snr": [data[endpoint][channel]["snr"]],
                    "OV#": [ov_no],
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
                    raise Exception(
                        f"There are already more than one rows "
                        f"for the given endpoint ({endpoint}), "
                        f"channel ({channel}) and OV# ({ov_no})"
                        ". Something went wrong."
                    )

                elif len(matching_rows_indices) == 1:
                    if overwrite:

                        row_index = matching_rows_indices[0]

                        new_row = { key : new_row[key][0] for key in new_row.keys() }  

                        if actually_save:
                            df.loc[row_index, :] = new_row

                    else:
                        print(
                            f"Skipping the entry for endpoint "
                            f"{endpoint}, channel {channel} and"
                            f" OV# {ov_no} ...")

                else: # len(matching_rows_indices) == 0
                    if actually_save:
                        df = pd.concat([df, pd.DataFrame(new_row)], axis = 0, ignore_index = True)
                        df.reset_index()

<<<<<<< HEAD
        df.to_pickle(path_to_output_file)

def compute_timestamp(day, month, year):
    """This function generates a timestamp from a date. 
    The date is given as three integer values which match 
    the day, the month and the year, respectively. The 
    timestamp is the number of seconds since 
    1970-01-01 00:00:00 UTC.
    
    """
    dt = datetime.datetime(year, month, day)
    
    return dt.timestamp()


def prepare_data_time(
    apas: list,
    pdes: list,
    batches: list,
    variable: str,
    general_df: pd.DataFrame
):
    
    data = {}

    for k in range(len(apas)):
        
        apa_no=apas[k]
        data[apa_no] = {}
        
        for i in range(len(pdes)):
            
            current_df = general_df[
                (general_df['APA'] == apa_no) &
                (general_df['PDE'] == pdes[i])]
            
            data[apa_no][pdes[i]] = {}

            possible_channel_iterators = current_df['channel_iterator'].unique()
            
            for channel_iterator in possible_channel_iterators:
                
                aux = current_df[current_df['channel_iterator'] == channel_iterator]
                time_ordered_values_of_variable = []

                # Here's why the data is ordered by batch number, i.e. ordered by time
                for batch_no in batches:

                    aux2 = aux[aux['batch_no'] == batch_no]
                    if len(aux2) == 0:
                        print(f"Warning: Found no entry for APA {apa_no}, PDE {pdes[i]}, batch {batch_no} and channel iterator {channel_iterator}.")
                    elif len(aux2) == 1:
                        time_ordered_values_of_variable.append(
                            aux2[variable].values[0])
                    else:
                        raise Exception(f"Found more than one entry for APA {apa_no}, PDE {pdes[i]}, batch {batch_no} and channel iterator {channel_iterator}.")
                        
                data[apa_no][pdes[i]][channel_iterator] = time_ordered_values_of_variable
    
    return data
=======
        df.to_pickle(path_to_output_file)

def compute_timestamp(day, month, year):
    """This function generates a timestamp from a date. 
    The date is given as three integer values which match 
    the day, the month and the year, respectively. The 
    timestamp is the number of seconds since 
    1970-01-01 00:00:00 UTC.
    
    """
    dt = datetime.datetime(year, month, day)
    
    return dt.timestamp()


def prepare_data_time(
    apas: list,
    pdes: list,
    batches: list,
    variable: str,
    general_df: pd.DataFrame
):
    
    data = {}

    for k in range(len(apas)):
        
        apa_no=apas[k]
        data[apa_no] = {}
        
        for i in range(len(pdes)):
            
            current_df = general_df[
                (general_df['APA'] == apa_no) &
                (general_df['PDE'] == pdes[i])]
            
            data[apa_no][pdes[i]] = {}

            possible_channel_iterators = current_df['channel_iterator'].unique()
            
            for channel_iterator in possible_channel_iterators:
                
                aux = current_df[current_df['channel_iterator'] == channel_iterator]
                time_ordered_values_of_variable = []

                # Here's why the data is ordered by batch number, i.e. ordered by time
                for batch_no in batches:

                    aux2 = aux[aux['batch_no'] == batch_no]
                    if len(aux2) == 0:
                        print(f"Warning: Found no entry for APA {apa_no}, PDE {pdes[i]}, batch {batch_no} and channel iterator {channel_iterator}.")
                    elif len(aux2) == 1:
                        time_ordered_values_of_variable.append(
                            aux2[variable].values[0])
                    else:
                        raise Exception(f"Found more than one entry for APA {apa_no}, PDE {pdes[i]}, batch {batch_no} and channel iterator {channel_iterator}.")
                        
                data[apa_no][pdes[i]][channel_iterator] = time_ordered_values_of_variable
    
    return data
