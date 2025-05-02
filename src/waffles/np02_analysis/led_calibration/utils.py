import os
import numpy as np
import pandas as pd
from typing import Union

from waffles.core.utils import build_parameters_dictionary

from waffles.input_output.hdf5_structured import load_structured_waveformset

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.IPDict import IPDict
from waffles.np04_utils.utils import get_channel_iterator

from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map, cat_geometry_map
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle

import plotly.graph_objects as go 
from plotly.subplots import make_subplots

input_parameters = build_parameters_dictionary('params.yml')

def get_input_filepath(
        run: int
    ) -> str:

    return  f"data/run_0{run}.hdf5"


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

def read_data(
        input_path: str,
    ):

    try:
        new_wfset=load_structured_waveformset(input_path)   
    except FileNotFoundError:
        raise FileNotFoundError(f"File {input_path} was not found.")

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
                    "endpoint": [endpoint],
                    "channel": [channel],
                    "channel_iterator": [get_channel_iterator(
                        Analysis1_object.apa,
                        endpoint,
                        channel
                    )],
                    "PDE": [Analysis1_object.pde],
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

        df.to_pickle(path_to_output_file)
        
def get_wfs(wfs: list,                
            ep: Union[int, list] = -1,
            ch: Union[int, list] = -1,
            nwfs: int = -1,
            tmin: int = -1,
            tmax: int = -1,
            rec: list = [-1],
            adc_max_threshold: int = None):  

    if type(ch) == int:
        ch = [ch]

    if type(ep) == int:
        ep = [ep]
        
    waveforms = []
    n = 0
    for wf in wfs:
        t = np.float32(np.int64(wf.timestamp) - np.int64(wf.daq_window_timestamp))
        max_adc = np.max(wf.adcs)

        if (wf.endpoint      in ep  or ep[0] == -1) and \
           (wf.channel       in ch  or ch[0] == -1) and \
           (wf.record_number in rec or rec[0] == -1) and \
           ((t > tmin and t < tmax) or (tmin == -1 and tmax == -1)) and \
           (adc_max_threshold is None or max_adc <= adc_max_threshold):  # Filtro por umbral ADC
            n += 1
            waveforms.append(wf)
        if n >= nwfs and nwfs != -1:
            break

    return waveforms, WaveformSet(*waveforms)

def get_det_id_name(det_id: int):

    if   det_id == 1: det_id_name='nonTCO' 
    elif det_id ==2 : det_id_name= 'TCO'      
        
    return det_id_name

def get_endpoints(det:str, det_id: int):

    eps=[]
    
    if   det == 'Membrane': eps = [107]
    elif det == 'Cathode':  eps = [106] 
    elif det == 'PMTs':     eps = []
            
    # Change according to the NP02 mapping
    
    return eps

def get_grid(wfs: list,                
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
                [30, 17],
                [10, 37],
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
        fig.show()


