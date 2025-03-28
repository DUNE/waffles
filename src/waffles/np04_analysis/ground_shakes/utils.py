import numpy as np
from typing import Union
import warnings 
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map
from waffles.np04_data.ProtoDUNE_HD_APA_maps_APA1_104 import APA_map as APA_map_2
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.WaveformSet import WaveformSet

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.core.utils import build_parameters_dictionary
from waffles.data_classes.IPDict import IPDict

input_parameters = build_parameters_dictionary('params.yml')
    
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
    )
    analysis_input_parameters['int_ll'] = int_ll
    analysis_input_parameters['int_ul'] = \
        int_ll + input_parameters['integ_window']
    analysis_input_parameters['amp_ll'] = int_ll
    analysis_input_parameters['amp_ul'] = \
        int_ll + input_parameters['integ_window']

    return analysis_input_parameters

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

def get_endpoints(apa: int):

    eps=[]

    if    apa == 1: eps =[104,105,107]
    elif  apa == 2: eps =[109]
    elif  apa == 3: eps =[111]
    elif  apa == 4: eps =[112,113]

    return eps

def get_wfs(wfs: list,                
            ep: Union[int, list]=-1,
            ch: Union[int, list]=-1,
            nwfs: int = -1,
            tmin: int = -1,
            tmax: int = -1,
            rec: list = [-1]):

    # plot all waveforms in a given endpoint and channel

    if type(ch) == int:
        ch = [ch]

    if type(ep) == int:
        ep = [ep]
        
    waveforms = []
    n=0
    for wf in wfs:
        t = np.float32(np.int64(wf.timestamp)-np.int64(wf.daq_window_timestamp))
        if (wf.endpoint      in ep  or  ep[0]==-1) and \
           (wf.channel       in ch  or  ch[0]==-1) and \
           (wf.record_number in rec or rec[0]==-1) and \
           ((t > tmin and t< tmax) or (tmin==-1 and tmax==-1)):
            n=n+1
            waveforms.append(wf)
        if n>=nwfs and nwfs!=-1:
            break

    return waveforms, WaveformSet(*waveforms)

def get_wfs_interval(wfs: list,                
            tmin: int = -1,
            tmax: int = -1,
            nwfs: int = -1):
        
    waveforms = []
    n=0
    for wf in wfs:
        t = np.float32(np.int64(wf.timestamp)-np.int64(wf.daq_window_timestamp))
        if ((t > tmin and t< tmax) or (tmin==-1 and tmax==-1)):
            n=n+1
            waveforms.append(wf)
        if n>=nwfs and nwfs!=-1:
            break

    return waveforms

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
        line=dict(
            color=line_color,
            width=line_width,
            shape='hv'
        )
    )
    
    return histogram_trace

def get_grid(wfs: list,                
             apa: int = -1,
             run: int = -1):

    if run < 29927:
        grid_apa = ChannelWsGrid(APA_map[apa], WaveformSet(*wfs))
    else:
        grid_apa = ChannelWsGrid(APA_map_2[apa], WaveformSet(*wfs))        
        
    return grid_apa


def get_min_ticks(wfs, record_numbers):
    """
    Extrae los min_tick de cada waveform para cada record.
    
    Parámetros:
    - wfs: Lista de objetos Waveform.
    - record_numbers: Lista de record_numbers a procesar (opcional, si es None, se extraen de wfs).
    
    Retorna:
    - Un diccionario donde la clave es el record_number y el valor es un numpy array con los min_tick de cada waveform.
    """
    if record_numbers is None:
        record_numbers = set(wf.record for wf in wfs)  # Extraer records únicos automáticamente
    
    min_ticks_by_record = {}
    
    for rec in record_numbers:
        min_ticks = []
        for wf in wfs:
            if wf.record_number == rec:
                offset = np.float32(np.int64(wf.timestamp) - np.int64(wf.daq_window_timestamp))
                mt = np.argmin(wf.adcs)  # Índice del mínimo valor ADC
                
                if wf.adcs[mt] < 10:  # Comprobar saturación
                    min_ticks.append(mt + offset)
                else:
                    min_ticks.append(None)  # Valor cuando hay saturación
        
        if min_ticks:  # Solo guardar si hay datos para el record_number
            min_ticks_by_record[rec] = min_ticks  # Cambié a lista en lugar de np.array
    
    return min_ticks_by_record

def get_std_min_ticks(min_ticks_by_record):
    """
    Calcula la desviación estándar de los min_ticks para cada record.
    
    Retorna:
    - Un diccionario donde la clave es el record_number y el valor es la std de los min_tick.
    """
    std_by_record = {}
    
    for rec, min_ticks in min_ticks_by_record.items():
        # Filtrar los valores None
        valid_min_ticks = [tick for tick in min_ticks if tick is not None]
        
        if len(valid_min_ticks) > 1:  # Evitar cálculos inválidos
            std_by_record[rec] = np.std(valid_min_ticks)
        else:
            std_by_record[rec] = np.nan  # Si solo hay un dato, no se puede calcular std
    
    return std_by_record

# ------------ Plot a waveform ---------------

def plot_wf( waveform_adcs : WaveformAdcs,  
             figure,
             row,
             col,
             offset: bool = False,
             ) -> None:

    """
    Plot a single waveform
    """
    
    x0 = np.arange(  len(waveform_adcs.adcs),
                     dtype = np.float32)
    y0 = waveform_adcs.adcs

    if offset:        
        dt = np.float32(np.int64(waveform_adcs.timestamp)-np.int64(waveform_adcs.daq_window_timestamp))
    else:
        dt = 0

    wf_trace = go.Scatter(x = x0 + dt,   
                           y = y0,
                           mode = 'lines',
                           line=dict(width=0.5)
                           )
 

    figure.add_trace(wf_trace,row,col)

# ------------- Plot a set of waveforms ----------

def plot_wfs(channel_ws,  
             figure, 
             row, 
             col,           
             offset: bool = False,
             ):

    """
    Plot a list of waveforms
    """    

    # plot nwfs waveforms
    n=0        
    for wf in channel_ws.waveforms:
        n=n+1
        # plot the single waveform
        plot_wf(wf,figure, row, col, offset)

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
    
    
    '''
    waveform_sets = {
        "[-1000, -500]": get_wfs_interval(channel_ws.waveforms, -1000, -500),
        "[-450, -300]": get_wfs_interval(channel_ws.waveforms, -450, -300),
        "[0, 300]": get_wfs_interval(channel_ws.waveforms,0, 300),
        "[600, 1000]": get_wfs_interval(channel_ws.waveforms, 600, 1000),
        "[2000, 5000]": get_wfs_interval(channel_ws.waveforms, 2000, 5000)
    }
    
    np.seterr(divide='ignore')  
    
    # Different colors for each range
    colors = ['blue', 'red', 'green', 'purple', 'orange']  
    '''
    waveform_sets= {"All":get_wfs_interval(channel_ws.waveforms,-1, -1)}
    
    for i, (label, selected_wfs) in enumerate(waveform_sets.items()):
        if not selected_wfs:
            print(f"No waveforms found for range {label}")
            continue

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
            name=f"FFT {label}",
            line=dict(color='black', width=1),
        ), row=row, col=col)  
    
    return figure  


