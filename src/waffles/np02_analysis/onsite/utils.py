import numpy as np
from typing import Union, Dict
import warnings 
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map, cat_geometry_map
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.WaveformSet import WaveformSet

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.core.utils import build_parameters_dictionary
from waffles.data_classes.IPDict import IPDict

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
             det: str,
             det_id: list):

    if det == 'Membrane':
        grid = ChannelWsGrid(mem_geometry_map[det_id], WaveformSet(*wfs))
    elif det == 'Cathode':
        grid = ChannelWsGrid(cat_geometry_map[det_id], WaveformSet(*wfs))  
    elif det == 'PMTs':
        grid = None      
        
    return grid

def get_det_id_name(det_id: int):

    if   det_id == 1: det_id_name='nonTCO' 
    elif det_id ==2 : det_id_name= 'TCO'      
        
    return det_id_name

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
             nwfs: int = -1,
             xmin: int = -1,
             xmax: int = -1,
             tmin: int = -1,
             tmax: int = -1,
             offset: bool = False,
             ):

    """
    Plot a list of waveforms
    """
    
    # don't consider time intervals that will not appear in the plot
    if tmin == -1 and tmax == -1:
        tmin=xmin-1024    # harcoded number
        tmax=xmax        

    # plot nwfs waveforms
    n=0        
    for wf in channel_ws.waveforms:
        n=n+1
        # plot the single waveform
        plot_wf(wf,figure, row, col, offset)
        if n>=nwfs and nwfs!=-1:
            break

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