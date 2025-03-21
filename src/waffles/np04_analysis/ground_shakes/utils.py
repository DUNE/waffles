import numpy as np
from typing import Union
import warnings 
import plotly.graph_objs as go
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

import numpy as np
from typing import List, Union, Dict

def get_meansigma_per_channel(wfs: list,                
                               ep: Union[int, list] = -1,
                               ch: Union[int, list] = -1,
                               nwfs: int = -1,
                               tmin: int = -1,
                               tmax: int = -1,
                               rec: list = [-1]) -> Dict[int, float]:
    """
    Computes the standard deviation (sigma) of each waveform's ADCs relative to its mean within a specific interval,
    and returns the mean sigma per channel.
    """
    
    print("Fetching selected waveforms...")
    selected_wfs, _ = get_wfs(wfs, ep, ch, nwfs, tmin, tmax, rec)
    print(f"Number of selected waveforms: {len(selected_wfs)}")

    # Dictionary to store sigmas per channel
    channel_sigma = {}  # {channel_id: [sigma1, sigma2, ...]}

    for wf in selected_wfs:
        if not hasattr(wf, "channel"):  
            print("Error: waveform missing 'channel' attribute!")
            continue
        
        mean = np.mean(wf.adcs)  # Compute mean of ADC values
        sigma = np.sqrt(np.sum((wf.adcs - mean) ** 2) / len(wf.adcs))  # Standard deviation formula
        

        if wf.channel not in channel_sigma:
            channel_sigma[wf.channel] = []  # Create list if channel not seen before
        
        channel_sigma[wf.channel].append(sigma)  # Store sigma for this waveform

    # Compute the mean sigma per channel
    mean_sigma = {}
    for ch_idx, sigmas in channel_sigma.items():
        mean_sigma[ch_idx] = np.mean(sigmas) if sigmas else float("nan")  

    print("Final mean sigmas per channel:", mean_sigma)
    return mean_sigma



#-------------- Time offset histograms -----------

def plot_to_function(channel_ws, apa,idx, figure, row, col, nbins):

    # Compute the time offset
    times = [wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp for wf in channel_ws.waveforms]

    if not times:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return

    # Generaate the histogram
    histogram = get_histogram(times, nbins, line_width=0.5)

    # Return the axis titles and figure title along with the figure
    x_axis_title = "Time offset"
    y_axis_title = "Entries"
    figure_title = f"Time offset histograms for APA {apa}"
    
    if figure is None:
        return x_axis_title, y_axis_title, figure_title
    
    # Add the histogram to the corresponding channel
    figure.add_trace(histogram, row=row, col=col)
    
    return figure


# --------------- Sigma vs timestamp  --------------

def plot_sigma_vs_ts_function(channel_ws, apa,idx, figure, row, col,nbins):

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
    
    # Return the axis titles and figure title along with the figure
    x_axis_title = "Timestamp"
    y_axis_title = "Sigma"
    figure_title = f"Sigma vs timestamp for APA {apa}"
    
    if figure is None:
        return x_axis_title, y_axis_title, figure_title
    
    # Add the histogram to the corresponding channel
    figure.add_trace(go.Scatter(
        x=timestamps,
        y=sigmas,
        mode='markers',
        marker=dict(color='black', size=2.5)  
    ), row=row, col=col)
    
    return figure


# --------------- Sigma histograms  --------------
 
def plot_sigma_function(channel_ws, apa, idx, figure, row, col, nbins):
    
    # Compute the sigmas
    
    sigmas = [np.std(wf.adcs) for wf in channel_ws.waveforms]

    if not sigmas:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return None, None, None, None  # Return None if no data
    
        
    # Generate the histogram
    histogram = get_histogram(sigmas, nbins, line_width=0.5)

    # Return the axis titles and figure title along with the figure
    x_axis_title = "Sigma"
    y_axis_title = "Entries"
    figure_title = f"Sigma histograms for APA {apa}"
    
    if figure is None:
        return x_axis_title, y_axis_title, figure_title
    
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

def plot_meanfft_function(channel_ws, apa, idx, figure, row, col, nbins):

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
    
    # Return the axis titles and figure title along with the figure
    x_axis_title = "Frequency [MHz]"
    y_axis_title = "Power [dB]"
    figure_title = f"Superimposed FFT of Selected Waveforms for APA {apa}"
    
    if figure is None:
        return x_axis_title, y_axis_title, figure_title


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
            line=dict(color=colors[i % len(colors)], width=1)
        ), row=row, col=col)  
    
    return figure  