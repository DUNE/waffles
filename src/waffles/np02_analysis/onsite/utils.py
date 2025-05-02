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

from plotly.subplots import make_subplots

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
           (adc_max_threshold is None or max_adc <= adc_max_threshold):
            
            n += 1
            waveforms.append(wf)  # Guardamos el waveform junto con su baseline

        if n >= nwfs and nwfs != -1:
            break


    return waveforms, WaveformSet(*waveforms)


def baseline_cut(wfs: list):
    """
    Filtra waveforms cuyo baseline_rms está dentro de un rango alrededor de la media.
    baseline_tolerance: fracción (por ejemplo, 0.1 para ±10%)
    """
    
    baseline_rms = [wf.analyses['baseline_computation'].result['baseline_rms'] for wf in wfs]
    mean_rms = np.mean(baseline_rms)
    #print('mean_rms:', mean_rms)

    selected_wfs = [
        wf for wf in wfs
        if wf.analyses['baseline_computation'].result['baseline_rms'] < mean_rms
    ]

    return selected_wfs, WaveformSet(*selected_wfs)

def get_gain_and_snr(
        grid_apa: ChannelWsGrid
    ):

    data = {}

    for i in range(grid_apa.ch_map.rows):
        for j in range(grid_apa.ch_map.columns):

            endpoint = grid_apa.ch_map.data[i][j].endpoint
            channel = grid_apa.ch_map.data[i][j].channel

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


def get_det_id_name(det_id: int):

    if   det_id == 1: det_id_name='nonTCO' 
    elif det_id ==2 : det_id_name= 'TCO'      
        
    return det_id_name

# ------------ Plot a waveform ---------------

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
        line = dict(width=0.5)
    )

    figure.add_trace(wf_trace, row, col)

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
             baseline: bool = True,
             ) -> None:
    """
    Plot a list of waveforms. If baseline=True, subtract baseline from each waveform.
    """

    if tmin == -1 and tmax == -1:
        tmin = xmin - 1024
        tmax = xmax        

    n = 0        
    for i, wf in enumerate(channel_ws.waveforms):
        n += 1
        if baseline:
            bl = wf.analyses['baseline_computation'].result['baseline']
        else:
            bl = None
        plot_wf(wf, figure, row, col, baseline=bl, offset=offset)
        if n >= nwfs and nwfs != -1:
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

def plot_avg_waveform_with_peak_and_intervals(channel_ws, figure, row, col, intervals):
    import plotly.graph_objects as go
    import numpy as np

    all_wfs = np.array([wf.adcs for wf in channel_ws.waveforms])
    avg_waveform = np.mean(all_wfs, axis=0)
    peak_index = np.argmax(avg_waveform)
    peak_value = avg_waveform[peak_index]

    # Mostrar la waveform promedio
    figure.add_trace(go.Scatter(
        x=np.arange(len(avg_waveform)),
        y=avg_waveform,
        mode='lines',
        name=f"Avg WF - Ch {channel_ws.channel}",
        line=dict(color='blue')
    ), row=row, col=col)

    # Punto rojo en el pico
    figure.add_trace(go.Scatter(
        x=[peak_index],
        y=[peak_value],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Peak'
    ), row=row, col=col)

    # Zonas sombreadas de carga y baseline
    for interval in intervals:
        left = int(interval * 0.3)
        right = interval - left

        # Intervalo de integral (zona verde)
        start_sig = max(0, peak_index - left)
        end_sig = min(len(avg_waveform), peak_index + right)

        figure.add_shape(
            type="rect",
            x0=start_sig,
            x1=end_sig,
            y0=min(avg_waveform),
            y1=max(avg_waveform),
            fillcolor="rgba(0,255,0,0.15)",  # verde
            line=dict(width=0),
            layer="below",
            row=row,
            col=col
        )

        # Intervalo de baseline (zona naranja): primeros 50 puntos
        base_start = 0
        base_end = min(50, len(avg_waveform))

        figure.add_shape(
            type="rect",
            x0=base_start,
            x1=base_end,
            y0=min(avg_waveform),
            y1=max(avg_waveform),
            fillcolor="rgba(255,165,0,0.2)",  # naranja
            line=dict(width=0),
            layer="below",
            row=row,
            col=col
        )

    return figure

def plot_sigma_to_noise_vs_interval_histfit(channel_ws, figure, row, col, intervals, baseline_window=50, bins=40):
    import numpy as np
    import plotly.graph_objects as go
    from scipy.optimize import curve_fit
    from scipy.stats import norm

    def gaussian(x, a, mu, sigma):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    all_wfs = np.array([wf.adcs for wf in channel_ws.waveforms])
    wf_len = all_wfs.shape[1]
    avg_waveform = np.mean(all_wfs, axis=0)
    peak_index = np.argmax(avg_waveform)

    interval_labels = []
    snr_list = []

    for interval in intervals:
        left = int(interval * 0.3)
        right = interval - left

        sig_start = max(0, peak_index - left)
        sig_end = min(wf_len, peak_index + right)

        base_start = 0
        base_end = min(baseline_window, wf_len)

        signal_charges = np.sum(all_wfs[:, sig_start:sig_end], axis=1)
        baseline_charges = np.sum(all_wfs[:, base_start:base_end], axis=1)

        # Ajuste gaussiano a histogramas
        hist_s, bins_s = np.histogram(signal_charges, bins=bins, density=True)
        bin_centers_s = 0.5 * (bins_s[1:] + bins_s[:-1])
        popt_s, _ = curve_fit(gaussian, bin_centers_s, hist_s, p0=[1, np.mean(signal_charges), np.std(signal_charges)])

        hist_b, bins_b = np.histogram(baseline_charges, bins=bins, density=True)
        bin_centers_b = 0.5 * (bins_b[1:] + bins_b[:-1])
        popt_b, _ = curve_fit(gaussian, bin_centers_b, hist_b, p0=[1, np.mean(baseline_charges), np.std(baseline_charges)])

        mu_signal, sigma_signal = popt_s[1], popt_s[2]
        mu_baseline, sigma_baseline = popt_b[1], popt_b[2]

        snr = (mu_signal - mu_baseline) / sigma_baseline if sigma_baseline != 0 else 0
        snr_list.append(snr)
        interval_labels.append(f"[-{left},+{right}]")

    figure.add_trace(go.Scatter(
        x=interval_labels,
        y=snr_list,
        mode='lines+markers',
        name=f"SNR Fit - Ch {channel_ws.channel}",
        line=dict(width=2)
    ), row=row, col=col)

    return figure

def plot_sigma_to_noise_vs_interval(channel_ws, figure, row, col, intervals):
    import numpy as np
    import plotly.graph_objects as go

    all_wfs = np.array([wf.adcs for wf in channel_ws.waveforms])
    avg_waveform = np.mean(all_wfs, axis=0)
    peak_index = np.argmax(avg_waveform)
    wf_len = all_wfs.shape[1]

    sigma_to_noise_ratios = []
    interval_labels = []

    for interval in intervals:
        left = int(interval * 0.3)
        right = interval - left

        sig_start = max(0, peak_index - left)
        sig_end = min(wf_len, peak_index + right)

        base_start = 0
        base_end = min(50, sig_start)

        signal_charges = np.sum(all_wfs[:, sig_start:sig_end], axis=1)
        baseline_charges = np.sum(all_wfs[:, base_start:base_end], axis=1)

        mu_signal = np.mean(signal_charges)
        mu_baseline = np.mean(baseline_charges)
        sigma_baseline = np.std(baseline_charges)

        snr = (mu_signal - mu_baseline) / sigma_baseline if sigma_baseline != 0 else 0
        sigma_to_noise_ratios.append(snr)
        interval_labels.append(f"[-{left},+{right}]")

    figure.add_trace(go.Scatter(
        x=interval_labels,
        y=sigma_to_noise_ratios,
        mode='lines+markers',
        marker=dict(size=6),
        line=dict(width=2),
        name=f"SNR - Ch {channel_ws.channel}"
    ), row=row, col=col)

    return figure
