import numpy as np
from plotly import graph_objects as go
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.np02_utils.AutoMap import dict_uniqch_to_module

import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from tqdm import tqdm
from waffles.data_classes.WaveformSet import WaveformSet

from typing import Optional
import os

from scipy.stats import skewnorm
from waffles.utils.numerical_utils import skewed_gaussian


def plot_averages_w_peaks_rise_fall(peaks_all, fig:go.Figure, g:ChannelWsGrid, x_range=None, rise_fall:bool = True):

    """
    Plot average waveforms for each channel in a ChannelWsGrid with peak and rise/fall indicators.

    For each valid channel in the grid, this function:
      - Plots the averaged waveform in the corresponding subplot.
      - Optionally restricts the x-axis to a specified range.
      - Optionally adds vertical dashed lines at key points: t10 and t90 of rise, t90 and t10 of fall.
      - Annotates each subplot with channel info, peak amplitude, and optionally rise/fall times.

    Parameters
    ----------
    peaks_all : dict
        Dictionary returned by `compute_peaks_rise_fall_ch`. Keys are (endpoint, channel),
        values contain peak, rise/fall times, and averaged waveform data.
    fig : go.Figure
        Plotly Figure object containing the subplots where waveforms will be plotted.
    g : ChannelWsGrid
        Grid object containing the channel waveform sets.
    x_range : tuple of two floats, optional
        (min, max) time range to display on the x-axis for all subplots. 
        If None, the full waveform is shown.
    rise_fall : bool, True by default.
        If True the rise and fall time dashed lines are shown and the values are added in the legend

    Returns
    -------
    go.Figure
        The updated Plotly Figure with all average waveforms plotted, vertical lines for 
        rise/fall times, and annotations with peak and timing information.
    """

    ncols = len(g.ch_map.data[0])  
    max_row, max_col = 0, 0

    fig.data = []
    fig.layout.annotations = []

    for (row, col), uch in np.ndenumerate(g.ch_map.data):
        row += 1
        col += 1
        max_row = max(max_row, row)
        max_col = max(max_col, col)

        subplot_idx = (row - 1) * ncols + col
        
        if str(uch) not in dict_uniqch_to_module:
            continue
        if uch.channel not in g.ch_wf_sets[uch.endpoint]:
            continue

        vals = peaks_all[(uch.endpoint, uch.channel)]
        time = vals["time"]
        avg = vals["avg"]
        peak_time = vals["peak_time"]
        peak_value = vals["peak_value"]

        if x_range is not None:
            x_min, x_max = x_range
            mask = (time >= x_min) & (time <= x_max)
            time = time[mask]
            avg = avg[mask]
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=avg,
                mode="lines",
                name=f"{uch.endpoint}-{uch.channel}"
            ),
            row=row, col=col
        )

        if rise_fall:

            for t, color, label in [
                (vals["t_low"], "green", "t10 rise"),
                (vals["t_high"], "blue", "t90 rise"),
                (vals["t_high_fall"], "orange", "t90 fall"),
                (vals["t_low_fall"], "purple", "t10 fall"),
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=[t, t],
                        y=[0, peak_value], 
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                        showlegend=False,  
                    ),
                    row=row, col=col
                ) 
                    
        if subplot_idx == 1:
            xref = "x domain"
            yref = "y domain"
        else:
            xref = f"x{subplot_idx} domain"
            yref = f"y{subplot_idx} domain"

        key = f"{uch.endpoint}-{uch.channel}"
        module_name = dict_uniqch_to_module.get(key, None)

        if rise_fall:
            fig.add_annotation(
                x=0.98,
                y=0.95,
                xref=xref,
                yref=yref,
                text=(
                    f"{module_name}<br>"
                    f"Peak = {peak_value:.1f} ADC<br>"
                    f"Rise time = {vals['rise_time']:.0f} ticks<br>"
                    f"Fall time = {vals['fall_time']:.0f} ticks"
                ),
                showarrow=False,
                align="left",
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="black",
                borderwidth=1
            )

        else:
            fig.add_annotation(
                x=0.98,
                y=0.95,
                xref=xref,
                yref=yref,
                text=(
                    f"{module_name}<br>"
                    f"Peak = {peak_value:.1f} ADC<br>"
                ),
                showarrow=False,
                align="left",
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="black",
                borderwidth=1
            )

        if x_range is not None:
            fig.update_xaxes(range=[x_min, x_max])

    for row in range(1, max_row + 1):
        fig.update_yaxes(title_text="Amplitude [ADC]", row=row, col=1)

    for col in range(1, max_col + 1):
        fig.update_xaxes(title_text="Time [ticks]", row=max_row, col=col)
        
    return fig


def compute_mpv_waveforms(wfch: WaveformSet, analysis_label="std", specific_tick=None):
    """
    Compute MPV at each tick for a channel.
    Possibility to specify a single tick.
    """
    all_adc_values = []
    for wf in wfch.waveforms:
        adcs_float = wf.adcs.astype(np.float32)
        if analysis_label in wf.analyses:
            baseline = wf.analyses[analysis_label].result["baseline"]
            adcs_float = adcs_float - baseline
        else:
            continue
        all_adc_values.append(adcs_float)  
    
    all_adc_values = np.array(all_adc_values)

    # Handle specific_tick case
    if specific_tick is not None:
        adc_values = all_adc_values[:, specific_tick]
        
        if len(adc_values) == 0:
            return {
                'adc_values': np.array([]),
                'minb': 0,
                'maxb': 0,
                'nbins': 50,
                'mpv': np.nan,
                'fit_success': False
            }

        try:
            minb, maxb = np.quantile(adc_values, [0.01, 0.99])
            adc_values_filtered = adc_values[(adc_values > minb) & (adc_values < maxb)]

            nbins = min(int(np.sqrt(len(adc_values_filtered))), 75)
            bins = np.linspace(minb, maxb, nbins + 1)
            counts, bins_edges = np.histogram(adc_values_filtered, bins=bins)
            binscenter = (bins_edges[1:] + bins_edges[:-1]) * 0.5

            A_guess = np.max(counts)
            mu_guess = binscenter[np.argmax(counts)]
            sigma_guess = np.std(adc_values_filtered)
            alpha_guess = stats.skew(adc_values_filtered)
            p0 = [A_guess, mu_guess, sigma_guess, alpha_guess]

            popt, _ = curve_fit(skewed_gaussian, binscenter, counts, p0=p0, maxfev=5000)
            A, mu, sigma, alpha = popt

            # Create smooth curve for plotting
            x_range = np.linspace(minb, maxb, 1000)
            fit_y = skewed_gaussian(x_range, A, mu, sigma, alpha)
            mpv = x_range[np.argmax(fit_y)]
            
            return {
                'adc_values': adc_values_filtered,
                'minb': minb,
                'maxb': maxb,
                'nbins': nbins,
                'x_range': x_range,
                'fit_y': fit_y,
                'mpv': mpv,
                'fit_success': True
            }

        except Exception as e:
            print(f"Warning: Fit failed at tick {specific_tick}: {e}, using median")
            return {
                'adc_values': adc_values_filtered if 'adc_values_filtered' in locals() else adc_values,
                'minb': minb if 'minb' in locals() else adc_values.min(),
                'maxb': maxb if 'maxb' in locals() else adc_values.max(),
                'nbins': nbins if 'nbins' in locals() else 50,
                'mpv': np.median(adc_values),
                'fit_success': False
            }  

    # Normal case: compute for all ticks
    n_ticks = len(wfch.waveforms[0].adcs)
    mpv_waveform = np.zeros(n_ticks)
    print(f"Computing MPV for {n_ticks} ticks from {len(wfch.waveforms)} waveforms...")
    
    for tick in tqdm(range(n_ticks), desc="Processing ticks"):
        adc_values = all_adc_values[:, tick]
        if len(adc_values) == 0:
            mpv_waveform[tick] = np.nan
            continue

        try:
            minb, maxb = np.quantile(adc_values, [0.01, 0.99])
            adc_values = adc_values[(adc_values > minb) & (adc_values < maxb)]

            nbins = min(int(np.sqrt(len(adc_values))), 75)
            bins = np.linspace(minb, maxb, nbins + 1)
            counts, bins_edges = np.histogram(adc_values, bins=bins)
            binscenter = (bins_edges[1:] + bins_edges[:-1]) * 0.5

            A_guess = np.max(counts)
            mu_guess = binscenter[np.argmax(counts)]
            sigma_guess = np.std(adc_values)
            alpha_guess = stats.skew(adc_values)
            p0 = [A_guess, mu_guess, sigma_guess, alpha_guess]

            popt, _ = curve_fit(skewed_gaussian, binscenter, counts, p0=p0, maxfev=5000)
            A, mu, sigma, alpha = popt

            fit_y = skewed_gaussian(binscenter, A, mu, sigma, alpha)
            mpv = binscenter[np.argmax(fit_y)]
            mpv_waveform[tick] = mpv

        except Exception as e:
            print(f"Warning: Fit failed at tick {tick}: {e}, using median")
            mpv = np.median(adc_values)
            mpv_waveform[tick] = mpv
    
    return mpv_waveform  

def plot_tick_distributions_with_fit(fig:go.Figure, g:ChannelWsGrid, tick_index):
    max_row, max_col = 0, 0
    ncols = len(g.ch_map.data[0]) 
    
    for (row, col), uch in np.ndenumerate(g.ch_map.data):
        row += 1
        col += 1
        max_row = max(max_row, row)
        max_col = max(max_col, col)
        subplot_idx = (row - 1) * ncols + col
        
        if str(uch) not in dict_uniqch_to_module:
            continue
        if uch.channel not in g.ch_wf_sets[uch.endpoint]:
            continue
        
        wfch = g.ch_wf_sets[uch.endpoint][uch.channel]
        
        # Get fit info from compute_mpv_waveforms
        fit_info = compute_mpv_waveforms(wfch, specific_tick=tick_index)
        
        key = f"{uch.endpoint}-{uch.channel}"
        module_name = dict_uniqch_to_module.get(key, None)
        
        if module_name is None:
            print(f"No module mapping found for endpoint {uch.endpoint}, channel {uch.channel}")
            continue

        # Plot histogram
        fig.add_trace(
            go.Histogram(
                x=fit_info['adc_values'],
                #nbinsx=fit_info['nbins'],
                xbins=dict(
                    start=fit_info['minb'], 
                    end=fit_info['maxb'], 
                    size=(fit_info['maxb'] - fit_info['minb']) / fit_info['nbins']
                ),
                name='Data',
                showlegend=False,
                opacity=0.6
            ),
            row=row, col=col
        )

        # Plot fit curve if successful
        if fit_info['fit_success']:
            fig.add_trace(
                go.Scatter(
                    x=fit_info['x_range'],
                    y=fit_info['fit_y'],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Fit',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Add MPV line
        fig.add_vline(
            x=fit_info['mpv'],
            line=dict(color='green', dash='dash', width=2),
            row=row, col=col
        )

        # Add annotation
        stats_text = (f"{module_name}<br>"
                      f"MPV: {fit_info['mpv']:.2f}")
        
        if subplot_idx == 1:
            xref = "x domain"
            yref = "y domain"
        else:
            xref = f"x{subplot_idx} domain"
            yref = f"y{subplot_idx} domain"

        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref=xref,
            yref=yref,
            text=stats_text,
            showarrow=False,
            align="left",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.7)"
        )

    for row in range(1, max_row + 1):
        fig.update_yaxes(title_text="Count", row=row, col=1)
    for col in range(1, max_col + 1):
        fig.update_xaxes(title_text="ADC", row=max_row, col=col)
        
        
def plot_mpv_waveforms(fig:go.Figure, g:ChannelWsGrid):
    """
    Plot MPV waveform for each channel in a grid
    """
    max_row, max_col = 0, 0
    
    for (row, col), uch in np.ndenumerate(g.ch_map.data):
        row += 1
        col += 1
        max_row = max(max_row, row)
        max_col = max(max_col, col)
        
        if str(uch) not in dict_uniqch_to_module:
            continue
        if uch.channel not in g.ch_wf_sets[uch.endpoint]:
            continue
        
        print(f"Processing channel {uch.channel}...")
        wfch = g.ch_wf_sets[uch.endpoint][uch.channel] 

        mpv_waveform = compute_mpv_waveforms(wfch)

        time = np.arange(len(mpv_waveform))

        fig.add_trace(
            go.Scatter(
                x=time,
                y=mpv_waveform,
                mode="lines",
                showlegend=False
            ),
            row=row, col=col
        )
   
    for row in range(1, max_row + 1):
        fig.update_yaxes(title_text="MPV [ADC]", row=row, col=1)
    for col in range(1, max_col + 1):
        fig.update_xaxes(title_text="Time [ticks]", row=max_row, col=col)        

def plot_mpv_waveforms_normalized(fig:go.Figure, g:ChannelWsGrid, calibration_data:dict, 
                                   n_ticks=1024, save: bool = False, save_dir: Optional[str] = None):
    """
    Plot normalized MPV waveforms for each channel in a ChannelWsGrid.
    
    """
    if save:
        if save_dir is None:
            raise ValueError("save_dir must be specified")
        os.makedirs(save_dir, exist_ok=True)
    
    max_row, max_col = 0, 0
    ncols = len(g.ch_map.data[0])
    
    for (row, col), uch in np.ndenumerate(g.ch_map.data):
        row += 1
        col += 1
        max_row = max(max_row, row)
        max_col = max(max_col, col)
        subplot_idx = (row - 1) * ncols + col
        
        if str(uch) not in dict_uniqch_to_module:
            continue
        if uch.channel not in g.ch_wf_sets[uch.endpoint]:
            continue
        
        print(f"Processing channel {uch.channel}...")
        wfch = g.ch_wf_sets[uch.endpoint][uch.channel]
        
        mpv_waveform = compute_mpv_waveforms(wfch)
        
        time = np.arange(len(mpv_waveform))
        ch = uch.channel
        run = list(wfch.runs)[0]
        
        peak_mpv = np.nanmax(mpv_waveform)
        
        if peak_mpv == 0 or np.isnan(peak_mpv):
            print(f"Zero or NaN peak for channel {ch}")
            continue
        
        if ch not in calibration_data[uch.endpoint]:
            print(f"Channel {ch} not found in calibration file")
            continue
        
        spe_amp = calibration_data[uch.endpoint][ch]["SpeAmpl"]
        
        mpv_norm = mpv_waveform * (spe_amp / peak_mpv)
        
        key = f"{uch.endpoint}-{uch.channel}"
        module_name = dict_uniqch_to_module.get(key, None)
        
        if module_name is None:
            print(f"No module mapping found for endpoint {uch.endpoint}, channel {uch.channel}")
            continue
        
        if save:
            module_for_title = module_name[:2]
            channel_for_title = module_name[3]
            filename = f"template_{run}_{module_for_title}_{channel_for_title}.txt"
            filepath = os.path.join(save_dir, filename)
            np.savetxt(filepath, mpv_norm, fmt="%.9e")
            
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                print(f"File saved: {filepath}")
            else:
                print(f"Failed to save: {filepath}")
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=mpv_norm,
                mode="lines",
                showlegend=False
            ),
            row=row, col=col
        )
        
        if subplot_idx == 1:
            xref = "x domain"
            yref = "y domain"
        else:
            xref = f"x{subplot_idx} domain"
            yref = f"y{subplot_idx} domain"
        
        fig.add_annotation(
            x=0.98,
            y=0.95,
            xref=xref,
            yref=yref,
            text=f"{module_name}<br>",
            showarrow=False,
            align="left",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.7)"
        )
    
    for row in range(1, max_row + 1):
        fig.update_yaxes(title_text="Amplitude [ADC]", row=row, col=1)
    for col in range(1, max_col + 1):
        fig.update_xaxes(title_text="Time [ticks]", row=max_row, col=col)