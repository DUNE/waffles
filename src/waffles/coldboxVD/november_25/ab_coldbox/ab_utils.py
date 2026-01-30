import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from waffles.np02_utils.PlotUtils import np02_gen_grids
from waffles.data_classes.UniqueChannel import UniqueChannel

from waffles.data_classes.Waveform import Waveform



def plotting_overlap_wf(wfset, n_wf = None, index_list = None, show : bool = True, save : bool = False, x_min=None, x_max=None, y_min=None, y_max=None, int_ll=None, int_ul=None, baseline=None, analysis_label : str = '', output_folder : str = 'output'):
    """
    Plot an overlap of multiple waveforms using Plotly.

    The function allows plotting either:
    - the first n_wf waveforms, or
    - a specific subset of waveforms selected via index_list.

    Optional vertical and horizontal reference lines (integration limits and baseline)
    can be added to the plot. Axis ranges can be manually specified, and the figure
    can be shown interactively and/or saved to disk.

    Parameters
    ----------
    wfset : object
        Waveform container holding the waveforms and their metadata.
    n_wf : int, optional
        Number of waveforms to plot (ignored if index_list is provided).
    index_list : list[int], optional
        Explicit list of waveform indices to plot.
    show : bool
        Whether to display the plot interactively.
    save : bool
        Whether to save the plot as an image.
    x_min, x_max, y_min, y_max : float, optional
        Axis limits.
    int_ll, int_ul : float, optional
        Lower and upper integration limits (vertical lines).
    baseline : float, optional
        Baseline value (horizontal line).
    analysis_label : str
        Label used to access analysis results (if baseline is needed).
    output_folder : str
        Folder where the output image is saved.
    """

    
    if index_list is None and n_wf is None:
        raise ValueError("You must provide either n_wf or index_list")

    if index_list is not None:
        indices = [i for i in index_list if 0 <= i < len(wfset.waveforms)]
    else:
        indices = range(min(n_wf, len(wfset.waveforms)))
    
    fig = go.Figure()

    for i in indices:
        
        y = wfset.waveforms[i].adcs
            
        fig.add_trace(go.Scatter(
            x=np.arange(len(y)) + wfset.waveforms[i].time_offset,
            y=y,
            mode='lines',
            line=dict(width=0.5),
            showlegend=True, 
            name = f"Waveform {i}"))

    xaxis_range = dict(range=[x_min, x_max]) if x_min is not None and x_max is not None else {}
    yaxis_range = dict(range=[y_min, y_max]) if y_min is not None and y_max is not None else {}

    fig.update_layout(
        xaxis_title="Time ticks (AU)",
        yaxis_title="Amplitude (Adcs)",
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
        fig.write_image(f"{output_folder}/waveform_plot.png", scale=2)
    
    if show:
        fig.show()

#########################################################################################################

def coldbox_single_channel_grid(wfset, config_channel):
    """
    Generate a waveform grid for a single specified detector channel.

    This function creates a ChannelWsGrid for one specific channel, 
    based on the waveform dataset provided.

    Parameters
    ----------
    wfset : object
        Waveform dataset containing multiple channels and associated metadata.
    config_channel : int or str
        Identifier of the channel to generate the grid for.

    Returns
    -------
    grid : ChannelWsGrid
        A waveform grid object corresponding to the specified channel.
    """
    
    kwargs = {}
    detector = [UniqueChannel(10, x) for x in [config_channel]]
    grid = np02_gen_grids(wfset, detector, rows=kwargs.pop("rows", 0), cols=kwargs.pop("cols", 0))
    return grid['Custom']


#########################################################################################################

def baseline_std_selection(waveform: Waveform, baseline_analysis_label: str, mean_std: float, n_std: float = 1.0) -> bool:
    """
    Check if a waveform passes a baseline standard deviation threshold, defined as
    mean_std multiplied by n_std.

    Parameters
    ----------
    waveform : object
        The waveform object containing analysis results.
    baseline_analysis_label : str
        Label identifying which baseline analysis to use.
    mean_std : float
        Reference standard deviation used as a threshold.
    n_std : float, optional
        Multiplicative factor for the threshold (default is 1.0).

    Returns
    -------
    bool
        True if the waveform's baseline_std is less than mean_std * n_std,
        False otherwise.
    """

    if waveform.analyses[baseline_analysis_label].result['baseline_std'] < mean_std*n_std:
        return True
    else:
        return False