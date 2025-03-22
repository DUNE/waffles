# grid_plots.py

from typing import Callable, List, Union
import numpy as np
import plotly.graph_objects as go

from waffles.plotting.session import PlotSession
from waffles.plotting.drawing_tools_utils import (
    compute_charge_histogram, compute_charge_histogram_params,
    get_endpoints, get_grid, get_histogram, get_wfs
)
from waffles import WaveformSet



# from grid_plots import (
#     plot_param_vs_channel,
#     plot_chist_param_vs_var,
#     plot_to_interval,
#     plot_grid_histogram
# )
# from waffles.plotting.session import PlotSession

# # Example: Gain vs channel
# session = PlotSession()
# plot_param_vs_channel(wset, session, ep=0, chs=list(range(128)), param='gain')
# session.write("gain_vs_channel.html")

# # Example: Time offset vs APA
# session = PlotSession()
# plot_to_interval(wset, session, apa=[1,2], nbins=120, tmin=-500, tmax=500)
# session.write("to_per_apa.png")

# # Example: spe_mean vs voltage setting
# session = PlotSession()
# plot_chist_param_vs_var(wset_map, session, ep=1, ch=10, param='spe_mean', var='bias_voltage')
# session.write("spe_vs_voltage.html")


def plot_param_vs_channel(wset: WaveformSet,
                          session: PlotSession,
                          ep: int = -1,
                          chs: List[int] = None,
                          param: str = 'gain',
                          op: str = ''):
    """Plot gain, sn, or spe_mean vs channel"""
    ch_values = []
    param_values = []

    for ch in chs:
        calibh = compute_charge_histogram(wset, ep, ch, 135, 165, 200, -5000, 20000, op=op + ' peaks')
        gain, sn, spe_mean = compute_charge_histogram_params(calibh)

        ch_values.append(ch)
        if param == 'gain':
            param_values.append(gain)
        elif param == 'sn':
            param_values.append(sn)
        elif param == 'spe_mean':
            param_values.append(spe_mean)

    trace = go.Scatter(x=ch_values, y=param_values, mode='markers')
    session.add_trace(trace)
    session.set_labels("channel", param)
    session.set_title(f"{param} vs channel")


def plot_chist_param_vs_var(wset_map,
                             session: PlotSession,
                             ep: int = -1,
                             ch: int = -1,
                             param: str = 'gain',
                             var: str = None,
                             op: str = ''):
    """Plot a histogram parameter (gain, sn, spe_mean) vs a user-defined variable."""
    par_values = []
    var_values = []

    for wset, var_val in wset_map:
        calibh = compute_charge_histogram(wset, ep, ch, 128, 170, 300, -5000, 40000, op="peaks")
        gain, sn, spe_mean = compute_charge_histogram_params(calibh)
        var_values.append(var_val)

        if param == 'gain':
            par_values.append(gain)
        elif param == 'sn':
            par_values.append(sn)
        elif param == 'spe_mean':
            par_values.append(spe_mean)

    trace = go.Scatter(x=var_values, y=par_values)
    session.add_trace(trace)
    session.set_labels(var, param)
    session.set_title(f"{param} vs {var}")


def plot_to_interval(wset: WaveformSet,
                     session: PlotSession,
                     apa: Union[int, List[int]] = -1,
                     ch: Union[int, List[int]] = -1,
                     nbins: int = 125,
                     tmin: int = None,
                     tmax: int = None,
                     xmin: int = None,
                     xmax: int = None,
                     rec: List[int] = [-1]):
    """Plot time offset histograms grouped by APA."""
    if isinstance(apa, list):
        apa_list = apa
    else:
        apa_list = [apa]

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for idx, apa_val in enumerate(apa_list):
        eps = get_endpoints(apa_val)
        selected_wfs = get_wfs(wset.waveforms, eps, ch, -1, tmin, tmax, rec)

        times = [
            wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp
            for wf in selected_wfs
            if (ch == -1 or wf.channel in (ch if isinstance(ch, list) else [ch]))
        ]

        color = colors[idx % len(colors)]
        trace = get_histogram(times, nbins, tmin, tmax, color)
        trace.name = f"APA {apa_val}"
        session.add_trace(trace)

        print(f"APA {apa_val}: {len(selected_wfs)} waveforms")

    session.set_labels("Time offset", "Entries")
    session.set_title("Time offset histogram for each APA")


def plot_grid_histogram(wfset: WaveformSet,
                        session: PlotSession,
                        wf_func: Callable,
                        apa: int = -1,
                        ch: Union[int, List[int]] = -1,
                        nbins: int = 100,
                        nwfs: int = -1,
                        xmin: int = None,
                        xmax: int = None,
                        tmin: int = -1,
                        tmax: int = -1,
                        rec: List[int] = [-1]):
    """
    Plot a WaveformSet in grid mode, generating a histogram per channel.
    """
    if tmin == -1 and tmax == -1:
        tmin = xmin
        tmax = xmax

    eps = get_endpoints(apa)
    selected_wfs = get_wfs(wfset.waveforms, eps, ch, nwfs, tmin, tmax, rec)

    if not selected_wfs:
        print(f"No waveforms found for APA={apa}, Channel={ch}, Time range=({tmin}, {tmax})")
        return

    run = wfset.waveforms[0].run_number
    grid = get_grid(selected_wfs, apa, run)

    from waffles.plotting.drawing_tools_utils import plot_CustomChannelGrid, plot_histogram_function_user

    session.fig = plot_CustomChannelGrid(
        grid,
        plot_function=lambda channel_wfs, idx, figure_, row, col, func, *args, **kwargs:
            plot_histogram_function_user(wf_func, channel_wfs, idx, figure_, row, col, nbins, xmin, xmax),
        x_axis_title='Time offset',
        y_axis_title='Entries',
        figure_title=f'Time offset histogram for APA {apa}',
        share_x_scale=True,
        share_y_scale=True,
        wf_func=wf_func
    )