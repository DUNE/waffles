from typing import Union, Optional, List
import numpy as np
import plotly.graph_objects as go
from waffles.plotting.session import PlotSession
from waffles.plotting.drawing_tools_utils import get_wfs, get_wfs_in_channel, get_wfs_with_integral_in_range
from waffles import Waveform, WaveformAdcs, WaveformSet, ChannelWs


# Intended use:
# from waveform_plotters import plot, plot_avg
# from waffles.plotting.session import PlotSession

# session = PlotSession(mode='html')
# plot(wfset, session, ep=0, ch=1, nwfs=5)
# session.write('my_waveform_plot.html')

# session = PlotSession(mode='png')
# plot_avg(wfset, session, ep=0, ch=1)
# session.write('avg_wf.png')

def plot(obj: Union[Waveform, WaveformAdcs, WaveformSet, List[Waveform]],
         session: PlotSession,
         ep: Union[int, List[int]] = -1,
         ch: Union[int, List[int]] = -1,
         nwfs: int = -1,
         xmin: int = -1,
         xmax: int = -1,
         ymin: int = -1,
         ymax: int = -1,
         tmin: int = -1,
         tmax: int = -1,
         offset: bool = False,
         rec: List[int] = [-1],
         op: str = ''):
    """Plot a single or multiple waveforms."""
    if isinstance(obj, Waveform):
        plot_wfs([obj], session, ep, ch, nwfs, xmin, xmax, ymin, ymax, tmin, tmax, offset, rec, op)
    elif isinstance(obj, WaveformAdcs):
        plot_wfs([obj], session, -100, -100, nwfs, xmin, xmax, ymin, ymax, tmin, tmax, offset, rec, op)
    elif isinstance(obj, list) and all(isinstance(o, Waveform) for o in obj):
        plot_wfs(obj, session, ep, ch, nwfs, xmin, xmax, ymin, ymax, tmin, tmax, offset, rec, op)
    elif isinstance(obj, WaveformSet):
        plot_wfs(obj.waveforms, session, ep, ch, nwfs, xmin, xmax, ymin, ymax, tmin, tmax, offset, rec, op)
    else:
        raise TypeError("Unsupported object type for plotting")

def plot_wfs(wfs: List[WaveformAdcs],
             session: PlotSession,
             ep: Union[int, List[int]] = -1,
             ch: Union[int, List[int]] = -1,
             nwfs: int = -1,
             xmin: int = -1,
             xmax: int = -1,
             ymin: int = -1,
             ymax: int = -1,
             tmin: int = -1,
             tmax: int = -1,
             offset: bool = False,
             rec: List[int] = [-1],
             op: str = ''):
    """Plot a list of waveforms."""

    # Handle default time range
    if tmin == -1 and tmax == -1:
        if xmin != -1 and xmax != -1:
            tmin = xmin - 1024
            tmax = xmax
        else:
            tmin, tmax = -99999, 99999

    # Select waveforms
    selected_wfs = get_wfs(wfs, ep, ch, nwfs, tmin, tmax, rec)
    print(f"[DEBUG] selected {len(selected_wfs)} waveform(s) from input wfs")

    count = 0
    for wf in selected_wfs:
        plot_wf(wf, session, offset=offset)
        count += 1
        if nwfs != -1 and count >= nwfs:
            break

    session.set_labels("time tick", "adcs")
    session.update_axes(
        x_range=(xmin, xmax) if xmin != -1 and xmax != -1 else None,
        y_range=(ymin, ymax) if ymin != -1 and ymax != -1 else None
    )

def plot_wf(waveform_adcs: WaveformAdcs,
            session: PlotSession,
            offset: bool = False,
            name: Optional[str] = None):
    """Plot a single waveform."""
    x0 = np.arange(len(waveform_adcs.adcs)).astype(float)
    y0 = np.array(waveform_adcs.adcs).astype(float)

    if offset:
        dt = float(getattr(waveform_adcs, "timestamp", 0) -
                   getattr(waveform_adcs, "daq_window_timestamp", 0))
    else:
        dt = 0

    # Use custom name or fallback gracefully
    channel_str = getattr(waveform_adcs, "channel", "avg")
    trace = go.Scatter(
        x=x0 + dt,
        y=y0,
        mode='lines',
        line=dict(color=session.line_color, width=0.5),
        name=name or f"ch {channel_str}"
    )
    session.add_trace(trace)

def plot_avg(wset: WaveformSet,
             session: PlotSession,
             ep: int = -1,
             ch: int = -1,
             nwfs: int = -1,
             imin: float = None,
             imax: float = None,
             op: str = ''):
    """Plot the average waveform from a channel."""
    wset2 = get_wfs_in_channel(wset, ep, ch)
    if imin is not None:
        wset2 = get_wfs_with_integral_in_range(wset2, imin, imax)

    ch_ws = ChannelWs(*wset2.waveforms)
    mean_wf = ch_ws.compute_mean_waveform()

    plot_wf(mean_wf, session)

    session.set_labels('time tick', 'average adcs')