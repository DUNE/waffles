# histograms.py

from typing import List, Union
import numpy as np
import plotly.graph_objects as go

from waffles.plotting.session import PlotSession
from waffles.plotting.drawing_tools_utils import (
    get_histogram, has_option, compute_charge_histogram,
    compute_charge_histogram_params, get_wfs_in_channel, get_wfs_with_integral_in_range
)
from waffles.data_classes import Event, WaveformSet, CalibrationHistogram



# from histograms import plot_evt_nch, plot_evt_time, plot_to, plot_charge, plot_charge_peaks
# from waffles.plotting.session import PlotSession

# # Example: channels per event
# session = PlotSession()
# plot_evt_nch(my_events, session)
# session.write("nchannels_hist.html")

# # Example: time offset
# session = PlotSession()
# plot_to(wfset, session, ep=2, ch=5)
# session.write("time_offset.png")

# # Example: charge histogram with fitted peaks
# session = PlotSession()
# chist = plot_charge(wfset, session, ep=0, ch=2)
# plot_charge_peaks(chist, session)
# session.write("charge_with_peaks.html")


def plot_evt_nch(events: List[Event.Event], session: PlotSession, nbins: int = 100,
                 xmin: float = None, xmax: float = None, title: str = "Channels firing per event"):
    nchs = [ev.get_nchannels() for ev in events]
    trace = get_histogram(nchs, nbins, xmin, xmax)
    session.add_trace(trace)
    session.set_labels("# channels", "entries")
    session.set_title(title)


def plot_evt_time(events: List[Event.Event], session: PlotSession, type: str = 'ref',
                  nbins: int = 100, xmin: float = None, xmax: float = None):
    if type == 'ref':
        times = [ev.ref_timestamp * 1e-9 * 16 for ev in events]
    elif type == 'first':
        times = [ev.first_timestamp * 1e-9 * 16 for ev in events]
    elif type == 'last':
        times = [ev.last_timestamp * 1e-9 * 16 for ev in events]
    else:
        raise ValueError(f"Unknown timestamp type: {type}")

    trace = get_histogram(times, nbins, xmin, xmax)
    session.add_trace(trace)
    session.set_labels(f"{type}_timestamp", "entries")
    session.set_title(f"{type.capitalize()} timestamp histogram")


def plot_to(wset: WaveformSet, session: PlotSession, ep: int = -1, ch: int = -1,
            nbins: int = 100, xmin: float = None, xmax: float = None):
    """Time offset histogram for a WaveformSet."""
    times = [
        wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp
        for wf in wset.waveforms
        if (wf.endpoint == ep or ep == -1) and (wf.channel == ch or ch == -1)
    ]
    trace = get_histogram(times, nbins, xmin, xmax)
    session.add_trace(trace)
    session.set_labels("time offset", "entries")
    session.set_title("Time offset distribution")


def plot_charge(wset: WaveformSet, session: PlotSession,
                ep: int = -1, ch: int = -1, int_ll: int = 135, int_ul: int = 165,
                nb: int = 200, hl: int = -5000, hu: int = 50000,
                b_ll: int = 0, b_ul: int = 100, nwfs: int = -1,
                variable: str = 'integral', op: str = '') -> CalibrationHistogram:

    chist = compute_charge_histogram(wset, ep, ch, int_ll, int_ul, nb, hl, hu, b_ll, b_ul, nwfs, variable, op + ' print')

    from waffles.plotting.drawing_tools_utils import plot_CalibrationHistogram  # used in original

    plot_CalibrationHistogram(chist, session.fig, 'Charge Spectrum', None, None, True, 200)
    session.set_labels(variable, "entries")
    session.set_title(f"Charge histogram for ch={ch}")
    return chist


def plot_charge_peaks(calibh: CalibrationHistogram, session: PlotSession,
                      npeaks: int = 2, prominence: float = 0.2,
                      half_points_to_fit: int = 10, op: str = ''):

    from waffles.plotting.drawing_tools_utils import (
        compute_peaks, plot_CalibrationHistogram
    )

    compute_peaks(calibh, npeaks, prominence, half_points_to_fit, op)
    plot_CalibrationHistogram(calibh, session.fig, 'Peaks', None, None, True, 200)
    session.set_title("Charge peaks")