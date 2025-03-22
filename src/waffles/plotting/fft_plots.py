# fft_plots.py

import numpy as np
import plotly.graph_objects as go
import warnings
from typing import List, Union
from waffles.plotting.session import PlotSession
from waffles import Waveform

# from fft_plots import plot_fft, plot_meanfft
# from waffles.plotting.session import PlotSession

# # FFT of one waveform
# session = PlotSession()
# plot_fft(wf, session)
# session.write("fft_single.html")

# # Mean FFT over several waveforms
# session = PlotSession()
# plot_meanfft(wset.waveforms, session, tmin=0, tmax=500)
# session.write("fft_mean.html")


def compute_fft(signal: np.ndarray, dt: float = 16e-9):
    """Compute FFT and return frequency [MHz] and power [dB]."""
    np.seterr(divide='ignore')
    
    if signal.shape[0] % 2 != 0:
        warnings.warn("Signal length not even, trimming last point.")
        signal = signal[:-1]

    n = len(signal)
    freqs = np.fft.fftfreq(n, d=dt)
    fft_vals = np.fft.fft(signal) / n
    pos_mask = freqs > 0

    freq_axis = freqs[pos_mask] / 1e6  # MHz
    power_db = 20 * np.log10(np.abs(2 * fft_vals[pos_mask]) / 2**14)

    return freq_axis, power_db


def plot_fft(waveform: Waveform, session: PlotSession, dt: float = 16e-9):
    """Plot FFT of a single waveform."""
    freq, power = compute_fft(waveform.adcs, dt)
    trace = go.Scatter(
        x=freq, y=power,
        mode='lines',
        line=dict(color=session.line_color, width=0.5)
    )
    session.add_trace(trace)
    session.set_labels("Frequency [MHz]", "Power [dB]")
    session.set_title(f"FFT of channel {waveform.channel}")
    session.fig.update_layout(xaxis_type='log')


def plot_meanfft(wfs: List[Waveform],
                 session: PlotSession,
                 tmin: int = -1,
                 tmax: int = -1):
    """Plot the mean FFT of a group of waveforms."""
    if not wfs:
        print("No waveforms provided.")
        return

    fft_xs = []
    fft_ys = []

    for wf in wfs:
        if tmin != -1 and tmax != -1:
            offset = wf.timestamp - wf.daq_window_timestamp
            if not (tmin <= offset <= tmax):
                continue

        fx, fy = compute_fft(wf.adcs)
        fft_xs.append(fx)
        fft_ys.append(fy)

    if not fft_ys:
        print("No waveforms passed time filter.")
        return

    mean_freq = np.mean(fft_xs, axis=0)
    mean_power = np.mean(fft_ys, axis=0)

    trace = go.Scatter(
        x=mean_freq,
        y=mean_power,
        mode='lines',
        line=dict(color='blue', width=1),
        name='Mean FFT'
    )

    session.add_trace(trace)
    session.set_labels("Frequency [MHz]", "Power [dB]")
    session.set_title("Mean FFT of selected waveforms")
    session.fig.update_layout(xaxis_type='log')


def plot_meanfft_grid(channel_ws, apa, idx, fig, row, col, nbins):
    """Grid version: plot mean FFT over time intervals."""
    waveform_sets = {
        "[-1000, -500]": get_wfs_interval(channel_ws.waveforms, -1000, -500),
        "[0, 300]": get_wfs_interval(channel_ws.waveforms, 0, 300),
        "[600, 1000]": get_wfs_interval(channel_ws.waveforms, 600, 1000)
    }

    colors = ['blue', 'green', 'red']
    x_axis_title = "Frequency [MHz]"
    y_axis_title = "Power [dB]"
    title = f"Mean FFT per interval - APA {apa}"

    for i, (label, subset) in enumerate(waveform_sets.items()):
        if not subset:
            continue
        xs, ys = [], []
        for wf in subset:
            x, y = compute_fft(wf.adcs)
            xs.append(x)
            ys.append(y)
        if not ys:
            continue
        mean_x = np.mean(xs, axis=0)
        mean_y = np.mean(ys, axis=0)

        fig.add_trace(go.Scatter(
            x=mean_x,
            y=mean_y,
            mode='lines',
            name=label,
            line=dict(color=colors[i % len(colors)], width=1)
        ), row=row, col=col)

    return x_axis_title, y_axis_title, title


# Helper: filter by time offset
def get_wfs_interval(wfs: List[Waveform], tmin: int, tmax: int) -> List[Waveform]:
    return [
        wf for wf in wfs
        if tmin <= (wf.timestamp - wf.daq_window_timestamp) <= tmax
    ]