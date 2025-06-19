import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

import waffles
from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.WaveformSet import WaveformSet

# Load waveform data from HDF5 files
def load_waveforms(path, channel, max_samples):
    wfset = load_structured_waveformset(path)
    waveforms = [wf.adcs[:max_samples] for wf in wfset.waveforms if wf.channel == channel and len(wf.adcs) >= max_samples]
    return wfset, np.array(waveforms)

def process_waveforms(led_path, channel, max_samples=1024):
    _, led_wfs = load_waveforms(led_path, channel, max_samples)

    xt = np.arange(max_samples)  # NumPy’s arange creates a 1D array of evenly spaced integers

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("LED data all", "LED data 0"))

    # primo subplot: tutte le waveforms
    for i, wf in enumerate(led_wfs):
        fig.add_trace(
            go.Scatter(x=xt, y=wf, mode="lines",
                       name=f"WF {i}", showlegend=False),
            row=1, col=1
        )
    
    # secondo subplot: la media di tutte
    mean_wf = led_wfs.mean(axis=0)
    fig.add_trace(
        go.Scatter(x=xt, y=mean_wf, mode="lines",
                   name="Mean LED", line=dict(color='red')),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Sample Index", row=2, col=1)
    fig.update_yaxes(title_text="ADC Counts", row=1, col=1)
    fig.update_yaxes(title_text="ADC Counts", row=2, col=1)
    fig.update_layout(height=600, width=800, title_text="LED Waveforms")
    fig.show()


    # Compute rise time for the first waveform
    wf = led_wfs[2]
    baseline = np.median(wf[:50])
    wf_corrected = wf - baseline
    peak = wf_corrected.max()

    threshold_10 = 0.1 * peak
    threshold_90 = 0.9 * peak

    print(f"threshold_10: {threshold_10} threshold_90: {threshold_90}")

    idx_10 = np.argmax(wf_corrected > threshold_10)
    idx_90 = np.argmax(wf_corrected > threshold_90)

    print(f"idx_10: {idx_10} idx_90: {idx_90}")

    rise_time_ticks = idx_90 - idx_10
    rise_time_ns = rise_time_ticks * 16

    print(f"Rise time: {rise_time_ticks} ticks = {rise_time_ns} ns")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("LED data 0", "LED baseline subtracted"))

    # primo subplot: solo la prima waveform
    fig.add_trace(
        go.Scatter(x=xt, y=led_wfs[2], mode="lines",
                   name="WF 0", showlegend=True),
        row=1, col=1
    )
    
    # secondo subplot: baseline subtracted
    fig.add_trace(
        go.Scatter(x=xt, y=wf_corrected, mode="lines",
                   name="WF 0 baseline subtracted", line=dict(color='red')),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Sample Index", row=2, col=1)
    fig.update_yaxes(title_text="ADC Counts", row=1, col=1)
    fig.update_yaxes(title_text="ADC Counts", row=2, col=1)
    fig.update_layout(height=600, width=800, title_text="LED Waveforms")
    fig.show()

if __name__ == "__main__":

    led_path = "data/processed_np02vd_raw_run036026_0000_df-s04-d0_dw_0_20250425T094006.hdf5_structured.hdf5"
    channel = 45
    process_waveforms(led_path, channel)