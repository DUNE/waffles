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

    idx_10_value = []
    idx_90_value = []
    rise_times_ticks = []
    rise_times_ns = []
    for wf in led_wfs: #led_wfs[:10]:
    
        baseline = np.median(wf[:50])
        wf_corr = wf - baseline
        peak_idx = np.argmax(wf_corr)
        peak = wf_corr[peak_idx]

        threshold_10 = 0.1 * peak
        threshold_90 = 0.9 * peak

        start = 0  # skip the first ticks

        rising_region = wf_corr[start : peak_idx + 1]

        idx10_all = start + np.where(rising_region >= threshold_10)[0]
        idx90_all = np.where(rising_region >= threshold_90)[0]
    
        if len(idx10_all) == 0 or len(idx90_all) == 0:
            # handle the case where the waveform never actually crosses the threshold
            continue  

        idx_10 = idx10_all[0]
        idx_90 = idx90_all[0]

        rt_ticks = idx_90 - idx_10
        rt_ns    = rt_ticks * 16

        idx_10_value.append(idx_10)
        idx_90_value.append(idx_90)
        rise_times_ticks.append(rt_ticks)
        rise_times_ns.append(rt_ns)

    #for i, (id10, id90, ticks, ns) in enumerate(zip(idx_10_value, idx_90_value, rise_times_ticks, rise_times_ns)):
    #    print(f"Waveform {i:2d}: 10% at = {id10} 90% at = {id90} Rise time = {ticks:3d} ticks  = {ns:5.1f} ns")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("LED data 0", "LED baseline subtracted"))

    # primo subplot: solo la prima waveform
    fig.add_trace(
        go.Scatter(x=xt, y=led_wfs[0], mode="lines",
                   name="WF 0", showlegend=True),
        row=1, col=1
    )
    
    # secondo subplot: baseline subtracted
    fig.add_trace(
        go.Scatter(x=xt, y=wf_corr, mode="lines",
                   name="WF 0 baseline subtracted", line=dict(color='red')),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Sample Index", row=2, col=1)
    fig.update_yaxes(title_text="ADC Counts", row=1, col=1)
    fig.update_yaxes(title_text="ADC Counts", row=2, col=1)
    fig.update_layout(height=600, width=800, title_text="LED Waveforms")
    fig.show()

    grid_wfs = led_wfs[:9]
    num = len(grid_wfs)

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f"WF {i}" for i in range(num)],
        horizontal_spacing=0.05,
        vertical_spacing=0.08
    )

    xt = np.arange(grid_wfs.shape[1])

    for i, wf in enumerate(grid_wfs):
        baseline = np.mean(wf[:50])
        wf_corr  = wf - baseline
        r = i // 3 + 1
        c = i % 3 + 1
        fig.add_trace(
            go.Scatter(x=xt, y=wf_corr, mode="lines", showlegend=False),
            row=r, col=c
        )
        if r == 3:
            fig.update_xaxes(title_text="Sample Index", row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text="ADC Counts", row=r, col=c)

    fig.update_layout(
        height=900, width=1300,
        title_text="First 9 Waveforms (3×3 Grid)"
    )
    fig.show()    

    hist_fig = go.Figure()

    hist_fig.add_trace(
        go.Histogram(
            x=idx_90_value,
            nbinsx=100,
            name="Max position (time ticks)"
        )
    )

    hist_fig.update_layout(
        title="Histogram of Max position - run 36030 - ch 41", #- ch 43",
        xaxis_title="Max position (time ticks)",
        yaxis_title="Count",
        bargap=0.1
    )

    hist_fig.show()


if __name__ == "__main__":

    led_path = "data/processed_np02vd_raw_run036103_0000_df-s04-d0_dw_0_20250425T132156.hdf5_structured.hdf5"
    channel = 41 #43
    process_waveforms(led_path, channel)