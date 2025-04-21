import numpy as np
import plotly.graph_objects as go
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.input_output.hdf5_structured import load_structured_waveformset

def subtract_baseline(adcs, n_samples=100, sigma=3.0):
    """
    Subtract a baseline from the waveform by:
      1) Taking the first n_samples as a "pedestal region".
      2) Computing mean & std dev in that region.
      3) Sigma-clipping out points beyond +/- sigma * std.
      4) Taking the mean of the remaining points as the baseline.
      5) If too few points remain after clipping, use the median of the first n_samples.
      6) Subtract the final baseline from the entire waveform.
    
    :param adcs: 1D np.array (waveform)
    :param n_samples: number of samples from the start to use for baseline
    :param sigma: how many std devs for clipping
    :return: Baseline-subtracted waveform (1D np.array)
    """
    if len(adcs) < n_samples:
        raise ValueError("Waveform is shorter than the number of baseline samples.")

    # 1) Extract the pedestal region
    pedestal_region = adcs[:n_samples]

    # 2) Compute initial mean & std
    mean0 = np.mean(pedestal_region)
    std0  = np.std(pedestal_region)

    # 3) Sigma-clip
    mask = np.abs(pedestal_region - mean0) < sigma * std0

    # 4) If too few remain, fall back to median, else compute mean of clipped data
    if np.count_nonzero(mask) < 5:
        baseline = np.median(pedestal_region)
    else:
        baseline = np.mean(pedestal_region[mask])

    # 5) Subtract from entire waveform
    return adcs - baseline


def deconvolve_waveform(cosmic_adcs, led_adcs_avg, epsilon=1e-5):
    """
    Perform a simple frequency-domain deconvolution:
      1) FFT(cosmic)
      2) FFT(average LED)
      3) ratio = FFT(cosmic) / (FFT(LED) + epsilon)
      4) inverse FFT(ratio)
    """
    cosmic_fft = np.fft.rfft(cosmic_adcs)
    led_fft    = np.fft.rfft(led_adcs_avg)

    ratio_fft = cosmic_fft / (led_fft + epsilon)
    deconv_wf = np.fft.irfft(ratio_fft, n=len(cosmic_adcs))
    return deconv_wf


def average_led_waveform(led_subtracted):
    """
    Given a list of baseline-subtracted LED waveforms (all from the same channel),
    compute the average waveform sample-by-sample.
    
    :param led_subtracted: list of 1D numpy arrays (baseline-subtracted LED waveforms)
    :return: The averaged LED waveform (1D numpy array)
    """
    adcs_2d = np.array(led_subtracted, dtype=np.float64)
    avg_wf  = np.mean(adcs_2d, axis=0)
    return avg_wf


def run_deconvolution_example(
    cosmics_path, 
    led_path, 
    target_channel=0, 
    max_cosmic_wfs=None, 
    n_baseline_samples=100,
    sigma_clip=3.0
):
    """
    1) Load cosmic and LED HDF5 waveform sets
    2) Filter by 'target_channel'
    3) Baseline-subtract each waveform (with sigma-clipping)
    4) Compute average LED waveform
    5) Deconvolve each cosmic waveform
    6) Average all deconvolved cosmic waveforms
    7) Return final results
    """
    # 1) Load waveforms
    wfset_cosmics = load_structured_waveformset(cosmics_path, max_waveforms=max_cosmic_wfs)
    wfset_led     = load_structured_waveformset(led_path)

    # 2) Filter by channel
    cosmics_channel = [wf for wf in wfset_cosmics.waveforms if wf.channel == target_channel]
    led_channel     = [wf for wf in wfset_led.waveforms     if wf.channel == target_channel]

    if len(cosmics_channel) == 0:
        raise ValueError(f"No cosmic waveforms found for channel={target_channel} in {cosmics_path}.")
    if len(led_channel) == 0:
        raise ValueError(f"No LED waveforms found for channel={target_channel} in {led_path}.")

    # 3) Baseline-subtract cosmic and LED waveforms with sigma-clipping
    cosmic_subtracted = [
        subtract_baseline(wf.adcs.astype(np.float64), n_samples=n_baseline_samples, sigma=sigma_clip)
        for wf in cosmics_channel
    ]
    led_subtracted = [
        subtract_baseline(wf.adcs.astype(np.float64), n_samples=n_baseline_samples, sigma=sigma_clip)
        for wf in led_channel
    ]

    # 4) Compute the average LED waveform
    avg_led = average_led_waveform(led_subtracted)

    # 5) Deconvolve each cosmic waveform
    deconv_cosmics = []
    for cosmic_adcs in cosmic_subtracted:
        deconv_wf = deconvolve_waveform(cosmic_adcs, avg_led, epsilon=1e-5)
        deconv_cosmics.append(deconv_wf)

    # 6) Average all deconvolved cosmic waveforms
    deconv_stack = np.array(deconv_cosmics, dtype=np.float64)
    avg_deconv   = np.mean(deconv_stack, axis=0)

    return {
        "avg_led_adcs":     avg_led,
        "cosmic_subtracted": cosmic_subtracted,
        "deconv_cosmics":    deconv_cosmics,
        "avg_deconv":        avg_deconv,
        "cosmics_channel":   cosmics_channel,
        "led_channel":       led_channel
    }


if __name__ == "__main__":

    # Example usage (update these file paths and channel as needed)
    cosmic_file = "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035730_0000_df-s04-d0_dw_0_20250404T144939.hdf5_structured.hdf5"
    led_file    = "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035747_0000_df-s04-d0_dw_0_20250405T225146.hdf5_structured.hdf5"
    channel     = 20

    # Run the analysis
    results = run_deconvolution_example(
        cosmics_path=cosmic_file,
        led_path=led_file,
        target_channel=channel,
        max_cosmic_wfs=None,       # Limit waveforms or None for all
        n_baseline_samples=250,    # Baseline from first 100 samples
        sigma_clip=1.0             # Sigma for clipping outliers in baseline region
    )

    # Unpack results
    avg_led         = results["avg_led_adcs"]
    cosmic_subbed   = results["cosmic_subtracted"]
    deconv_cosmics  = results["deconv_cosmics"]
    avg_deconv      = results["avg_deconv"]
    # cosmic_waveforms = results["cosmics_channel"]  # If needed for metadata

    # Plot example outputs
    # 1) Average LED (baseline-subtracted)
    fig_avg_led = go.Figure()
    fig_avg_led.add_trace(
        go.Scatter(
            y=avg_led, 
            mode='lines', 
            name='Average LED (sigma-clip baseline-sub)'
        )
    )
    fig_avg_led.update_layout(
        title=f"Average LED Waveform (Channel={channel})",
        xaxis_title="Sample Index",
        yaxis_title="ADC"
    )
    fig_avg_led.show()
    fig_avg_led.write_html("avg_led_waveform.html")

    # 2) Average Cosmic Waveform (baseline-subtracted)
    adcs_2d_cosmics = np.array(cosmic_subbed, dtype=np.float64)
    avg_cosmic_bs = np.mean(adcs_2d_cosmics, axis=0)

    fig_avg_cosmic = go.Figure()
    fig_avg_cosmic.add_trace(
        go.Scatter(
            y=avg_cosmic_bs, 
            mode='lines', 
            name='Average Cosmic (sigma-clip baseline-sub)'
        )
    )
    fig_avg_cosmic.update_layout(
        title=f"Average Cosmic Waveform (Channel={channel})",
        xaxis_title="Sample Index",
        yaxis_title="ADC"
    )
    fig_avg_cosmic.show()
    fig_avg_cosmic.write_html("avg_cosmic_waveform.html")

    # 3) One example cosmic vs. its deconvolution
    cosmic_idx = 0
    fig_cosmic = go.Figure()
    fig_cosmic.add_trace(
        go.Scatter(
            y=cosmic_subbed[cosmic_idx], 
            mode='lines', 
            name=f'Cosmic baseline-sub (example {cosmic_idx})'
        )
    )
    fig_cosmic.add_trace(
        go.Scatter(
            y=deconv_cosmics[cosmic_idx], 
            mode='lines', 
            name=f'Deconvolved (example {cosmic_idx})'
        )
    )
    fig_cosmic.update_layout(
        title=f"Baseline-Subtracted vs. Deconvolved Cosmic Waveform (Ch={channel})",
        xaxis_title="Sample Index",
        yaxis_title="ADC"
    )
    fig_cosmic.show()
    fig_cosmic.write_html("example_cosmic_deconv.html")

    # 4) Average of all deconvolved cosmic waveforms
    fig_avg_deconv = go.Figure()
    fig_avg_deconv.add_trace(
        go.Scatter(y=avg_deconv, mode='lines', name='Average Deconvolution')
    )
    fig_avg_deconv.update_layout(
        title=f"Average Deconvolution (Channel={channel})",
        xaxis_title="Sample Index",
        yaxis_title="ADC"
    )
    fig_avg_deconv.show()
    fig_avg_deconv.write_html("avg_deconvolution.html")