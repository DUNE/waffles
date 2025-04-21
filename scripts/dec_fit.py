import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.input_output.hdf5_structured import load_structured_waveformset


####################
# Utility functions
####################

def baseline_and_normalize(adcs, n_baseline_samples=100):
    """
    1) Baseline subtraction:
       - Compute average of the first `n_baseline_samples` 
         as pedestal and subtract it from the entire waveform.
    2) Amplitude normalization:
       - Divide by the maximum of the baseline-subtracted waveform.
         (If max is 0 or negative, do nothing or skip.)
    
    :param adcs: 1D numpy array of raw waveform ADC counts
    :param n_baseline_samples: int, number of samples from the start used for baseline
    :return: new 1D numpy array, baseline-subtracted and normalized
    """
    if len(adcs) < n_baseline_samples:
        # edge case if waveforms are shorter
        raise ValueError("Waveform shorter than n_baseline_samples!")

    baseline = np.mean(adcs[:n_baseline_samples])
    wave_corr = adcs - baseline

    max_val = np.max(wave_corr)
    if max_val > 0:
        wave_corr = wave_corr / max_val
    
    return wave_corr


def double_exp(t, A1, tau1, A2, tau2, baseline):
    """
    Double-exponential decay + constant offset:
      f(t) = A1*exp(-t/tau1) + A2*exp(-t/tau2) + baseline
    
    :param t: array-like, time axis (ns or µs)
    :param A1, tau1: amplitude and decay constant of the first component
    :param A2, tau2: amplitude and decay constant of the second component
    :param baseline: constant offset
    """
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + baseline


def deconvolve_waveform(cosmic_adcs, led_adcs_avg, epsilon=1e-5):
    """
    Frequency-domain deconvolution:
      1) FFT(cosmic)
      2) FFT(average LED)
      3) ratio = FFT(cosmic) / (FFT(LED) + epsilon)
      4) inverse FFT(ratio)
    """
    cosmic_fft = np.fft.rfft(cosmic_adcs)
    led_fft = np.fft.rfft(led_adcs_avg)

    ratio_fft = cosmic_fft / (led_fft + epsilon)
    deconv_wf = np.fft.irfft(ratio_fft, n=len(cosmic_adcs))
    
    return deconv_wf


def run_deconvolution_example(
    cosmics_path, 
    led_path, 
    target_channel=0, 
    max_cosmic_wfs=None,
    n_baseline_samples=100
):
    from waffles.input_output.hdf5_structured import load_structured_waveformset
    
    wfset_cosmics = load_structured_waveformset(cosmics_path, max_waveforms=max_cosmic_wfs)
    wfset_led     = load_structured_waveformset(led_path)

    # Filter waveforms
    cosmics_channel = [wf for wf in wfset_cosmics.waveforms if wf.channel == target_channel]
    led_channel     = [wf for wf in wfset_led.waveforms if wf.channel == target_channel]

    if len(cosmics_channel) == 0:
        raise ValueError(f"No cosmic waveforms found for channel={target_channel}.")
    if len(led_channel) == 0:
        raise ValueError(f"No LED waveforms found for channel={target_channel}.")

    # --- Convert to arrays, baseline subtract, normalize ---
    cosmic_arrays = []
    for wf in cosmics_channel:
        arr = baseline_and_normalize(wf.adcs, n_baseline_samples=n_baseline_samples)
        cosmic_arrays.append(arr)

    led_arrays = []
    for wf in led_channel:
        arr = baseline_and_normalize(wf.adcs, n_baseline_samples=n_baseline_samples)
        led_arrays.append(arr)

    # --- Compute average LED ---
    adcs_2d_led = np.array(led_arrays, dtype=np.float64)
    avg_led = np.mean(adcs_2d_led, axis=0)

    # --- Deconvolve each cosmic waveform using avg LED ---
    deconv_cosmics = []
    for cosmic_adcs in cosmic_arrays:
        deconv_wf = deconvolve_waveform(cosmic_adcs, avg_led, epsilon=1e-5)
        deconv_cosmics.append(deconv_wf)

    # --- Average all deconvolved waveforms ---
    deconv_stack = np.array(deconv_cosmics, dtype=np.float64)
    avg_deconv   = np.mean(deconv_stack, axis=0)

    return {
        "avg_led_adcs": avg_led,
        "deconv_cosmics": deconv_cosmics,
        "avg_deconv": avg_deconv,
        "cosmics_waveforms": cosmics_channel,
        "led_waveforms": led_channel
    }


def fit_and_plot_double_exp(avg_deconv, sampling_ns=16.0, 
                            log_min_threshold=1e-5,
                            title="Avg Deconvolution with Double Exp Fit"):
    """
    1) Create time axis from sampling_ns
    2) Mask out near-zero or negative points for log-scale
    3) Fit a double-exponential model
    4) Plot both data and fit on a log-scale y-axis
    5) Return fit parameters
    """
    n_samples = len(avg_deconv)
    time_ns = np.arange(n_samples) * sampling_ns

    # -- 2) Mask out near-zero or negative points
    mask = (avg_deconv > log_min_threshold)
    t_fit = time_ns[mask]
    y_fit = avg_deconv[mask]

    # -- 3) Fit with double_exp. Provide initial guesses:
    # A1, tau1, A2, tau2, baseline
    # You may adjust these guesses if your data is very different
    p0 = [1.0, 300.0, 0.5, 1000.0, 1e-3]  # example guesses
    popt, pcov = curve_fit(double_exp, t_fit, y_fit, p0=p0)
    A1, tau1, A2, tau2, baseline = popt

    # Build a smooth fit curve
    fit_curve = double_exp(time_ns, A1, tau1, A2, tau2, baseline)

    # -- 4) Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_ns, y=avg_deconv, 
        mode='markers', 
        name='Avg Deconvolution'
    ))
    fig.add_trace(go.Scatter(
        x=time_ns, y=fit_curve,
        mode='lines',
        name=(f"Double Exp Fit:<br>"
              f"A1={A1:.3g}, tau1={tau1:.2f}ns<br>"
              f"A2={A2:.3g}, tau2={tau2:.2f}ns<br>"
              f"baseline={baseline:.3g}")
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (ns)",
        yaxis_title="Amplitude (norm.)",
        # Set log scale for y-axis
        yaxis_type="log",
        # Optionally adjust the range so the data is fully visible:
        # e.g., if the data is between ~1e-5 and 1
        yaxis_range=[-5, 0]  # log10 range, i.e. from 10^-5 to 10^0
    )
    fig.show()

    return popt, pcov


##########################
# Main usage / example
##########################

if __name__ == "__main__":

    # Example paths - replace with your real HDF5 files
    # Example usage:
    cosmic_file = "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035730_0001_df-s04-d0_dw_0_20250404T145714.hdf5_structured.hdf5"  # your cosmic waveformset HDF5
    led_file    = "/nfs/home/marroyav/hdf5_update/scripts/output_07/processed_np02vd_raw_run035738_0000_df-s04-d0_dw_0_20250405T210441.hdf5_structured.hdf5"      # your LED waveformset HDF5
    channel     = 45                 # whichever channel you want


    # Run the main deconvolution workflow
    results = run_deconvolution_example(
        cosmics_path=cosmic_file,
        led_path=led_file,
        target_channel=channel,
        max_cosmic_wfs=None,        # or set a smaller number if you want to limit
        n_baseline_samples=100      # baseline from first 100 samples
    )

    avg_deconv = results["avg_deconv"]

    # Fit and plot double-exponential decay in log scale
    popt, pcov = fit_and_plot_double_exp(
        avg_deconv,
        sampling_ns=16.0,  # sampling period in ns
        log_min_threshold=1e-5,
        title=f"Channel {channel}: Double Exp Fit on Deconvolved Avg"
    )

    # popt = [A1, tau1, A2, tau2, baseline]
    print("Fitted double-exponential parameters:", popt)
    print("Covariance matrix:", pcov)