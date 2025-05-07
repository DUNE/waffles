import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from waffles.input_output.hdf5_structured import load_structured_waveformset

# Dual-component scintillation model
def double_exp_model(x, A1, A2, L, T1, T2, x0):
    mask = x >= x0
    model = np.zeros_like(x)
    model[mask] = (A1 * (np.exp(-(x[mask]-x0)/L) - np.exp(-(x[mask]-x0)/T1)) +
                   A2 * (np.exp(-(x[mask]-x0)/L) - np.exp(-(x[mask]-x0)/T2)))
    return model

# Load waveform data from HDF5 files
def load_waveforms(path, channel, max_samples):
    wfset = load_structured_waveformset(path)
    waveforms = [wf.adcs[:max_samples] for wf in wfset.waveforms if wf.channel == channel and len(wf.adcs) >= max_samples]
    return np.array(waveforms)

# Baseline subtraction (robust)
def subtract_baseline(wf, baseline_region=50):
    baseline = np.median(wf[:baseline_region])
    return wf - baseline

# Alignment using cross-correlation
def align_waveforms(reference, target):
    corr = np.correlate(target - np.mean(target), reference - np.mean(reference), mode='same')
    shift = np.argmax(corr) - len(reference)//2
    return np.roll(target, -shift)

# Main processing function
def process_waveforms(cosmic_path, led_path, noise_path, channel, max_samples=1024):
    cosmic_wfs = load_waveforms(cosmic_path, channel, max_samples)
    led_wfs = load_waveforms(led_path, channel, max_samples)
    noise_wfs = load_waveforms(noise_path, channel, max_samples)

    min_count = min(len(cosmic_wfs), len(led_wfs), len(noise_wfs))
    cosmic_wfs, led_wfs, noise_wfs = cosmic_wfs[:min_count], led_wfs[:min_count], noise_wfs[:min_count]

    aligned_cosmics, aligned_leds = [], []

    for c_wf, l_wf in zip(cosmic_wfs, led_wfs):
        c_wf = subtract_baseline(c_wf)
        l_wf = subtract_baseline(l_wf)
        aligned_led = align_waveforms(c_wf, l_wf)
        eps = 1e-15
        aligned_led *= np.max(c_wf) / (np.max(aligned_led) + eps)
        aligned_cosmics.append(c_wf)
        aligned_leds.append(aligned_led)

    avg_cosmic = np.mean(aligned_cosmics, axis=0)
    avg_led = np.mean(aligned_leds, axis=0)

    x = np.arange(max_samples)
    p0 = [np.max(avg_cosmic), np.max(avg_cosmic)*2, 200, 30, 600, np.argmax(avg_cosmic)-20]
    bounds = ([0, 0, 10, 10, 10, 0], [np.inf, np.inf, 400, 40, 1500, max_samples])

    popt, _ = curve_fit(double_exp_model, x, avg_cosmic, p0=p0, bounds=bounds)

    fit_curve = double_exp_model(x, *popt)
    mask = x >= popt[5]
    fast = np.zeros_like(x)
    slow = np.zeros_like(x)
    fast[mask] = popt[0] * (np.exp(-(x[mask]-popt[5])/popt[2]) - np.exp(-(x[mask]-popt[5])/popt[3]))
    slow[mask] = popt[1] * (np.exp(-(x[mask]-popt[5])/popt[2]) - np.exp(-(x[mask]-popt[5])/popt[4]))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Linear Scale", "Logarithmic Scale"))

    fig.add_trace(go.Scatter(y=avg_cosmic, name="Avg Cosmic", line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=fit_curve, name="Fit", line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(y=fast, name="Fast Comp.", line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(y=slow, name="Slow Comp.", line=dict(dash='dash')), row=1, col=1)

    fig.add_trace(go.Scatter(y=np.abs(avg_cosmic)+1e-6, name="Avg Cosmic (log)", line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=np.abs(fit_curve)+1e-6, name="Fit (log)", line=dict(color='red')), row=2, col=1)

    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_layout(title="Waveform Analysis", height=800, xaxis_title="Samples (16 ns/sample)", yaxis_title="Amplitude")
    fig.show()

    # Print parameters
    print("Fit Parameters:")
    print(f"A1: {popt[0]:.2f}")
    print(f"A2: {popt[1]:.2f}")
    print(f"L: {popt[2]*16e-3:.2f} µs")
    print(f"T1: {popt[3]*16e-3:.2f} µs")
    print(f"T2: {popt[4]*16e-3:.2f} µs")
    print(f"x0: {popt[5]} samples")

if __name__ == "__main__":
    cosmic_path = "data/cosmic.hdf5"
    led_path = "data/led.hdf5"
    noise_path = "data/noise.hdf5"
    channel = 7
    process_waveforms(cosmic_path, led_path, noise_path, channel)