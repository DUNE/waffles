import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

import waffles
from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.WaveformSet import WaveformSet

# Dual-component scintillation model
def double_exp_model(x, A1, A2, L, T1, T2, x0):
    mask = x >= x0
    model = np.zeros_like(x)
    model[mask] = (A1 * (np.exp(-(x[mask]-x0)/L) - np.exp(-(x[mask]-x0)/T1)) +
                   A2 * (np.exp(-(x[mask]-x0)/L) - np.exp(-(x[mask]-x0)/T2)))
    return model

# 1) Gaussiana + 2 esponenziali
def model_gauss_2exp(x, A_fast, mu, sigma, A_int, tau_int, A_slow, tau_slow, x0):
    y = np.zeros_like(x)
    mask = x >= x0
    t = x[mask]
    y[mask] += A_fast * np.exp(- (t - mu)**2 / (2 * sigma**2))
    y[mask] += A_int * np.exp(-(t - x0) / tau_int)
    y[mask] += A_slow * np.exp(-(t - x0) / tau_slow)
    return y

# 2) Tre esponenziali
def model_3exp(x, A1, tau1, A2, tau2, A3, tau3, x0):
    y = np.zeros_like(x)
    mask = x >= x0
    t = x[mask]
    y[mask] = (
        A1 * np.exp(-(t - x0) / tau1) +
        A2 * np.exp(-(t - x0) / tau2) +
        A3 * np.exp(-(t - x0) / tau3)
    )
    return y

# Load waveform data from HDF5 files
def load_waveforms(path, channel, max_samples):
    wfset = load_structured_waveformset(path)
    waveforms = [wf.adcs[:max_samples] for wf in wfset.waveforms if wf.channel == channel and len(wf.adcs) >= max_samples]
    return wfset, np.array(waveforms)

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
    cosmic_wfset, cosmic_wfs = load_waveforms(cosmic_path, channel, max_samples)
    _, led_wfs = load_waveforms(led_path, channel, max_samples)
    _, noise_wfs = load_waveforms(noise_path, channel, max_samples)

    mean_cosmic = waffles.WaveformSet.compute_mean_waveform(cosmic_wfset)
    mean_array = mean_cosmic.adcs  
#    mean_led = waffles.WaveformSet.compute_mean_waveform(led_wfs)

    xt = np.arange(max_samples)  # NumPy’s arange creates a 1D array of evenly spaced integers

    # Plotting a single waveform
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Cosmic data", "Mean cosmic"))

        # primo subplot: tutte le waveforms
    for i, wf in enumerate(cosmic_wfs):
        fig.add_trace(
            go.Scatter(x=xt, y=wf, mode="lines",
                       name=f"WF {i}", showlegend=False),
            row=1, col=1
        )

    # secondo subplot: la media di tutte
    mean_wf = cosmic_wfs.mean(axis=0)
    fig.add_trace(
        go.Scatter(x=xt, y=mean_wf, mode="lines",
                   name="Mean Cosmic", line=dict(color='red')),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Sample Index", row=2, col=1)
    fig.update_yaxes(title_text="ADC Counts", row=1, col=1)
    fig.update_yaxes(title_text="ADC Counts", row=2, col=1)
    fig.update_layout(height=600, width=800, title_text="Cosmic Waveforms")
    fig.show()
  
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

    x = np.arange(max_samples)  # NumPy’s arange creates a 1D array of evenly spaced integers
    p0 = [np.max(avg_cosmic), np.max(avg_cosmic)*2, 200, 30, 600, np.argmax(avg_cosmic)-20]
    bounds = ([0, 0, 10, 10, 10, 0], [np.inf, np.inf, 400, 40, 1500, max_samples])

    popt, _ = curve_fit(double_exp_model, x, avg_cosmic, p0=p0, bounds=bounds)  #

    # popt is the array of optimized parameters: the optimizer’s best-guess values for each of the six fitting parameters
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

    popt1, popt2 = compare_fits(x, avg_cosmic)

    # --- funzione di confronto fit --- #
def compare_fits(x, avg_cosmic):
    # Initial guesses and bounds for model_gauss_2exp
    A_fast_init = np.min(avg_cosmic)
    mu_init = x[np.argmin(avg_cosmic)]
    sigma_init = 5
    A_int_init = A_fast_init / 2
    tau_int_init = 50
    A_slow_init = A_fast_init / 3
    tau_slow_init = 200
    x0_init = mu_init - 10

    p0_1 = [A_fast_init, mu_init, sigma_init,
            A_int_init, tau_int_init, A_slow_init, tau_slow_init, x0_init]
    bounds_1 = (
        [-np.inf, x[0], 1, -np.inf, 1, -np.inf, 10, x[0]],
        [0, x[-1], 20, 0, 500, 0, 1000, x[-1]]
    )

    popt1, _ = curve_fit(
        model_gauss_2exp,
        x, avg_cosmic,
        p0=p0_1,
        bounds=bounds_1
    )

    fit1 = model_gauss_2exp(x, *popt1)

    # Initial guesses and bounds for model_3exp
    A1_init = A_fast_init
    tau1_init = 10
    A2_init = A_fast_init / 2
    tau2_init = 50
    A3_init = A_fast_init / 3
    tau3_init = 200
    x0_2_init = x0_init

    p0_2 = [A1_init, tau1_init, A2_init, tau2_init, A3_init, tau3_init, x0_2_init]
    bounds_2 = (
        [-np.inf, 1, -np.inf, 1, -np.inf, 10, x[0]],
        [0, 100, 0, 300, 0, 1000, x[-1]]
    )

    popt2, _ = curve_fit(
        model_3exp,
        x, avg_cosmic,
        p0=p0_2,
        bounds=bounds_2
    )

    fit2 = model_3exp(x, *popt2)

    # --- compute components of fit1 (Gauss + 2exp) ---
    A_fast, mu, sigma = popt1[0:3]
    A_int, tau_int = popt1[3:5]
    A_slow, tau_slow = popt1[5:7]
    x0 = popt1[7]

    mask = x >= x0
    fast_component_1 = np.zeros_like(x)
    intermediate_component_1 = np.zeros_like(x)
    slow_component_1 = np.zeros_like(x)

    fast_component_1[mask] = A_fast * np.exp(- (x[mask] - mu)**2 / (2 * sigma**2))
    intermediate_component_1[mask] = A_int * np.exp(-(x[mask] - x0) / tau_int)
    slow_component_1[mask] = A_slow * np.exp(-(x[mask] - x0) / tau_slow)

    # --- compute components of fit2 (3exp) ---
    A1, tau1 = popt2[0:2]
    A2, tau2 = popt2[2:4]
    A3, tau3 = popt2[4:6]
    x0_2 = popt2[6]

    mask2 = x >= x0_2
    comp1 = np.zeros_like(x)
    comp2 = np.zeros_like(x)
    comp3 = np.zeros_like(x)

    comp1[mask2] = A1 * np.exp(-(x[mask2] - x0_2) / tau1)
    comp2[mask2] = A2 * np.exp(-(x[mask2] - x0_2) / tau2)
    comp3[mask2] = A3 * np.exp(-(x[mask2] - x0_2) / tau3)

    # --- Plotting both models in subplots ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(x, avg_cosmic, label="Average waveform", color='black')
    axs[0].plot(x, fit1, label="Fit: Gauss + 2 exp", color='blue')
    axs[0].plot(x, fast_component_1, '--', label='Fast (Gauss)', color='cyan')
    axs[0].plot(x, intermediate_component_1, '--', label='Intermediate (exp)', color='orange')
    axs[0].plot(x, slow_component_1, '--', label='Slow (exp)', color='green')
    axs[0].set_title("Fit with Gaussian + 2 exponentials")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(x, avg_cosmic, label="Average waveform", color='black')
    axs[1].plot(x, fit2, label="Fit: 3 exponentials", color='red')
    axs[1].plot(x, comp1, '--', label='Fast (exp)', color='cyan')
    axs[1].plot(x, comp2, '--', label='Intermediate (exp)', color='orange')
    axs[1].plot(x, comp3, '--', label='Slow (exp)', color='green')
    axs[1].set_title("Fit with 3 exponentials")
    axs[1].legend()
    axs[1].grid()

    axs[1].set_xlabel("Sample")
    plt.tight_layout()
    plt.show()

    return popt1, popt2

if __name__ == "__main__":
    cosmic_path = "data/cosmic.hdf5"
    led_path = "data/led.hdf5"
    #led_path = "data/SPE36335_ch30.hdf5"
    noise_path = "data/noise.hdf5"
    channel = 30
    process_waveforms(cosmic_path, led_path, noise_path, channel)