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
    y = np.zeros_like(x, dtype=float)
    mask = x >= x0
    t = x[mask] - x0
    # componente fast (gaussiana)
    y[mask] += A_fast * np.exp(- (t - mu)**2 / (2 * sigma**2))
    # componente intermedia (esponenziale)
    y[mask] += A_int * np.exp(- t / tau_int)
    # componente slow (esponenziale)
    y[mask] += A_slow * np.exp(- t / tau_slow)
    return y

# 2) Tre esponenziali con smoothing comune L
def model_3exp(x, A1, tau1, A2, tau2, A3, tau3, L, x0):
    y = np.zeros_like(x, dtype=float)
    mask = x >= x0
    t = x[mask] - x0
    for A, tau in [(A1, tau1), (A2, tau2), (A3, tau3)]:
        y[mask] += A * (np.exp(-t / L) - np.exp(-t / tau))
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
#    _, noise_wfs = load_waveforms(noise_path, channel, max_samples)

    mean_cosmic = waffles.WaveformSet.compute_mean_waveform(cosmic_wfset)
    mean_array = mean_cosmic.adcs  
#    mean_led = waffles.WaveformSet.compute_mean_waveform(led_wfs)

    xt = np.arange(max_samples)  # NumPy’s arange creates a 1D array of evenly spaced integers

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

    min_count = min(len(cosmic_wfs), len(led_wfs))
    cosmic_wfs, led_wfs = cosmic_wfs[:min_count], led_wfs[:min_count]
#    min_count = min(len(cosmic_wfs), len(led_wfs), len(noise_wfs))
#    cosmic_wfs, led_wfs, noise_wfs = cosmic_wfs[:min_count], led_wfs[:min_count], noise_wfs[:min_count]

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

def compare_fits(x, avg_cosmic):

    # --- 1) Fit Gaus + 2 Exp --- #
    i0 = int(np.argmax(avg_cosmic))

    A_fast_init = np.max(avg_cosmic)
    mu_init = 5
#    mu_init = int(np.argmax(avg_cosmic))
    sigma_init = 2 #it was 5
    A_int_init = A_fast_init / 2
    tau_int_init = 200
    A_slow_init = A_fast_init / 4
    tau_slow_init = 600
    x0_init = mu_init - 10
    p0_1 = [
        A_fast_init,         
        mu_init,             
        sigma_init,          
        A_int_init,
        tau_int_init,
        A_slow_init,
        tau_slow_init,
        i0
    ]
    bounds_1 = (
        [0,   0, 1,   0, 10,   0, 10,   i0-5],
        [np.inf, 20, 20, np.inf, 500, np.inf, 2000, i0+5]
    )
    popt1, pcov1 = curve_fit(
        model_gauss_2exp, x, avg_cosmic,
        p0=p0_1, bounds=bounds_1
    )
    perr1 = np.sqrt(np.diag(pcov1))
    fit1 = model_gauss_2exp(x, *popt1)

    # pull out your trigger and time‐axis
    x0_2 = popt1[7]
    x   = np.arange(len(avg_cosmic))

    # define windows relative to trigger x0_2
    fast_win    = (x >= x0_2) & (x <  x0_2 + 30)
    inter_win   = (x >= x0_2+30) & (x <  x0_2 +150)
    slow_win    =  x >= x0_2 +150

    # compute *absolute* areas so we get positive amplitudes
    area_fast  = abs(np.trapz(avg_cosmic[ fast_win ], x[ fast_win ]))
    area_inter = abs(np.trapz(avg_cosmic[ inter_win ], x[ inter_win ]))
    area_slow  = abs(np.trapz(avg_cosmic[ slow_win ],  x[ slow_win  ]))

    # now set p0_2 to those positive proxies
    p0_2 = [
        area_fast,    10,       # A1, τ1 init
        area_inter,  100,       # A2, τ2 init
        area_slow,   600,       # A3, τ3 init
        200,          x0_2      # L, x0
    ]

    # bounds ±50% around those *positive* areas, and small wiggle on x0_2
    eps = 1
    bounds_2 = (
        [0.5*area_fast,   1,   0.5*area_inter,  10,  0.5*area_slow, 100,  10, x0_2 - eps],
        [1.5*area_fast,  50,   1.5*area_inter, 500,  1.5*area_slow,2000, 500, x0_2 + eps]
    )

    print("area_fast, area_inter, area_slow =", area_fast, area_inter, area_slow)
    print("p0_2:", p0_2)
    print("bounds_2 lower:", bounds_2[0])
    print("bounds_2 upper:", bounds_2[1])

    popt2, pcov2 = curve_fit(
        model_3exp, x, avg_cosmic,
        p0=p0_2, bounds=bounds_2,
        max_nfev=10000
    )

    perr2 = np.sqrt(np.diag(pcov2))
    fit2 = model_3exp(x, *popt2)

    # --- compute components of fit1 (Gauss + 2exp) ---
    A_fast, mu_rel, sigma = popt1[0:3]
    A_int, tau_int   = popt1[3:5]
    A_slow, tau_slow = popt1[5:7]
    x0               = popt1[7]

    # prepare masks and shifted time axis
    mask = x >= x0
    t    = x[mask] - x0  # time *after* the trigger

    fast_component_1 = np.zeros_like(x, dtype=float)
    intermediate_component_1 = np.zeros_like(x, dtype=float)
    slow_component_1 = np.zeros_like(x, dtype=float)

    # now mu_rel is how many samples after x0 the Gaussian peaks
    fast_component_1[mask] = A_fast * np.exp(- (t - mu_rel)**2 / (2 * sigma**2))
    intermediate_component_1[mask] = A_int * np.exp(- t / tau_int)
    slow_component_1[mask] = A_slow * np.exp(- t / tau_slow)

    # --- compute components of fit2 (3exp) --- #
    A1, tau1 = popt2[0], popt2[1]
    A2, tau2 = popt2[2], popt2[3]
    A3, tau3 = popt2[4], popt2[5]
    L        = popt2[6]
    x0_2     = popt2[7]

    mask2 = x >= x0_2
    t2    = x[mask2] - x0_2

    comp1 = np.zeros_like(x, dtype=float)
    comp2 = np.zeros_like(x, dtype=float)
    comp3 = np.zeros_like(x, dtype=float)

    # each smoothed exactly as in the model
    comp1[mask2] = A1 * (np.exp(-t2 / L) - np.exp(-t2 / tau1))
    comp2[mask2] = A2 * (np.exp(-t2 / L) - np.exp(-t2 / tau2))
    comp3[mask2] = A3 * (np.exp(-t2 / L) - np.exp(-t2 / tau3))

    # Plots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Fit Gauss+2exp", "Fit 3exp")
    )

    # gauss+2exp
    fig.add_trace(
        go.Scatter(x=x, y=avg_cosmic, mode="lines",
                   name="Avg Cosmic", line=dict(color='black')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=fit1, mode="lines",
                   name="Gauss+2exp", line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=fast_component_1, name="Fast Comp.", line=dict(dash='dash', color='green')), 
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x, y=intermediate_component_1, name="Int Comp.", line=dict(dash='dash', color='orange')), 
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x, y=slow_component_1, name="Slow Comp.", line=dict(dash='dash', color='purple')), 
        row=1, col=1)

    # 3 exp
    fig.add_trace(
        go.Scatter(x=x, y=avg_cosmic, mode="lines",
                   name="Avg Cosmic", line=dict(color='black'),
                   showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=fit2, mode="lines",
                   name="3exp", line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=comp1, name="Fast Comp.", line=dict(dash='dash', color='green')), 
        row=2, col=1)
    fig.add_trace(
        go.Scatter(x=x, y=comp2, name="Int Comp.", line=dict(dash='dash', color='orange')), 
        row=2, col=1)
    fig.add_trace(
        go.Scatter(x=x, y=comp3, name="Slow Comp.", line=dict(dash='dash', color='purple')), 
        row=2, col=1)

    fig.update_xaxes(title_text="Samples 16ns/sample", row=2, col=1)
    fig.update_yaxes(title_text="ADC Counts", row=1, col=1)
    fig.update_yaxes(title_text="ADC Counts", row=2, col=1)
    fig.update_layout(
        height=700, width=800,
        title="Gaussiana+2exp vs 3exp",
        legend=dict(
                x=0.98, y=0.98,
                xanchor='right', yanchor='top'
            )
    )

        # --- Parameter‐box annotations --- #
    # (make sure perr1 and perr2 are defined via np.sqrt(np.diag(pcov1/2)))
    text1 = (
        f"x0 = {popt1[7]:.1f}±{perr1[7]:.1f}<br>"
        f"A_fast = {popt1[0]:.1f}±{perr1[0]:.1f}<br>"
        f"μ = {popt1[1]:.1f}±{perr1[1]:.1f}<br>"
        f"σ = {popt1[2]:.1f}±{perr1[2]:.1f}<br>"
        f"τ_int = {popt1[4]:.1f}±{perr1[4]:.1f}<br>"
        f"τ_slow = {popt1[6]:.1f}±{perr1[6]:.1f}"
    )
    text2 = (
        f"x0 = {popt2[6]:.1f}±{perr2[6]:.1f}<br>"
        f"τ₁ = {popt2[1]:.1f}±{perr2[1]:.1f}<br>"
        f"τ₂ = {popt2[3]:.1f}±{perr2[3]:.1f}<br>"
        f"τ₃ = {popt2[5]:.1f}±{perr2[5]:.1f}"
    )

    # 1) Centered in the *first* subplot
    fig.add_annotation(
        dict(
            x=0.5, y=0.5,                  # center of the domain
            xref='x domain', yref='y domain',
            text=text1,
            showarrow=False,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
            borderwidth=1,
            align='center'
        ),
        row=1, col=1
    )

    # 2) Centered in the *second* subplot
    fig.add_annotation(
        dict(
            x=0.5, y=0.5,                 
            xref='x2 domain', yref='y2 domain',
            text=text2,
            showarrow=False,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
            borderwidth=1,
            align='center'
        ),
        row=2, col=1
    )

    fig.show()

    return popt1, popt2

if __name__ == "__main__":
    cosmic_path = "data/cosmic.hdf5"
    #led_path = "data/led.hdf5"
    led_path = "data/SPE36335_ch30.hdf5"
    noise_path = "data/noise.hdf5"
    channel = 30
    process_waveforms(cosmic_path, led_path, noise_path, channel)