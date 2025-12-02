import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.signal import welch as psd
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import pandas as pd
from scipy.fft import fft, ifft

def scintillation(t, A_s, A_t, tau_s, tau_t):
    Is = (A_s) * np.exp(-(t)/tau_s)
    It = (A_t) * np.exp(-(t)/tau_t)
    #Ir = (A_r) * np.exp(-(t)/tau_r)
    #Ir = (A_r)/((1+(t)/tau_r)**2)         
    return Is + It

def scintillation_norm(t, A_s, A_t, tau_s, tau_t):
    """Compute the scintillation function and normalize it."""
    Is = A_s * np.exp(-t / tau_s)
    It = A_t * np.exp(-t / tau_t)
    scint = Is + It
    return scint / np.max(scint)  # Normalize to peak value of 1

def convolution(t, A_s, A_t, tau_s, tau_t, offset, template): # doesn't work properly
    scint = scintillation(t, A_s, A_t, tau_s, tau_t)
    return np.fft.ifft(np.fft.fft(template) * np.fft.fft(scint)).real + offset

def direct_convolution(t, A_s, A_t, tau_s, tau_t, offset, template, dt):
    scint = scintillation(t, A_s, A_t, tau_s, tau_t)
    conv_result = np.convolve(template, scint, mode='full')[:len(template)] * dt  

    return conv_result + offset

def fft_convolution(t, A_s, A_t, tau_s, tau_t, offset, template, dt):
    """Perform convolution using FFT (faster than direct summation)."""
    # Compute the scintillation function
    scint = scintillation(t, A_s, A_t, tau_s, tau_t)

    # Compute FFT convolution (zero-padding ensures correct length)
    N = len(template) + len(scint) - 1
    conv_result = ifft(fft(template, N) * fft(scint, N)).real[:len(template)]  # Keep original length

    return conv_result * dt + offset  # Scale by dt

#######################################################################################################

def fft_convolution_norm(t, A_s, A_t, tau_s, tau_t, offset, template, dt):
    """Perform convolution using FFT after normalizing scintillation function."""
    # Compute and normalize scintillation function
    scint = scintillation_norm(t, A_s, A_t, tau_s, tau_t)
    
    # Compute FFT convolution (zero-padding ensures correct length)
    N = len(template) + len(scint) - 1
    conv_result = ifft(fft(template, N) * fft(scint, N)).real[:len(template)]  # Keep original length

    # Rescale convolution result properly
    norm_factor = A_s * tau_s + A_t * tau_t  # Rescale using the original integral
    return (conv_result * norm_factor * dt) + offset  # Ensure physical consistency

def conv_fit(signal, template, end_time, debug = False):
    
    tau_s_guess = 7
    tau_t_guess = 1400

    A_t_guess = 1
    A_s_guess = 1

    begin, end = 0, end_time#600#900
    data_tofit, time = signal[begin:end], 16 * np.arange(begin, end) 
    offset_guess = 0

    params_list = ['A_s', 'A_t', 'tau_s_guess', 'tau_t_guess', 'offset']
    step = 1
    while step <= 2:

        if step == 1:
            guess  = ( A_s_guess, A_t_guess, offset_guess)
            bounds = ((0, 0, 0),
                     ( np.inf,  np.inf, np.inf)) 
            params, covariance = curve_fit(lambda time, A_s, A_t, offset: 
                                convolution(time, A_s, A_t, tau_s_guess, tau_t_guess, offset, template[begin:end]), 
                                time, data_tofit, p0 = guess, bounds=bounds)
            errors = np.sqrt(np.diag(covariance))
            if debug:
                print('First aproximation of As, At and offset:')
                for i, p in enumerate(params):
                    print( f' {p} +- {errors[i]}')
            
        if step == 2:
            guess  = (params[0], params[1], tau_s_guess, tau_t_guess, params[2])
            bounds = ((0, 0, 2, 1000, params[2]-errors[2]),
                      (params[0]+errors[0], params[1]+errors[1], 15, 1800, params[2]+errors[2]))

            params, covariance = curve_fit(lambda time, A_s, A_t, tau_s, tau_t, offset: 
                                convolution(time, A_s, A_t, tau_s, tau_t, offset, template[begin:end]), 
                                time, data_tofit, p0 = guess, bounds = bounds, maxfev=1000)
            errors = np.sqrt(np.diag(covariance))

            fitted_data = convolution(time, *params, template[begin:end])
            residuals = data_tofit - fitted_data
            ss_tot = np.sum((data_tofit - np.mean(data_tofit)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            if debug:
                print('Second aproximation:')
                for i, p in enumerate(params):
                    print( f'{params_list[i]} = {p} +- {errors[i]}')
                print(f' R² = {r_squared}')
                    
        step += 1
                      
    return params, errors

def conv_fit_v2(signal, template, end_time, debug = False):
    
    tau_s_guess = 7
    tau_t_guess = 1400

    A_t_guess = 1
    A_s_guess = 1

    begin, end = 0, end_time#600#900
    data_tofit, time = signal[begin:end], 16 * np.arange(begin, end) 
    template = template[begin:end]

    # Create new time array with 4x more points
    new_time = np.linspace(time[0], time[-1], int(16 * len(time)))
    
    # Interpolation functions (use cubic for smoothness)
    data_interp_func = interp1d(time, data_tofit, kind='cubic')
    template_interp_func = interp1d(time, template, kind='cubic')
    
    # Get upsampled data
    data_tofit_upsampled = data_interp_func(new_time)
    template_upsampled = template_interp_func(new_time)

    offset_guess = 0

    params_list = ['A_s', 'A_t', 'tau_s_guess', 'tau_t_guess', 'offset']
    step = 1
    while step <= 2:

        if step == 1:
            guess  = ( A_s_guess, A_t_guess, offset_guess)
            bounds = ((0, 0, 0),
                     ( np.inf,  np.inf, np.inf)) 
            params, covariance = curve_fit(lambda new_time, A_s, A_t, offset: 
                                convolution(new_time, A_s, A_t, tau_s_guess, tau_t_guess, offset, template_upsampled), 
                                new_time, data_tofit_upsampled, p0 = guess, bounds=bounds)
            errors = np.sqrt(np.diag(covariance))
            if debug:
                print('First aproximation of As, At and offset:')
                for i, p in enumerate(params):
                    print( f' {p} +- {errors[i]}')
            
        if step == 2:
            guess  = (params[0], params[1], tau_s_guess, tau_t_guess, params[2])
            bounds = ((0, 0, 2, 1000, params[2]-errors[2]),
                      (params[0]+errors[0], params[1]+errors[1], 15, 1800, params[2]+errors[2]))

            params, covariance = curve_fit(lambda new_time, A_s, A_t, tau_s, tau_t, offset: 
                                convolution(new_time, A_s, A_t, tau_s, tau_t, offset, template_upsampled), 
                                new_time, data_tofit_upsampled, p0 = guess, bounds = bounds, maxfev=1000)
            errors = np.sqrt(np.diag(covariance))

            fitted_data = convolution(new_time, *params, template_upsampled)
            residuals = data_tofit_upsampled - fitted_data
            ss_tot = np.sum((data_tofit_upsampled - np.mean(data_tofit_upsampled)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            if debug:
                print('Second aproximation:')
                for i, p in enumerate(params):
                    print( f'{params_list[i]} = {p} +- {errors[i]}')
                print(f' R² = {r_squared}')
                    
        step += 1

    if debug:
        plt.plot(new_time,data_tofit_upsampled/np.max(data_tofit_upsampled), marker='.', label = 'Signal')
        plt.plot(new_time,template_upsampled/np.max(template_upsampled), label = 'SPE')
        plt.plot(new_time,fitted_data/np.max(fitted_data), label = 'Fit')
        plt.legend()
        plt.show()
    return params, errors, r_squared

def conv_fit_v3(signal, template, end_time, factor=100, debug=False):
    
    tau_s_guess = 7
    tau_t_guess = 1400
    A_s_guess = 1
    A_t_guess = 1

    begin, end = 0, end_time
    data_tofit, time = signal[begin:end], 16 * np.arange(begin, end)
    template = template[begin:end]

    # Upsampling
    new_time = np.linspace(time[0], time[-1], int(factor * len(time)))
    dt = (new_time[1] - new_time[0])  # Time step after upsampling

    # Interpolation
    data_interp = interp1d(time, data_tofit, kind='linear')
    template_interp = interp1d(time, template, kind='linear')

    data_tofit_upsampled = data_interp(new_time)
    template_upsampled = template_interp(new_time)

    offset_guess = 0
    params_list = ['A_s', 'A_t', 'tau_s', 'tau_t', 'offset']
    
    step = 1
    while step <= 2:

        if step == 1:
            guess = (A_s_guess, A_t_guess, offset_guess)
            bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))

            params, covariance = curve_fit(lambda new_time, A_s, A_t, offset: 
                                           direct_convolution(new_time, A_s, A_t, tau_s_guess, tau_t_guess, offset, template_upsampled, dt), 
                                           new_time, data_tofit_upsampled, p0=guess, bounds=bounds)
            errors = np.sqrt(np.diag(covariance))
            if debug:
                print('First approximation of As, At, and offset:')
                for i, p in enumerate(params):
                    print(f' {p} +- {errors[i]}')
            
        if step == 2:
            guess = (params[0], params[1], tau_s_guess, tau_t_guess, params[2])
            bounds = ((0, 0, 2, 1000, params[2]-errors[2]),
                      (params[0]+errors[0], params[1]+errors[1], 15, 1800, params[2]+errors[2]))

            params, covariance = curve_fit(lambda new_time, A_s, A_t, tau_s, tau_t, offset: 
                                           direct_convolution(new_time, A_s, A_t, tau_s, tau_t, offset, template_upsampled, dt), 
                                           new_time, data_tofit_upsampled, p0=guess, bounds=bounds)
            errors = np.sqrt(np.diag(covariance))

            fitted_data = direct_convolution(new_time, *params, template_upsampled, dt)
            residuals = data_tofit_upsampled - fitted_data
            ss_tot = np.sum((data_tofit_upsampled - np.mean(data_tofit_upsampled)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Properly scale photons
            photons = (params[0] * params[2] + params[1] * params[3]) #* dt  # Scale by dt
            Ist = params[0] * params[2]/(params[1] * params[3])
            if debug:
                print('Second approximation:')
                for i, p in enumerate(params):
                    print(f'{params_list[i]} = {p} +- {errors[i]}')
                print(f'Ist = {Ist}')
                print(f'Photons = {photons}')
                print(f'R² = {r_squared}')

                # Debugging: Check fit
                plt.figure(figsize=(10,5))
                plt.plot(new_time, data_tofit_upsampled, 'k-', label='Signal (Interpolated)')
                plt.plot(new_time, fitted_data, 'r--', label='Direct Convolution Fit')
                plt.legend()
                plt.title("Direct Convolution Fit")
                plt.show()
                    
        step += 1

    if debug:
        plt.figure(figsize=(10,5))
        plt.plot(new_time, data_tofit_upsampled/np.max(data_tofit_upsampled), marker='.', label='Signal (Normalized)')
        plt.plot(new_time, template_upsampled/np.max(template_upsampled), label='SPE (Normalized)')
        plt.plot(new_time, fitted_data/np.max(fitted_data), label='Direct Fit')
        plt.legend()
        plt.show()

    return params, errors

def conv_fit_v4(signal, template, end_time, factor=3, debug=False):
    
    tau_s_guess = 7
    tau_t_guess = 1400
    A_s_guess = 1
    A_t_guess = 1

    begin, end = 0, end_time
    data_tofit, time = signal[begin:end], 16 * np.arange(begin, end)
    template = template[begin:end]

    # Upsampling
    new_time = np.linspace(time[0], time[-1], int(factor * len(time)))
    dt = (new_time[1] - new_time[0])  # Time step after upsampling

    # Interpolation
    data_interp = interp1d(time, data_tofit, kind='linear')
    template_interp = interp1d(time, template, kind='linear')

    data_tofit_upsampled = data_interp(new_time)
    template_upsampled = template_interp(new_time)

    offset_guess = 0
    params_list = ['A_s', 'A_t', 'tau_s', 'tau_t', 'offset']
    
    step = 1
    while step <= 2:

        if step == 1:
            guess = (A_s_guess, A_t_guess, offset_guess)
            bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))

            params, covariance = curve_fit(lambda new_time, A_s, A_t, offset: 
                                           fft_convolution(new_time, A_s, A_t, tau_s_guess, tau_t_guess, offset, template_upsampled, dt), 
                                           new_time, data_tofit_upsampled, p0=guess, bounds=bounds)
            errors = np.sqrt(np.diag(covariance))
            if debug:
                print('First approximation of As, At, and offset:')
                for i, p in enumerate(params):
                    print(f' {p} +- {errors[i]}')
            
        if step == 2:
            guess = (params[0], params[1], tau_s_guess, tau_t_guess, params[2])
            bounds = ((0, 0, 2, 1000, params[2]-errors[2]),
                      (params[0]+errors[0], params[1]+errors[1], 15, 1800, params[2]+errors[2]))

            params, covariance = curve_fit(lambda new_time, A_s, A_t, tau_s, tau_t, offset: 
                                           fft_convolution(new_time, A_s, A_t, tau_s, tau_t, offset, template_upsampled, dt), 
                                           new_time, data_tofit_upsampled, p0=guess, bounds=bounds)
            errors = np.sqrt(np.diag(covariance))

            fitted_data = fft_convolution(new_time, *params, template_upsampled, dt)
            residuals = data_tofit_upsampled - fitted_data
            ss_tot = np.sum((data_tofit_upsampled - np.mean(data_tofit_upsampled)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Properly scale photons
            photons = (params[0] * params[2] + params[1] * params[3]) #* dt  # Scale by dt
            Ist = params[0] * params[2]/(params[1] * params[3])
            if debug:
                print('Second approximation:')
                for i, p in enumerate(params):
                    print(f'{params_list[i]} = {p} +- {errors[i]}')
                print(f'Ist = {Ist}')
                print(f'Photons = {photons}')
                print(f'R² = {r_squared}')

                # Debugging: Check fit
                plt.figure(figsize=(10,5))
                plt.plot(new_time, data_tofit_upsampled, 'k-', label='Signal (Interpolated)')
                plt.plot(new_time, fitted_data, 'r--', label='Direct Convolution Fit')
                plt.legend()
                plt.title("Direct Convolution Fit")
                plt.show()
                    
        step += 1

    if debug:
        plt.figure(figsize=(6,4))
        plt.plot(new_time, data_tofit_upsampled/np.max(data_tofit_upsampled), marker='.', label='Signal (Normalized)')
        plt.plot(new_time, template_upsampled/np.max(template_upsampled), label='SPE (Normalized)')
        plt.plot(new_time, fitted_data/np.max(fitted_data), label='Direct Fit')
        plt.legend()
        plt.show()

    return params, errors

def conv_fit_v5(signal, template, peak_lim, end_time, factor=2.5, debug=False):
    
    tau_s_guess = 7
    tau_t_guess = 1400
    A_s_guess = 1
    A_t_guess = 1

    begin, end = 0, end_time
    data_tofit, time = signal[begin:end], 16 * np.arange(begin, end)
    template = template[begin:end]

    data_norm_factor = np.max(data_tofit[:peak_lim])
    template_norm_factor = np.max(template)

    # Upsampling
    new_time = np.linspace(time[0], time[-1], int(factor * len(time)))
    dt = (new_time[1] - new_time[0])  # Time step after upsampling

    # Interpolation
    data_interp = interp1d(time, data_tofit, kind='cubic', fill_value="extrapolate")
    template_interp = interp1d(time, template, kind='cubic', fill_value="extrapolate")

    
    data_tofit_upsampled = data_interp(new_time)
    template_upsampled = template_interp(new_time)

    # Normalize both signal and template before fitting
    data_tofit_upsampled_norm = data_tofit_upsampled/data_norm_factor
    template_upsampled_norm = template_upsampled/template_norm_factor

    offset_guess = 0
    params_list = ['A_s', 'A_t', 'tau_s', 'tau_t', 'offset']
    
    step = 1
    while step <= 3:

        if step == 1:
            guess = (A_s_guess, A_t_guess, offset_guess)
            bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))

            params, covariance = curve_fit(lambda new_time, A_s, A_t, offset: 
                                           fft_convolution_norm(new_time, A_s, A_t, tau_s_guess, tau_t_guess, offset, template_upsampled_norm, dt), 
                                           new_time, data_tofit_upsampled_norm, p0=guess, bounds=bounds)
            errors = np.sqrt(np.diag(covariance))
            if debug:
                print('First approximation of As, At, and offset:')
                for i, p in enumerate(params):
                    print(f' {p} +- {errors[i]}')
          
            
        if step == 2:
            guess = (params[0], params[1], tau_s_guess, tau_t_guess, params[2])
            bounds = ((params[0]-errors[0], 0, 2, 1000, params[2]-errors[2]),
                      (params[0]+errors[0], params[1]+errors[1], 15, 1800, params[2]+errors[2]))

            params, covariance = curve_fit(lambda new_time, A_s, A_t, tau_s, tau_t, offset: 
                                           fft_convolution_norm(new_time, A_s, A_t, tau_s, tau_t, offset, template_upsampled_norm, dt), 
                                           new_time, data_tofit_upsampled_norm, p0=guess, bounds=bounds)
            errors = np.sqrt(np.diag(covariance))

            if debug:
                print('Second approximation of As, At, and offset:')
                for i, p in enumerate(params):
                    print(f' {p} +- {errors[i]}')

        if step == 3:
            if round(errors[2],2) == 0: errors[2] = 0.1
            if round(errors[3],2) == 0: errors[3] = 0.1
            if round(errors[4],2) == 0: errors[4] = 0.1

            if params[2] == 2: params[2] = 3
            if params[3] == 1000: params[3] = 1001

            if params[2] == 15: params[2] = 14
            if params[3] == 1800: params[3] = 1799
                
            
            min_taus = params[2]-errors[2]/10
            if min_taus<2 or min_taus>=15: 
                min_taus  = 2
    
            min_taut = params[3]-errors[3]/10
            if min_taut<1000 or min_taut >= 1800: 
                min_taut = 1000

            max_taus = params[2]+errors[2]/10
            if max_taus>15: 
                max_taus  = 15
                
            max_taut = params[3]+errors[3]/10
            if max_taut>1800: 
                max_taut = 1800

            guess = (A_s_guess, A_t_guess, params[2], params[3], params[4])
            bounds = ((0, 0, min_taus, min_taut, params[4]-errors[4]),
                      (np.inf, np.inf, max_taus, max_taut, params[4]+errors[4]))

            params, covariance = curve_fit(lambda new_time, A_s, A_t, tau_s, tau_t, offset: 
                                           fft_convolution(new_time, A_s, A_t, tau_s, tau_t, offset, template_upsampled, dt), 
                                           new_time, data_tofit_upsampled, p0=guess, bounds=bounds)
            errors = np.sqrt(np.diag(covariance))
            
            fitted_data = fft_convolution(new_time, *params, template_upsampled, dt)
            print(list(fitted_data))
            residuals = data_tofit_upsampled - fitted_data
            ss_tot = np.sum((data_tofit_upsampled - np.mean(data_tofit_upsampled)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Compute the correct photon count
            photons = (params[0] * params[2] + params[1] * params[3])
            Ist = params[0] * params[2] / (params[1] * params[3])
            e_I  = Ist*np.sqrt((errors[0]/params[0])**2 + (errors[1]/params[1])**2 +(errors[2]/params[2])**2 + (errors[3]/ params[3])**2)
            
            if debug:
                print('Third approximation:')
                for i, p in enumerate(params):
                    print(f'{params_list[i]} = {p} +- {errors[i]}')
                print(f'Ist = {Ist} +- {e_I}')
                print(f'Photons = {photons}')
                print(f'R² = {r_squared}')

                # Debugging: Check fit
                plt.figure(figsize=(6,4))
                plt.plot(new_time, data_tofit_upsampled, 'k-', label='Signal')
                plt.plot(new_time, fitted_data, 'r--', label='Convolution Fit')
                plt.legend()
                plt.title("Direct Convolution Fit")
                plt.show()
                    
        step += 1

    if debug:
        plt.figure(figsize=(6,4))
        plt.plot(new_time, data_tofit_upsampled_norm, marker='.', label='Signal (Normalized)')
        plt.plot(new_time, template_upsampled_norm, label='SPE (Normalized)')
        plt.plot(new_time, fitted_data/np.max(fitted_data), label='Direct Fit')
        plt.ylabel('Normalized Amplitude')
        plt.xlabel('Time (ns)')
        plt.xlim(new_time[0], new_time[len(new_time)-1])
        plt.legend()
        plt.show()

    return params, errors, r_squared

####################################################################################################



# Shifts the template by a given position
def shift_template(template, pos):
    shifted_template = np.zeros(len(template))
    noise = np.std(template[:50])
    baseline = np.mean(template[:50])
    
    template_peak = np.argmax(template)
    delta = pos - template_peak
    
    if delta >= 0 and delta < len(template):  # Ensure delta does not exceed template length
        template_cropped = template[:len(template) - delta]
        fill_to = len(template) - len(template_cropped)
        random_values = np.random.randint(-1, 2, size=fill_to)  # Generate [-1, 1] random values
        shifted_template[:fill_to] = baseline + random_values * noise
        shifted_template[fill_to:] = template_cropped[:len(template) - delta]
    else:
        shifted_template[:] = template  # If delta is out of bounds, use the original template
    
    return shifted_template

# Multi-signal generator function A
def multi_signalA(t, pulse, A_s, A_t, tau_s, tau_t, offset, template):
    return pulse + convolution(t, A_s, A_t, tau_s, tau_t, offset, template)

# Multi-signal generator function B
def multi_signalB(t, p, template_list):
    pulse = np.zeros(len(template))
    offset = p[0]
    tau_s  = p[1]
    tau_t  = p[2]
    amp    = p[3:]

    for i in range(0, len(amp), 2):
        A_s = amp[i]
        A_t = amp[i+1]
        pulse += convolution(t, A_s, A_t, tau_s, tau_t, offset, template_list[i // 2])
        
    return pulse

# The main convolution fitting function
def multi_convB(signal, template, peak_pos, degree, debug = False):
    
    tau_s_guess = 7
    tau_t_guess = 1400
    A_s_guess = 1
    A_t_guess = 0

    upper_limit = [pos-10 for i, pos in enumerate(peak_pos) if i > 0]
    time = 16 * np.arange(len(template))
    offset_guess = 0

    guess  = [A_s_guess, A_t_guess]
    bounds = ((0, 0), (np.inf, np.inf))
    
    begin = 0 
    params_list = []
    error_list = []

    step = 1
    amplitude = []
    amplitude_errors = []
    template_list = []
    
    while step <= 2:

        if step == 1:
    
            for i, pos in enumerate(peak_pos):
                
                end = upper_limit[i] if i < len(upper_limit) else len(template)      
                peak = peak_pos[:i+1]
                template_ = shift_template(template, pos)
                    
                if i == 0:
                    params, errors = conv_fit(signal[50:], template[50:], debug=False)
                    pulse = convolution(time, *params, template)
                    A_s, A_t, tau_s, tau_t, offset = params
                    eA_s, eA_t, etau_s, etau_t, eoffset = errors
                    amplitude.append(A_s)
                    amplitude.append(A_t)
                    amplitude_errors.append(errors[0])
                    amplitude_errors.append(errors[1])
                    template_list.append(template_)
                else:
                    params, covariance = curve_fit(
                        lambda t, *p: multi_signalA(t, pulse[begin:end], p[0], p[1], tau_s, tau_t, offset, template_[begin:end]),
                        time[begin:end], signal[begin:end],
                        p0=guess, bounds=bounds, maxfev=10000)
                    errors = np.sqrt(np.diag(covariance))
                    pulse = multi_signalA(time, pulse, params[0], params[1], tau_s, tau_t, offset, template_)

                    Ist = params[0]*tau_s/(params[1]*tau_t)

                    #if Ist < 1:
                    amplitude.append(params[0])
                    amplitude.append(params[1])
                    amplitude_errors.append(errors[0])
                    amplitude_errors.append(errors[1])
                    template_list.append(template_)
                

            if degree == 1:
                params = [offset, tau_s, tau_t, *amplitude]
                errors  = [eoffset, etau_s, etau_t, *amplitude_errors]

                return params, errors
        
        if step == 2:

            guess = [offset, tau_s, tau_t, *amplitude]

            if tau_s - etau_s < 2:
                min_taus = 2
            else:
                min_taus = tau_s - etau_s

            if tau_s + etau_s > 15:
                max_taus = 15
            else:
                max_taus = tau_s + etau_s

            if tau_t - etau_t < 1000:
                min_taut = 1000
            else:
                min_taut = tau_t - etau_t

            if tau_t + etau_t > 1800:
                max_taut = 1800
            else:
                max_taut = tau_t + etau_t
            
            min_bound = np.zeros(len(amplitude_errors))
            max_bound = np.array(amplitude) + np.array(amplitude_errors)
            bounds = ((-np.inf, min_taus, min_taut, *min_bound), (np.inf, max_taus, max_taut, *max_bound)) 
            
            params, covariance = curve_fit(
                lambda t, *p: multi_signalB(t, p, template_list),
                time, signal,
                p0=guess, bounds=bounds, maxfev=10000)
            errors = np.sqrt(np.diag(covariance))
            pulse = multi_signalB(time, params, template_list)
            residuals = signal - pulse
            ss_tot = np.sum((signal - np.mean(signal)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            if degree == 2:
                return params, errors
                 
        step += 1
        
        
    #return params, errors
def dec_gauss(f, f0, fc):
    return np.exp(-0.5*((f-f0)/fc)**2)

def deconvolution(signal, template, noise = None, filter_type = 'Gauss', limits = [6.25*(10**6),18.75*(10**6)],sample_rate = 62.5*(10**6), cutoff_rate = 10*(10**6), isplot = False):
    
    size   = len(template)
    signal = signal[:size]

    fft_signal   = np.fft.rfft(signal)
    fft_template = np.fft.rfft(template)
    frequencies  = np.fft.rfftfreq(size,1/sample_rate)

    deconv_signal = fft_signal/fft_template
    
    if noise is not None:
        noise     = noise[:size]
        fft_noise = np.fft.rfft(noise)
        wiener    = abs(fft_template)**2/(abs(fft_template)**2+abs(fft_noise)**2)

        if filter_type == 'Gauss':
            try:
                #min_index = np.argwhere(frequencies == limits[0])[0][0]
                #max_index = np.argwhere(frequencies == limits[1])[0][0]
                #param, covariance = curve_fit(dec_gauss, frequencies[min_index:max_index], wiener[min_index:max_index], p0 = (1,cutoff_rate))
            
                param, covariance = curve_fit(dec_gauss, frequencies, wiener, p0 = (1,cutoff_rate))
                signal_filter = dec_gauss(frequencies, 0, cutoff_rate)
                signal_filter[0] = 0
                label = 'Gauss Filter'
            except:
                print('Filter Method: Wiener')
                signal_filter = wiener
                label = 'Wiener Filter'

        if filter_type == 'Wiener':
            signal_filter =  wiener

        if isplot:
            f_signal, P_signal     = psd(signal, fs = sample_rate, nperseg=size)
            f_template, P_template = psd(template, fs = sample_rate, nperseg=size)
            f_noise, P_noise       = psd(noise, fs = sample_rate, nperseg=size)

            signal_trace   = go.Scatter(x=f_signal, y=P_signal, mode='lines', name='Signal')
            template_trace = go.Scatter(x=f_template, y=P_template, mode='lines', name='Template')
            noise_trace    = go.Scatter(x=f_noise, y=P_noise, mode='lines', name='Noise')
            wiener_trace   = go.Scatter(x=frequencies, y=wiener, mode='lines', name='Wiener')
            filter_trace   = go.Scatter(x=frequencies, y=signal_filter, mode='lines', name=filter_type)
            
            # Create the figure
            fig = go.Figure()
            
            # Add the traces to the figure
            fig.add_trace(signal_trace)
            fig.add_trace(template_trace)
            fig.add_trace(noise_trace)
            fig.add_trace(wiener_trace)
            fig.add_trace(filter_trace)
            
            # Update the layout
            fig.update_layout(title = 'Power Spectrum Density (PSD)',
                              xaxis = dict(title='Frequency (Hz)',type='log'),
                              yaxis = dict(title='PSD',type='log'),
                              legend= dict(x=0.02,y=0.98),
                              template='plotly_white')

            # Show the plot
            fig.show()
            
        return np.fft.irfft(signal_filter*deconv_signal)
        
    else:
        return np.fft.irfft(deconv_signal)

def dec_fit_FastSlow(t, As, taus, Af, tauf, sigma, t0, offset):
    return (As / np.sqrt(2)) * np.exp((sigma**2) / (2 * taus**2)) * erfc(((t0 - t) / sigma) + (sigma / taus)) * np.exp((t0 - t) / taus) + \
           (Af / np.sqrt(2)) * np.exp((sigma**2) / (2 * tauf**2)) * erfc(((t0 - t) / sigma) + (sigma / tauf)) * np.exp((t0 - t) / tauf) - offset

def dec_fit_FastSlowIntermediate(t, As, taus, Ai, taui, Af, tauf, sigma, t0, offset):
    return (As / np.sqrt(2)) * np.exp((sigma**2) / (2 * taus**2)) * erfc(((t0 - t) / sigma) + (sigma / taus)) * np.exp((t0 - t) / taus) + \
           (Ai / np.sqrt(2)) * np.exp((sigma**2) / (2 * taui**2)) * erfc(((t0 - t) / sigma) + (sigma / taui)) * np.exp((t0 - t) / taui) + \
           (Af / np.sqrt(2)) * np.exp((sigma**2) / (2 * tauf**2)) * erfc(((t0 - t) / sigma) + (sigma / tauf)) * np.exp((t0 - t) / tauf) - offset

# Error function
def func_error(error_matrix):
    N = len(error_matrix)
    return np.sqrt(np.sum(error_matrix * error_matrix, axis=0)) / N

def dec_fit(dec_signal, original_signal, show_parameters = False):

    tau_fast_guess = 1500
    tau_intr_guess = 50
    tau_slow_guess = 7
    sigma_guess    = 10
    offset_guess   = 0
    Af_guess       = 0.7*np.max(dec_signal)
    As_guess       = 0.3*np.max(dec_signal)

    begin, end       = 50, 2* len(dec_signal) // 3
    data_tofit, time = dec_signal[begin:end], 16 * np.arange(begin, end) #ns
    t0_guess         = time[np.argmax(data_tofit)]
    
    try:
        #initial_guess      = (0.1, tau_fast_guess, np.max(original_signal), tau_slow_guess, sigma_guess, t0_guess, offset_guess)
        
        initial_guess = (Af_guess, tau_fast_guess, As_guess, tau_slow_guess, sigma_guess, t0_guess, offset_guess)
        bounds        = ((0, 1100, 0, 0, 0, 0, -np.inf), 
                         (np.inf, 1900, np.inf, np.inf, np.inf, np.inf, np.inf))
        
        params, covariance = curve_fit(dec_fit_FastSlow, time, data_tofit, p0 = initial_guess, bounds = bounds)
        errors             = np.sqrt(np.diag(covariance))

        if show_parameters:
                print(f'As (ADC) = {params[0]} +- {errors[0]} ({100*params[0]/np.max(dec_signal)} %)')
                print(f'Af (ADC) = {params[2]} +- {errors[2]} ({100*params[2]/np.max(dec_signal)} %)')
                print(f'Tau_Slow (ns) = {params[1]} +- {errors[1]}')
                print(f'Tau_Fast (ns) = {params[3]} +- {errors[3]}')
        
        #if (params[3]-errors[3]) <= 10:# and errors[3] < params[3]:   
        params = [params[0], params[1], 0, np.inf, params[2], params[3], params[4], params[5], params[6]]
        errors = [errors[0], errors[1], 0, 0, errors[2], errors[3], errors[4], errors[5], errors[6]]

        return params, errors

        '''
        if (params[3]-errors[3]) > 10:
            #except:
        #print('Requires tri-decay')

            try:
                #second_guess = (0.1, tau_fast_guess, 1, tau_intr_guess, np.max(dec_signal), tau_slow_guess, sigma_guess , t0_guess, offset_guess)
                second_guess = (Af_guess, tau_fast_guess, 0.1, tau_intr_guess, As_guess, tau_slow_guess, sigma_guess, t0_guess, offset_guess)
                bounds       = ((0, 1100, 0, 11, 0, 2, 10, t0_guess - 100 , -np.inf),
                                (np.inf, 1900, np.inf, 900, np.inf, 10, np.inf,  np.inf , np.inf))
                
                params, covariance = curve_fit(dec_fit_FastSlowIntermediate, time, data_tofit, p0 = second_guess, bounds = bounds)
                errors = np.sqrt(np.diag(covariance))
        
                
                
                if errors[5] > params[5]:
                    third_guess = (0.1, tau_fast_guess, 1, tau_intr_guess, np.max(original_signal), tau_slow_guess, 10 , t0_guess, offset_guess)
                    p, c = curve_fit(dec_fit_FastSlowIntermediate, time, data_tofit, p0=third_guess,
                                                   bounds=((0, tau_fast_guess - 100, 0, 11, 0, 2, 10, t0_guess - 100 , -np.inf),
                                                           (np.inf, tau_fast_guess + 100, np.inf, 900, np.inf, 10, np.inf,  np.inf , np.inf)))
                    e = np.sqrt(np.diag(c))
            
                    if e[5] < p[5] and e[1] < p[1]:
                        params = p
                        errors = e
                
                if show_parameters:
                    print(f'As (ADC) = {params[0]} +- {errors[0]} ({100*params[0]/np.max(dec_signal)} %)')
                    print(f'Af (ADC) = {params[4]} +- {errors[4]} ({100*params[4]/np.max(dec_signal)} %)')
                    print(f'Ai (ADC) = {params[2]} +- {errors[2]} ({100*params[2]/np.max(dec_signal)} %)')
                    print(f'Tau_Slow (ns) = {params[1]} +- {errors[1]}')
                    print(f'Tau_Fast (ns) = {params[5]} +- {errors[5]}')
                    print(f'Tau_Intermediary (ns) = {params[3]} +- {errors[3]}')
        
                return params, errors
                    
            except:
                print('Failed')
            '''
    except:
                print('Failed')

def PID(TC, ToF = None, energy = None):

    TC = [str(item) for item in TC]
    tc_list = [t for t in TC if t != "Type.kCTBBeam"]
    p_list = []

    if ToF:
        if energy == '1GeV' or energy == '-1GeV':
            electron_list = ['Type.kCTBBeamChkvL', 'Type.kCTBBeamChkvHL', 'Type.kCTBBeamChkvHxL'] 
            muon_list = ['Type.kCTBBeamChkvLx', 'Type.kCTBBeamChkvHLx', 'Type.kCTBBeamChkvHxLx']   
            proton_list = ['Type.kCTBBeamChkvLx', 'Type.kCTBBeamChkvHLx', 'Type.kCTBBeamChkvHxLx']
        
            tof_ = ToF           
            for t in tc_list:
                
                if tof_ >= 0 and tof_ <= 105:
                    if t in electron_list:
                        p_list.append('electron')
                    if t in muon_list:
                        p_list.append('muon')
        
                elif tof_ >= 105 and tof_ <= 110 and t in muon_list:
                    p_list.append('muon')
        
                elif tof_ > 110 and tof_ <= 160 and t in proton_list:
                     p_list.append('proton')
        
                else:
                    p_list.append('unknown')
        
            if len(p_list) == 0:
                return(['unknown'])
            if len(p_list) == 1:
                return(p_list[0])
            if len(set(p_list)) == 1:
                return(p_list[0])
            else:
                return(p_list)            
    else:
        if energy == '5GeV' or energy == '-5GeV':
            kaon_list = ['Type.kCTBBeamChkvHLx'] 
            proton_list = ['Type.kCTBBeamChkvHxLx'] 
            muon_list = ['Type.kCTBBeamChkvHL']

            for t in tc_list:
                if t in kaon_list:
                    p_list.append('kaon')
                if t in proton_list:
                    p_list.append('proton')
                if t in muon_list:
                    p_list.append('muon')
                
            if len(p_list) == 0:
                return(['unknown'])
            else:
                return(p_list[0])

def gauss(x, a, b, c):#, j, k, l):
     return a*np.exp(-(x-b)**2/(2*(c**2)))

def gauss_fit(count, bins, nbin):
    
    ref = max(nbin // 8, 1)  # Ensure ref is at least 1 to avoid zero range
    i0 = np.argmax(count)

    imin = max(i0 - ref, 0)  # Prevent out-of-bounds indexing
    imax = min(i0 + ref, len(count))  # Prevent exceeding array size

    x = np.linspace(min(bins), max(bins), 1000)
    #x = np.linspace(bins[imin], bins[imax], 1000)
    count = count[imin:imax]
    bins = bins[imin:imax+1]  # Ensure bin edges align

    if len(count) < 3:
        raise ValueError("Not enough data points for fitting.")
    
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Convert to bin centers

    guess = [0.7*max(count), bin_centers[np.argmax(count)], 2*(bin_centers[len(bin_centers)//2]-bin_centers[0])]
    
    params, covariance = curve_fit(gauss, bin_centers, count, p0=guess)
    errors = np.sqrt(np.diag(covariance))
    fitted_curve = gauss(x, *params)
    return params, errors, fitted_curve, x
   

def custom_function(x, a, b, c, d, e, f):
     return a*np.exp(-(x-b)**2/(2*(c**2))) + d*np.exp(-(x-e)**2/(2*(f**2)))

def custom_fit(count, bins):
    bin_centers = (bins[:-1] + bins[1:]) / 2

    guess = [0.5*max(count),100, bin_centers[np.argmax(count)],
            0.5*max(count), 300, bin_centers[np.argmax(count)]]
    
    params, covariance = curve_fit(custom_function, bin_centers, count, p0=guess)#, maxfev=1000)
    errors = np.sqrt(np.diag(covariance))
    
    x = np.linspace(min(bins), max(bins), 1000)
    fitted_curve = custom_function(x, *params)

    return params, errors, fitted_curve, x
