import numpy as np
from scipy.signal import welch as psd
from scipy.optimize import curve_fit

import waffles.utils.deconvolution.deconvolution_utils as wudd

def deconvolve(signal      : np.ndarray,
                template    : np.ndarray,
                noise       : np.ndarray = None,
                filter_type : str = 'Gauss',
                sample_rate : float = 62.5e6,
                cutoff_rate : float = 10e6,
                isplot      : bool = False    
                )-> np.ndarray:
    """
    Method to deconvolve a signal using a given template and optional noise.
    
    Parameters
    ----------
    signal (np.ndarray): The input signal to be deconvolved.
    template (np.ndarray): The template used for deconvolution.
    noise (np.ndarray, optional): The noise array. Defaults to None (no filter is applied).
    filter_type (str): The type of filter to use ('Gauss' by default).
    sample_rate (float): The sampling rate in Hz. Defaults to 62.5e6 (62.5 MHz).
    cutoff_rate (float): The cutoff rate in Hz. Defaults to 10e6 (10 MHz).
    isplot (bool): To plot the results the PSDs.

    Returns:
    ----------
    np.ndarray: The deconvolved signal.
    """

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
                param, covariance = curve_fit(wudd.dec_gauss, frequencies, wiener, p0=cutoff_rate)
                signal_filter = wudd.dec_gauss(frequencies, *param)
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


def dec_fit(dec_signal : np.ndarray , 
            original_signal: np.ndarray):

    """
    Method to fit the deconvolution function.
    First the slow component is estimating by fitting a bi-exponential function
    and, then, a tri-exponential fitting is applied to estimate the fast 
    component.
    
    Parameters
    ----------
    dec_signal (np.ndarray): Deconvolved signal.
    original_signal (np.ndarray): Waveform before being deconvolved.
    
    Returns:
    ----------
    params: array with the free parameters from the fitting [ As, taus, Ai, taui, Af, tauf, sigma, t0, offset]
    errors: array with the errors of each parameter
    """
    
    # Estimation of the slow tau
    try:
        begin, end = 0, 2*len(dec_signal) // 3
        data_tofit, time = dec_signal[begin:end], 16 * np.arange(begin, end) #ns
        initial_guess = (0.1, 1500, np.max(original_signal), 7, 10, time[np.argmax(data_tofit)], np.mean(dec_signal[:20]))
        params, covariance = curve_fit(wudd.dec_fit_FastSlow, time, data_tofit, p0 = initial_guess,
                                        bounds=((0, 1000, 0, 0, 0, time[np.argmax(data_tofit)] - 100, -np.inf), (np.inf, 2000, np.inf, np.inf, np.inf, np.inf, np.inf)))
        errors = np.sqrt(np.diag(covariance))
        
        tau_fast_guess, tau_fast_guess_error = params[1], errors[1]
        t0_guess, t0_guess_error             = params[5], errors[5]
        offset_guess                         = params[6]
    
    except:
        tau_fast_guess, tau_fast_guess_error = 1500, 500
        t0_guess, t0_error_guess             = time[np.argmax(data_tofit)], np.inf
        offset_guess                         = np.mean(dec_signal[:20])
        
    try:
        data_tofit, time = dec_signal, 16 * np.arange(0, len(dec_signal))
        initial_guess = (0.1, tau_fast_guess, 1, 50, np.max(dec_signal), 7, 10 , t0_guess, offset_guess)
        params, covariance = curve_fit(wudd.dec_fit_FastSlowIntermediate, time, data_tofit, p0=initial_guess,
                                        bounds=((0, tau_fast_guess - tau_fast_guess_error, 0, 11, 0, 2, 10, t0_guess - 100 , -np.inf),
                                                (np.inf, tau_fast_guess + tau_fast_guess_error, np.inf, 900, np.inf, 10, np.inf,  np.inf , np.inf)))
        errors = np.sqrt(np.diag(covariance))

        if errors[5] > params[5]:
            initial_guess = (0.1, tau_fast_guess, 1, 50, np.max(original_signal), 7, 10 , t0_guess, offset_guess)
            params, covariance = curve_fit(wudd.dec_fit_FastSlowIntermediate, time, data_tofit, p0=initial_guess,
                                            bounds=((0, tau_fast_guess - tau_fast_guess_error, 0, 11, 0, 2, 10, t0_guess - 100 , -np.inf),
                                                    (np.inf, tau_fast_guess + tau_fast_guess_error, np.inf, 900, np.inf, 10, np.inf,  np.inf , np.inf)))
            errors = np.sqrt(np.diag(covariance))
            
        print(f'Tau_Slow (ns) = {params[1]} +- {errors[1]}')
        print(f'Tau_Fast (ns) = {params[5]} +- {errors[5]}')
        print(f'Tau_Intermediary (ns) = {params[3]} +- {errors[3]}')
        
        return params, errors
        
    except:
        print('Failed')
        return 0, 0