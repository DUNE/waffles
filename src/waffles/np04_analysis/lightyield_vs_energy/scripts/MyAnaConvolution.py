# Analysis class to search for peaks and beam event timestamp 

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult
from waffles.Exceptions import GenerateExceptionMessage

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft


import waffles.Exceptions as we
import numpy as np
import pandas as pd

def find_saturation_regions(signal, saturation_threshold):
    # Identify saturated indices
    saturated_indices = np.where(signal >= saturation_threshold)[0]
    
    if len(saturated_indices) == 0:
        return []

    # Compute differences between consecutive indices
    diffs = np.diff(saturated_indices)
    
    # Find breaks in the sequence where difference is greater than 1
    breaks = np.where(diffs > 1)[0]
    
    # Split indices into continuous regions based on breaks
    regions = np.split(saturated_indices, breaks + 1)
    
    # Store the start and end of each continuous region
    region_boundaries = [(np.where(signal >= saturation_threshold)[0][0], region[-1]) for region in regions]
    
    return region_boundaries

def saturated_signal(wf):
    saturation_threshold = 8000
    region = find_saturation_regions(wf, saturation_threshold)[0]
    begin = region[0]
    ending = region[1]
    time = np.arange(0, len(wf))

    # Fit first line (from `begin-2` to `begin`)
    y1 = wf[begin-2: begin]
    x1 = time[begin-2: begin]
    c1 = np.polyfit(x1, y1, 1)
    
    # Fit second line (from `ending` to `ending+5`)
    y2 = wf[ending: ending+3]
    x2 = time[ending: ending+3]
    c2 = np.polyfit(x2, y2, 1)
    
    # Create time array for the full range
    t = time[begin:ending]
    
    # Calculate fitted values for the first and second lines
    f1 = np.polyval(c1, t)
    f2 = np.polyval(c2, t)
    
    # Find the index where the first line exceeds the second line
    i = np.where(f1 >= f2)[0][0]
    new_region = np.concatenate((f1[:i], f2[i:]))
    
    # Modify the signal based on the two fitted regions
    wf2 = np.concatenate((wf[:begin], new_region, wf[ending:]))
    return wf2

def is_saturated(wf):
    saturation_threshold = 8000
    saturated_indices = np.where(wf >= saturation_threshold)[0]
    if len(saturated_indices) > 2 and np.all(np.diff(saturated_indices) == 1):
        return True
    else:
        return False

def scintillation(t, A_s, A_t, tau_s, tau_t):
    Is = (A_s) * np.exp(-(t)/tau_s)
    It = (A_t) * np.exp(-(t)/tau_t)      
    return Is + It

def scintillation_norm(t, A_s, A_t, tau_s, tau_t):
    """Compute the scintillation function and normalize it."""
    Is = A_s * np.exp(-t / tau_s)
    It = A_t * np.exp(-t / tau_t)
    scint = Is + It
    return scint / np.max(scint)  # Normalize to peak value of 1


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

def fft_convolution(t, A_s, A_t, tau_s, tau_t, offset, template, dt):
    """Perform convolution using FFT (faster than direct summation)."""
    # Compute the scintillation function
    scint = scintillation(t, A_s, A_t, tau_s, tau_t)

    # Compute FFT convolution (zero-padding ensures correct length)
    N = len(template) + len(scint) - 1
    conv_result = ifft(fft(template, N) * fft(scint, N)).real[:len(template)]  # Keep original length

    return conv_result * dt + offset  # Scale by dt

def conv_fit_v5(signal, template, peak_lim, end_time, factor=2.5):
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
          
            
        if step == 2:
            guess = (params[0], params[1], tau_s_guess, tau_t_guess, params[2])
            bounds = ((params[0]-errors[0], 0, 2, 1000, params[2]-errors[2]),
                      (params[0]+errors[0], params[1]+errors[1], 15, 1800, params[2]+errors[2]))

            params, covariance = curve_fit(lambda new_time, A_s, A_t, tau_s, tau_t, offset: 
                                           fft_convolution_norm(new_time, A_s, A_t, tau_s, tau_t, offset, template_upsampled_norm, dt), 
                                           new_time, data_tofit_upsampled_norm, p0=guess, bounds=bounds)
            errors = np.sqrt(np.diag(covariance))

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
            residuals = data_tofit_upsampled - fitted_data
            ss_tot = np.sum((data_tofit_upsampled - np.mean(data_tofit_upsampled)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Compute the correct photon count
            photons = (params[0] * params[2] + params[1] * params[3])
            Ist = params[0] * params[2] / (params[1] * params[3])
            e_I  = Ist*np.sqrt((errors[0]/params[0])**2 + (errors[1]/params[1])**2 +(errors[2]/params[2])**2 + (errors[3]/ params[3])**2)
                    
        step += 1

    return params, errors, r_squared


class MyAnaConvolution(WfAna):
    def __init__(self, input_parameters: IPDict):
        self.__df_template = input_parameters['df_template']
        self.__tick = input_parameters.get('tick', 265) # ??
        
        

    def analyse(self, waveform: WaveformAdcs) -> None:
        """With respect to the given WaveformAdcs object, this 
        analyser method does the following:
          - search for template
          - check saturation and correct if necessary
          - align waveform with template
          - fit convolution to get #pe

        Parameters
        ---------
        waveform: WaveformAdcs
            The WaveformAdcs object which will be analysed

        Returns
        ----------
        None
        """
        try:
            template= self.__df_template.loc[(self.__df_template['endpoint'] == waveform.endpoint) & (self.__df_template['channel'] == waveform.channel), 'Template_avg'].iloc[0]
            template = np.array(template)
            template = template - np.median(template)
        except Exception as e:
            #print(f"Template not found for endpoint {waveform.endpoint} and channel {waveform.channel}: {e}")
            self._WfAna__result = WfAnaResult(n_pe = np.nan, e_n_pe = np.nan, fit_params = [np.nan, np.nan, np.nan, np.nan, np.nan], fit_errors = [np.nan, np.nan, np.nan, np.nan, np.nan],r_squared = np.nan)
            return

        wf = np.array(waveform.adcs)
        if is_saturated(wf):
            try:
                wf = saturated_signal(wf)
            except Exception as e:
                #print("Saturation correction failed:", e)
                self._WfAna__result = WfAnaResult(n_pe = np.nan, e_n_pe = np.nan, fit_params = [np.nan, np.nan, np.nan, np.nan, np.nan], fit_errors = [np.nan, np.nan, np.nan, np.nan, np.nan],r_squared = np.nan)
                return
            
        if (np.max(wf[:150])) < 0:
            #print("Max wf < 0")
            self._WfAna__result = WfAnaResult(n_pe = np.nan, e_n_pe = np.nan, fit_params = [np.nan, np.nan, np.nan, np.nan, np.nan], fit_errors = [np.nan, np.nan, np.nan, np.nan, np.nan],r_squared = np.nan)
            return
        
        try:
            template_half_max = np.max(template) / 2
            wf_half_max = np.max(wf[:150]) / 2
            template_idx = np.where(template >= template_half_max)[0][0]
        except Exception as e:
            #print("Half max calculation failed:", e)
            self._WfAna__result = WfAnaResult(n_pe = np.nan, e_n_pe = np.nan, fit_params = [np.nan, np.nan, np.nan, np.nan, np.nan], fit_errors = [np.nan, np.nan, np.nan, np.nan, np.nan],r_squared = np.nan)
            return
        
        wf_idx = np.where(wf[:150] >= wf_half_max)[0][0]
        delta = template_idx - wf_idx

        # Alignement of the wf with template (and cut the final part if necessary)
        wf = np.roll(wf, delta)
        wf = wf[:len(template)]
        wf = wf[100:]
        template = template[100:]

        if waveform.endpoint == 109:
            lim = 500
        else:
            if len(template) > 400:
                lim = 400
            else:
                lim = 200
            
        try:
            params, errors, r_squared = conv_fit_v5(wf, template, 150, lim, factor = 2.5)
            if r_squared < 0.5:
                #print("Low R^2:", r_squared)
                self._WfAna__result = WfAnaResult(n_pe = np.nan, e_n_pe = np.nan, fit_params = [np.nan, np.nan, np.nan, np.nan, np.nan], fit_errors = [np.nan, np.nan, np.nan, np.nan, np.nan],r_squared = np.nan)
                return
            else:
            
                A_s, A_t, tau_s, tau_t, offset = params
                eA_s, eA_t,  etau_s, etau_t,  eoffset = errors
            
                area   = A_s*tau_s*(1-np.exp(-16*self.__tick/(tau_s))) + (A_t*tau_t*(1-np.exp(-16*self.__tick/(tau_t))))
                e_area = np.sqrt((A_s*etau_s)**2 + (tau_s*eA_s)**2 + (A_t*etau_t)**2 + (tau_t*eA_t)**2)

                # Results 
                self._WfAna__result = WfAnaResult(
                    n_pe = area,
                    e_n_pe = e_area,
                    fit_params = params,
                    fit_errors = errors,
                    r_squared = r_squared
                )
                return
                    
        except Exception as e:
                #print("Fitting failed:", e)
                self._WfAna__result = WfAnaResult(n_pe = np.nan, e_n_pe = np.nan, fit_params = [np.nan, np.nan, np.nan, np.nan, np.nan], fit_errors = [np.nan, np.nan, np.nan, np.nan, np.nan],r_squared = np.nan)
                return
            

        

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(
            input_parameters: IPDict,
            points_no: int
    ) -> None:

        # No checks for now
        return
    

