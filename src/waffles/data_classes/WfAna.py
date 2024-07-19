from typing import Tuple, List, TYPE_CHECKING
from scipy import signal as spsi
from scipy import ndimage as spim
from scipy.signal import welch as psd
from scipy.optimize import curve_fit
import plotly.graph_objects as go

import numpy as np

if TYPE_CHECKING:                                # Import only for type-checking, so as
    from .WaveformAdcs import WaveformAdcs       # to avoid a runtime circular import
                                                    
from .WfAnaResult import WfAnaResult
from .WfPeak import WfPeak


class WfAna:

    """
    Stands for Waveform Analysis. This class 
    implements an analysis which has been performed 
    over a certain WaveformAdcs object.
    
    Attributes
    ----------
    BaselineLimits : list of int
        It must have an even number of integers which
        must meet BaselineLimits[i] < BaselineLimits[i + 1].
        Given a WaveformAdcs object, wf, the points which
        are used for baseline calculation are
        wf.Adcs[BaselineLimits[2*i] - wf.TimeOffset : BaselineLimits[(2*i) + 1] - wf.TimeOffset],
        with i = 0,1,...,(len(BaselineLimits)/2) - 1. The 
        upper limits are exclusive.
    IntLl (resp. IntUl) : int
        Stands for integration lower (resp. upper) limit.
        Iterator value for the first (resp. last) point 
        of the waveform that falls into the integration
        window. IntLl must be smaller than IntUl. These
        limits are inclusive. I.e., the points which are
        used for the integral calculation are
        wf.Adcs[IntLl - wf.TimeOffset : IntUl + 1 - wf.TimeOffset]. 
    AmpLl (resp. AmpUl) : int
        Stands for amplitude lower (resp. upper) limit.
        Iterator value for the first (resp. last) point 
        of the waveform that is considered to compute
        the amplitude of the waveform. AmpLl must be smaller 
        than AmpUl. These limits are inclusive. I.e., the 
        points which are used for the amplitude calculation 
        are wf.Adcs[AmpLl - wf.TimeOffset : AmpUl + 1 - wf.TimeOffset].
    Result : WfAnaResult
        The result of the analysis
    Passed : bool
        Useful for analyses which should implement a quality
        filter. True (resp. False) if analysis concluded that 
        the waveform passes (fails).

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  baseline_limits : List[int],
                        int_ll : int,
                        int_ul : int,
                        amp_ll : int,
                        amp_ul : int):
        
        """
        WfAna class initializer. It is assumed that it is
        the caller responsibility to check the well-formedness 
        of the input parameters, according to the attributes
        documentation in WfAna.__init__. No checks are perfomed
        here.
        
        Parameters
        ----------
        baseline_limits : list of int
        int_ll : int
        int_ul : int
        amp_ll : int
        amp_ul : int
        """
        
        self.__baseline_limits = baseline_limits
        self.__int_ll = int_ll
        self.__int_ul = int_ul
        self.__amp_ll = amp_ll
        self.__amp_ul = amp_ul
        
        self.__result = None    # To be determined a posteriori 
        self.__passed = None    # by an analyser method

    #Getters
    @property
    def BaselineLimits(self):
        return self.__baseline_limits
    
    @property
    def IntLl(self):
        return self.__int_ll
    
    @property
    def IntUl(self):
        return self.__int_ul
    
    @property
    def AmpLl(self):
        return self.__amp_ll
    
    @property
    def AmpUl(self):
        return self.__amp_ul

    @property
    def Result(self):
        return self.__result
    
    @property
    def Passed(self):
        return self.__passed
    
    #Setters
    @Result.setter                                  # Adding setters for the self.__result and
    def Result(self, input):                        # self.__passed attributes, since
                                                    # WaveformAdcs.analyse() should be able to
        ## Shall we add a type check here?          # set them. Not adding setters for the rest
                                                    # of the attributes, since they shall not
        self.__result = input                       # be set outside WfAna code
        return
    
    @Passed.setter
    def Passed(self, input):

        ## Shall we add a type check here?

        self.__passed = input
        return

    def analyser_template(  self,
                            waveform : 'WaveformAdcs',
                            *args,
                            **kwargs) -> Tuple[WfAnaResult, bool, dict]:
        
        """
        This method implements a template for an analyser 
        method.

        Parameters
        ----------
        waveform : WaveformAdcs
            WaveformAdcs object which will be analysed.
        *args
            Additional positional arguments
        **kwargs
            Additional keyword arguments

        Returns
        ----------
        output_1 : WfAnaResult
            The result of the analysis
        output_2 : bool
            True (resp. False) if analysis concluded
            that the waveform passes (resp. fails).
        output_3 : dict
            Additional information about the analysis.
            If there is no additional information,
            then this dictionary is empty.
        """

        output_1 = WfAnaResult( baseline = 0,
                                baseline_min = 0,
                                baseline_max = 10,
                                baseline_rms = 1,
                                peaks = [WfPeak(1), WfPeak(12), WfPeak(300)],
                                integral = 10.0,
                                amplitude = 5.0,
                                deconvoluted_adcs = np.array([0.,-1.,1.,0.,-1.]))
        output_2 = True
        output_3 = {}

        return output_1, output_2, output_3
    
    def standard_analyser(  self,
                            waveform : 'WaveformAdcs',  # The WaveformAdcs class is not defined at runtime, only
                                                        # during type-checking (see TYPE_CHECKING). Not enclosing
                                                        # the type in quotes would raise a `NameError: name
                                                        # 'WaveformAdcs' is not defined.`
                            *args,
                            **kwargs) -> Tuple[WfAnaResult, bool, dict]:
        
        """
        This method implements an analyser method
        which does the following:

            - It computes the baseline as the median of the points
            that are considered, according to the documentation of
            the self.__baseline_limits attribute.
            - It searches for peaks using scipy.signal.find_peaks.
            - It calculates the integral of 
            waveform.Adcs[IntLl - waveform.TimeOffset : IntUl + 1 - waveform.TimeOffset]. 
            To do so, it assumes that the temporal resolution of 
            the waveform is constant and approximates its integral 
            to waveform.TimeStep_ns*np.sum( -b + waveform.Adcs[IntLl - waveform.TimeOffset : IntUl + 1 - waveform.TimeOffset]),
            where b is the computed baseline.
            - It calculates the amplitude of 
            waveform.Adcs[AmpLl - waveform.TimeOffset : AmpUl + 1 - waveform.TimeOffset].

        Note that for these computations to be well-defined, it is
        assumed that

            - BaselineLimits[0] - wf.TimeOffset >= 0
            - BaselineLimits[-1] - wf.TimeOffset <= len(wf.Adcs)
            - IntLl - wf.TimeOffset >= 0
            - IntUl - wf.TimeOffset < len(wf.Adcs)
            - AmpLl - wf.TimeOffset >= 0
            - AmpUl - wf.TimeOffset < len(wf.Adcs)

        For the sake of efficiency, these checks are not done.
        It is the caller's responsibility to ensure that these 
        requirements are met.

        Parameters
        ----------
        waveform : WaveformAdcs
            WaveformAdcs object which will be analysed.
        *args
            These arguments are passed to 
            scipy.signal.find_peaks(waveform.Adcs, *args, **kwargs)
            as *args.        
        **kwargs
            If the 'return_peaks_properties' keyword argument is present
            in **kwargs and set to True, then this method returns 
            information about the spotted-peaks properties. The rest 
            of the keyword arguments are passed to 
            scipy.signal.find_peaks(waveform.Adcs, *args, **kwargs) 
            as **kwargs.

        Returns
        ----------
        output_1 : WfAnaResult
            The result of the analysis
        output_2 : bool
            True (resp. False) if the analysis concluded
            that the waveform passes (resp. fails).
        output_3 : dict
            Contains the 'properties' dictionary returned by
            scipy.signal.find_peaks, under the 'peaks_properties'
            key, if the 'return_peaks_properties' keyword argument
            is present in **kwargs and set to True. It is an empty
            dictionary if else.
        """

        return_peaks_properties = None

        try:
            return_peaks_properties = kwargs.pop('return_peaks_properties')     # If the 'return_peaks_properties' keyword
        except KeyError:                                                        # is present, pop it from kwargs before
            pass                                                                # giving them to scipy.signal.find_peaks
                    
        split_baseline_samples = [  waveform.Adcs[ self.__baseline_limits[2*i] - waveform.TimeOffset : self.__baseline_limits[(2*i) + 1] - waveform.TimeOffset]
                                    for i in range(len(self.__baseline_limits)//2) ]
        
        baseline_samples = np.concatenate(split_baseline_samples)
        baseline = np.median(baseline_samples)

        peaks, properties = spsi.find_peaks(-1.*waveform.Adcs, *args, **kwargs)     ## Assuming that the waveform is
                                                                                    ## inverted. We should find another 
                                                                                    ## way not to hardcode this
        output_1 = WfAnaResult( baseline = baseline,
                               
                                baseline_min = None,            # Might be set to np.min(baseline_samples) (resp. 
                                baseline_max = None,            # np.max(baseline_samples), ~np.std(baseline_samples)) 
                                baseline_rms = None,            # for a deeper analysis for which we need (and
                                                                # can afford the computation time) for this data

                                peaks = [ WfPeak(peaks[i]) for i in range(len(peaks)) ],
                                integral = waveform.TimeStep_ns*(((self.__int_ul - self.__int_ll + 1)*baseline) - np.sum(waveform.Adcs[ self.__int_ll - waveform.TimeOffset : self.__int_ul + 1 - waveform.TimeOffset])),   ## Assuming that the waveform is
                                                                                                                                                                                                                            ## inverted and using linearity
                                                                                                                                                                                                                            ## to avoid some multiplications
                                amplitude = + np.max(waveform.Adcs[self.__amp_ll - waveform.TimeOffset : self.__amp_ul + 1 - waveform.TimeOffset]) 
                                            - np.min(waveform.Adcs[self.__amp_ll - waveform.TimeOffset : self.__amp_ul + 1 - waveform.TimeOffset]),
                                            
                                deconvoluted_adcs=None)     # Not computed by this standard analyser
        
        output_2 = True         # This standard analyser does not implement a quality filter yet

        if return_peaks_properties is True:
            output_3 = {'peaks_properties': properties}
        else:
            output_3 = {}
            
        return output_1, output_2, output_3
    
    def baseline_is_available(self) -> bool:
        
        """
        This method returns True if self.__result
        is not None and self.__result.Baseline is
        not None. It returns False otherwise.

        Returns
        ----------
        bool
        """

        if self.__result is not None:
            if self.__result.Baseline is not None:
                return True
        
        return False

    def peaks_are_available(self) -> bool:
        
        """
        This method returns True if self.__result
        is not None and self.__result.Peaks is
        not None and len(self.__result.Peaks)
        is greater than 0. It returns False otherwise.

        Returns
        ----------
        bool
        """

        if self.__result is not None:
            if self.__result.Peaks is not None:
                if len(self.__result.Peaks) > 0:
                    return True
        
        return False
    
    @staticmethod
    def deconvolve(signal      : np.ndarray,
                   template    : np.ndarray,
                   noise       : np.ndarray = None,
                   filter_type : str = 'Gauss',
                   sample_rate : float = 62.5e6,
                   cutoff_rate : float = 10e6
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
                    param, covariance = curve_fit(dec_gauss, frequencies, wiener, p0=cutoff_rate)
                    signal_filter = dec_gauss(frequencies, *param)
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
            params, covariance = curve_fit(dec_fit_FastSlow, time, data_tofit, p0 = initial_guess,
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
            params, covariance = curve_fit(dec_fit_FastSlowIntermediate, time, data_tofit, p0=initial_guess,
                                           bounds=((0, tau_fast_guess - tau_fast_guess_error, 0, 11, 0, 2, 10, t0_guess - 100 , -np.inf),
                                                   (np.inf, tau_fast_guess + tau_fast_guess_error, np.inf, 900, np.inf, 10, np.inf,  np.inf , np.inf)))
            errors = np.sqrt(np.diag(covariance))
    
            if errors[5] > params[5]:
                initial_guess = (0.1, tau_fast_guess, 1, 50, np.max(original_signal), 7, 10 , t0_guess, offset_guess)
                params, covariance = curve_fit(dec_fit_FastSlowIntermediate, time, data_tofit, p0=initial_guess,
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
 