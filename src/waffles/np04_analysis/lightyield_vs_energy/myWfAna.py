import numpy as np

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.utils.check_utils as wuc
import waffles.Exceptions as we

from waffles.utils.baseline.baseline import SBaseline # Baseline computation from Henrique
from scipy.fft import fft, fftshift # For the deconvolution


def from_cutoff_to_sigma(cutoff_frequency):
    npoints = 1024//2 # The fft has half the length of the waveforms
    FFTFreq = 32.5
    binwidth = FFTFreq/npoints
    fc_in_ticks = cutoff_frequency/binwidth
    return fc_in_ticks/np.sqrt(np.log(2))

def gaus(x, sigma=20):
    return np.exp(-(x)**2/2/sigma**2)


class MyWfAna(WfAna):

    @we.handle_missing_data
    def __init__(self, input_parameters: IPDict):

        self.__baseline_start = input_parameters['baseline_start'] #before deconv
        self.__baseline_stop = input_parameters['baseline_stop'] #before deconv
        self.__int_ll = input_parameters['int_ll'] #after deconv
        self.__int_ul = input_parameters['int_ul'] #after deconv
        self.__gauss_cutoff = input_parameters['gauss_cutoff'] #MHz
        self.__template = input_parameters['template'] #adcs 

        super().__init__(input_parameters)

    # Getters
    @property
    def baseline_start(self):
        return self.__baseline_start
    
    @property
    def baseline_stop(self):
        return self.__baseline_stop

    @property
    def int_ll(self):
        return self.__int_ll

    @property
    def int_ul(self):
        return self.__int_ul
    
    @property
    def gauss_cutoff(self):
        return self.__gauss_cutoff

    @property
    def template(self):
        return self.__template

    def analyse(self, waveform: WaveformAdcs) -> None:

        # Henrique baseline 
        baseliner = SBaseline()
        baseliner.binsbase       = np.linspace(0, 2**14-1, 2**14)
        baseliner.threshold      = 6
        baseliner.wait           = 25
        baseliner.minimumfrac    = 0.166666
        baseliner.baselinestart  = self.__baseline_start
        baseliner.baselinefinish = self.__baseline_stop
        
        baseline, optimal = baseliner.wfset_baseline(waveform)
        new_wf_adcs = waveform.adcs.astype(float) - baseline
        new_wf_adcs = -new_wf_adcs
        
        # First deconvolution
        signal_fft = np.fft.fft(new_wf_adcs)
        maritza_template_fft = np.fft.fft(self.__template, n=len(new_wf_adcs))
        martiza_deconvolved_fft = signal_fft / maritza_template_fft    
        martiza_deconvolved_wf = np.fft.ifft(martiza_deconvolved_fft).real  
        
        _x = np.linspace(0, 1024, 1024, endpoint=False)
        sigma = from_cutoff_to_sigma(self.__gauss_cutoff)
        filter_gaus = np.array([gaus(x, sigma=sigma) for x in _x])
        
        filtered_martiza_deconvolved_fft = martiza_deconvolved_fft * filter_gaus
        filtered_martiza_deconvolved_wf = np.fft.ifft(filtered_martiza_deconvolved_fft).real  

        integral_before_deconvolution = waveform.time_step_ns * (((
                self.__int_ul - self.__int_ll + 1) * baseline) - np.sum(
                waveform.adcs[
                    self.__int_ll - waveform.time_offset:
                    self.__int_ul + 1 - waveform.time_offset]))

        integral_after_deconvolution = waveform.time_step_ns * (((
                self.__int_ul - self.__int_ll + 1) * baseline) - np.sum(
                new_wf_adcs[
                    self.__int_ll - waveform.time_offset:
                    self.__int_ul + 1 - waveform.time_offset]))

        self._WfAna__result = WfAnaResult(
            baseline = baseline,
            optimal_baseline  = optimal,
            integral_before_deconvolution= integral_before_deconvolution,
            integral_after_deconvolution = integral_after_deconvolution,
            deconvolved_wf_adc = new_wf_adcs
        )
        return

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(
            input_parameters: IPDict,
            points_no: int
    ) -> None:

        return