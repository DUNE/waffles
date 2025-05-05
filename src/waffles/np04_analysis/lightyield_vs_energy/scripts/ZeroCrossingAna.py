import numpy as np

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.utils.check_utils as wuc
import waffles.Exceptions as we

from waffles.np04_analysis.light_vs_hv.imports import *


import numpy as np

class ZeroCrossingAna(WfAna):

    @we.handle_missing_data
    def __init__(self, input_parameters: IPDict):
        """BasicWfAna class initializer. It is assumed that it is
        the caller responsibility to check the well-formedness
        of the input parameters, according to the attributes
        documentation in the class documentation. No checks
        are perfomed here.

        Parameters
        ----------
        input_parameters: IPDict
            This IPDict must contain the following keys:
                - 'baseline_limits' (list of int)
                - 'int_ll' (int)
                - 'int_ul' (int)
                - 'amp_ll' (int)
                - 'amp_ul' (int)
        """

        self.__baseline_ll = input_parameters['baseline_ll']
        self.__baseline_ul = input_parameters['baseline_ul']
        self.__zero_ll = input_parameters['zero_ll']
        self.__zero_ul = input_parameters['zero_ul']
        self.__int_ll = input_parameters['int_ll']
        self.__int_ul = input_parameters['int_ul']
        self.__amp_ll = input_parameters['amp_ll']
        self.__amp_ul = input_parameters['amp_ul']
        self.__fprompt_ul= input_parameters['fprompt_ul']
        self.__t0_wf_ul= input_parameters['t0_wf_ul']
        self.__template_path= input_parameters['template_path']
        super().__init__(input_parameters)

    # Getters
    @property
    def baseline_ll(self):
        return self.__baseline_ll

    @property
    def baseline_ul(self):
        return self.__baseline_ul

    @property
    def zero_ll(self):
        return self.__zero_ll

    @property
    def zero_ul(self):
        return self.__zero_ul
    
    @property
    def int_ll(self):
        return self.__int_ll

    @property
    def int_ul(self):
        return self.__int_ul
    
    @property
    def amp_ll(self):
        return self.__amp_ll

    @property
    def amp_ul(self):
        return self.__amp_ul
    
    @property
    def fprompt_ul(self):
        return self.__fprompt_ul

    @property
    def t0_wf_ul(self):
        return self.__t0_wf_ul
    
    @property
    def template_path(self):
        return self.__template_path


    def analyse(self, waveform: WaveformAdcs) -> None:

        #print("analyseee")
        aux_baseline=waveform.adcs[self.__baseline_ll:self.__baseline_ul]
        baseline=np.mean(aux_baseline)
        noise=np.std(aux_baseline)

        template=[]

        with open(self.__template_path, "r") as template_data:
            template.append( [float(line.strip()) for line in template_data] )

        waveform_aux = np.asarray((waveform.adcs-baseline)[self.__zero_ll:self.__zero_ul])
    
        # Find the indices where the sign changes
        zero_crossing = np.where(np.diff(np.sign(waveform_aux)))[0]+self.__zero_ll

        if len(zero_crossing)>=1:
            value_0=zero_crossing[0]
        else:
            value_0=-1

        waveform_aux = np.asarray((waveform.adcs-baseline)[0:self.__t0_wf_ul])

        # Find the indices where the sign changes
        zero_crossing = np.where(np.diff(np.sign(waveform_aux)))[0]

        if len(zero_crossing)>=1:
            start_value=zero_crossing[len(zero_crossing)-1]
        else:
            start_value=-1
       

        waveform_aux = np.asarray((waveform.adcs-baseline)[self.__int_ll:self.__int_ul])
        integral=-np.sum(waveform_aux)

        if start_value != -1:
            waveform_aux = np.asarray((waveform.adcs-baseline)[start_value:self.__fprompt_ul])
            integral_fast=-np.sum(waveform_aux)
        else:
            integral_fast=-1

        waveform_aux = np.asarray((waveform.adcs-baseline)[self.__amp_ll:self.__amp_ul])
        amplitude=-np.min(waveform_aux)

        if value_0!=-1 and start_value!=-1:
            waveform_aux=np.asarray((waveform.adcs-baseline)[start_value:value_0])
            integral_0=-np.sum(waveform_aux)
        else:
            integral_0=-1

        if integral_fast !=-1 and integral_0 !=-1:
            fprompt = integral_fast/integral_0
        else:
            fprompt = -1

        if value_0!=-1:
            waveform_aux =  np.asarray((waveform.adcs-baseline)[value_0:])
            second_peak= -np.min(waveform_aux)
        else:
            second_peak=-1


        roll=+130
        N=1024
        _x =  np.fft.fftfreq(N) * N #np.linspace(0, 1024, 1024, endpoint=False)
        
        my_waveform = np.array(waveform.adcs-baseline)

        filter_gaus = [ gaus(x,35) for x in _x]
        signall=np.concatenate([np.zeros(0),my_waveform,np.zeros(0)])
        template_menos=-np.array(template)
        signal_fft = np.fft.fft(my_waveform)
        template_menos_fft = np.fft.fft(template_menos, n=len(signal_fft))  # Match signal length
        deconvolved_fft = signal_fft/ (template_menos_fft)     # Division in frequency domain
        for j, _ in enumerate(deconvolved_fft):
            deconvolved_fft[j] *= filter_gaus[j]
        
        other_deconvolved_aux = np.fft.ifft(deconvolved_fft)
        
        deconvolved_wf=other_deconvolved_aux.real
      
        deconvolved_wf = np.roll(np.array(deconvolved_wf),roll)
        
        self._WfAna__result = WfAnaResult(
            baseline=baseline,
            noise=noise,
            zero_crossing=value_0,
            t0=start_value,
            amplitude=amplitude,
            integral=integral,
            integral_0=integral_0 ,
            integral_fast=integral_fast ,
            fprompt = fprompt,
            second_peak = second_peak,
            deconvolved_waveform = deconvolved_wf
        )
        return

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(
            input_parameters: IPDict
    ) -> None:
        return