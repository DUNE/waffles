import numpy as np

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.utils.check_utils as wuc
import waffles.Exceptions as we

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


    def analyse(self, waveform: WaveformAdcs) -> None:
        """With respect to the given WaveformAdcs object, this 
        analyser method does the following:

            - It computes the baseline as the median of the points
            that are considered, according to the documentation of
            the self.__baseline_limits attribute.
            - It calculates the integral of
            waveform.adcs[int_ll - waveform.time_offset:
            int_ul + 1 - waveform.time_offset].
            To do so, it assumes that the temporal resolution of
            the waveform is constant and approximates its integral
            to waveform.time_step_ns*np.sum(-b + waveform.adcs[int_ll -
            waveform.time_offset: int_ul + 1 - waveform.time_offset]),
            where b is the computed baseline.
            - It calculates the amplitude of
            waveform.adcs[amp_ll - waveform.time_offset: amp_ul + 1 -
            waveform.time_offset].

        Note that for these computations to be well-defined, it is
        assumed that

            - baseline_limits[0] - wf.time_offset >= 0
            - baseline_limits[-1] - wf.time_offset <= len(wf.adcs)
            - int_ll - wf.time_offset >= 0
            - int_ul - wf.time_offset < len(wf.adcs)
            - amp_ll - wf.time_offset >= 0
            - amp_ul - wf.time_offset < len(wf.adcs)

        For the sake of efficiency, these checks are not done.
        It is the caller's responsibility to ensure that these
        requirements are met.

        Parameters
        ----------
        waveform : WaveformAdcs
            The WaveformAdcs object which will be analysed

        Returns
        ----------
        None
        """

        aux_baseline=waveform.adcs[self.__baseline_ll:self.__baseline_ul]
        baseline=np.mean(aux_baseline)
        noise=np.std(aux_baseline)

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
            second_peak =second_peak
        )
        return

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(
            input_parameters: IPDict
    ) -> None:
        return