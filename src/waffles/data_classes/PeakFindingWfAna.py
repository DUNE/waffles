
from scipy import signal as spsi

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.WfPeak import WfPeak

import waffles.Exceptions as we


class PeakFindingWfAna(BasicWfAna):
    """This class inherits from BasicWfAna. This 
    class implements an analysis which, on top of 
    the features of the BasicWfAna analysis, 
    implements a peak-finding algorithm based on
    scipy.signal.find_peaks().

    Attributes
    ----------
    input_parameters: IPDict (inherited from WfAna)
    baseline_limits: list of int (inherited from BasicWfAna)
    int_ll (resp. int_ul): int (inherited from BasicWfAna)
    amp_ll (resp. amp_ul): int (inherited from BasicWfAna)
    peak_finding_kwargs: dict
        Dictionary of keyword arguments which are passed to
        scipy.signal.find_peaks(Waveform.adcs, **peak_finding_kwargs)
        by the analyse() method.
    result: WfAnaResult (inherited from WfAna)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

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
                - 'peak_finding_kwargs' (dict)
            N.B.: The last key must also be defined,
            even if it is an empty dictionary.
        """

        self.__peak_finding_kwargs = input_parameters['peak_finding_kwargs']

        super().__init__(input_parameters)

    # Getters
    @property
    def peak_finding_kwargs(self):
        return self.__peak_finding_kwargs

    def analyse(
        self, 
        waveform: WaveformAdcs,
        return_peaks_properties: bool = False
    ) -> dict:
        """With respect to the given WaveformAdcs object, this analyser
        method does the following:

            - It computes the baseline as the median of the points
            that are considered, according to the documentation of
            the baseline_limits attribute.
            - It searches for peaks over the inverted Waveform,
            by calling

                scipy.signal.find_peaks(-1.*waveform.adcs,
                                        **self.__peak_finding_kwargs)

            - It calculates the integral of
            waveform.adcs[int_ll - waveform.time_offset :
            int_ul + 1 - waveform.time_offset].
            To do so, it assumes that the temporal resolution of
            the Waveform is constant and approximates its integral
            to waveform.time_step_ns*np.sum( -b + waveform.adcs[int_ll -
            waveform.time_offset : int_ul + 1 - waveform.time_offset]),
            where b is the computed baseline.
            - It calculates the amplitude of
            waveform.adcs[amp_ll - waveform.time_offset :
            amp_ul + 1 - waveform.time_offset].

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
        requirements are met. Also, regarding
        self.__peak_finding_kwargs, it is the caller's
        responsibility to ensure that it is well-defined.

        Parameters
        ----------
        waveform : WaveformAdcs
            The WaveformAdcs object which will be analysed
        return_peaks_properties : bool
            If True, then this method returns information about
            the spotted-peaks properties.

        Returns
        ----------
        output : dict
            If return_peaks_properties is False, then this
            dictionary is empty. If return_peaks_properties is
            True, then this is a dictionary containing the
            properties for the spotted peaks.
        """
        
        # Takes care of baseline, integral
        # and amplitude computations
        super().analyse(waveform)   
        
        # Assuming that the Waveform is
        # inverted.
        peaks, properties = spsi.find_peaks(
            -1.*waveform.adcs,
            **self.__peak_finding_kwargs)


        self._wf_ana__result['peaks'] = [
            WfPeak(peaks[i]) for i in range(len(peaks))]

        if return_peaks_properties is True:
            output = properties
        else:
            output = {}

        return output

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(
        input_parameters: IPDict,
        points_no: int
    ) -> None:
        """Apart from calling the base class check_input_parameters()
        method which performs some checks, this method adds
        another one to check whether the keys in
        input_parameters['peak_finding_kwargs'] belong to the
        set of valid keys for scipy.signal.find_peaks(), i.e.

            [   'height',
                'threshold',
                'distance',
                'prominence',
                'width',
                'wlen',
                'rel_height',
                'plateau_size'  ]

        If any of these checks fail, an exception is raised.

        Parameters
        ----------
        input_parameters : IPDict
            The input parameters to be checked. It is the IPDict
            that can be potentially given to BasciWfAna.__init__
            to instantiate a BasicWfAna object.
        points_no : int
            The number of points in any Waveform that could be
            analysed. It is assumed to be the same for all the
            waveforms.

        Returns
        ----------
        None
        """
        
        # Not using the super() syntax because
        # BasicWfAna.check_input_parameters is static
        BasicWfAna.check_input_parameters(
            input_parameters,   
            points_no)
                  
        aux = [
            'height',
            'threshold',
            'distance',
            'prominence',
            'width',
            'wlen',
            'rel_height',
            'plateau_size']

        for kwarg in input_parameters['peak_finding_kwargs']:
            if kwarg not in aux:
                raise Exception(we.GenerateExceptionMessage(
                    1,
                    'PeakFindingWfAna.check_input_parameters()',
                    f"A non-valid keyword argument ('{kwarg}')"
                    " was given to the 'peak_finding_kwargs' input "
                    "parameter."))

    # The following method is not supported
    # and may be removed in the near future
    def peaks_are_available(self) -> bool:
        """This method returns True if self.result
        is not None and self.result.Peaks is
        not None and len(self.result.Peaks)
        is greater than 0. It returns False otherwise.

        Returns
        ----------
        bool
        """

        if self.result is not None:
            if self.result.Peaks is not None:
                if len(self.result.Peaks) > 0:
                    return True

        return False