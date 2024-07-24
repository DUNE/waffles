
from scipy import signal as spsi
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                                   # Import only for type-checking, so as
    from waffles.data_classes.WaveformAdcs import WaveformAdcs      # to avoid a runtime circular import
                                                    
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.WfPeak import WfPeak

import waffles.Exceptions as we

class PeakFindingWfAna(BasicWfAna):

    """
    This class inherits from BasicWfAna. This class
    implements an analysis which, on top of the
    features of the BasicWfAna analysis, implements
    a peak-finding algorithm based on 
    scipy.signal.find_peaks().
    
    Attributes
    ----------
    InputParameters : IPDict (inherited from WfAna)
    BaselineLimits : list of int (inherited from BasicWfAna)
    IntLl (resp. IntUl) : int (inherited from BasicWfAna)
    AmpLl (resp. AmpUl) : int (inherited from BasicWfAna)
    PeakFindingKwargs : dict
        Dictionary of keyword arguments which are passed to
        scipy.signal.find_peaks(waveform.Adcs, **PeakFindingKwargs)
        by the analyse() method.
    Result : WfAnaResult (inherited from WfAna)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    @we.handle_missing_data
    def __init__(self,  input_parameters : IPDict):
        
        """
        BasicWfAna class initializer. It is assumed that it is
        the caller responsibility to check the well-formedness 
        of the input parameters, according to the attributes
        documentation in the class documentation. No checks 
        are perfomed here.
        
        Parameters
        ----------
        input_parameters : IPDict
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

    #Getters    
    @property
    def PeakFindingKwargs(self):
        return self.__peak_finding_kwargs
    
    def analyse(self,   waveform : 'WaveformAdcs',  
                        return_peaks_properties : bool = False) -> dict:
        
        """
        With respect to the given WaveformAdcs object, this analyser 
        method does the following:

            - It computes the baseline as the median of the points
            that are considered, according to the documentation of
            the BaselineLimits attribute.
            - It searches for peaks over the inverted waveform, 
            by calling 
            
                scipy.signal.find_peaks(-1.*waveform.Adcs,
                                        **self.__peak_finding_kwargs)

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
        super().analyse(waveform)   # Takes care of baseline, integral
                                    # and amplitude computations

        peaks, properties = spsi.find_peaks(-1.*waveform.Adcs,              ## Assuming that the waveform is
                                            **self.__peak_finding_kwargs)   ## inverted. We should find another 
                                                                            ## way not to hardcode this

        self._WfAna__result['peaks'] = [ WfPeak(peaks[i]) for i in range(len(peaks)) ]

        if return_peaks_properties is True:
            output = properties
        else:
            output = {}
            
        return output

    ## The following method is not supported
    ## and may be removed in the near future

    def peaks_are_available(self) -> bool:
        
        """
        This method returns True if self.Result
        is not None and self.Result.Peaks is
        not None and len(self.Result.Peaks)
        is greater than 0. It returns False otherwise.

        Returns
        ----------
        bool
        """

        if self.Result is not None:
            if self.Result.Peaks is not None:
                if len(self.Result.Peaks) > 0:
                    return True
        
        return False