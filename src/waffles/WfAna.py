from typing import Tuple, List, TYPE_CHECKING

import numpy as np
from scipy import signal as spsi

if TYPE_CHECKING:                                           # Import only for type-checking, so as
    from src.waffles.WaveformAdcs import WaveformAdcs       # to avoid a runtime circular import
                                                    
from src.waffles.WfAnaResult import WfAnaResult
from src.waffles.WfPeak import WfPeak

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
                        int_ul : int):
        
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
        """
        
        self.__baseline_limits = baseline_limits
        self.__int_ll = int_ll
        self.__int_ul = int_ul
        
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

        output_1 = WfAnaResult( 0,                              # baseline
                                0,                              # baseline_min
                                10,                             # baseline_max
                                1,                              # baseline_rms
                                np.array([10,20]),              # peaks_pos
                                np.array([30.,10.]),            # peaks_ampl
                                10.0,                           # integral
                                np.array([0.,-1.,1.,0.,-1.]))   # deconvoluted_adcs
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

        Note that for these computations to be well-defined, it is
        assumed that

            - BaselineLimits[0] - wf.TimeOffset >= 0
            - BaselineLimits[-1] - wf.TimeOffset <= len(wf.Adcs)
            - IntLl - wf.TimeOffset >= 0
            - IntUl - wf.TimeOffset < len(wf.Adcs)

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
                                                                                    ## way not to hardcode this.
        output_1 = WfAnaResult( baseline=baseline,
                                baseline_min=None,          # np.max(baseline_samples)
                                baseline_max=None,          # np.min(baseline_samples)
                                baseline_rms=None,          # np.std(baseline_samples)  ## Not definitive, computes STD not RMS
                                peaks= [ WfPeak(peaks[i]) for i in range(len(peaks)) ],
                                integral=None,              # waveform.TimeStep_ns*(np.sum(waveform.Adcs[self.__int_ll:self.__int_ul + 1]) - baseline)
                                deconvoluted_adcs=None)     ## deconvoluted_adcs , not computed by this standard analyser
        
        output_2 = True             ## This standard analyser does not implement a quality filter yet

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