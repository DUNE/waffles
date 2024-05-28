from typing import Tuple, List

import numpy as np
from scipy import signal as spsi

from src.waffles.WfAnaResult import WfAnaResult

class WfAna:

    """
    Stands for Waveform Analysis. This class 
    implements an analysis which has been performed 
    over a certain waveform.
    
    Attributes
    ----------
    BaselineLimits : list of int
        It must have an even number of integers which
        must meet BaselineLimits[i]<BaselineLimits[i+1].
        Given a Waveform object whose adcs array is x, 
        the points which are used for baseline calculation
        are x[BaselineLimits[2*i]:BaselineLimits[2*i+1]],
        with i=0,1,...,(len(BaselineLimits)/2)-1. The 
        upper limits are exclusive.
    IntLl (resp. IntUl) : int
        Stands for integration lower (resp. upper) limit.
        Iterator value for the first (resp. last) point 
        of the waveform that falls into the integration
        window. IntLl must be smaller than IntUl. These
        limits are inclusive.
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
        
        """WfAna class initializer. It is assumed that it is
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
        self.__passed = None    # by an analyzer method

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
                                                    # Waveform.analyze() should be able to set
        ## Shall we add a type check here?          # them. Not adding setters for the rest of 
                                                    # the attributes, since they shall not be
        self.__result = input                       # set outside WfAna code
        return
    
    @Passed.setter
    def Passed(self, input):

        ## Shall we add a type check here?

        self.__passed = input
        return

    def analyzer_template(  self,
                            waveform : 'Waveform',
                            *args,
                            **kwargs) -> Tuple[WfAnaResult, bool]:
        
        """
        This method implements a template for an analyzer 
        method.

        Parameters
        ----------
        waveform : Waveform
            Waveform object which will be analyzed.
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
        return output_1, output_2
    
    def standard_analyzer(  self,
                            waveform : 'Waveform',
                            *args,
                            **kwargs) -> Tuple[WfAnaResult, bool]:
        
        """
        This method implements an analyzer method
        which does the following:

            - It computes the baseline as the median of the points
            that are considered, according to the self.__baseline_limits
            attribute.
            - It searches for peaks using scipy.signal.find_peaks.
            - It calculates the integral of 
            waveform.Adcs[IntLl:IntUl+1]. To do so, it assumes that
            the temporal resolution of the waveform is constant and
            and approximates its integral to 
            waveform.TimeStep_ns*np.sum(-b+waveform.Adcs[IntLl:IntUl+1]),
            where b is the comptued baseline.

        Parameters
        ----------
        waveform : Waveform
            Waveform object which will be analyzed.
        *args, **kwargs
            These arguments are passed to 
            scipy.signal.find_peaks(waveform.Adcs, *args, **kwargs)

        Returns
        ----------
        output_1 : WfAnaResult
            The result of the analysis
        output_2 : bool
            True (resp. False) if analysis concluded
            that the waveform passes (resp. fails).
        """
        
        split_baseline_samples = [  waveform.Adcs[ self.__baseline_limits[2*i] : self.__baseline_limits[(2*i)+1]]
                                    for i in range(len(self.__baseline_limits)//2) ]
        
        baseline_samples = np.concatenate(split_baseline_samples)
        baseline = np.median(baseline_samples)

        peaks, properties = spsi.find_peaks(waveform.Adcs, *args, **kwargs)

        output_1 =  WfAnaResult(baseline,
                                np.min(baseline_samples),
                                np.max(baseline_samples),
                                np.std(baseline_samples),   ## Not definitive, computes STD not RMS
                                peaks,
                                np.array([ waveform.Adcs[it]-baseline for it in peaks ]),
                                waveform.TimeStep_ns*(np.sum(waveform.Adcs[self.__int_ll:self.__int_ul+1])-baseline),
                                None)                       ## deconvoluted_adcs , not computed by this standard analyzer
        
        output_2 = True ## This standard analyzer does not implement a quality filter yet
        return output_1, output_2