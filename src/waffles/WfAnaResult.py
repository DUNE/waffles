from typing import List

import numpy as np

from src.waffles.WfPeak import WfPeak
from src.waffles.Exceptions import generate_exception_message

class WfAnaResult:

    """
    Stands for Waveform Analysis Result. This class 
    wraps the results of an analysis which has been
    performed over a certain waveform. If the analysis
    that is run does not contemplate a certain attribute,
    then such attribute will remain set to None,
    which should be interpreted as unavailable data.
    
    Attributes
    ----------
    Baseline : float
        The resulting value for the waveform baseline
    BaselineMin (resp. BaselineMax) : float
        The minimum (resp. maximum) value of the 
        waveform chunk which was considered for the 
        baseline calculation
    BaselineRms : float
        The RMS of the chunk of the waveform which has
        been used for the baseline calculation
    Peaks : list of WfPeak objects
        The peaks which have been spotted in the waveform
    Integral : float
        The integral of the waveform Adcs
    DeconvolutedAdcs : unidimensional numpy array of floats         ## For the moment, this is all of the information
        The deconvoluted waveform Adcs                              ## resulting from the deconvolution analysis that 
                                                                    ## we keep, since this is a bottleneck for such 
                                                                    ## analysis. However, in the future we could add 
                                                                    ## any further attribute which implements any FoM 
                                                                    ## that comes from the deconvolution analysis, such 
                                                                    ## as the purity.
    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  baseline : float,
                        baseline_min : float,
                        baseline_max : float,
                        baseline_rms : float,
                        peaks : List[WfPeak],
                        integral : float,
                        deconvoluted_adcs : np.ndarray):
        
        """
        WfAnaResult class initializer
        
        Parameters
        ----------
        baseline : float
        baseline_min : float
        baseline_max : float
            baseline_max must be bigger than baseline_min
        baseline_rms : float
        peaks : list of WfPeak objects
        integral : float
        deconvoluted_adcs : unidimensional numpy array of floats

        """
        ## Shall we add type checks here?

        self.__baseline = baseline

        if baseline_min >= baseline_max:                                            # If this check makes the execution time 
            raise Exception(generate_exception_message( 1,                          # be prohibitively high, it may be removed
                                                        'WfAnaResult.__init__()',
                                                        f"'baseline_min' ({baseline_min}) cannot be bigger or equal to 'baseline_max' ({baseline_max})."))

        self.__baseline_min = baseline_min
        self.__baseline_max = baseline_max
        self.__baseline_rms = baseline_rms
        self.__peaks = peaks
        self.__integral = integral
        self.__deconvoluted_adcs = deconvoluted_adcs

    #Getters
    @property
    def Baseline(self):
        return self.__baseline
    
    @property
    def BaselineMin(self):
        return self.__baseline_min
    
    @property
    def BaselineMax(self):
        return self.__baseline_max
    
    @property
    def BaselineRms(self):
        return self.__baseline_rms
    
    @property
    def Peaks(self):
        return self.__peaks
    
    @property
    def Integral(self):
        return self.__integral
    
    @property
    def DeconvolutedAdcs(self):
        return self.__deconvoluted_adcs
    
    # There are no setters for these attributes, since they
    # should be solely set by an analyser method of WfAna
    # when it creates this WfAnaResult object.