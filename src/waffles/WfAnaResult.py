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
    PeaksPos : unidimensional numpy array of integers
        Iterator values for the points within the waveform
        Adcs for which a peak was spotted
    PeaksAmpl : unidimensional numpy array of floats
        PeaksAmpl[i] is the amplitude of the i-th peak
        with respect to the computed baseline
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

    def __init__(self,  baseline,
                        baseline_min,
                        baseline_max,
                        baseline_rms,
                        peaks_pos,
                        peaks_ampl,
                        integral,
                        deconvoluted_adcs):
        
        """WfAnaResult class initializer
        
        Parameters
        ----------
        baseline : float
        baseline_min : float
        baseline_max : float
        baseline_rms : float
        peaks_pos : unidimensional numpy array of integers
        peaks_ampl : unidimensional numpy array of floats
        integral : float
        deconvoluted_adcs : unidimensional numpy array of floats

        """
        ## Shall we add type checks here?

        self.__baseline = baseline
        self.__baseline_min = baseline_min
        self.__baseline_max = baseline_max
        self.__baseline_rms = baseline_rms
        self.__peaks_pos = peaks_pos
        self.__peaks_ampl = peaks_ampl
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
    def PeaksPos(self):
        return self.__peaks_pos
    
    @property
    def PeaksAmpl(self):
        return self.__peaks_ampl
    
    @property
    def Integral(self):
        return self.__integral
    
    @property
    def DeconvolutedAdcs(self):
        return self.__deconvoluted_adcs
    
    # There are no setters for these attributes, since they
    # should be solely set by an analyser method of WfAna
    # when it creates this WfAnaResult object.