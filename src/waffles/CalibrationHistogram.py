import numpy as np
from typing import List, Optional

from .WaveformSet import WaveformSet
from .Exceptions import generate_exception_message

class CalibrationHistogram:

    """
    Stands for Calibration Histogram. This class
    implements a histogram which is used for
    SiPM-based detector calibration. A well formed
    calibration histogram displays a number of
    well defined peaks, which match the 0-PE, 1-PE,
    ..., N-PE waveforms, for some N>=1. This 
    histogram keeps track of which Waveform 
    objects contribute to which bin, by keeping 
    its indices with respect to some assumed 
    ordering.
    
    Attributes
    ----------
    BinsNumber : int
        Number of bins in the histogram. It must
        be greater than 1.
    Edges : unidimensional numpy array of floats
        Its length must match BinsNumber + 1. The
        i-th bin, with i = 0, ..., BinsNumber - 1, 
        contains the number of occurrences between 
        Edges[i] and Edges[i+1].
    MeanBinWidth : float
        The mean difference between two consecutive
        edges. It is computed as
        (Edges[BinsNumber] - Edges[0]) / BinsNumber.
        For calibration histograms with an uniform
        binning, this value matches 
        (Edges[i+1] - Edges[i]) for all i.
    Counts : unidimensional numpy array of integers
        Its length must match BinsNumber. Counts[i] 
        gives the number of occurrences in the i-th 
        bin, with i = 0, ..., BinsNumber - 1.
    Indices : list of lists of integers
        Its length must match BinsNumber. Indices[i] 
        gives the list of indices, with respect to 
        some ordering, of the Waveform objects which 
        contributed to the i-th bin. Note that the 
        length of Indices[i] must match Counts[i], 
        for i = 0, ..., BinsNumber - 1.
    GaussianFitsParameters : dict of list of tuples of floats
        The keys for this dictionary is 
        'scale', 'mean', and 'std'. The value for
        each key is a list of tuples. The i-th
        element of the list whose key is 'scale'
        (resp. 'mean', 'std'), gives a tuple with
        two floats, where the first element is the 
        scaling factor (resp. mean, standard 
        deviation) of the i-th gaussian fit of this 
        histogram, and the second one is the error
        of such fit parameter.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  bins_number : int,
                        edges : np.ndarray,
                        counts : np.ndarray,
                        indices : List[List[int]]):
        
        """
        CalibrationHistogram class initializer. It is the 
        caller's responsibility to check the types of the 
        input parameters. No type checks are perfomed here.
        
        Parameters
        ----------
        bins_number : int
        edges : unidimensional numpy array of floats
        counts : unidimensional numpy array of integers
        indices : list of lists of integers
        """

        if bins_number < 2:
            raise Exception(generate_exception_message( 1,
                                                        'CalibrationHistogram.__init__()',
                                                        f"The given bins number ({bins_number}) must be greater than 1."))
        if len(edges) != bins_number + 1:
            raise Exception(generate_exception_message( 2,
                                                        'CalibrationHistogram.__init__()',
                                                        f"The length of the 'edges' parameter ({len(edges)}) must match 'bins_number + 1' ({bins_number + 1})."))
        if len(counts) != bins_number:
            raise Exception(generate_exception_message( 3,
                                                        'CalibrationHistogram.__init__()',
                                                        f"The length of the 'counts' parameter ({len(counts)}) must match 'bins_number' ({bins_number})."))
        if len(indices) != bins_number:
            raise Exception(generate_exception_message( 4,
                                                        'CalibrationHistogram.__init__()',
                                                        f"The length of the 'indices' parameter ({len(indices)}) must match 'bins_number' ({bins_number})."))
        for i in range(bins_number):
            if len(indices[i]) != counts[i]:
                raise Exception(generate_exception_message( 5,
                                                            'CalibrationHistogram.__init__()',
                                                            f"The length of 'indices[{i}]' parameter ({len(indices[i])}) must match 'counts[{i}]' ({counts[i]})."))
        self.__bins_number = bins_number
        self.__edges = edges
        self.__mean_bin_width = ( self.__edges[self.__bins_number] - self.__edges[0] ) / self.__bins_number
        self.__counts = counts
        self.__indices = indices

        self.__gaussian_fits_parameters = None
        self.__reset_gaussian_fit_parameters()

    #Getters
    @property
    def BinsNumber(self):
        return self.__bins_number
    
    @property
    def Edges(self):
        return self.__edges
    
    @property
    def MeanBinWidth(self):
        return self.__mean_bin_width
    
    @property
    def Counts(self):
        return self.__counts
    
    @property
    def Indices(self):
        return self.__indices
    
    @property
    def GaussianFitsParameters(self):
        return self.__gaussian_fits_parameters
    
    def __reset_gaussian_fit_parameters(self) -> None:

        """
        This method is not intended for user usage. It
        resets the GaussianFitsParameters attribute to
        its initial state.
        """

        self.__gaussian_fits_parameters = { 'scale':[], 
                                            'mean':[], 
                                            'std':[]}
        return
    
    def __add_gaussian_fit_parameters(self, scale : float,
                                            scale_err : float,
                                            mean : float,
                                            mean_err : float,
                                            std : float,
                                            std_err : float) -> None:
        
        """
        This method is not intended for user usage.
        It takes care of adding the given fit parameters 
        to the GaussianFitsParameters attribute according 
        to its structure. No checks are performed in this 
        function regarding the values of the input 
        parameters.

        Parameters
        ----------
        scale : float
            The scaling factor of the gaussian fit
        scale_err : float
            The error of the scaling factor of the 
            gaussian fit
        mean : float
            The mean value of the gaussian fit
        mean_err : float
            The error of the mean value of the 
            gaussian fit
        std : float
            The standard deviation of the gaussian fit
        std_err : float
            The error of the standard deviation of the 
            gaussian fit

        Returns
        ----------
        None
        """

        self.__gaussian_fits_parameters['scale'].append((scale, scale_err))
        self.__gaussian_fits_parameters['mean'].append((mean, mean_err))
        self.__gaussian_fits_parameters['std'].append((std, std_err))

        return
    
    @classmethod
    def from_WaveformSet(cls,   waveform_set : WaveformSet,
                                bins_number : int,
                                domain : np.ndarray,
                                variable : str = 'integral',
                                analysis_label : Optional[str] = None):
        
        """
        This method creates a CalibrationHistogram object by
        adding the values, of the given variable, of the
        Waveform objects from the given WaveformSet object.
        It is the caller's responsibility to ensure that the
        type of the input parameters is suited. No type checks
        are performed here.

        Parameters
        ----------
        waveform_set : WaveformSet
            The WaveformSet object from where to take the
            Waveform objects to add to the calibration
            histogram.
        bins_number : int
            The number of bins for the created calibration
            histogram. It must be greater than 1.
        domain : np.ndarray
            A 2x1 numpy array where (domain[0], domain[1])
            gives the range to consider for the created
            calibration histogram. Any sample which falls 
            outside this range is ignored.
        variable : str
            If variable is set to 'integral', then the 
            calibration histogram will be computed using 
            the integral of the waveforms, up to the 
            input given to the 'analysis_label' parameter. 
            If variable is set to 'amplitude', then the 
            calibration histogram will be computed using 
            the amplitude of the waveforms. The default                     ## Not implemented yet
            behaviour, which is used if the input is 
            different from 'integral' or 'amplitude', 
            is that of 'integral'.
        analysis_label : str
            This parameter only makes a difference if
            'variable' is set to 'integral'. In such 
            case, this parameter gives the key for 
            the WfAna object within the Analyses 
            attribute of each considered waveform 
            from where to take the integral value to 
            add to the calibration histogram. Namely, 
            if such WfAna object is x, then 
            x.Result.Integral is the considered
            integral. If 'analysis_label' is None, 
            then the last analysis added to 
            the Analyses attribute will be the used 
            one. If there is not even one analysis, 
            then an exception will be raised.

        Returns
        ----------
        output : CalibrationHistogram
            The created calibration histogram
        """
        
        if bins_number < 2:
            raise Exception(generate_exception_message( 1,
                                                        'CalibrationHistogram.from_WaveformSet()',
                                                        f"The given bins number ({bins_number}) must be greater than 1."))
        if np.ndim(domain) != 1 or len(domain) != 2:
            raise Exception(generate_exception_message( 2,
                                                        'CalibrationHistogram.from_WaveformSet()',
                                                        f"The 'domain' parameter must be a 2x1 numpy array."))
        fUseIntegral = False
        if variable != 'amplitude':
            fUseIntegral = True

        if fUseIntegral:
            return cls.__from_integrals_of_WaveformSet( waveform_set,
                                                        bins_number,
                                                        domain,
                                                        analysis_label)
        else:
            raise Exception(generate_exception_message( 3,
                                                        'CalibrationHistogram.from_WaveformSet()',
                                                        f'Amplitude histograms are not implemented yet'))

    @classmethod
    def __from_integrals_of_WaveformSet(cls,    waveform_set : WaveformSet,
                                                bins_number : int,
                                                domain : np.ndarray,
                                                analysis_label : Optional[str] = None) -> 'CalibrationHistogram':
        
        """
        This method is not intended for user usage. It must
        only be called by the 
        CalibrationHistogram.from_WaveformSet() class
        method, which ensures that the input parameters
        are well-formed. No checks are perfomed here.

        Parameters
        ----------
        waveform_set : WaveformSet
        bins_number : int
            It is given to the 'bins' parameter of
            the WaveformSet.histogram1d() static method.
        domain : np.ndarray
            It is given to the 'domain' parameter of 
            the WaveformSet.histogram1d() static method.
        analysis_label : str

        Returns
        ----------
        output : CalibrationHistogram
            The created calibration histogram
        """

        edges = np.linspace(domain[0],
                            domain[1], 
                            num = bins_number + 1,
                            endpoint = True)
        
        samples = np.array([ waveform_set.Waveforms[idx].get_analysis(analysis_label).Result.Integral for idx in range(len(waveform_set.Waveforms)) ])  ## Trying to grab the WfAna object
                                                                                                                                                        ## waveform by waveform using 
                                                                                                                                                        ## WaveformAdcs.get_analysis()
                                                                                                                                                        ## might be slow. Find a different
                                                                                                                                                        ## solution if this becomes a 
                                                                                                                                                        ## a problem at some point.        
        counts, indices = WaveformSet.histogram1d(  samples,
                                                    bins_number,
                                                    domain,
                                                    keep_track_of_idcs = True)
        return cls( bins_number,
                    edges,
                    counts,
                    indices)
    
    @staticmethod
    def gaussian(   x : float, 
                    scale : float, 
                    mean : float, 
                    std : float) -> float:

        """
        Evaluates an scaled gaussian function
        at x. The function is defined as:
        
        f(x) = scale * exp( -1 * (( x - mean ) / ( 2 * std )) ** 2)

        This function supports numpy arrays as input.
        
        Parameters
        ----------
        x : float
            The point at which the function is evaluated.
        scale : float
            The scale factor of the gaussian function
        mean : float
            The mean value of the gaussian function
        std : float
            The standard deviation of the gaussian function

        Returns
        -------
        float
            The value of the function at x
        """
    
        return scale * np.exp( -1. * (np.power( ( x - mean ) / ( 2 * std ), 2)))