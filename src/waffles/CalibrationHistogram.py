import math

import numpy as np
from scipy import signal as spsi
from scipy import optimize as spopt
from plotly import graph_objects as pgo
from typing import Tuple, List, Dict, Optional

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
    
    def __spot_first_peaks(self,    max_peaks : int,
                                    prominence : float,
                                    initial_percentage = 0.1,
                                    percentage_step = 0.1) -> Tuple[bool, Tuple]:
        
        """
        This method is not intended for user usage. It must
        be only called by CalibrationHistogram.fit_peaks(),
        where the well-formedness checks of the input
        parameters have been performed. This method tries 
        to find peaks over the signal which is computed as

            signal = (self.Counts - np.min(self.Counts))/np.max(self.Counts)

        This method iteratively calls

            scipy.signal.find_peaks(signal[0:points], 
                                    prominence = prominence)

        to spot, at most, max_peaks peaks. To do so, at the 
        first iteration, points is computed as 
        math.floor(initial_percentage * len(signal)). If the 
        number of spotted peaks is less than max_peaks, then 
        points is increased by 
        math.floor(percentage_step * len(signal)) and the 
        scipy.signal.find_peaks() method is called again. This 
        process is repeated until the number of spotted peaks
        is equal to max_peaks, or until the number of points 
        reaches len(signal). If the number of points reaches 
        len(signal), then scipy.signal.find_peaks() is called 
        one last time as

            scipy.signal.find_peaks(signal, 
                                    prominence = prominence)
        
        If the last call found a number of peaks smaller than
        max_peaks, then this function returns (False, peaks),
        where peaks is the output of the last call to 
        scipy.signal.find_peaks(). If the last call found a
        number of peaks greater than or equal to max_peaks, 
        then the function returns (True, peaks), where peaks 
        is the output of scipy.signal.find_peaks() but 
        truncated to the first max_peaks found peaks.

        Parameters
        ----------
        max_peaks : int
            The maximum number of peaks to spot. It must 
            be a positive integer. This is not checked here, 
            it is the caller's responsibility to ensure this.
        prominence : float
            The prominence parameter to pass to the
            scipy.signal.find_peaks() method. Since the
            signal is normalized, this prominence can be
            understood as a fraction of the total amplitude
            of the signal. P.e. setting prominence to 0.5,
            will prevent scipy.signal.find_peaks() from
            spotting peaks whose amplitude is less than
            half of the total amplitude of the signal.
        initial_percentage : float
            The initial percentage of the signal to
            consider. It must be greater than 0.0
            and smaller than 1.0.
        percentage_step : float
            The percentage step to increase the signal
            to consider in successive calls of 
            scipy.signal.find_peaks(). It must be greater 
            than 0.0 and smaller than 1.0.

        Returns
        -------
        output : tuple of ( bool,  tuple, )
            The first entry is a boolean which is True if 
            the number of peaks found is greater than or 
            equal to max_peaks, and False otherwise. The
            second entry is the output of the last call to
            scipy.signal.find_peaks() if the first entry
            of the tuple is False. If the first entry of
            the tuple is True, then the second entry is
            the output of scipy.signal.find_peaks() but
            truncated to the first max_peaks found peaks. 
            For more information, check the 
            scipy.signal.find_peaks() documentation.
        """

        signal = ( self.Counts - np.min(self.Counts) ) / np.max(self.Counts)

        fFoundMax = False
        fReachedEnd = False
        points = math.floor( initial_percentage * len(signal) )

        while not fFoundMax and not fReachedEnd:

            points = min(points, len(signal))

            spsi_output = spsi.find_peaks(  signal[0:points],
                                            prominence = prominence,
                                            width = 0,                  # Adding a minimal 0 width, which constraints nothing,
                                            rel_height = 0.5)           # but which makes scipy.signal.find_peaks() return
                                                                        # information about each peak-width at half its height.
            if len(spsi_output[0]) >= max_peaks:

                spsi_output = CalibrationHistogram.trim_spsi_find_peaks_output_to_max_peaks(spsi_output,
                                                                                            max_peaks)
                fFoundMax = True

            if points == len(signal):
                fReachedEnd = True

            points += math.floor( percentage_step * len(signal) )

        if fFoundMax:
            return (True, spsi_output)
        else:
            return (False, spsi_output)
        
    @staticmethod
    def trim_spsi_find_peaks_output_to_max_peaks(   spsi_output : Tuple[np.ndarray, Dict[str, np.ndarray]],
                                                    max_peaks : int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        
        """
        This function gets the output of a certain call to 
        scipy.signal.find_peaks() and returns the same output, 
        but truncated to the first max_peaks found peaks. 

        Parameters
        ----------
        spsi_output : tuple of ( np.ndarray,  dict, )
            The output of a call to scipy.signal.find_peaks().
            No checks are performed here regarding the
            well-formedness of this input.
        max_peaks : int
            The maximum number of peaks to keep. It must be a
            positive integer. This is not checked here. It is
            the caller's responsibility to ensure this. If the 
            given spsi_output has a number less or equal to 
            max_peaks peaks, then the output of this function 
            will be the same as the given spsi_output. If it
            has more than max_peaks peaks, then the output of
            this function will contain only the information
            regarding the first max_peaks peaks of the given 
            spsi_output.

        Returns
        -------
        output : tuple of ( np.ndarray, dict, )
            The first element is an unidimensional numpy array 
            which contains the first max_peaks elements of the 
            first element of the given spsi_output. The second 
            element is a dictionary which contains the same keys 
            as the second element of the given spsi_output, but 
            the values (which are unidimensional numpy arrays) 
            contain only the first max_peaks elements of the 
            values of the second element of the given spsi_output.
        """

        if len(spsi_output[0]) <= max_peaks:
            return spsi_output
        else:
            return (spsi_output[0][0:max_peaks], {key: value[0:max_peaks] for key, value in spsi_output[1].items()})
        
    def fit_peaks(self, max_peaks : int,
                        prominence : float,
                        half_points_to_fit : int,
                        initial_percentage = 0.1,
                        percentage_step = 0.1) -> bool:
            
        """
        This method
            
            -   tries to find the first max_peaks of 
                this calibration histogram whose prominence
                is greater than the given prominence parameter,
                using the scipy.signal.find_peaks() function 
                iteratively. This function delegates this
                task to the 
                CalibrationHistogram.__spot_first_peaks() 
                method.

            -   Then, it fits a gaussian function to each one
                of the found peaks using the output of 
                the last call to scipy.signal.find_peaks()
                (which is returned by 
                CalibrationHistogram.__spot_first_peaks())
                as a seed for the fit.

            -   Finally, it stores the fit parameters in the
                GaussianFitsParameters attribute, according
                to its structure, which can be found in the
                class documentation.

        This method returns True if the number of found peaks
        matches the given max_peaks parameter, and False
        if it is smaller than max_peaks.
        
        Parameters
        ----------
        max_peaks : int
            It must be a positive integer. It gives the
            maximum number of peaks that could be possibly
            fit. This parameter is passed to the 'max_peaks'
            parameter of the 
            CalibrationHistogram.__spot_first_peaks() method.
        prominence : float
            It must be greater than 0.0 and smaller than 1.0.
            It gives the minimal prominence of the peaks to 
            spot. This parameter is passed to the 'prominence' 
            parameter of the 
            CalibrationHistogram.__spot_first_peaks() method,
            where it is interpreted as the fraction of the 
            total amplitude of the histogram which is required 
            for a peak to be spotted as such. P.e. setting 
            prominence to 0.5, will prevent scipy.signal.find_peaks() 
            from spotting peaks whose amplitude is less than
            half of the total amplitude of the histogram.
        half_points_to_fit : int
            It must be a positive integer. For each peak, it 
            gives the number of points to consider on either 
            side of the peak maximum, to fit each gaussian 
            function. I.e. if i is the iterator value for
            self.Counts of the i-th peak, then the histogram
            bins which will be considered for the fit are
            given by the slice 
            self.Counts[i - half_points_to_fit : i + half_points_to_fit + 1].
        initial_percentage : float
            It must be greater than 0.0 and smaller than 1.0.
            This parameter is passed to the 'initial_percentage' 
            parameter of the 
            CalibrationHistogram.__spot_first_peaks() method. 
            For more information, check the documentation of 
            such method.
        percentage_step : float
            It must be greater than 0.0 and smaller than 1.0.
            This parameter is passed to the 'percentage_step' 
            parameter of the 
            CalibrationHistogram.__spot_first_peaks() method. 
            For more information, check the documentation of 
            such method.

        Returns
        -------
        bool
            True if the number of found peaks matches the given
            max_peaks parameter, and False if it is smaller than
            max_peaks.
        """

        if max_peaks < 1:
            raise Exception(generate_exception_message( 1,
                                                        'CalibrationHistogram.fit_peaks()',
                                                        f"The given max_peaks ({max_peaks}) must be greater than 0."))
        if prominence <= 0.0 or prominence >= 1.0:
            raise Exception(generate_exception_message( 2,
                                                        'CalibrationHistogram.fit_peaks()',
                                                        f"The given prominence ({prominence}) must be greater than 0.0 and smaller than 1.0."))
        
        if initial_percentage <= 0.0 or initial_percentage >= 1.0:
            raise Exception(generate_exception_message( 3,
                                                        'CalibrationHistogram.fit_peaks()',
                                                        f"The given initial_percentage ({initial_percentage}) must be greater than 0.0 and smaller than 1.0."))
    
        if percentage_step <= 0.0 or percentage_step >= 1.0:
            raise Exception(generate_exception_message( 4,
                                                        'CalibrationHistogram.fit_peaks()',
                                                        f"The given percentage_step ({percentage_step}) must be greater than 0.0 and smaller than 1.0."))
        self.__reset_gaussian_fit_parameters()

        fFoundMax, spsi_output = self.__spot_first_peaks(   max_peaks,
                                                            prominence,
                                                            initial_percentage,
                                                            percentage_step)
        peaks_n_to_fit = len(spsi_output[0])

        for i in range(peaks_n_to_fit):
                
            aux_idx  = spsi_output[0][i]

            aux_seeds = [   self.Counts[aux_idx],                                           # Scale seed
                            (self.Edges[aux_idx] + self.Edges[aux_idx + 1]) / 2.,           # Mean seed
                            spsi_output[1]['widths'][i] * self.__mean_bin_width / 2.355]    # Std seed : Note that 
                                                                                            # CalibrationHistogram.__spot_first_peaks() 
                                                                                            # is computing the widths of the peaks, in
                                                                                            # number of samples, at half of their height 
                                                                                            # (rel_height = 0.5). 2.355 is approximately 
                                                                                            # the conversion factor between the standard 
                                                                                            # deviation and the FWHM. Also, note that here
                                                                                            # we are assuming that the binning is uniform.
            aux_lower_lim = max(0,
                                aux_idx - half_points_to_fit)   # Restrict the fit lower limit to 0
            
            aux_upper_lim = min(len(self.Counts) - 1,                   # The upper limit should be restricted len(self.Counts).
                                aux_idx + half_points_to_fit + 1)       # Making it be further restricted to len(self.Counts) - 1
                                                                        # so that there is always available data to compute the
                                                                        # center of the bins, in the following line.
            
            aux_bin_centers = ( self.Edges[aux_lower_lim : aux_upper_lim] + self.Edges[aux_lower_lim + 1 : aux_upper_lim + 1] ) / 2.
            aux_counts = self.Counts[aux_lower_lim : aux_upper_lim]

            try:
                aux_optimal_parameters, aux_covariance_matrix  = spopt.curve_fit(   CalibrationHistogram.gaussian, 
                                                                                    aux_bin_centers, 
                                                                                    aux_counts, 
                                                                                    p0 = aux_seeds)
            except RuntimeError:    # Happens if scipy.optimize.curve_fit()
                                    # could not converge to a solution
        
                fFoundMax = False   # In this case, we will skip this peak
                                    # (so, in case fFoundMax was True, now 
                                    # it must be false) and we will continue 
                                    # with the next one, if any
                continue    

            aux_errors = np.sqrt(np.diag(aux_covariance_matrix))

            self.__add_gaussian_fit_parameters( aux_optimal_parameters[0],
                                                aux_errors[0],
                                                aux_optimal_parameters[1],
                                                aux_errors[1],
                                                aux_optimal_parameters[2],
                                                aux_errors[2])
        return fFoundMax

    def plot(self,  figure : pgo.Figure,
                    name : Optional[str] = None,
                    row : Optional[int] = None,
                    col : Optional[int] = None,
                    plot_fits : bool = False,
                    fit_npoints : int = 200) -> None:
        
        """
        This method plots this calibration histogram in the given 
        figure.
        
        Parameters
        ----------
        figure : plotly.graph_objects.Figure
            The figure in which the calibration histogram (CH) 
            will be plotted
        name : str
            The name for the CH trace which will be added to 
            the given figure.
        row (resp. col) : int
            The row (resp. column) in which the CH will be 
            plotted. This parameter is directly handled to
            the 'row' (resp. 'col') parameter of
            plotly.graph_objects.Figure.add_trace(). It is the
            caller's responsibility to ensure two things:
                
                - if the given 'figure' parameter does not contain
                  a subplot grid (p.e. it was not created by
                  plotly.subplots.make_subplots()) then 'row' and
                  'col' must be None.
                   
                - if the given 'figure' parameter contains a subplot
                  grid, then 'row' and 'col' must be valid 1-indexed
                  integers.
        plot_fits : bool
            If True, then the gaussian fits of the peaks, if any, 
            will be plotted over the CH. If False, then only the 
            CH will be plotted. Note that if no fit has been performed
            yet, then the self.__gaussian_fits_parameters attribute
            will be empty and no fit will be plotted.
        fit_npoints : int
            This parameter only makes a difference if 'plot_fits'
            is set to True. In that case, it gives the number of
            points to use to plot each gaussian fit. Note that
            the plot range of the fit will be the same as the
            range of the CH. It must be greater than 1. It is
            the caller's responsibility to ensure this.
        """

        histogram_trace = pgo.Scatter(  x = self.Edges,
                                        y = self.Counts,
                                        mode = 'lines',
                                        line=dict(  color = 'black', 
                                                    width = 0.5,
                                                    shape = 'hv'),
                                        name = name)
        
        figure.add_trace(   histogram_trace,
                            row = row,
                            col = col)
        if plot_fits:

            for i in range(len(self.GaussianFitsParameters['scale'])):

                fit_x = np.linspace(self.Edges[0],
                                    self.Edges[-1],
                                    num = fit_npoints)
                
                fit_y = CalibrationHistogram.gaussian(  fit_x,
                                                        self.GaussianFitsParameters['scale'][i][0],
                                                        self.GaussianFitsParameters['mean'][i][0],
                                                        self.GaussianFitsParameters['std'][i][0])
                fit_trace = pgo.Scatter(x = fit_x,
                                        y = fit_y,
                                        mode = 'lines',
                                        line=dict(  color = 'red', 
                                                    width = 0.5),
                                        name = f"{name} (Fit {i})")
                
                figure.add_trace(   fit_trace,
                                    row = row,
                                    col = col)  
        return