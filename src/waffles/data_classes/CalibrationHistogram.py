import numba
import numpy as np
from typing import List, Optional, Union

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.TrackedHistogram import TrackedHistogram

import waffles.utils.numerical_utils as wun
import waffles.Exceptions as we

from iminuit import Minuit
from iminuit.cost import LeastSquares
from waffles.data_classes.CrossTalk import CrossTalk

class CalibrationHistogram(TrackedHistogram):
    """This class implements a histogram which is used
    for SiPM-based detector calibration. A well formed
    calibration histogram displays a number of
    well defined peaks, which match the 0-PE, 1-PE,
    ..., N-PE waveforms, for some N>=1. As it inherits
    from TrackedHistogram, this histogram keeps track
    of which Waveform objects contribute to which bin,
    by keeping its indices with respect to some assumed
    ordering.

    Attributes
    ----------
    bins_number: int (inherited from TrackedHistogram)
    edges: unidimensional numpy array of floats
    (inherited from TrackedHistogram)
    mean: float (inherited from TrackedHistogram)
    nentries: int (inherited from TrackedHistogram)
    mean_bin_width: float (inherited from TrackedHistogram)
    counts: unidimensional numpy array of integers
    (inherited from tracked_Histogram)
    indices: list of lists of integers (inherited from TrackedHistogram)
    gaussian_fits_parameters: dict of list of tuples of floats
        The keys for this dictionary are
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

    def __init__(
        self, 
        bins_number: int,
        edges: np.ndarray,
        counts: np.ndarray,
        indices: List[List[int]],
        normalization: float = 1.0
    ):
        """CalibrationHistogram class initializer. It is the
        caller's responsibility to check the types of the
        input parameters. No type checks are perfomed here.

        Parameters
        ----------
        bins_number: int
        edges: unidimensional numpy array of floats
        counts: unidimensional numpy array of integers
        indices: list of lists of integers
        """

        if not np.any(counts):
            raise we.EmptyCalibrationHistogram(
                we.GenerateExceptionMessage(
                    1,
                    'CalibrationHistogram.__init__()',
                    "The given calibration histogram is empty."
                )
            )

        super().__init__(
            bins_number,
            edges,
            counts,
            indices)

        self.normalization = normalization
        self.__gaussian_fits_parameters = {}
        self.__reset_gaussian_fit_parameters()

    @property
    def gaussian_fits_parameters(self):
        return self.__gaussian_fits_parameters

    def __reset_gaussian_fit_parameters(self) -> None:
        """This method is not intended for user usage. 
        It resets the gaussian_fits_parameters attribute 
        to its initial state.
        """

        self.__gaussian_fits_parameters = {
            'scale': [],
            'mean': [],
            'std': []}
        return

    def __add_gaussian_fit_parameters(
        self, 
        scale: float,
        scale_err: float,
        mean: float,
        mean_err: float,
        std: float,
        std_err: float
    ) -> None:
        """This method is not intended for user usage.
        It takes care of adding the given fit parameters
        to the gaussian_fits_parameters attribute according
        to its structure. No checks are performed in this
        function regarding the values of the input
        parameters.

        Parameters
        ----------
        scale: float
            The scaling factor of the gaussian fit
        scale_err: float
            The error of the scaling factor of the
            gaussian fit
        mean: float
            The mean value of the gaussian fit
        mean_err: float
            The error of the mean value of the
            gaussian fit
        std: float
            The standard deviation of the gaussian fit
        std_err: float
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
    def from_WaveformSet(
        cls, 
        waveform_set: WaveformSet,
        bins_number: int,
        domain: np.ndarray,
        variable: str,
        analysis_label: Optional[str] = None,
        normalize_histogram: bool = False
    ):
        """This method creates a CalibrationHistogram object
        by taking one sample per Waveform from the given
        WaveformSet object. For each Waveform, the sample
        is taken by subscribing one of their analyses (up to
        the analysis_label input parameter) with the given
        variable. It is the caller's responsibility to
        ensure that the type of the input parameters is
        suited. No type checks are performed here.

        Parameters
        ----------
        waveform_set: WaveformSet
            The WaveformSet object from where to take the
            Waveform objects to add to the calibration
            histogram.
        bins_number: int
            The number of bins for the created calibration
            histogram. It must be greater than 1.
        domain: np.ndarray
            A 2x1 numpy array where (domain[0], domain[1])
            gives the range to consider for the created
            calibration histogram. Any sample which falls
            outside this range is ignored.
        variable: str
            For each Waveform object within the given
            Waveform set, this parameter gives the key
            for the considered WfAna object (up to the
            analysis_label input parameter) from where
            to take the sample to add to the calibration
            histogram. Namely, for a WfAna object x,
            x.result[variable] is the considered
            sample. It is the caller's responsibility to
            ensure that the values for the given variable
            (key) are scalars, i.e. that they are valid
            samples for a 1D histogram.
        analysis_label: str
            For each considered Waveform object, this
            parameter gives the key for the WfAna
            object within the analyses attribute from
            where to take the sample to add to the
            calibration histogram. If 'analysis_label'
            is None, then the last analysis added to the
            analyses attribute will be the used one. If
            there is not even one analysis, then an
            exception will be raised.

        Returns
        ----------
        output: CalibrationHistogram
            The created calibration histogram
        """

        if bins_number < 2:
            raise Exception(
                we.GenerateExceptionMessage(
                    1,
                    'CalibrationHistogram.from_WaveformSet()',
                    f"The given bins number ({bins_number}) must be"
                    " greater than 1."
                )
            )
        if np.ndim(domain) != 1 or len(domain) != 2:
            raise Exception(
                we.GenerateExceptionMessage(
                    2,
                    'CalibrationHistogram.from_WaveformSet()',
                    "The 'domain' parameter must be a 2x1 numpy array."
                )
            )

        # Trying to grab the WfAna object Waveform by Waveform using
        # WaveformAdcs.get_analysis() might be slow. Find a different
        # solution if this becomes a problem at some point.
        samples = [
            waveform_set.waveforms[idx].get_analysis(
                analysis_label
            ).result[variable]  for idx in range(    
                len(waveform_set.waveforms))
            if waveform_set.waveforms[idx].get_analysis( analysis_label).result[variable] is not np.nan
        ]
        meansample = 1.0
        if normalize_histogram:
            meansample = np.mean(samples)
            samples = [ s/meansample for s in samples ]
        try:
            return cls.__from_samples(
                samples,
                bins_number,
                domain,
                normalization=meansample
            )
        except numba.errors.TypingError:

            raise Exception(
                we.GenerateExceptionMessage(
                    3,
                    'CalibrationHistogram.from_WaveformSet()',
                    f"The given variable ('{variable}') does not give"
                    " suited samples for a 1D histogram."
                )
            )

    @classmethod
    def __from_samples(
        cls, 
        samples: List[Union[int, float]],
        bins_number: int,
        domain: np.ndarray,
        normalization: float = 1.0
    ) -> 'CalibrationHistogram':
        """This method is not intended for user usage. It 
        must be only called by the
        CalibrationHistogram.from_WaveformSet() class
        method, which ensures that the input parameters
        are well-formed. No checks are perfomed here.

        Parameters
        ----------
        samples: list of int or float
            The samples to add to the calibration histogram
        bins_number: int
            It is given to the 'bins' parameter of
            the waffles histogram1d() helper function.
        domain: np.ndarray
            It is given to the 'domain' parameter of
            the waffles histogram1d() helper function

        Returns
        ----------
        output: CalibrationHistogram
            The created calibration histogram
        """

        edges = np.linspace(domain[0],
                            domain[1],
                            num=bins_number + 1,
                            endpoint=True)

        counts, indices = wun.histogram1d(
            np.array(samples),
            bins_number,
            domain,
            keep_track_of_idcs=True)
        
        return cls(
            bins_number,
            edges,
            counts,
            indices,
            normalization=normalization
        )

    def compute_cross_talk(self) -> None:
        """
        This method computes the cross-talk of the SiPM
        from the fitted gaussian peaks of the calibration
        histogram. It relies on the Vinogradov et al. method
        described in Henrique's thesis arXiv:2112.02967v1
        page 108.
        It also create a CrossTalk object to store the results
        of the cross-talk analysis.
        Returns
        -------
        cross_talk: float
            The estimated cross-talk value
        err_cross_talk: float
            The uncertainty in the cross-talk estimate
        """
        if len(self.gaussian_fits_parameters["mean"]) < 3:
            print("Peak fitting failed.")
            self.CrossTalk = CrossTalk()
            return

        fraction_events_in_peaks = [] 
        err_fraction_events_in_peaks = []
        peak_numbers = []
        n_fitted_peaks = len(self.gaussian_fits_parameters["mean"])
        spe_charge = self.gaussian_fits_parameters["mean"][1][0] - self.gaussian_fits_parameters["mean"][0][0]
        n_cx_peaks = min(int(self.edges[-1] // spe_charge)+1, n_fitted_peaks)
        norm_factor = 1./(self.nentries * self.mean_bin_width)

        for peak in range(0, n_cx_peaks):
            scale = self.gaussian_fits_parameters["scale"][peak][0]
            if scale < 0:
                continue
            std   = self.gaussian_fits_parameters["std"][peak][0]
            fraction_events_in_peaks.append(scale * std * np.sqrt(2 * np.pi) * norm_factor)
            err_scale = self.gaussian_fits_parameters["scale"][peak][1]
            err_std   = self.gaussian_fits_parameters["std"][peak][1]
            err_fit   = wun.error_propagation(scale, err_scale, std, err_std, "mul")
            err_fraction_events_in_peaks.append(err_fit * np.sqrt(2*np.pi) * norm_factor)
            peak_numbers.append(peak)
        fraction_events_in_peaks = np.array(fraction_events_in_peaks)
        err_fraction_events_in_peaks = np.array(err_fraction_events_in_peaks)
        peak_numbers = np.array(peak_numbers)

        iminuitparams = [self.mean, 0.1, 1.0]
        paramnames = ["L", "p", "N"]

        chi2 = LeastSquares(peak_numbers,
                            fraction_events_in_peaks,
                            err_fraction_events_in_peaks,
                            wun.CX_fit_function)
        mm = Minuit(chi2, *iminuitparams, name=paramnames)
        mm.migrad()
        mm.hesse()
        fitstatus = mm.fmin.is_valid if mm.fmin else False

        if fitstatus == False:
            print("Cross-talk fit failed.")
            self.CrossTalk = CrossTalk()
            return 

        avg_photons     = mm.params[0].value
        err_avg_photons = mm.params[0].error
        cross_talk      = mm.params[1].value
        err_cross_talk  = mm.params[1].error
        norm_factor     = mm.params[2].value
        err_norm_factor = mm.params[2].error

        self.CrossTalk = CrossTalk(
            avg_photons,
            err_avg_photons,
            cross_talk,
            err_cross_talk,
            norm_factor,
            err_norm_factor,
            n_cx_peaks,
            peak_numbers,
            fraction_events_in_peaks,
            err_fraction_events_in_peaks
        )

        return

    # @property
    def get_cross_talk(self, recompute: bool=True) -> CrossTalk:
        if hasattr(self, 'CrossTalk') and not recompute:
            return self.CrossTalk
        else:
            self.compute_cross_talk()
            return self.CrossTalk
