import numpy as np

class CrossTalk:
    def __init__(self,
                 avg_photons: float = np.nan,
                 avg_photons_error: float = np.nan,
                 CX: float = np.nan,
                 CX_error: float = np.nan,
                 norm_factor: float = np.nan,
                 norm_factor_error: float = np.nan,
                 n_cx_peaks: int = 0,
                 peak_numbers: np.ndarray = np.array([]),
                 fraction_events_in_peaks: np.ndarray = np.array([]),
                 err_fraction_events_in_peaks: np.ndarray = np.array([])
    ):
        """
        Class to store cross-talk analysis results.

        Args:
            avg_photons (float): Average number of photons detected.
            avg_photons_error (float): Uncertainty in the average number of photons.
            CX (float): Estimated cross-talk value.
            CX_error (float): Uncertainty in the cross-talk estimate.
            norm_factor (float): Normalization factor used in the analysis.
            norm_factor_error (float): Uncertainty in the normalization factor.
            n_cx_peaks (int): Number of photo-electron peaks analyzed.
            peak_numbers (np.ndarray): Array of peak numbers (x-axis).
            fraction_events_in_peaks (np.ndarray): Fraction of events in each peak (y-axis).
            err_fraction_events_in_peaks (np.ndarray): Uncertainty in the fraction of events in each peak.
        """
        self.avg_photons = avg_photons
        self.avg_photons_error = avg_photons_error
        self.CX = CX
        self.CX_error = CX_error
        self.norm_factor = norm_factor
        self.norm_factor_error = norm_factor_error
        self.n_cx_peaks = n_cx_peaks
        self.peak_numbers = peak_numbers
        self.fraction_events_in_peaks = fraction_events_in_peaks
        self.err_fraction_events_in_peaks = err_fraction_events_in_peaks
