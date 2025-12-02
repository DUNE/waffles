import numpy as np

class CrossTalk:
    def __init__(self,
                 mean_photons: float,
                 err_mean_photons: float,
                 CX: float,
                 err_CX: float,
                 norm_factor: float,
                 err_norm_factor: float,
                 n_cx_peaks: int,
                 peak_numbers: np.ndarray,
                 fraction_events_in_peaks: np.ndarray,
                 err_fraction_events_in_peaks: np.ndarray
    ):
        """
        Class to store cross-talk analysis results.

        Args:
            mean_photons (float): Average number of photons detected.
            err_mean_photons (float): Uncertainty in the average number of photons.
            CX (float): Estimated cross-talk value.
            err_CX (float): Uncertainty in the cross-talk estimate.
            norm_factor (float): Normalization factor used in the analysis.
            err_norm_factor (float): Uncertainty in the normalization factor.
            n_cx_peaks (int): Number of photo-electron peaks analyzed.
            peak_numbers (np.ndarray): Array of peak numbers (x-axis).
            fraction_events_in_peaks (np.ndarray): Fraction of events in each peak (y-axis).
            err_fraction_events_in_peaks (np.ndarray): Uncertainty in the fraction of events in each peak.
        """
        self.n_avg_photons = mean_photons
        self.err_n_avg_photons = err_mean_photons
        self.CX = CX
        self.err_CX = err_CX
        self.norm_factor = norm_factor
        self.err_norm_factor = err_norm_factor
        self.n_cx_peaks = n_cx_peaks
        self.peak_numbers = peak_numbers
        self.fraction_events_in_peaks = fraction_events_in_peaks
        self.err_fraction_events_in_peaks = err_fraction_events_in_peaks
