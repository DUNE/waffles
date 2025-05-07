import numpy as np
from numba import njit
from waffles.data_classes.Waveform import Waveform
from waffles.utils.denoising.tv1ddenoise import Denoise

class SBaseline:
    def __init__(self, binsbase = None, threshold:float = 6, wait:int = 25, baselinestart:int = 0, baselinefinish:int = 112, minimumfrac:float = 1/6., default_filtering = None):
        """This class is used to compute the baseline of a Waveform.adcs or over all Waveforms. Description of the method for computing baseline in `compute_baseline`.

        Parameters
        ----------
        binsbase: int or sequence or str
            Numpy histogram bins for computing MPV of baseline
         threshold: float
            Threshold to avoid during computation of baseline
         wait: int
            Time to wait if `threshold is reached`
         baselinestart: int
            Start of baseline
         baselinefinish: int
            Finish of baseline
         minimumfrac: float
            Minimum fraction of the baseline that needs to be used in the
            `mean` computation. If inside this fraction, `optimal` will be True
         default_filtering: float
            For EXTERNAL USAGE, the value is stored in `self.filtering` 
            Used only in automatic methods such as BasicWfAna: this will be the filtering applied.
            For methods of the SBaseline class, the filtering value needs to be passed

        Methods
        ----------
        compute_baseline
            Evaluates baseline for a Waveform.adcs or numpy array object.
            Returns the baseline and if it is optimal
        wfset_baseline
            Evaluates baseline for all Waveforms. Stores it in a new
            member called `baseline`
        """
        self.binsbase = binsbase
        if not self.binsbase:
            self.binsbase = np.linspace(0,2**14-1,2**14)
        self.threshold = threshold
        self.wait = wait
        self.baselinestart = baselinestart
        self.baselinefinish = baselinefinish
        self.minimumfrac = minimumfrac
        self.denoiser = Denoise()
        self.filtering = None
        if default_filtering is not None:
            self.filtering = default_filtering

        self.write_filtered_waveform = True

    @staticmethod
    @njit
    def compute_base_mean(wvf: np.ndarray, res0:int, threshold, baselinestart, baselinefinish, wait, minimumfrac) -> tuple[float, bool]:
        i = 0
        res = 0
        counts = 0
        for _ in wvf:
            if (i>=baselinefinish):
                break
            val = wvf[i]
            if ((val > res0+threshold) | (val < res0 - threshold)):
                i+=wait
            else:
                res+=val
                counts+=1
                i+=1
        if (counts>0):
            res /= counts
        if(counts > (baselinefinish - baselinestart)*minimumfrac):
            return res, True
        else:
            return res0, False

    def compute_baseline(self, wvf_base: np.ndarray, filtering = None) -> tuple[float, bool]:
        """ Computes baseline...
        The code works as following:
            1. Insert all values of
               wvf_base[baselinestart:baselinefinish] in a histogram and
               computes the MPV (res0).
            2. Using res0, cycle throw wvf_base and skipping `wait` ticks if
               a value is bigger than `res0+threshold`. If after `wait` ticks
               it is still above threshold, do `wait` again. The mean of all
               points outside of `wait` intervals is computed
            3. If the fraction of accepcted points is bigger then
               `minimumfrac`, the baseline is optimal Parameters
        ----------
            wvf_base: Waveform.adcs
                Waveform that you want to compute the baseline

        Returns
        ----------
            base: float
                The baseline computed
            optimal
                If the baseline is optimazed or not
        """
        if filtering is not None:
            wvf_base = self.denoiser.apply_denoise(wvf_base, filtering)

        # # find the MPV so we can estimate the offset
        hist, bin_edges = np.histogram(wvf_base[self.baselinestart:self.baselinefinish], bins=self.binsbase)
        # first estimative of baseline
        res0 = bin_edges[np.argmax(hist)]
        return self.compute_base_mean(wvf_base, res0, self.threshold, self.baselinestart, self.baselinefinish, self.wait, self.minimumfrac)


    def wfset_baseline(self, waveform: Waveform, filtering: float = 2) -> tuple[float, bool]:
        """ Evaluates baseline for all Waveforms. Stores it in a new member
        called `baseline`

        Parameters
        ----------
            waveform: Waveform
                Waveforms that you want to compute the baseline
            filtering: float
                Filtering that you want to apply before evaluating the baseline
        """
        wvf: np.ndarray = waveform.adcs
        response = self.denoiser.apply_denoise(wvf, filtering)
        wvf_base = response[self.baselinestart:self.baselinefinish]
        res0, optimal = self.compute_baseline(wvf_base)
        if self.write_filtered_waveform:
            waveform.filtered = response

        return res0, optimal

    def __repr__(self):
        return (
            f"SBaseline(\n"
            f"  threshold={self.threshold},\n"
            f"  wait={self.wait},\n"
            f"  baselinestart={self.baselinestart},\n"
            f"  baselinefinish={self.baselinefinish},\n"
            f"  minimumfrac={self.minimumfrac},\n"
            f"  write_filtered_waveform={self.write_filtered_waveform}\n"
            f"  filtering = {self.filtering} (external usage),\n"
            f")"
        )



