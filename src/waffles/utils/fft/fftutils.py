from scipy.fft import fft
import numpy as np

class FFTWaffles:
    def __init__(self, npoints:int = 1024, filter_type:str = 'Gauss', sample_rate_MHz:float = 62.5, cutoff_MHz:float = 10):
        """
        FFTWaffles class for performing Fast Fourier Transform (FFT) operations.

        Parameters
        ----------
        npoints : int
            Number of points for the FFT.
        filter_type : str
            Type of filter to apply ('Gauss' by default).
        sample_rate_MHz : float
            Sampling rate in MHz (default is 62.5 MHz).
        cutoff_MHz : float
            Cutoff frequency in MHz (default is 10 MHz).
        """
        self.setup_deconv_params(npoints, filter_type, sample_rate_MHz, cutoff_MHz)

    def setup_deconv_params(self, npoints:int, filter_type:str, sample_rate_MHz:float, cutoff_MHz:float):
        self.filter_type = filter_type
        self.sample_rate = sample_rate_MHz
        self.cutoff = cutoff_MHz
        self.npoints = npoints
        self.x_freq = self.getXFreq(self.sample_rate, self.npoints)
        self.set_filters(self.cutoff)

    def set_filters(self, cutoff_MHz:float):
        self.gauss_sigma = cutoff_MHz/ (np.sqrt(np.log(2))) # This way, gain of 1/sqrt(2) at cutoff frequency
        if self.filter_type == 'Gauss':
            self.gauss_filter = self.gauss_filter(self.x_freq, self.gauss_sigma)

    @classmethod
    def getXFreq(cls, sample_rate, npoints): 
        return np.fft.rfftfreq(npoints, 1/sample_rate)

    @classmethod
    def getFFT(cls, waveform: np.ndarray) -> np.ndarray:
        """ 
        Performs Fast Fourier Transform on the given waveform.
        Uses only half+1 of the spectrum. This is done because for real input,
        the negative frequencies are the same
        The +1 is to perfect get the fft back
        Does not generate x_freq, as this can be done only once. Check `getXFreq` method.

        """
        yf = np.fft.rfft(waveform)
        return yf

    @classmethod
    def backFFT(cls, yf: np.ndarray):
        # Inverse FFT to get back to time domain
        return np.fft.irfft(yf, n=len(yf)*2-2).real

    @classmethod
    def convolveFFT(cls, yf1:np.ndarray, yf2:np.ndarray) -> np.ndarray:
        return yf1 * yf2 

    @classmethod
    def gauss_filter(cls, x, sigma):
        return np.exp(-0.5 * (x/sigma)**2)

    def deconvolve(self,
                   signal:np.ndarray,
                   template:np.ndarray,
                   )-> np.ndarray:

        signalfft = self.getFFT(signal)
        templatefft = self.getFFT(template)
        deconv_signal:np.ndarray =  signalfft/templatefft
        if self.filter_type == 'Gauss':
            deconv_signal = deconv_signal*self.gauss_filter

        return deconv_signal

