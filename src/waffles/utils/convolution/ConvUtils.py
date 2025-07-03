from scipy.fft import fft, fftfreq
import numpy as np

def getFFT(waveform: np.ndarray):
    # Number of sample points
    yf = fft(waveform)
    return yf

def backFFT(yf: np.ndarray):
    # Inverse FFT to get back to time domain
    return np.fft.ifft(yf)

def convolveFFT(yf1, yf2):
    return yf1 * yf2 


    

