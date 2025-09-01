import waffles
import numpy as np
from scipy.optimize import curve_fit, brentq
from waffles.utils.denoising.tv1ddenoise import Denoise


def allow_channel_wfs(waveform: waffles.Waveform, channel: int) -> bool:
    return waveform.endpoint == (channel//100) and waveform.channel == (channel%100)
 
# --- Waveform manipulation methods ------------------------------------------
def create_float_waveforms(waveforms: waffles.Waveform) -> None:
    for wf in waveforms:
        wf.adcs_float = wf.adcs.astype(np.float64)

def sub_baseline_to_wfs(waveforms: waffles.Waveform, prepulse_ticks: int):
    norm = 1./prepulse_ticks
    for wf in waveforms:
        baseline = np.sum(wf.adcs_float[:prepulse_ticks])*norm
        wf.adcs_float -= baseline
        wf.adcs_float *= -1

def create_filtered_waveforms(waveforms: waffles.Waveform,
                              filt_level: float) -> None:
    denoiser = Denoise()
    for wf in waveforms:
        wf.adcs_filt = denoiser.apply_denoise(wf.adcs_float, filt_level)



# --- t0 methods -------------------------------------------------------------
def find_threshold_crossing(y: np.array,
                            prepulse_ticks: int,
                            postpulse_ticks: int,
                            threshold: float) -> float:
    """
    Find the x-position where the data in y first surpasses the threshold by interpolating.
    
    Args:
        y (np.ndarray): Array of y-values (must be the same length as x).
        threshold (float): The threshold value to find the crossing for.
        
    Returns:
        float: The interpolated x position where y first surpasses the threshold.
    """
    # Find the index where the value in y first surpasses the threshold
    above_threshold = np.where(y[prepulse_ticks:postpulse_ticks] > threshold)[0]+prepulse_ticks
    
    if not above_threshold.size:
        return None  # No crossing found
    
    idx = above_threshold[0]  # First index where y > threshold
    
    # Linear interpolation between idx-1 and idx
    x1, y1 = idx - 1, y[idx - 1]
    x2, y2 = idx, y[idx]
    
    # Interpolate the exact crossing point
    x_cross = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
    return x_cross

# --- Zero crossing methods ---------------------------------------------------
def pol1(x, a, b):
    return a + b*x

def pol2(x, a, b, c):
    return a + b*x + c*x**2

def pol3(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3

def find_zero_crossing(y: np.array,
                       postpulse_ticks: int) -> float:
    """
    Perform a linear interpolation between pre and post pulse ticks to find the zero crossing.
    """
    # Limit y to postpulse_ticks
    y = y[:postpulse_ticks]
    
    # Find the position of the maximum value in y
    max_pos = np.argmax(y)

    # Efficiently find the zero-crossing index using np.searchsorted()
    zero_cross = len(y[:max_pos]) - np.searchsorted(y[:max_pos][::-1] < 0, True)
    # zero_cross = max_pos - zero_cross if zero_cross > 0 else 0

    # Extract relevant portion of y
    y_segment = y[zero_cross-1:max_pos]
    x_segment = np.arange(zero_cross-1, max_pos)

    # Fit a quadratic polynomial
    # b = (y_segment[-1] - y_segment[0]) / (x_segment[-1] - x_segment[0])
    # a = -b * x_segment[0]
    # init_param = [a, b, 0]
    # param, _ = curve_fit(pol2, x_segment, y_segment, p0=init_param)
    param, _ = curve_fit(pol1, x_segment, y_segment)

    # Find the root (zero crossing) of the fitted curve
    try:
        root = brentq(pol1, x_segment[0] - 2, x_segment[-1], args=tuple(param))
        return root
    except ValueError:
        print("Zero crossing not found")
        return 0  # If root-finding fails




def smooth_wfs(waveforms: waffles.Waveform, sigma: int) -> None:
    """

    """
    gx = np.linspace(-4*sigma, 4*sigma, 8*sigma+1)
    gauss = np.exp(-0.5*((gx/sigma)**2))*(1/(sigma*(2*np.pi)**0.5))

    for wf in waveforms:
        wf = np.convolve(wf,gauss,"same")

