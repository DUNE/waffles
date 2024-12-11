import waffles
import numpy as np
import TimeResolution as tr

def allow_channel_wfs(waveform: waffles.Waveform, endpoint: int, channel: int) -> bool:
    return waveform.endpoint == endpoint and waveform.channel == channel

def create_float_waveforms(waveforms: waffles.Waveform) -> None:
    for wf in waveforms:
        wf.adcs_float = wf.adcs.astype(np.float64)

def sub_baseline_to_wfs(waveforms: waffles.Waveform, prepulse_ticks: int):
    norm = 1./prepulse_ticks
    for wf in waveforms:
        baseline = np.sum(wf.adcs_float[:prepulse_ticks])*norm
        wf.adcs_float -= baseline
        wf.adcs_float *= -1

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

def smooth_wfs(waveforms: waffles.Waveform, sigma: int) -> None:
    """

    """
    gx = np.linspace(-4*sigma, 4*sigma, 8*sigma+1)
    gauss = np.exp(-0.5*((gx/sigma)**2))*(1/(sigma*(2*np.pi)**0.5))

    for wf in waveforms:
        wf = np.convolve(wf,gauss,"same")

