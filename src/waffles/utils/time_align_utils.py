import numpy as np
from scipy import interpolate

def shift_waveform_continuous_forwards(waveform, shift_amount):

    if (shift_amount%1 == 0):
        return np.roll(waveform, shift_amount, axis=0)

    # raise Exception("Nope...")
    # Create an array of indices corresponding to the original waveform
    original_indices = np.arange(len(waveform))

    # The new indices we want to interpolate at, shifted by the shift_amount
    new_indices = original_indices - shift_amount

    # Use scipy's interp1d to create a linear interpolator
    interpolator = interpolate.interp1d(original_indices, waveform, kind='linear', fill_value="extrapolate")

    # Interpolate the waveform at the new shifted indices
    shifted_waveform = interpolator(new_indices)

    return shifted_waveform

def find_threshold_crossing(waveform, threshold_per_cent:float):
    threshold = threshold_per_cent*np.max(waveform)
    # Find indices where the waveform crosses the threshold (positive slope)
    crossings = np.where(np.diff(np.sign(waveform - threshold)) > 0)[0]
    
    if len(crossings) == 0:
        raise ValueError("No threshold crossing found in the waveform.")
    
    # For simplicity, let's take the first crossing point
    # Interpolate between the two points that straddle the threshold crossing
    idx_before = crossings[0]
    y1, y2 = waveform[idx_before], waveform[idx_before + 1]
    x1, x2 = idx_before, idx_before + 1
    
    # Linear interpolation to find the exact fractional index of crossing
    fractional_crossing = x1 + (threshold - y1) / (y2 - y1)
    
    return fractional_crossing


def shift_waveform_to_align_threshold(waveform, threshold, reduce_offset, target_index=-1):
    # Find the crossing point (fractional index)
    crossing_point = find_threshold_crossing(waveform, threshold)
    
    # Compute how much we need to shift the waveform to align the crossing point to `target_index`
    if (target_index == -1):
        shift_amount = crossing_point % 1
    shift_amount = round(target_index - crossing_point)
    if (reduce_offset):
        shift_amount-=2
    
    # Shift the waveform by this amount using linear interpolation
    return shift_waveform_continuous_forwards(waveform, shift_amount), round(shift_amount)
