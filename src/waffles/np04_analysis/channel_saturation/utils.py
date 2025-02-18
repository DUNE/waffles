# See waffles.np04_analysis.lightyield_vs_energy.utils 

from waffles.np04_analysis.lightyield_vs_energy.imports import *

# To select saturated events
def saturation_filter(waveform : Waveform, min_adc_saturation : int = 0) -> bool:
    if min(waveform.adcs) <= min_adc_saturation :
        return True
    else:
        return False