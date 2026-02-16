from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
import numpy as np
import yaml
from waffles.utils.denoising.tv1ddenoise import Denoise
from waffles.utils.baseline.baseline import SBaseline

"""    
This module defines the `WaveformSelector` class, which applies waveform
selection cuts based on a YAML configuration, optionally denoises waveforms.

The selection cuts can be configured per endpoint and channel, and support
upper/lower thresholds over specific time windows. 
"""

class WaveformSelector:
    """
    Apply selection cuts and optionally denoise waveforms.

    This class is designed to process `Waveform` objects by applying a series of 
    user-defined cuts specified in a YAML file. Each cut can select waveforms 
    based on the minimum or maximum signal within a specific time range. 
    The class can also denoise waveforms before applying the cuts.

    Attributes
    ----------
    yamlfile : str
        Path to the YAML file containing cut definitions.
        Example of accepted YAML file structure for endpoint 107 and channel 27:
        107: 
            27: 
                cuts:
                - { t0:  10, tf:   245, threshold: 130, type: lower, filter:  0, npop: max }
                - { t0:  250, tf:   340, threshold: 1100, type:  lower, filter:  0, npop: max }
        
    cutsdata : dict
        Dictionary storing the parsed cuts for each endpoint and channel.
    denoiser : Denoise
        Object for waveform denoising.

    Methods
    -------
    applycuts(waveform, analysis_label='std') -> bool
        Apply all cuts defined in the YAML file to a single waveform. Returns
        True if the waveform passes all cuts, False otherwise.
    loadcuts()
        Load cut definitions from the YAML file into `cutsdata`.
    """

    def __init__(self, yamlfile):

        self.numpyoperations = {
            "max": np.max,
            "min": np.min,
        }

        self.yamlfile = yamlfile
        self.loadcuts()
        self.denoiser = Denoise()
        
    def applycuts(self, waveform: Waveform, analysis_label="std") -> bool:
        """
            Uses the cuts speficied in a yaml file to select the proper waveforms
        """

        if waveform.endpoint not in self.cutsdata:
            return True
        if waveform.channel not in self.cutsdata[waveform.endpoint]:
            return True
        cuts = self.cutsdata[waveform.endpoint][waveform.channel]['cuts']
        
        for cut in cuts:
            t0 = cut['t0']
            tf = cut['tf']
            threshold = cut['threshold']
            cut_type  = cut['type']
            filter    = cut['filter']
            
            # Substract baseline and denoise before getting the reference value for the cut
            if analysis_label in waveform.analyses:
                wf_base = waveform.adcs-waveform.analyses[analysis_label].result["baseline"]
                wf_cut = self.denoiser.apply_denoise_inplace(wf_base, filter) # this method avoids memory leaks
            else:
                raise ValueError(f"Analysis label '{analysis_label}' not found in waveform analyses. Cannot apply cuts.")

            # get the reference value in the time range specified [t0, tf]
            # the type of reference value is given by cut['npop'] = 'max, 'min' 
            ref_val = self.numpyoperations[cut['npop']](wf_cut[t0:tf])

            # perform an upper or lower cut depending on the cut type
            if cut_type == 'higher':
                if ref_val < threshold:
                    return False
            elif cut_type =='lower':
                if ref_val > threshold:
                    return False


        return True

    def check_and_fix_cutsdata(self):

        for ep, chcuts in self.cutsdata.items():
            for ch, dictcuts in chcuts.items():
                cuts = dictcuts.get('cuts', [])
                if isinstance(dict, list):
                    # Ok...
                    continue
                elif isinstance(cuts, dict):
                    # If cuts is a dict, convert it to a list of dicts
                    self.cutsdata[ep][ch]['cuts'] = [ v for k, v in cuts.items() ]
                    

        
    def loadcuts(self):
        try:
            with open(self.yamlfile, 'r') as f:
                self.cutsdata = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file {self.yamlfile} not found. Cannot apply cuts.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {self.yamlfile}: {e}. No cuts will be applied.")

        self.check_and_fix_cutsdata()
