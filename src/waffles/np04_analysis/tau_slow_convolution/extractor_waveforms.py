from waffles.data_classes.Waveform import Waveform
import numpy as np
import yaml
from waffles.utils.denoising.tv1ddenoise import Denoise
from waffles.utils.baseline.baseline import SBaseline
# import all tunable parameters


class Extractor:
    def __init__(self, params, selection_type:str, current_run:int = None, factor:float = 1.0):
        """This class extract either responses or templates from the rawfiles

        Parameters
        ----------
        selectiontype: str
            `template` or `response`
        current_run: int
            The current run being analyzed
        factor: float
            Multiplicative factor for the waveforms. Set it to -1 if signals are negative polarity.

        """

        self.numpyoperations = {
            "max": np.max,
            "min": np.min,
        }

        self.selection_type = selection_type
        self.loadcuts()
        self.skeepcuts = False
        self.factor = factor
        self.current_run = current_run

        self.denoiser = Denoise()


        self.baseliner = SBaseline()
        # Setting up baseline parameters
        self.baseliner.binsbase       = np.linspace(0,2**14-1,2**14)
        self.baseliner.threshold      = params.baseline_threshold
        self.baseliner.wait           = params.baseline_wait
        self.baseliner.minimumfrac    = params.baseline_minimum_frac
        self.baseliner.baselinestart  = params.baseline_start
        self.baseliner.baselinefinish = params.baseline_finish_template
        if self.selection_type=='response':
            self.baseliner.baselinefinish = params.baseline_finish_response
        

        self.channel_correction=False

    def applycuts(self, waveform: Waveform, ch:int) -> bool:
        """Uses the cuts speficied in a yaml file to select the proper waveforms
        """
        if self.channel_correction:
            ch = 100*waveform.endpoint + ch
        try:
            cuts = self.cutsdata[ch]['cuts']
        except Exception as error:
            return True
        
        for cut in cuts:
            t0 = cut['t0']
            tf = cut['tf']
            if self.current_run and self.selection_type == 'response':
                if self.current_run < 27393:
                    if t0 > 2:
                        t0 -= 2
                    if tf != 1024:
                        tf -= 2
            threshold = cut['threshold']
            cut_type  = cut['type']
            filter    = cut['filter']
            stop      = cut['stop']

            # Substract baseline, invert and denoise before getting the reference value for the cut
            wf_cut = self.denoiser.apply_denoise((waveform.adcs-waveform.baseline), filter)*self.factor

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

            # stop the loop after the cut. This is to avoid running further cuts (Henrique ??)
            if stop:
                break


        return True

    def allow_certain_endpoints_channels(self, waveform: Waveform, allowed_endpoints:list, allowed_channels:list) -> bool:
        """ This fuction needs to be called, ex:
            wfset_ch = WaveformSet.from_filtered_WaveformSet( wfset, extractor.allow_certain_endpoints_channels, [endpoint] , [ch], show_progress=args['showp'])

        """
        if waveform.endpoint in allowed_endpoints:
            if waveform.channel in allowed_channels:                
                return True
        return False

    def apply_cuts(self, waveform: Waveform) -> bool:
        """ This fuction needs to be called, ex:
            wfset_ch = WaveformSet.from_filtered_WaveformSet( wfset, extractor.allow_certain_endpoints_channels, [endpoint] , [ch], show_progress=args['showp'])

        """

        # check if this waveform has an optimal baseline
        base, optimal = self.baseliner.wfset_baseline(waveform)
        waveform.baseline = base
        waveform.optimal = optimal
        if not optimal: 
            return False
        
        if self.skeepcuts: 
            return True
        else:
            return self.applycuts(waveform, waveform.channel)
        
    def loadcuts(self):
        try:
            with open(f'configs/cuts_{self.selection_type}.yaml', 'r') as f:
                self.cutsdata = yaml.safe_load(f)
        except:
            print("Could not load yaml file..., creating fake cut, all waveforms will be applied")
            self.cutsdata['nothing'] = 0
