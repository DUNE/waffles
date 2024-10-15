from waffles.data_classes.Waveform import Waveform
import numpy as np
import yaml
from waffles.utils.denoising.tv1ddenoise import Denoise
from waffles.utils.baseline.baseline import SBaseline

class Extractor:
    def __init__(self, selectiontype:str, current_run:int = None):
        """This class extract either responses or templates from the rawfiles

        Parameters
        ----------
        selectiontype: str
            `template` or `response`
        current_run: int
            The current run being analyzed

        """

        self.numpyoperations = {
            "max": np.max,
            "min": np.min,
        }

        self.__selectiontype = selectiontype
        self.loadcuts()
        self.skeepcuts = False
        self.current_run = current_run


        self.denoiser = Denoise()


        self.baseliner = SBaseline()
        # Setting up baseline parameters
        self.baseliner.binsbase = np.linspace(0,2**14-1,2**14)
        self.baseliner.threshold = 6
        self.baseliner.wait = 25
        self.baseliner.baselinefinish = 112
        self.baseliner.baselinestart = 0
        if self.__selectiontype=='response':
            self.baseliner.baselinefinish = 60
        self.baseliner.minimumfrac = 1/6.


        self.channel_correction=False


    def applycuts(self, waveform: Waveform, ch:int) -> bool:
        """Uses the cuts speficied in a yaml file to select the proper waveforms
        """
        if self.channel_correction:
            ch = 100*waveform.endpoint + ch.astype(np.int32)
        try:
            cuts = self.cutsdata[ch]['cuts']
        except Exception as error:
            return True
        for cut in cuts:
            t0 = cut['t0']
            tf = cut['tf']
            if self.current_run and self.__selectiontype == 'response':
                if self.current_run < 27393:
                    if t0 > 2:
                        t0 -= 2
                    if tf != 1024:
                        tf -= 2
            thre = cut['threshold']
            cuttype = cut['type']
            filter = cut['filter']
            stop = cut['stop']
            wvfcut = self.denoiser.apply_denoise((waveform.adcs-waveform.baseline), filter)*(-1)
            refval = self.numpyoperations[cut['npop']](wvfcut[t0:tf])
            if cuttype == 'higher':
                if refval < thre:
                    return False
            elif cuttype =='lower':
                if refval > thre:
                    return False

            if stop:
                break
        return True

    def allow_certain_endpoints_channels(self, waveform: Waveform, allowed_endpoints:list, allowed_channels:list) -> bool:
        """ This fuction needs to be called, ex:
            wfset_ch = WaveformSet.from_filtered_WaveformSet( wfset, extractor.allow_certain_endpoints_channels, [endpoint] , [ch], show_progress=args['showp'])

        """
        if waveform.endpoint in allowed_endpoints:
            if waveform.channel in allowed_channels:
                base, optimal = self.baseliner.wfset_baseline(waveform)
                waveform.baseline = base
                waveform.optimal = optimal
                if not optimal: return False
                if self.skeepcuts: return True
                outcuts = self.applycuts(waveform, waveform.channel)
                if outcuts:
                    return True
        return False

    def loadcuts(self):
        try:
            with open(f'cuts_{self.__selectiontype}.yaml', 'r') as f:
                self.cutsdata = yaml.safe_load(f)
        except:
            print("Could not load yaml file..., creating fake cut, all waveforms will be applied")
            self.cutsdata['nothing'] = 0
