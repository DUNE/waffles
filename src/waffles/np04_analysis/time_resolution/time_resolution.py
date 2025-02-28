import waffles
import numpy as np
from typing import Literal
from waffles.utils.denoising.tv1ddenoise import Denoise


from utils import *


################################################################
############## CLASS IMPLEMENTATION ############################
class TimeResolution:
    def __init__(self,
                 wf_set: waffles.WaveformSet,
                 prepulse_ticks: int,
                 int_low: int,
                 int_up: int,
                 postpulse_ticks: int,
                 spe_charge: float,
                 spe_ampl: float,
                 min_pes : float,
                 baseline_rms: float,
                 ep = 0,  ch = 0,
                 ) -> None:
        """
        This class is used to estimate the time resolution.
        """
        self.wf_set = wf_set

        self.prepulse_ticks = prepulse_ticks
        self.postpulse_ticks = postpulse_ticks
        self.baseline_rms = baseline_rms
        self.spe_charge = spe_charge
        self.spe_ampl = spe_ampl
        self.min_pes = min_pes
        self.int_low = int_low
        self.int_up = int_up
        # self.qq = qq
        self.denoiser = Denoise()

        self.ep = ep        #Endpoint reference channel
        self.ch = ch        #channel
        self.wfs = []           #waveforms
        self.denoisedwfs = []   #waveforms
        self.n_select_wfs = 0   #number of selected wfs
        self.t0s = []               #t0 values
        self.pes = []               #pes values
        self.t0 = 0.            #Average t0 among the selected wfs
        self.t0_std = 0.        #Standard deviation to t0
        


    def create_wfs(self) -> None:
        t_wfset = waffles.WaveformSet.from_filtered_WaveformSet(self.wf_set, allow_channel_wfs, self.ep, self.ch)
        self.wfs = t_wfset.waveforms
        create_float_waveforms(self.wfs)
        sub_baseline_to_wfs(self.wfs, self.prepulse_ticks)

    def create_denoised_wfs(self, filt_level: float) -> None:
        create_filtered_waveforms(self.wfs, filt_level)
        

    def select_time_resolution_wfs(self) -> None:
        """
        Args:
        - waveforms: self.wfs

        Returns:
        - waveforms.time_resolution_selection: boolean variable to mark if the wf satisfy the selection
        """
        waveforms = self.wfs

        n_selected = 0

        for wf in waveforms:
            max_el_pre = np.max(wf.adcs_float[:self.prepulse_ticks])
            min_el_pre = np.min(wf.adcs_float[:self.prepulse_ticks])

            # Check if the baseline condition is satisfied
            if max_el_pre < 4*self.baseline_rms and min_el_pre > -(4*self.baseline_rms):
                # Calculate max and min in the signal region (after the pre region)
                max_el_signal = np.max(wf.adcs_float[self.prepulse_ticks:self.postpulse_ticks])
                ampl_post = wf.adcs_float[self.postpulse_ticks]
                wf.pe = wf.adcs_float[self.int_low:self.int_up].sum()/self.spe_charge

                # Check if the signal is within saturation limits
                if (ampl_post < 0.8*max_el_signal
                    and wf.pe > self.min_pes):
                    wf.time_resolution_selection = True
                    n_selected += 1

                else:
                    wf.time_resolution_selection = False

            else:
                wf.time_resolution_selection = False
        
        self.n_select_wfs = n_selected

    def set_wfs_t0(self,
                   method: Literal["half_amplitude","denoise"],
                   relative_thr = 0.5,
                   ) -> None:
        """
        Set the t0 of the selected waveforms
        Args:
        - waveforms: self.wfs         
        Returns:
        - waveforms.t0 for each of the selected wvfs
        - waveforms.avg_t0
        """
        waveforms = self.wfs
        self.t0s = []
        self.pes = []
        
        t0_list = []
        pe_list = []
        for wf in waveforms:
            if (wf.time_resolution_selection == True):
                thr = relative_thr*self.spe_ampl*wf.pe
                
                if method == "half_amplitude":
                    # thr = relative_thr*np.max(wf.adcs_float[self.prepulse_ticks:self.postpulse_ticks])
                    wf.t0 = find_threshold_crossing(wf.adcs_float, self.prepulse_ticks, self.postpulse_ticks, thr)
                if method == "denoise":
                    # thr = relative_thr*np.max(wf.adcs_filt[self.prepulse_ticks:self.postpulse_ticks])
                    wf.t0 = find_threshold_crossing(wf.adcs_filt, self.prepulse_ticks, self.postpulse_ticks, thr)
              
                if wf.t0 is not None:
                    t0_list.append(wf.t0)
                    pe_list.append(wf.pe)

        if len(t0_list) > 10:
            t0 = np.average(t0_list)
            std= np.std(t0_list)
            self.t0s = np.array(t0_list)
            self.pes = np.array(pe_list)
            self.t0 = t0
            self.t0_std = std
