import waffles
import numpy as np
from typing import Literal, Tuple
from waffles.utils.denoising.tv1ddenoise import Denoise


from utils import *


################################################################
############## CLASS IMPLEMENTATION ############################
class TimeResolution:
    def __init__(self,
                 wf_set: waffles.WaveformSet,
                 prepulse_ticks= 0,
                 int_low= 0,
                 int_up= 0,
                 postpulse_ticks= 0,
                 spe_charge= 0.,
                 spe_ampl= 0.,
                 min_pes = 0.,
                 baseline_rms= 0.,
                 ch = 0,
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

        self.ch = ch        #channel
        self.wfs = []           #waveforms
        self.denoisedwfs = []   #waveforms
        self.n_select_wfs = 0   #number of selected wfs
        self.t0 = 0.            #Average t0 among the selected wfs
        self.t0_std = 0.        #Standard deviation to t0


    def sanity_check(self) -> None:
        """
        Sanity check for the parameters
        """
        if self.prepulse_ticks >= self.postpulse_ticks:
            raise ValueError("prepulse_ticks must be smaller than postpulse_ticks")
        if self.int_low >= self.int_up:
            raise ValueError("int_low must be smaller than int_up")
        if self.spe_charge <= 0:
            raise ValueError("spe_charge must be greater than 0")
        if self.spe_ampl <= 0:
            raise ValueError("spe_ampl must be greater than 0")
        if self.min_pes <= 0:
            raise ValueError("min_pes must be greater than 0")
        if self.baseline_rms <= 0:
            raise ValueError("baseline_rms must be greater than 0")
    
    def set_analysis_parameters(self, 
                                ch: int,
                                prepulse_ticks: int,
                                postpulse_ticks: int,
                                int_low: int,
                                int_up: int,
                                spe_charge: float,
                                spe_ampl: float,
                                min_pes: float,
                                baseline_rms: float) -> None:
        """
        Set the analysis parameters and do sanity checks
        """
        self.ch = ch
        self.prepulse_ticks = prepulse_ticks
        self.postpulse_ticks = postpulse_ticks
        self.int_low = int_low
        self.int_up = int_up
        self.spe_charge = spe_charge
        self.spe_ampl = spe_ampl
        self.min_pes = min_pes
        self.baseline_rms = baseline_rms

        try:
            self.sanity_check()
        except ValueError as e:
            print(f"Error: {e}")
            raise



    

    def create_wfs(self) -> None:
        t_wfset = waffles.WaveformSet.from_filtered_WaveformSet(self.wf_set, allow_channel_wfs, self.ch)
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
                   method: Literal["amplitude", "integral", "denoise"],
                   relative_thr = 0.5,
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Set the t0 of the selected waveforms
        Args:
        - waveforms: self.wfs         
        Returns:
        - waveforms.t0 for each of the selected wvfs
        - waveforms.avg_t0
        """
        waveforms = self.wfs
        
        t0_list = []
        pe_list = []
        ts_list = []
        for wf in waveforms:
            if (wf.time_resolution_selection == True):
                
                if method == "amplitude":
                    thr = relative_thr*np.max(wf.adcs_float[self.prepulse_ticks:self.postpulse_ticks])
                    wf.t0 = find_threshold_crossing(wf.adcs_float, self.prepulse_ticks, self.postpulse_ticks, thr)
                
                elif method == "integral":
                    thr = relative_thr*self.spe_ampl*wf.pe
                    wf.t0 = find_threshold_crossing(wf.adcs_float, self.prepulse_ticks, self.postpulse_ticks, thr)

                elif method == "denoise":
                    thr = relative_thr*np.max(wf.adcs_filt[self.prepulse_ticks:self.postpulse_ticks])
                    wf.t0 = find_threshold_crossing(wf.adcs_filt, self.prepulse_ticks, self.postpulse_ticks, thr)
              
                if wf.t0 is not None:
                    t0_list.append(wf.t0)
                    pe_list.append(wf.pe)
                    ts_list.append(wf.timestamp)

        t0s = np.array(t0_list)
        pes = np.array(pe_list)
        tss = np.array(ts_list)

        return t0s, pes, tss
