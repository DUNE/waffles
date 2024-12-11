import waffles
import numpy as np
from typing import Literal

from TimeResolution_Utils import *


################################################################
############## CLASS IMPLEMENTATION ############################
class TimeResolution:
    def __init__(self,
                 wf_set: waffles.WaveformSet,
                 prepulse_ticks: int,
                 postpulse_ticks: int,
                 min_amplitude: int,
                 max_amplitude: int,
                 baseline_rms: float
                 ref_ep = 0,  ref_ch = 0,
                 com_ep = 0,  com_ch = 0,
                 ) -> None:
        """
        This class is used to estimate the time resolution.
        """
        self.wf_set = wf_set

        self.prepulse_ticks = prepulse_ticks
        self.postpulse_ticks = postpulse_ticks
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.baseline_rms = baseline_rms
        # self.qq = qq
        # self.qq = qq

        self.ref_ep = ref_ep        #Endpoint reference channel
        self.ref_ch = ref_ch        #channel
        self.ref_wfs = []           #waveforms
        self.ref_n_select_wfs = 0   #number of selected wfs
        self.ref_t0 = 0.            #Average t0 among the selected wfs
        self.ref_t0_std = 0.        #Standard deviation to t0
        
        self.com_ep = com_ep        #Same for comparison channel
        self.com_ch = com_ch
        self.com_wfs = []
        self.com_n_select_wfs = 0
        self.com_t0 = 0.
        self.com_t0_std = 0.


    def create_wfs(self, tag: Literal["ref","com"]) -> None:
        if tag == "ref":
            t_wfset = waffles.WaveformSet.from_filtered_WaveformSet(self.wf_set, allow_channel_wfs, self.ref_ep, self.ref_ch)
            self.ref_wfs = t_wfset.waveforms
            create_float_waveforms(self.ref_wfs)
            sub_baseline_to_wfs(self.ref_wfs, self.prepulse_ticks)
        if tag == "com":
            t_wfset = waffles.WaveformSet.from_filtered_WaveformSet(self.wf_set, allow_channel_wfs, self.com_ep, self.com_ch)
            self.com_wfs = t_wfset.waveforms
            create_float_waveforms(self.com_wfs)
            sub_baseline_to_wfs(self.com_wfs, self.prepulse_ticks)
 

    def select_time_resolution_wfs(self, tag: Literal["ref","com"]) -> None:
        """
        Args:
        - waveforms: self.ref_wfs or self.com_wfs

        Returns:
        - waveforms.time_resolution_selection: boolean variable to mark if the wf satisfy the selection
        """
        if tag == "ref":
            waveforms = self.ref_wfs
        if tag == "com":
            waveforms = self.com_wfs

        n_selected = 0

        for wf in waveforms:
            max_el_pre = np.max(wf.adcs_float[:self.prepulse_ticks])
            min_el_pre = np.min(wf.adcs_float[:self.prepulse_ticks])

            # Check if the baseline condition is satisfied
            if max_el_pre < 4*self.baseline_rms and min_el_pre > -(4*self.baseline_rms):
                # Calculate max and min in the signal region (after the pre region)
                max_el_signal = np.max(wf.adcs_float[self.prepulse_ticks:self.postpulse_ticks])
                ampl_post = wf.adcs_float[self.postpulse_ticks]

                # Check if the signal is within saturation limits
                if (max_el_signal < self.max_amplitude and
                    max_el_signal > self.min_amplitude and
                    ampl_post < 0.8*max_el_signal):
                    wf.time_resolution_selection = True
                    n_selected += 1

                else:
                    wf.time_resolution_selection = False

            else:
                wf.time_resolution_selection = False
        
        if tag == "ref":
            self.ref_n_select_wfs = n_selected
        if tag == "com":
            self.com_n_select_wfs = n_selected

    def set_wfs_t0(self, tag: Literal["ref","com"]) -> None:
        """
        Set the t0 of the selected waveforms
        Args:
        - waveforms: self.ref_wfs or self.com_wfs
        
        Returns:
        - waveforms.t0 for each of the selected wvfs
        - wavefomrs.avg_t0
        """
        if tag == "ref":
            waveforms = self.ref_wfs
        if tag == "com":
            waveforms = self.com_wfs
        
        t0_list = []
        for wf in waveforms:
            if (wf.time_resolution_selection == True):
                half = 0.5*np.max(wf.adcs_float[self.prepulse_ticks:self.postpulse_ticks])
                wf.t0 = find_threshold_crossing(wf.adcs_float, self.prepulse_ticks, self.postpulse_ticks, half)
                t0_list.append(wf.t0)

        if len(t0_list) > 10:
            t0 = np.average(t0_list)
            std= np.std(t0_list)
            if tag == "ref":
                self.ref_t0 = t0
                self.ref_t0_std = std
            if tag == "com":
                self.com_t0 = t0
                self.com_t0_std = std


    def calculate_t0_differences(self) -> np.array:
        """
        Calculate differences in t0 values for wf objects with matching ts values and selection==True.
        Args:
        
        Returns:
            np.ndarray: Array of t0 differences for matching ts values.
        """
        
        # Filter wf objects where selection is True
        wf1_filtered = {wf.timestamp: wf.t0 for wf in self.ref_wfs if wf.time_resolution_selection}
        wf2_filtered = {wf.timestamp: wf.t0 for wf in self.com_wfs if wf.time_resolution_selection}
        
        # Find common ts values and calculate t0 differences
        common_ts = set(wf1_filtered.keys()).intersection(wf2_filtered.keys())
        t0_differences = [wf1_filtered[ts] - wf2_filtered[ts] for ts in common_ts]
        
        return np.array(t0_differences)
