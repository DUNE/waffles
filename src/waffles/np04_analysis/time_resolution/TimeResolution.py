import waffles
import numpy as np

from TimeResolution_Utils import *


################################################################
############## CLASS IMPLEMENTATION ############################
class TimeResolution:
    def __init__(self,
                 wf_set: waffles.WaveformSet,
                 ref_ep: int, ref_ch: int,
                 tag_ep: int, tag_ch: int,
                 prepulse_ticks: int,
                 postpulse_ticks: int,
                 min_amplitude: int,
                 max_amplitude: int,
                 baseline_rms: float) -> None:
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

        self.ref_ep = ref_ep
        self.ref_ch = ref_ch
        self.tag_ep = tag_ep
        self.tag_ch = tag_ch
        self.ref_wfs = []
        self.tag_wfs = []

    def create_wfs(self, endpoint: int, channel: int):
        """General method to create filtered waveforms based on endpoint and channel."""
        filtered_wfset = waffles.WaveformSet.from_filtered_WaveformSet(self.wf_set, allow_channel_wfs, endpoint, channel)
        return filtered_wfset.waveforms

    def create_ref_wfs(self) -> None:
        self.ref_wfs = self.create_wfs(self.ref_ep, self.ref_ch)
        create_float_waveforms(self.ref_wfs)
        sub_baseline_to_wfs(self.ref_wfs, self.prepulse_ticks)
        
    def create_tag_wfs(self) -> None:
        self.tag_wfs = self.create_wfs(self.tag_ep, self.tag_ch)
        create_float_waveforms(self.tag_wfs)
        sub_baseline_to_wfs(self.tag_wfs, self.prepulse_ticks)

    def select_time_resolution_wfs(self, waveforms) -> None:
        """
        Args:
        - waveforms: self.ref_wfs or self.tag_wfs

        Returns:
        - waveforms.time_resolution_selection: boolean variable to mark if the wf satisfy the selection
        """
        for wf in waveforms:
            max_el_pre = np.max(wf.adcs_float[:self.prepulse_ticks])
            min_el_pre = np.min(wf.adcs_float[:self.prepulse_ticks])

            # Check if the baseline condition is satisfied
            if max_el_pre < 4*self.baseline_rms and min_el_pre > -(4*self.baseline_rms):
                # Calculate max and min in the signal region (after the pre region)
                max_el_signal = np.max(wf.adcs_float[self.prepulse_ticks:self.postpulse_ticks])

                # Check if the signal is within saturation limits
                if max_el_signal < self.max_amplitude and max_el_signal > self.min_amplitude:
                    wf.time_resolution_selection = True

                else:
                    wf.time_resolution_selection = False

            else:
                wf.time_resolution_selection = False

    def set_wfs_t0(self, waveforms: waffles.Waveform) -> None:
        """
        Set the t0 of the selected waveforms
        Args:
        - waveforms: self.ref_wfs or self.tag_wfs
        
        Returns:
        - waveforms.t0 for each of the selected wvfs
        - wavefomrs.avg_t0
        """
        avg_t0 = 0.
        t0_counts = 0
        for wf in waveforms:
            if (wf.time_resolution_selection == True):
                half = 0.5*np.max(wf.adcs_float[self.prepulse_ticks:self.postpulse_ticks])
                wf.t0 = find_threshold_crossing(wf.adcs_float, self.prepulse_ticks, self.postpulse_ticks, half)
                avg_t0 += wf.t0
                t0_counts += 1

        waveforms.avg_t0 = avg_t0/t0_counts


    def calculate_t0_differences(self) -> np.array:
        """
        Calculate differences in t0 values for wf objects with matching ts values and selection==True.
        Args:
        
        Returns:
            np.ndarray: Array of t0 differences for matching ts values.
        """
        
        # Filter wf objects where selection is True
        wf1_filtered = {wf.timestamp: wf.t0 for wf in self.ref_wfs if wf.time_resolution_selection}
        wf2_filtered = {wf.timestamp: wf.t0 for wf in self.tag_wfs if wf.time_resolution_selection}
        
        # Find common ts values and calculate t0 differences
        common_ts = set(wf1_filtered.keys()).intersection(wf2_filtered.keys())
        t0_differences = [wf1_filtered[ts] - wf2_filtered[ts] for ts in common_ts]
        
        return np.array(t0_differences)
