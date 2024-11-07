import waffles
import numpy as np

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
        self.qq = qq
        self.qq = qq

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
        self.ref_wfs.adcs_float = create_float_waveforms(self.ref_wfs)
        sub_baseline_to_wfs(self.ref_wfs, self.prepulse_ticks)
        
    def create_tag_wfs(self) -> None:
        self.tag_wfs = self.create_wfs(self.tag_ep, self.tag_ch)
        self.tag_wfs.adcs_float = create_float_waveforms(self.tag_wfs)
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