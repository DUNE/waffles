from typing import Optional
from waffles.data_classes.Waveform import Waveform
import numpy as np

class BeamWaveform(Waveform):
    def __init__(
        self,
        timestamp: int,
        time_step_ns: float,
        daq_window_timestamp: int,
        adcs: np.ndarray,
        run_number: int,
        record_number: int,
        endpoint: int,
        channel: int,
        tof: Optional[float] = None,
        time_offset: int = 0,
        starting_tick: int = 0,
        trigger_type: Optional[int] = None
    ):
        super().__init__(
            timestamp,
            time_step_ns,
            daq_window_timestamp,
            adcs,
            run_number,
            record_number,
            endpoint,
            channel,
            time_offset,
            starting_tick,
            trigger_type
        )
        self._tof = tof

    @property
    def tof(self):
        return self._tof

    @classmethod
    def from_waveform(cls, waveform: Waveform, tof: Optional[float] = None):
        return cls(
            waveform.timestamp,
            waveform.time_step_ns,
            waveform.daq_window_timestamp,
            waveform.adcs,
            waveform.run_number,
            waveform.record_number,
            waveform.endpoint,
            waveform.channel,
            tof=tof,
            time_offset=getattr(waveform, 'time_offset', 0),
            starting_tick=getattr(waveform, 'starting_tick', 0),
            trigger_type=getattr(waveform, 'trigger_type', None)
        )

    def __repr__(self):
        base = super().__repr__()
        return base.strip() + f", TOF: {self._tof} ns"