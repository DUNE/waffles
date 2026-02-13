import numpy as np

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from dataclasses import dataclass
from typing import Optional


@dataclass
class NeededWaveformParams:
    timestamp: int = 0
    time_step_ns: float = 16.
    daq_window_timestamp: int = 0
    adcs: np.ndarray = np.zeros(1024, dtype=np.int16)
    run_number: int = 0
    record_number: int = 0
    endpoint: int = 0
    channel: int = 0
    time_offset: int = 0
    starting_tick: int = 0
    trigger_type: Optional[int] = None

class EasyWaveformCreator:
    """
    This class provides an easy interface to create a WaveformSet. 
    It dynamically creates "empty" Waveforms with predefined parameters or passed by the user.
    Different methods can interact differenly.
    Default values will create a simple WaveformSet with 1 channel, 1 endpoint, 1 record, 1 run, and 1024 ticks with a time offset of 0.
    These values can be controlled anyway by the user.
    """
    def __init__(self,
                 timestamp: int = 0,
                 time_step_ns: float = 16.,
                 daq_window_timestamp: int = 0,
                 adcs: np.ndarray = np.zeros(1024, dtype=np.int16),
                 run_number: int = 0,
                 record_number: int = 0,
                 endpoint: int = 0,
                 channel: int = 0,
                 time_offset: int = 0,
                 starting_tick: int = 0,
                 trigger_type: Optional[int] = None
                 ):

        self.params = NeededWaveformParams(
            timestamp=timestamp,
            time_step_ns=time_step_ns,
            daq_window_timestamp=daq_window_timestamp,
            adcs=adcs,
            run_number=run_number,
            record_number=record_number,
            endpoint=endpoint,
            channel=channel,
            time_offset=time_offset,
            starting_tick=starting_tick,
            trigger_type=trigger_type
        )

    @classmethod
    def create_WaveformSet(cls, **kwargs) -> WaveformSet:
        """
        Creates a WaveformSet with a single Waveform using the parameters passed by the user or the default ones.
        Parameters
        ----------
            **kwargs: dict
                The parameters to create the Waveform. They are the same as the parameters of the NeededWaveformParams dataclass.
        Returns
        -------
            WaveformSet
                A WaveformSet with a single Waveform created with the parameters passed by the user or the default ones.
        """
        channels = kwargs.pop('channel', 0)
        if isinstance(channels, int):
            channels = [channels]

        waveforms = []
        for channel in channels:
            kwargs['channel'] = channel
            params = NeededWaveformParams(**kwargs)

            waveform = Waveform(
                timestamp=params.timestamp,
                time_step_ns=params.time_step_ns,
                daq_window_timestamp=params.daq_window_timestamp,
                adcs=params.adcs,
                run_number=params.run_number,
                record_number=params.record_number,
                endpoint=params.endpoint,
                channel=int(params.channel),
                time_offset=params.time_offset,
                starting_tick=params.starting_tick,
                trigger_type=params.trigger_type
            )
            waveforms.append(waveform)

        return WaveformSet(*waveforms)

    @classmethod
    def create_WaveformSet_dictEndpointCh(
            cls,
            dict_endpoint_ch: dict,
            **kwargs
        ) -> WaveformSet:
        """
        Creates a WaveformSet with multiple Waveforms using the parameters passed by the user or the default ones.
        Parameters
        ----------
            dict_endpoint_ch: dict

        Returns
        -------
            WaveformSet
                A WaveformSet with multiple Waveforms created with the parameters passed by the user or the default ones.
        """
        waveforms = []
        for endpoint, channels in dict_endpoint_ch.items():
            for channel in channels:
                kwargs['endpoint'] = endpoint
                kwargs['channel'] = channel
                params = NeededWaveformParams(**kwargs)

                waveform = Waveform(
                    timestamp=params.timestamp,
                    time_step_ns=params.time_step_ns,
                    daq_window_timestamp=params.daq_window_timestamp,
                    adcs=params.adcs,
                    run_number=params.run_number,
                    record_number=params.record_number,
                    endpoint=int(params.endpoint),
                    channel=int(params.channel),
                    time_offset=params.time_offset,
                    starting_tick=params.starting_tick,
                    trigger_type=params.trigger_type
                )

                waveforms.append(waveform)
        return WaveformSet(*waveforms)


            



