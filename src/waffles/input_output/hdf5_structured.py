import numpy as np
import h5py
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform

def save_structured_waveformset(wfset: WaveformSet, filepath: str, compression="gzip", compression_opts=5):
    waveforms = wfset.waveforms
    n_waveforms = len(waveforms)
    n_samples = wfset.points_per_wf

    adcs_array = np.zeros((n_waveforms, n_samples), dtype=np.uint16)
    timestamps = np.zeros(n_waveforms, dtype=np.uint64)
    daq_timestamps = np.zeros(n_waveforms, dtype=np.uint64)
    run_numbers = np.zeros(n_waveforms, dtype=np.int32)
    record_numbers = np.zeros(n_waveforms, dtype=np.int32)
    channels = np.zeros(n_waveforms, dtype=np.uint8)
    endpoints = np.zeros(n_waveforms, dtype=np.int32)

    for i, wf in enumerate(waveforms):
        adcs_array[i] = wf._WaveformAdcs__adcs
        timestamps[i] = wf._Waveform__timestamp
        daq_timestamps[i] = wf._Waveform__daq_window_timestamp
        run_numbers[i] = wf._Waveform__run_number
        record_numbers[i] = wf._Waveform__record_number
        channels[i] = wf._Waveform__channel
        endpoints[i] = wf._Waveform__endpoint

    with h5py.File(filepath, "w") as f:
        f.create_dataset("adcs", data=adcs_array, compression=compression, compression_opts=compression_opts, chunks=True, shuffle=True)
        f.create_dataset("timestamps", data=timestamps, compression=compression, compression_opts=compression_opts)
        f.create_dataset("daq_timestamps", data=daq_timestamps, compression=compression, compression_opts=compression_opts)
        f.create_dataset("run_numbers", data=run_numbers, compression=compression, compression_opts=compression_opts)
        f.create_dataset("record_numbers", data=record_numbers, compression=compression, compression_opts=compression_opts)
        f.create_dataset("channels", data=channels, compression=compression, compression_opts=compression_opts)
        f.create_dataset("endpoints", data=endpoints, compression=compression, compression_opts=compression_opts)

        f.attrs["n_waveforms"] = n_waveforms
        f.attrs["n_samples"] = n_samples
        f.attrs["time_step_ns"] = waveforms[0]._WaveformAdcs__time_step_ns
        f.attrs["time_offset"] = waveforms[0]._WaveformAdcs__time_offset


def load_structured_waveformset(filepath: str, run_filter=None, endpoint_filter=None, max_waveforms=None) -> WaveformSet:
    with h5py.File(filepath, "r") as f:
        adcs_array = f["adcs"][:]
        timestamps = f["timestamps"][:]
        daq_timestamps = f["daq_timestamps"][:]
        run_numbers = f["run_numbers"][:]
        record_numbers = f["record_numbers"][:]
        channels = f["channels"][:]
        endpoints = f["endpoints"][:]
        time_step_ns = f.attrs["time_step_ns"]
        time_offset = f.attrs["time_offset"]

    indices = np.arange(len(adcs_array))

    if run_filter is not None:
        run_filter = np.atleast_1d(run_filter)
        indices = indices[np.isin(run_numbers[indices], run_filter)]

    if endpoint_filter is not None:
        endpoint_filter = np.atleast_1d(endpoint_filter)
        indices = indices[np.isin(endpoints[indices], endpoint_filter)]

    if max_waveforms is not None:
        indices = indices[:max_waveforms]

    waveforms = []
    for i in indices:
        wf = Waveform(
            run_number=int(run_numbers[i]),
            record_number=int(record_numbers[i]),
            endpoint=int(endpoints[i]),
            channel=int(channels[i]),
            timestamp=int(timestamps[i]),
            daq_window_timestamp=int(daq_timestamps[i]),
            starting_tick=0,
            adcs=adcs_array[i],
            time_step_ns=float(time_step_ns),
            time_offset=int(time_offset),
        )
        waveforms.append(wf)

    return WaveformSet(waveforms)