import numpy as np
import h5py

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform

def save_structured_waveformset(
    wfset,
    filepath: str,
    compression: str = "gzip",
    compression_opts: int = 5
) -> None:
    """
    Saves a WaveformSet (or list of Waveform objects) into a structured HDF5 file,
    preserving all essential waveform info.
    """

    # Convert a list of Waveform objects to a WaveformSet if needed
    if isinstance(wfset, list):
        if all(isinstance(w, Waveform) for w in wfset):
            print("🛠️ Auto-wrapping list of Waveforms into WaveformSet (inside hdf5_structured.py)")
            wfset = WaveformSet(*wfset)
        else:
            raise TypeError("Expected a WaveformSet or list[Waveform].")

    elif not isinstance(wfset, WaveformSet):
        raise TypeError(f"Expected WaveformSet or list of Waveforms, got {type(wfset)}")

    waveforms = wfset.waveforms
    n_waveforms = len(waveforms)
    n_samples = wfset.points_per_wf

    # Prepare arrays
    adcs = np.zeros((n_waveforms, n_samples), dtype=np.uint16)
    timestamps = np.zeros(n_waveforms, dtype=np.uint64)
    daq_timestamps = np.zeros(n_waveforms, dtype=np.uint64)
    run_numbers = np.zeros(n_waveforms, dtype=np.int32)
    record_numbers = np.zeros(n_waveforms, dtype=np.int32)
    channels = np.zeros(n_waveforms, dtype=np.uint8)
    endpoints = np.zeros(n_waveforms, dtype=np.int32)
    trigger_types = np.zeros(n_waveforms, dtype=np.uint64)

    # Fill arrays from each Waveform
    for i, wf in enumerate(waveforms):
        adcs[i] = wf.adcs
        timestamps[i] = wf.timestamp
        daq_timestamps[i] = wf.daq_window_timestamp
        run_numbers[i] = wf.run_number
        record_numbers[i] = wf.record_number
        channels[i] = wf.channel
        endpoints[i] = wf.endpoint
        trigger_types[i] = getattr(wf, "trigger_type", 0)

    print(f"✅ Saving {n_waveforms} waveforms to {filepath}")
    print(f"   Sample type: {type(waveforms[0])}, ADC shape: {waveforms[0].adcs.shape}")

    time_step_ns = waveforms[0].time_step_ns
    time_offset = waveforms[0].time_offset

    # Write to HDF5
    with h5py.File(filepath, "w") as f:
        f.create_dataset(
            "adcs", data=adcs,
            compression=compression, compression_opts=compression_opts,
            chunks=True, shuffle=True
        )
        f.create_dataset("timestamps", data=timestamps, compression=compression, compression_opts=compression_opts)
        f.create_dataset("daq_timestamps", data=daq_timestamps, compression=compression, compression_opts=compression_opts)
        f.create_dataset("run_numbers", data=run_numbers, compression=compression, compression_opts=compression_opts)
        f.create_dataset("record_numbers", data=record_numbers, compression=compression, compression_opts=compression_opts)
        f.create_dataset("channels", data=channels, compression=compression, compression_opts=compression_opts)
        f.create_dataset("endpoints", data=endpoints, compression=compression, compression_opts=compression_opts)
        f.create_dataset("trigger_types", data=trigger_types, compression=compression, compression_opts=compression_opts)

        f.attrs["n_waveforms"] = n_waveforms
        f.attrs["n_samples"] = n_samples
        f.attrs["time_step_ns"] = time_step_ns
        f.attrs["time_offset"] = time_offset

    print(f"✅ Saved! {n_waveforms} waveforms to {filepath}")


def load_structured_waveformset(
    filepath: str,
    run_filter=None,
    endpoint_filter=None,
    max_waveforms=None,
    max_to_load=None,
) -> WaveformSet:
    """
    Loads a structured HDF5 file into a WaveformSet, optionally filtering
    by run number, endpoint, and max waveforms.
    Parameters
    ----------
    filepath: str
        Path to the HDF5 file to load.
    run_filter: int, list of int, or None
    endpoint_filter: int, list of int, or None
    max_waveforms: int or None
        Maximum number of waveforms to process. If None, loads all.
        This will take run_filter and endpoint_filter into account
    max_to_load: int or None
        Maximum number of waveforms to load from the file.
        If None, loads all available waveforms.
        The maximum number don't consider any filters. This should be used for
        quick checks only
    """


    if run_filter is None and endpoint_filter is None:
        if max_to_load is None:
            # No reason to load everything...
            max_to_load = max_waveforms

    with h5py.File(filepath, "r") as f:
        # Read datasets
        adcs_array = f["adcs"][:max_to_load]
        timestamps = f["timestamps"][:max_to_load]
        daq_timestamps = f["daq_timestamps"][:max_to_load]
        run_numbers = f["run_numbers"][:max_to_load]
        record_numbers = f["record_numbers"][:max_to_load]
        channels = f["channels"][:max_to_load]
        endpoints = f["endpoints"][:max_to_load]
        trigger_types = f["trigger_types"][:max_to_load] if "trigger_types" in f else np.zeros(len(endpoints), dtype=np.uint64)
        time_step_ns = f.attrs["time_step_ns"]
        time_offset = f.attrs["time_offset"]

    # Figure out which indices to include
    indices = np.arange(len(adcs_array))

    if run_filter is not None:
        run_filter = np.atleast_1d(run_filter)
        indices = indices[np.isin(run_numbers[indices], run_filter)]

    if endpoint_filter is not None:
        endpoint_filter = np.atleast_1d(endpoint_filter)
        indices = indices[np.isin(endpoints[indices], endpoint_filter)]

    if max_waveforms is not None:
        indices = indices[:max_waveforms]

    # Build the Waveform objects
    waveforms = []
    for i in indices:
        waveforms.append(
            Waveform(
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
                trigger_type=int(trigger_types[i])
            )
        )

    # Expand waveforms with * so that WaveformSet sees them as varargs
    wfset = WaveformSet(*waveforms)
    print(f"📤 load_structured_waveformset returning type: {type(wfset)} with {len(wfset.waveforms)} waveforms")

    return wfset
