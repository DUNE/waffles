import numpy as np
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform

def read_cb_25_spybuffer_file(filename, WFs, length):
    if WFs < 0:
        WFs = 100000

    try:
        data = np.fromfile(filename, dtype="<u2")  # big-endian uint16
    except OSError:
        print("Error opening file: ", filename)
        return np.empty((0, length))

    total = len(data) // length
    WFs = min(WFs, total)

    data = data[:WFs * length]
    waveform_adcs_array = data.reshape(WFs, length)

    print("The file has been correctly read")
    return waveform_adcs_array

def create_waveform_set_from_spybuffer(filename: str, 
                                       WFs: int=-1,
                                       length: int=1024, 
                                       config_channel: int=0) -> WaveformSet:
    adcs_array = read_cb_25_spybuffer_file(filename, WFs, length)

    # if channel = 0 find in in filename that ends with channel_CH.dat
    if config_channel == -1:
        config_channel = int(filename.split("channel_")[-1].split(".dat")[0])

    waveforms = [
        Waveform(
            run_number=int(0),
            record_number=int(i),
            endpoint=int(10),
            channel=config_channel,
            timestamp=int(0),
            daq_window_timestamp=int(0),
            starting_tick=0,
            adcs=adcs_array[i],
            time_step_ns=float(16),
            time_offset=int(0),
            trigger_type=int(0)
        )
        for i in range(adcs_array.shape[0])
    ]

    # Expand waveforms with * so that WaveformSet sees them as varargs
    wfset = WaveformSet(*waveforms)

    return wfset
