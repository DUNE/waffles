# waveformset_dataframe_utils.py

import pandas as pd
import numpy as np
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform

def waveformset_to_dataframe(wfset: WaveformSet, flatten_adc: bool = False) -> pd.DataFrame:
    rows = []
    for wf in wfset.waveforms:
        row = {
            "run_number": wf._Waveform__run_number,
            "record_number": wf._Waveform__record_number,
            "channel": int(wf._Waveform__channel),
            "endpoint": wf._Waveform__endpoint,
            "timestamp": wf._Waveform__timestamp,
            "daq_timestamp": wf._Waveform__daq_window_timestamp,
            "time_step_ns": wf._WaveformAdcs__time_step_ns,
            "time_offset": wf._WaveformAdcs__time_offset,
        }

        if flatten_adc:
            for i, adc in enumerate(wf._WaveformAdcs__adcs):
                row[f"adc_{i}"] = adc
        else:
            row["adcs"] = wf._WaveformAdcs__adcs.copy()

        rows.append(row)

    return pd.DataFrame(rows)

def dataframe_to_waveformset(df: pd.DataFrame) -> WaveformSet:
    waveforms = []

    if "adcs" not in df.columns:
        raise ValueError("DataFrame must contain a column named 'adcs'")

    for _, row in df.iterrows():
        wf = Waveform(
            run_number=int(row["run_number"]),
            record_number=int(row["record_number"]),
            endpoint=int(row["endpoint"]),
            channel=int(row["channel"]),
            timestamp=int(row["timestamp"]),
            daq_window_timestamp=int(row["daq_timestamp"]),
            starting_tick=0,
            adcs=np.array(row["adcs"], dtype=np.uint16),
            time_step_ns=float(row["time_step_ns"]),
            time_offset=int(row["time_offset"]),
        )
        waveforms.append(wf)

    return WaveformSet(waveforms)
