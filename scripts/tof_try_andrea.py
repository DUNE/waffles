import numpy as np
from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.BeamWaveform import BeamWaveform
from datetime import datetime, timezone, timedelta
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.BeamWaveform import BeamWaveform
# This fuction should extract tof and tof_timestamp into a dictionary 

def format_timestamp_with_tz(dt):
    # Devuelve string tipo '2025-08-24T08:00:00-05:00'
    s = dt.strftime('%Y-%m-%dT%H:%M:%S%z')  # Ejemplo: '2025-08-24T080000-0500'
    return s[:-2] + ':' + s[-2:]

def cern_time_from_timestamp(timestamp_s):
    dt_utc = datetime.utcfromtimestamp(timestamp_s).replace(tzinfo=timezone.utc)
    # CERN está en UTC+2
    dt_cern = dt_utc.astimezone(timezone(timedelta(hours=2)))
    return dt_cern

def read_waveformset(input_path):
    print(f"Reading the WaveformSet from: {input_path}")
    try:   
        nwaveforms = 2000
        wfset = load_structured_waveformset(input_path, max_to_load=nwaveforms)
        
        # Tiempo inicial
        start_time = wfset.waveforms[0].daq_window_timestamp
        start_time_s = start_time * 16 / 1e9
        dt_cern_start = cern_time_from_timestamp(start_time_s)

        # Tiempo final
        end_time = wfset.waveforms[nwaveforms-1].daq_window_timestamp
        end_time_s = end_time * 16 / 1e9
        dt_cern_end = cern_time_from_timestamp(end_time_s)

        # Ajuste de +1 y -1 segundo
        dt_cern_start = dt_cern_start - timedelta(seconds=1)
        dt_cern_end = dt_cern_end + timedelta(seconds=1)

        # Conversión a zona horaria local UTC-5
        tz_utc_minus_5 = timezone(timedelta(hours=-5))
        dt_start_local = dt_cern_start.astimezone(tz_utc_minus_5)
        dt_end_local = dt_cern_end.astimezone(tz_utc_minus_5)

        # Formato de strings
        start_time_str = format_timestamp_with_tz(dt_start_local)
        end_time_str = format_timestamp_with_tz(dt_end_local)

    except FileNotFoundError:
        raise FileNotFoundError(f"File {input_path} was not found.")

    return start_time_str, end_time_str, wfset

def main():
    input_path="/afs/cern.ch/work/a/arochefe/private/repositories/waffles/scripts/processed_np02vd_raw_run039105_0000_df-s05-d0_dw_0_20250824T210850.hdf5.copied_structured_membrane.hdf5"

    # --- leer tiempos ---
    t0, t1, wfset = read_waveformset(input_path)
    print(t0, t1)

    # --- construir beamwaveformset ---
    beam_wfset = BeamWaveform.build_beamwaveformset(wfset, t0, t1)

    print(f"📦 BeamWaveformSet con {len(beam_wfset.waveforms)} waveforms")

    # --- imprimir algunas waveforms para ver tof y tof_ts ---
    for i, wf in enumerate(beam_wfset.waveforms):  # primeras 5
        print(f"[{i}] daq_window_ts={wf.daq_window_timestamp}, "
              f"tof={wf.tof}, "
              f"tof_ts={getattr(wf, '_tof_ts', None)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import logging
        logging.warning("Interrupted by user")














