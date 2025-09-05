from typing import Optional
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
import numpy as np
import requests
import urllib3


class BeamWaveform(Waveform):
    """
    BeamWaveform extends the standard Waveform by including beam-related
    information such as TOF. This class provides methods to fetch and
    parse IFBeam data, compute TOFs, and associate them with timestamps.

    Important:
    - All helper methods are defined as @classmethod or @staticmethod.
    - Internal calls should consistently use `cls.` to ensure correct resolution.
    - The get_tofs method now returns (timestamp, tof) pairs, where the
      timestamp corresponds to the detector with suffix 'B'.
    """

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
        self._tof_ts = tof_ts

    @property
    def tof(self):
        return self._tof

    @property
    def tof_ts(self):
        return self._tof_ts

    @classmethod
    def from_waveform(cls, waveform: Waveform, tof: Optional[float] = None, tof_ts: Optional[float] = None):
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
            tof_ts=tof_ts,
            time_offset=getattr(waveform, 'time_offset', 0),
            starting_tick=getattr(waveform, 'starting_tick', 0),
            trigger_type=getattr(waveform, 'trigger_type', None)
        )

    @classmethod
    def fetch_ifbeam(cls, var, event, t0, t1):
        url = (
            "https://dbdata3vm.fnal.gov:9443/ifbeam/data/data"
            f"?e={event}&v={var}&t0={t0}&t1={t1}&f=csv"
        )
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.get(url, verify=False)
        r.raise_for_status()
        return r.text.splitlines()

    @classmethod
    def parse_csv_ifbeam_value(cls, lines):
        values = []
        for row in lines[1:]:  # skip header
            parts = row.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                for i in range(5, len(parts)):
                    val = int(float(parts[i]))
                    values.append(val)
            except ValueError:
                continue
        return values

    @classmethod
    def get_var_values(cls, t0: str, t1: str, var_name: str):
        event = "z,pdune"
        lines = cls.fetch_ifbeam(var_name, event, t0, t1)
        data = cls.parse_csv_ifbeam_value(lines)
        return data

    @classmethod
    def get_relevant_values(cls, t0: str, t1: str, prefix: str):
        seconds_data = cls.get_var_values(t0, t1, prefix + ":seconds[]")
        coarse_data = cls.get_var_values(t0, t1, prefix + ":coarse[]")
        frac_data = cls.get_var_values(t0, t1, prefix + ":frac[]")
        count_var = cls.get_var_values(t0, t1, prefix + ":timestampCount")
        return count_var, seconds_data, coarse_data, frac_data

    @classmethod
    def get_tof_vars_values(cls, t0: str, t1: str, tof_var_name: str):
        prefix = "dip/acc/NORTH/NP02/BI/XTOF/" + tof_var_name
        return cls.get_relevant_values(t0, t1, prefix)

    @classmethod
    def get_trigger_values(cls, t0: str, t1: str):
        prefix = "dip/acc/NORTH/NP02/BI/TDC/GeneralTrigger"
        return cls.get_relevant_values(t0, t1, prefix)

    @classmethod
    def check_valid_tof(cls, tof_ref_sec, tof_ref_ns, tof_s, tof_c, tof_f, results: list):
        fUpstreamToDownstream = 500.0  # ns
        for k in range(len(tof_c)):
            tof_sec = tof_s[k * 2 + 1]
            tof_ns = tof_c[k] * 8 + tof_f[k] / 512
            if tof_sec == 0:
                break

            delta = 1e9 * (tof_ref_sec - tof_sec) + tof_ref_ns - tof_ns  # ns
            if 0 < delta < fUpstreamToDownstream:
                timestamp = tof_sec  # segundos
                results.append({
                    "tof_ts": timestamp,
                    "tof": delta
                })

    @classmethod
    def get_tofs(cls, t0: str, t1: str, delta_trig: float, offset: float):
        trig_n, trig_s, trig_c, trig_f = cls.get_trigger_values(t0, t1)
        tof_s, tof_c, tof_f = [], [], []
        tof_name = ['XBTF022638A','XBTF022638B','XBTF022670A','XBTF022670B']
        for name in tof_name:
            ni, si, ci, fi = cls.get_tof_vars_values(t0, t1, name)
            tof_s.append(si)
            tof_c.append(ci)
            tof_f.append(fi)

        fDownstreamToGenTrig = delta_trig
        results = []

        for i in range(len(trig_c)):
            trig_sec = trig_s[i*2+1] 
            trig_ns  = (trig_c[i]+offset)*8 + trig_f[i] / 512
            if trig_sec == 0: break
            for j in range(len(tof_c[2])):
                tof2A_sec = tof_s[2][j*2+1]
                tof2A_ns  = tof_c[2][j]*8 + tof_f[2][j] /512
                if tof2A_sec == 0: break
                delta_2A = 1e9*(trig_sec-tof2A_sec) + trig_ns - tof2A_ns
                if 0 < delta_2A < fDownstreamToGenTrig:
                    cls.check_valid_tof(tof2A_sec,tof2A_ns,tof_s[0],tof_c[0],tof_f[0],results)
                    cls.check_valid_tof(tof2A_sec,tof2A_ns,tof_s[1],tof_c[1],tof_f[1],results)
            for j in range(len(tof_c[3])):
                tof2B_sec = tof_s[3][j*2+1]
                tof2B_ns  = tof_c[3][j]*8 + tof_f[3][j] /512
                if tof2B_sec == 0: break
                delta_2B = 1e9*(trig_sec-tof2B_sec) + trig_ns - tof2B_ns
                if 0 < delta_2B < fDownstreamToGenTrig:
                    cls.check_valid_tof(tof2B_sec,tof2B_ns,tof_s[0],tof_c[0],tof_f[0],results)
                    cls.check_valid_tof(tof2B_sec,tof2B_ns,tof_s[1],tof_c[1],tof_f[1],results)

        return results

    @classmethod
    def get_BeamInfo_from_ifbeam(cls, t0: int, t1: int, fXCETDebug=False):
        beam_info = []
        tofs = cls.get_tofs(t0, t1, 60.0, 0.0)
        for tof_dict in tofs:  
            beam_info.append((tof_dict["tof_ts"], tof_dict["tof"]))
        return beam_info
    
    @classmethod
    def build_beamwaveformset(cls, wfset, t0: int, t1: int, tolerance: float = 0.05) -> WaveformSet:
        """
        Build a WaveformSet of BeamWaveforms using sequential matching:
        1. Compare each unique DAQ timestamp with TOF timestamps in order.
        2. Count a match if the difference is consistent with the first delta (within tolerance).
        3. Keep the combination with the highest count of consistent matches.
        """

        # --- Unique DAQ timestamps ---
        unique_daq_ts = sorted({wf.daq_window_timestamp for wf in wfset.waveforms})

        # --- Load beam info ---
        beam_info = cls.get_BeamInfo_from_ifbeam(t0, t1)  # list of (tof_ts, tof)
        tof_ts_list, tof_list = zip(*beam_info)

        best_combination = None
        best_count = -1

        # --- Try all possible starting alignments ---
        for start_idx in range(len(tof_ts_list) - len(unique_daq_ts) + 1):
            count = 0
            deltas = []
            for i, daq_ts in enumerate(unique_daq_ts):
                daq_s = daq_ts * 16 / 1e9
                tof_ts = tof_ts_list[start_idx + i]
                tof = tof_list[start_idx + i]
                delta = abs(daq_s - tof_ts)

                if i == 0:
                    delta_ref = delta
                    count += 1
                else:
                    if abs(delta - delta_ref) <= tolerance * delta_ref:
                        count += 1

                deltas.append(delta)

            if count > best_count:
                best_count = count
                best_combination = {
                    daq_ts: {"tof_ts": tof_ts_list[start_idx + i], "tof": tof_list[start_idx + i]}
                    for i, daq_ts in enumerate(unique_daq_ts)
                }

        # --- Create BeamWaveforms with matched TOF ---
        beam_waveforms = []
        for wf in wfset.waveforms:
            daq_ts = wf.daq_window_timestamp
            match = best_combination.get(daq_ts, None)
            if match is None:
                new_wf = cls.from_waveform(wf, tof=None)
            else:
                new_wf = cls.from_waveform(wf, tof=match["tof"], tof_ts=match["tof_ts"])
            beam_waveforms.append(new_wf)

        return WaveformSet(*beam_waveforms)

    def __repr__(self):
        base = super().__repr__()
        return base.strip() + f", TOF: {self._tof} ns" + f", TOF_ts: {self._tof_ts} ns" 
