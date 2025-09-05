#!/usr/bin/env python3
import json
import datetime
import logging
import urllib3
import requests
from waffles.input_output.hdf5_structured import load_structured_waveformset
from datetime import datetime, timezone, timedelta
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.BeamWaveform import BeamWaveform


def ticks_to_iso8601(ticks, base_date, tz_offset_hours=-5):
    from datetime import datetime, timezone, timedelta

    # Convertir ticks a segundos (1 tick = 16 ns = 16e-9 segundos)
    seconds_clock = ticks * 16e-9

    dt_base = datetime.strptime(base_date, "%Y-%m-%d")
    dt_with_seconds = dt_base + timedelta(seconds=seconds_clock)
    tzinfo = timezone(timedelta(hours=tz_offset_hours))
    dt_with_tz = dt_with_seconds.replace(tzinfo=tzinfo)
    return dt_with_tz.isoformat()

def format_timestamp_with_tz(dt):
    # Devuelve string tipo '2025-08-24T08:00:00-05:00'
    s = dt.strftime('%Y-%m-%dT%H:%M:%S%z')  # Ejemplo: '2025-08-24T080000-0500'
    return s[:-2] + ':' + s[-2:]

def format_timestamp(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def cern_time_from_timestamp(timestamp_s):
    dt_utc = datetime.utcfromtimestamp(timestamp_s).replace(tzinfo=timezone.utc)
    # CERN está en UTC+2
    dt_cern = dt_utc.astimezone(timezone(timedelta(hours=2)))
    return dt_cern

def chicago_time_from_timestamp(timestamp_s):
    dt_utc = datetime.utcfromtimestamp(timestamp_s).replace(tzinfo=timezone.utc)
    dt_chicago = dt_utc.astimezone(timezone(timedelta(hours=-5)))
    return dt_chicago


def read_waveformset(input_path):
    print(f"Reading the WaveformSet from: {input_path}")
 try:
        wfset = load_structured_waveformset(input_path)
        
        daq_timestamps = []

        for wf in wfset.waveforms:
            daq_timestamps.append(wf.daq_window_timestamp)

        if not daq_timestamps:
            print("No timestamps found.")
            return False

        # Primer timestamp (CERN time)
        start_time = daq_timestamps[0]
        start_time_s = start_time * 16 / 1e9
        dt_cern_start = cern_time_from_timestamp(start_time_s)
        dt_chicago_start = chicago_time_from_timestamp(start_time_s)

        # Último timestamp (cambia aquí para tomar el último válido, no el 3ro)
        end_time = daq_timestamps[500]
        end_time_s = end_time * 16 / 1e9
        dt_cern_end = cern_time_from_timestamp(end_time_s)
        dt_chicago_end = chicago_time_from_timestamp(end_time_s)

        tz_utc_minus_5 = timezone(timedelta(hours=-5))
        dt_start_local = dt_cern_start.astimezone(tz_utc_minus_5)
        dt_end_local = dt_cern_end.astimezone(tz_utc_minus_5)

        start_time_str = format_timestamp_with_tz(dt_start_local)
        end_time_str = format_timestamp_with_tz(dt_end_local)

        print('Start time CERN:', dt_cern_start)
        print('End time CERN:', dt_cern_end)
        print('Start time fermilab:', dt_chicago_start)
        print('End time fermilab:', dt_chicago_end)
        
        print('Start time unix:', start_time_s)
        print('End time unix:', end_time_s)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {input_path} was not found.")

    return start_time_str, end_time_str

def BeamInfo_from_ifbeam(t0: int, t1: int, fXCETDebug=False):
    beam_infos = []
    run = 0
    evt = 0
    t = 0

    # --- TOF values (recibimos tuplas) ---
    tofs = get_tofs(t0, t1, 60.0, 0.0)  

    # --- Cherenkov devices ---
    ckov1_counts, ckov1_trig, ckov1_press = get_ckov_values(t0, t1, "XCET021667")
    ckov2_counts, ckov2_trig, ckov2_press = get_ckov_values(t0, t1, "XCET021669")

    # --- XCET devices ---
    xcet1_sec, xcet1_frac, xcet1_coarse, fetched_XCET1 = get_xcet_values(t0, t1, "XCET021667", debug=fXCETDebug)
    xcet2_sec, xcet2_frac, xcet2_coarse, fetched_XCET2 = get_xcet_values(t0, t1, "XCET021669", debug=fXCETDebug)

    # --- Momentum values ---
    momentum_ref = get_var_values(t0, t1, "dip/acc/NORTH/NP02/POW/CALC/MOMENTUM:momentum_ref")
    momentum_meas = get_var_values(t0, t1, "dip/acc/NORTH/NP02/POW/CALC/MOMENTUM:momentum_meas")

    # --- Clock values ---
    seconds_clock = parse_clock(fetch_ifbeam("dip/acc/NORTH/NP02/BI/XTOF/XBTF022638A:timestampCount", "z,pdune", t0, t1))

    # --- Loop over TOFs ---
    for i, tof in enumerate(tofs):

        # CKOV1
        status1 = -1 if not fetched_XCET1 else 0
        timestamp1 = -1.0
        if fetched_XCET1:
            for ic1 in range(len(xcet1_sec)):
                # Compare XCET timestamp with TOF time here (adjust units accordingly)
                delta = 1e9 * (tof - xcet1_sec[ic1]) + (8. * xcet1_coarse[ic1] + xcet1_frac[ic1] / 512.)
                if abs(delta) < 500.:
                    status1 = 1
                    timestamp1 = delta
                    break
        # CKOV2
        status2 = -1 if not fetched_XCET2 else 0
        timestamp2 = -1.0
        if fetched_XCET2:
            for ic2 in range(len(xcet2_sec)):
                delta = 1e9 * (tof - xcet2_sec[ic2]) + (8. * xcet2_coarse[ic2] + xcet2_frac[ic2] / 512.)
                if abs(delta) < 500.:
                    status2 = 1
                    timestamp2 = delta
                    break

        # Use safe indexing for triggers and pressures
        c0 = ckov1_trig[i] if i < len(ckov1_trig) else 0
        c1 = ckov2_trig[i] if i < len(ckov2_trig) else 0
        p0 = ckov1_press[i] if i < len(ckov1_press) else 0
        p1 = ckov2_press[i] if i < len(ckov2_press) else 0

        # XCET safe values
        x1_sec = xcet1_sec[i] if fetched_XCET1 and i < len(xcet1_sec) else 0
        x1_frac = xcet1_frac[i] if fetched_XCET1 and i < len(xcet1_frac) else 0
        x1_coarse = xcet1_coarse[i] if fetched_XCET1 and i < len(xcet1_coarse) else 0
        x2_sec = xcet2_sec[i] if fetched_XCET2 and i < len(xcet2_sec) else 0
        x2_frac = xcet2_frac[i] if fetched_XCET2 and i < len(xcet2_frac) else 0
        x2_coarse = xcet2_coarse[i] if fetched_XCET2 and i < len(xcet2_coarse) else 0

        # Momenta
        mom_ref = momentum_ref[i] if i < len(momentum_ref) else 0
        mom_meas = momentum_meas[i] if i < len(momentum_meas) else 0
        mom_diff = mom_meas - mom_ref
        mom = mom_meas  # Update mom here!

        # Clock values safe
        sec_clk = seconds_clock[i] if i < len(seconds_clock) else 0


        beam_infos.append((
            run, evt, t, mom, tof,
            c0, c1, p0, p1,
            x1_sec, x1_frac, x1_coarse,
            x2_sec, x2_frac, x2_coarse,
            status1, timestamp1, status2, timestamp2,
            mom_ref, mom_meas, mom_diff,
            sec_clk
        ))

    return beam_infos

# -------------------------------
# CKOV / XCET / TOF / DB functions
# -------------------------------
def get_ckov_values(t0: str, t1: str, dev: str):
    prefix = f"dip/acc/NORTH/NP02/BI/XCET/{dev}"
    counts     = get_var_values(t0, t1, prefix + ":counts")
    countsTrig = get_var_values(t0, t1, prefix + ":countsTrig")
    pressures  = get_var_values(t0, t1, prefix + ":pressure")
    return counts, countsTrig, pressures

def get_xcet_values(t0: str, t1: str, dev: str, debug=False):
    prefix = f"dip/acc/NORTH/NP02/BI/{dev}"
    try:
        seconds = get_var_values(t0, t1, prefix + ":SECONDS")
        frac    = get_var_values(t0, t1, prefix + ":FRAC")
        coarse  = get_var_values(t0, t1, prefix + ":COARSE")
        if debug:
            for i in range(min(len(seconds), len(frac), len(coarse))):
                ns_val = 8.0 * coarse[i] + frac[i] * 8.0 / 512.0
                print(f"{dev} {i} sec={seconds[i]} ns={ns_val}")
        return seconds, frac, coarse, True
    except Exception as e:
        print(f"WARNING: Could not get {dev} info: {e}")
        return [], [], [], False


def get_tofs(t0: str, t1: str, delta_trig: float, offset: float):
    trig_n, trig_s, trig_c, trig_f = get_trigger_values(t0, t1)
    tof_s, tof_c, tof_f = [], [], []    
    tof_name = ['XBTF022638A','XBTF022638B','XBTF022670A','XBTF022670B']
    for name in tof_name:
        ni,si,ci,fi = get_tof_vars_values(t0, t1, name)
        tof_s.append(si); tof_c.append(ci); tof_f.append(fi)
    fDownstreamToGenTrig = delta_trig 
    tofs = []
    for i in range(len(trig_c)):
        trig_sec = trig_s[i*2+1] 
        trig_ns  = (trig_c[i]+offset)*8 + trig_f[i]/512.0
        if trig_sec == 0: break
        for j in range(len(tof_c[2])):
            tof2A_sec = tof_s[2][j*2+1]
            tof2A_ns  = tof_c[2][j]*8 + tof_f[2][j]/512.0
            if tof2A_sec == 0: break
            delta_2A = 1e9*(trig_sec-tof2A_sec) + trig_ns - tof2A_ns
            if 0 < delta_2A < fDownstreamToGenTrig:
                check_valid_tof(tof2A_sec,tof2A_ns,tof_s[0],tof_c[0],tof_f[0],tofs)
                check_valid_tof(tof2A_sec,tof2A_ns,tof_s[1],tof_c[1],tof_f[1],tofs)
        for j in range(len(tof_c[3])):
            tof2B_sec = tof_s[3][j*2+1]
            tof2B_ns  = tof_c[3][j]*8 + tof_f[3][j]/512.0
            if tof2B_sec == 0: break
            delta_2B = 1e9*(trig_sec-tof2B_sec) + trig_ns - tof2B_ns
            if 0 < delta_2B < fDownstreamToGenTrig:
                check_valid_tof(tof2B_sec,tof2B_ns,tof_s[0],tof_c[0],tof_f[0],tofs)
                check_valid_tof(tof2B_sec,tof2B_ns,tof_s[1],tof_c[1],tof_f[1],tofs)
    return tofs


def check_valid_tof(tof_ref_sec, tof_ref_ns, tof_s, tof_c, tof_f, tofs: []):
    fUpstreamToDownstream = 500.0
    for k in range(len(tof_c)):
        tof_sec = tof_s[k*2+1]
        tof_ns  = tof_c[k]*8 + tof_f[k]/512.0
        if tof_sec == 0: break
        delta = 1e9*(tof_ref_sec-tof_sec) + tof_ref_ns - tof_ns
        if 0 < delta < fUpstreamToDownstream:
            tofs.append(delta)

def get_tof_vars_values(t0: str, t1: str, tof_var_name: str):
    prefix = "dip/acc/NORTH/NP02/BI/XTOF/"+tof_var_name
    return get_relevant_values(t0, t1, prefix)

def get_trigger_values(t0: str, t1: str):
    prefix = "dip/acc/NORTH/NP02/BI/TDC/GeneralTrigger"
    return get_relevant_values(t0, t1, prefix)
    
def get_relevant_values(t0: str, t1: str, prefix: str):
    seconds_data = get_var_values(t0, t1, prefix+":seconds[]")
    coarse_data  = get_var_values(t0, t1, prefix+":coarse[]")
    frac_data    = get_var_values(t0, t1, prefix+":frac[]")
    count_var    = get_var_values(t0, t1, prefix+":timestampCount")
    
    return count_var, seconds_data, coarse_data, frac_data

def get_var_values(t0: str, t1: str, var_name: str):
    event  = "z,pdune"
    lines = fetch_ifbeam(var_name, event, t0, t1)
    data = parse_csv_value(lines)
    return data  

def fetch_ifbeam(var, event, t0, t1):
    url = (
        "https://dbdata3vm.fnal.gov:9443/ifbeam/data/data"
        f"?e={event}&v={var}&t0={t0}&t1={t1}&f=csv"
    )
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    r = requests.get(url, verify=False)
    r.raise_for_status()
    return r.text.splitlines()

def parse_csv_value(lines):
    values = []
    for row in lines[1:]:  # skip header
        parts = row.strip().split(",")
        if len(parts) < 4:
            continue
        try:
            for i in range(5,len(parts)):
                val = int(float(parts[i]))
                values.append(val)
        except ValueError:
            continue
    return values

def parse_clock(lines):
    """Extract the clock values from CSV lines"""
    clocks = []
    for row in lines[1:]:  # skip header
        parts = row.strip().split(",")
        if len(parts) < 4:
            continue
        clocks.append(parts[3])
    return clocks

def associate_beam_info_to_waveforms(wfset_original, beam_info, tolerance_percent=5):
    """
    Created a new WaveformSet with BeamWaveforms
    Each Waveform has a TOF associated if there is a compatible timestamp. 
    If not, TOF=None is included

    Args:
        wfset_original: WaveformSet with Waveforms
        beam_info: list of dicts with 'timestamp' and 'tof'
        tolerance_percent: tolerance in porcentage to do matching with timestamps
    
    Returns:
        new_wfset: WaveformSet with BeamWaveforms (regular Waveforms with Beam Information)
    """
    tolerance_factor = tolerance_percent / 100.0
    asociaciones = 0
    nuevas_waveforms = []

    for i, waveform in enumerate(wfset_original.waveforms):
        wf_ts = waveform.daq_window_timestamp
        tolerance_abs = wf_ts * tolerance_factor

        matched = None
        for entry in beam_info:
            try:
                beam_ts = float(entry['timestamp'])
                if abs(wf_ts - beam_ts) <= tolerance_abs:
                    matched = entry
                    break
            except (KeyError, TypeError, ValueError):
                continue

        if matched:
            tof = matched.get("tof")
            asociaciones += 1
            print(f"[OK] Waveform index {i}: Beam timestamp associated {beam_ts} with TOF {tof}")
        else:
            tof = None
            print(f"[  ] Waveform index {i}: No associated (TOF = None)")

        beam_wf = BeamWaveform.from_waveform(waveform, tof=tof)
        nuevas_waveforms.append(beam_wf)

    print(f"\n✅ Total of matchings: {asociaciones} of {len(wfset_original.waveforms)} waveforms")


    return WaveformSet(*nuevas_waveforms)


def main():
    import json

    with open("beam_config.json", "r") as f:
        config = json.load(f)

    input_path = config.get("input_path")
    if not input_path:
        print("Error: 'input_path' not found in beam_config.json")
        return

    wfset = load_structured_waveformset(input_path)
    print(f"WaveformSet loaded with {len(wfset.waveforms)} waveforms")

    t0, t1 = read_waveformset(input_path)

    beam_info_raw = BeamInfo_from_ifbeam(t0, t1)

    if not beam_info_raw:
        print("Information was not found in beam_info")
        return

    beam_info = []
    for entry in beam_info_raw:
        if entry[-1] != 0:
            beam_info.append({
                'timestamp': float(entry[-1]),
                'tof': float(entry[-2])  # ajusta si el índice es diferente
            })

    new_wfset=associate_beam_info_to_waveforms(wfset, beam_info, tolerance_percent=5)
    print(f"Returning {type(wfset)} with {len(wfset.waveforms)} waveforms")
    print(f"First waveform in the WaveformSet, {type(new_wfset.waveforms[0])} {new_wfset.waveforms[0].daq_window_timestamp}, {new_wfset.waveforms[0].tof}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        import logging
        logging.warning("Interrupted by user")
