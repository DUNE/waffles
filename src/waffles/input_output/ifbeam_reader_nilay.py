#!/usr/bin/env python3
import os
import math
import requests
import sys
import urllib3
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Beam info fetch function with trigger-matched CKOV/XCET and momentum
# -------------------------------
def BeamInfo_from_ifbeam(t0: int, t1: int, fXCETDebug=False):
    beam_infos = []
    run=0; evt=0; t=0; mom=0

    # --- TOF values ---
    tofs = get_tofs(t0, t1, 60.0, 0.0)

    # --- Cherenkov devices: counts, triggers, pressures ---
    ckov1_counts, ckov1_trig, ckov1_press = get_ckov_values(t0, t1, "XCET021667")
    ckov2_counts, ckov2_trig, ckov2_press = get_ckov_values(t0, t1, "XCET021669")

    # --- XCET devices (timestamps) ---
    xcet1_sec, xcet1_frac, xcet1_coarse, fetched_XCET1 = get_xcet_values(t0, t1, "XCET021667", debug=fXCETDebug)
    xcet2_sec, xcet2_frac, xcet2_coarse, fetched_XCET2 = get_xcet_values(t0, t1, "XCET021669", debug=fXCETDebug)

    # --- Momentum values (already in GeV/c in DB) ---
    momentum_ref  = get_var_values(t0, t1, "dip/acc/NORTH/NP02/POW/CALC/MOMENTUM:momentum_ref")
    momentum_meas = get_var_values(t0, t1, "dip/acc/NORTH/NP02/POW/CALC/MOMENTUM:momentum_meas")

    # --- Compute trigger-matched status and timestamp (delta) ---
    status1_list, timestamp1_list = [], []
    status2_list, timestamp2_list = [], []

    for i, tof in enumerate(tofs):
        # CKOV1
        status1 = -1 if not fetched_XCET1 else 0
        timestamp1 = -1.0
        if fetched_XCET1:
            for ic1 in range(len(xcet1_sec)):
                delta = 1e9*(0 - xcet1_sec[ic1]) + (0 - (8.*xcet1_coarse[ic1] + xcet1_frac[ic1]/512.))
                if abs(delta) < 500.:
                    status1 = 1
                    timestamp1 = delta
                    break
        status1_list.append(status1)
        timestamp1_list.append(timestamp1)

        # CKOV2
        status2 = -1 if not fetched_XCET2 else 0
        timestamp2 = -1.0
        if fetched_XCET2:
            for ic2 in range(len(xcet2_sec)):
                delta = 1e9*(0 - xcet2_sec[ic2]) + (0 - (8.*xcet2_coarse[ic2] + xcet2_frac[ic2]/512.))
                if abs(delta) < 500.:
                    status2 = 1
                    timestamp2 = delta
                    break
        status2_list.append(status2)
        timestamp2_list.append(timestamp2)

        # Append beam info
        c0 = ckov1_trig[i] if i < len(ckov1_trig) else 0
        c1 = ckov2_trig[i] if i < len(ckov2_trig) else 0
        p0 = ckov1_press[i] if i < len(ckov1_press) else 0
        p1 = ckov2_press[i] if i < len(ckov2_press) else 0
        x1_sec    = xcet1_sec[i]    if fetched_XCET1 and i < len(xcet1_sec) else 0
        x1_frac   = xcet1_frac[i]   if fetched_XCET1 and i < len(xcet1_frac) else 0
        x1_coarse = xcet1_coarse[i] if fetched_XCET1 and i < len(xcet1_coarse) else 0
        x2_sec    = xcet2_sec[i]    if fetched_XCET2 and i < len(xcet2_sec) else 0
        x2_frac   = xcet2_frac[i]   if fetched_XCET2 and i < len(xcet2_frac) else 0
        x2_coarse = xcet2_coarse[i] if fetched_XCET2 and i < len(xcet2_coarse) else 0

        # Momentum info
        mom_ref  = momentum_ref[i]  if i < len(momentum_ref)  else 0
        mom_meas = momentum_meas[i] if i < len(momentum_meas) else 0
        mom_diff = mom_meas - mom_ref

        beam_infos.append((run, evt, t, mom, tof, c0, c1, p0, p1,
                           x1_sec, x1_frac, x1_coarse,
                           x2_sec, x2_frac, x2_coarse,
                           status1, timestamp1, status2, timestamp2,
                           mom_ref, mom_meas, mom_diff))
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
                ns_val = 8.0 * coarse[i] + frac[i] / 512.0
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

# -------------------------------
# Main plotting function
# -------------------------------
def main():
    time_ranges = [
        ("2025-08-24T08:00:00-05:00", "2025-08-24T23:59:00-05:00"),
        ("2025-08-26T08:00:00-05:00", "2025-08-26T23:59:00-05:00"),
        ("2025-08-28T08:00:00-05:00", "2025-08-28T23:59:00-05:00")
    ]
    
    run_labels = [
        "2025-08-24",
        "2025-08-26",
        "2025-08-28"
    ]

    colors = ["blue", "red", "green"]

    # ---------------- TOF overlay ----------------
    plt.figure(figsize=(8,6))
    for i, ((t0, t1), run_label) in enumerate(zip(time_ranges, run_labels)):
        beam_infos = BeamInfo_from_ifbeam(t0, t1)
        tofs = [b[4] for b in beam_infos]
        if len(tofs) == 0: continue
        plt.hist(tofs, bins=60, range=(60, 90),
                 histtype="step", linewidth=2,
                 label=run_label, color=colors[i % len(colors)])
    plt.xlabel("TOF [ns]")
    plt.ylabel("Counts")
    plt.title("TOF Distribution (Overlay of Runs)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10, loc="upper right")
    plt.savefig("tof_all_runs.png", dpi=150)
    plt.close()

    # ---------------- CKOV combined figure (2x2 grid) ----------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for idx, name in enumerate(["CKOV1","CKOV2"]):
        for i, ((t0, t1), run_label) in enumerate(zip(time_ranges, run_labels)):
            beam_infos = BeamInfo_from_ifbeam(t0, t1)
            counts = [b[5 if idx==0 else 6] for b in beam_infos]
            press  = [b[7 if idx==0 else 8] for b in beam_infos]

            axs[idx,0].hist(counts, bins=50, histtype="step",
                            linewidth=1.5, label=run_label,
                            alpha=0.8, color=colors[i % len(colors)])
            axs[idx,1].hist(press, bins=50, histtype="step",
                            linewidth=1.5, label=run_label,
                            alpha=0.8, color=colors[i % len(colors)])

        axs[idx,0].set_xlabel(f"{name} Counts")
        axs[idx,0].set_ylabel("Entries")
        axs[idx,0].grid(True, linestyle="--", alpha=0.7)
        axs[idx,0].set_title(f"{name} Counts")

        axs[idx,1].set_xlabel(f"{name} Pressure")
        axs[idx,1].set_ylabel("Entries")
        axs[idx,1].grid(True, linestyle="--", alpha=0.7)
        axs[idx,1].set_title(f"{name} Pressure")

        axs[idx,0].legend(fontsize=8)
        axs[idx,1].legend(fontsize=8)

    plt.savefig("ckov_full_combined.png", dpi=150)
    plt.close()

    # ---------------- Momentum Ref / Meas / Diff ----------------
    plt.figure(figsize=(8,6))
    bins = np.linspace(0.05, 12, 100)
    for i, ((t0, t1), run_label) in enumerate(zip(time_ranges, run_labels)):
        beam_infos = BeamInfo_from_ifbeam(t0, t1)
        mom_ref  = [b[19] for b in beam_infos]
        mom_meas = [b[20] for b in beam_infos]
        mom_diff = [b[21] for b in beam_infos]
        if len(mom_ref) == 0: continue
        plt.hist(mom_ref,  bins=bins, histtype="step", linewidth=2,
                 label=f"{run_label} ref",  color=colors[i % len(colors)])
        plt.hist(mom_meas, bins=bins, histtype="step", linewidth=2,
                 label=f"{run_label} meas", color=colors[i % len(colors)], linestyle="--")
        plt.hist(mom_diff, bins=bins, histtype="step", linewidth=2,
                 label=f"{run_label} diff", color=colors[i % len(colors)], linestyle=":")
    plt.xlabel("Momentum [GeV/c]")
    plt.ylabel("Counts")
    plt.title("Momentum Reference / Measured / Difference")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=8)
    plt.savefig("momentum_ref_meas_diff.png", dpi=150)
    plt.close()

    # ---------------- 2D Heatmap: TOF vs Measured Momentum ----------------
    momentum_bins = np.linspace(0.05, 12, 24)
    for i, ((t0, t1), run_label) in enumerate(zip(time_ranges, run_labels)):
        beam_infos = BeamInfo_from_ifbeam(t0, t1)
        tofs = np.array([b[4] for b in beam_infos])
        mom_meas = np.array([b[20] for b in beam_infos])
        plt.figure(figsize=(10,6))
        plt.hist2d(mom_meas, tofs, bins=[momentum_bins, 300], range=[[0.05,12],[60,90]], cmap="plasma")
        plt.colorbar(label="Counts")
        plt.xlabel("Measured Momentum [GeV/c]")
        plt.ylabel("TOF [ns]")
        plt.title(f"TOF vs Momentum 2D Histogram: {run_label}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(f"tof_vs_momentum_2D_{i}.png", dpi=150)
        plt.close()

    print("All plots saved successfully!")

if __name__ == "__main__":
    main()

