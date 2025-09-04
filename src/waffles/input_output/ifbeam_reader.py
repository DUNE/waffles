import os
import math
from pathlib import Path
import csv
import requests
import sys
import urllib3
from math import fabs

import numpy as np
import uproot


def BeamInfo_from_ifbeam(t0: int, t1: int):

    beam_infos = []

    run=0
    evt=0
    t=0
    mom =0
    c0=0
    c1=0
    tofs = get_tofs(t0,t1)
    
    for idx in range(len(tofs)):
        beam_infos.append(BeamInfo(run,evt,t,mom,tofs[i],c0,c1))


    return beam_infos


def get_tofs(t0: str, t1: str, delta_trig: float, offset: float):

    # get general trigger values
    trig_n, trig_s, trig_c, trig_f = get_trigger_values(t0, t1)

    # get tof variable values    
    tof_s = []
    tof_c = []
    tof_f = []    

    tof_name =['XBTF022638A','XBTF022638B','XBTF022670A','XBTF022670B']
    
    for i in range(len(tof_name)):
        ni,si,ci,fi = get_tof_vars_values(t0, t1, tof_name[i])
        tof_s.append(si)
        tof_c.append(ci)
        tof_f.append(fi)

    # find valid tofs        
#    fDownstreamToGenTrig = 50.
    fDownstreamToGenTrig = delta_trig 

    tofs = []

    for i in range(len(trig_c)):
        trig_sec = trig_s[i*2+1] 
        trig_ns  = (trig_c[i]+offset)*8 + trig_f[i]/512.

#        print ("trig",i,trig_sec, trig_ns, trig_c[i])
        
        if trig_sec == 0:
            break

        # First check 2A
        for j in range(len(tof_c[2])):

            tof2A_sec = tof_s[2][j*2+1]
            tof2A_ns  = tof_c[2][j]*8 + tof_f[2][j]/512.             

            if tof2A_sec == 0:
                break
            
            delta_2A = 1e9*(trig_sec-tof2A_sec) + trig_ns - tof2A_ns
#            print ("2A",i,j,trig_sec, trig_ns, trig_c[i], tof2A_sec, tof2A_ns, tof_c[2][j], delta_2A)
            if  delta_2A < 0.: 
#                break
                continue
            elif delta_2A > fDownstreamToGenTrig:
                continue
            elif delta_2A>0 and delta_2A < fDownstreamToGenTrig:
            
#                print("Found match 2A to Gen")

#                print ("2A",i,j,trig_sec, trig_ns, trig_c[i], tof2A_sec, tof2A_ns, tof_c[2][j], delta_2A)                
                #check 1A-2A
                check_valid_tof(tof2A_sec, tof2A_ns, tof_s[0],tof_c[0],tof_f[0],tofs)
                #check 1B-2A
                check_valid_tof(tof2A_sec, tof2A_ns, tof_s[1],tof_c[1],tof_f[1],tofs)            

        # Then check 2B
        for j in range(len(tof_c[3])):

            tof2B_sec = tof_s[3][j*2+1]
            tof2B_ns  = tof_c[3][j]*8 + tof_f[3][j]/512.             

#            print (j,tof2B_sec)
            if tof2B_sec == 0:
                break
            
            delta_2B = 1e9*(trig_sec-tof2B_sec) + trig_ns - tof2B_ns
#            print ("2B",i,j,trig_sec, trig_ns, trig_c[i], tof2B_sec, tof2B_ns, tof_c[3][j], delta_2B)                
            #            print ("2B",i,j,trig_sec, trig_ns, tof2B_sec, tof2B_ns, delta_2B)
            if  delta_2B < 0.: 
#                break
                continue
            elif delta_2B > fDownstreamToGenTrig:
                continue
            elif delta_2B>0 and delta_2B < fDownstreamToGenTrig:
            
#                print("Found match 2B to Gen")
#                print ("2B",i,j,trig_sec, trig_ns, tof2B_sec, tof2B_ns, delta_2B)                
                
                #check 1A-2B
                check_valid_tof(tof2B_sec, tof2B_ns, tof_s[0],tof_c[0],tof_f[0],tofs)
                #check 1B-2B
                check_valid_tof(tof2B_sec, tof2B_ns, tof_s[1],tof_c[1],tof_f[1],tofs)            
                                
    return tofs

def check_valid_tof(tof_ref_sec, tof_ref_ns, tof_s, tof_c, tof_f, tofs: []):

    fUpstreamToDownstream =  500.
    
    for k in range(len(tof_c)):
        tof_sec = tof_s[k*2+1]
        tof_ns  = tof_c[k]*8 + tof_f[k]/512.

        if tof_sec == 0:
            break
        
        delta = 1e9*(tof_ref_sec-tof_sec) + tof_ref_ns - tof_ns
        if  delta < 0.: 
            continue
#            break
        elif delta > fUpstreamToDownstream:
            continue
        elif delta>0 and delta < fUpstreamToDownstream:
#            print("Found match")                                    
#            print (k,tof_ref_sec, tof_ref_ns, tof_sec, tof_ns, delta)
            tofs.append(delta)

def get_ckovs(t0: str, t1: str, delta_trig: float, offset: float):

    trig_n, trig_s, trig_c, trig_f = get_trigger_values(t0, t1)

    ckov_names = ['XCET022669', 'XCET022667']
    ckov_s, ckov_c, ckov_f = [], [], []
    pressures, counts = [], []
    fetched_ckov = []


    for name in ckov_names:
        try:
            count_var, si, ci, fi, pressure_list, count_list = get_ckov_vars_values(t0, t1, name)
            ckov_s.append(si)
            ckov_c.append(ci)
            ckov_f.append(fi)
            pressures.append(pressure_list[0] if pressure_list else 0.)
            counts.append(count_list[0] if count_list else 0.)
            fetched_ckov.append(True)
            print(f"[INFO] CKOV {name}: fetched data with counts={count_var}, pressure={pressure_list}")

        except Exception as e:
            print(f"[ERROR] Failed to fetch CKOV {name}: {e}")
            ckov_s.append([])
            ckov_c.append([])
            ckov_f.append([])
            pressures.append(0.)
            counts.append(0.)
            fetched_ckov.append(False)

    ckovs = []

    for i in range(len(trig_c)):
        trig_sec = trig_s[i * 2 + 1]
        trig_ns = (trig_c[i] + offset) * 8 + trig_f[i] / 512.

        if trig_sec == 0:
            break

        event_ckovs = []

        for det in range(2):  # CKOV1 y CKOV2
            if not fetched_ckov[det]:
                event_ckovs.append({
                    'ckov_id': det + 1,
                    'trigger': -1,
                    'timestamp': -1.0,
                    'pressure': pressures[det]
                })
                continue

            trigger = 0
            timestamp = -1.0

            for j in range(len(ckov_c[det])):
                ckov_sec = ckov_s[det][j * 2 + 1]
                ckov_ns = ckov_c[det][j] * 8 + ckov_f[det][j] / 512.

                if ckov_sec == 0:
                    continue

                delta = 1e9 * (trig_sec - (ckov_sec + offset)) + (trig_ns - ckov_ns)

                print(f"[DEBUG] Det {det+1}, Event {i}, Index {j} -> trig_sec={trig_sec}, ckov_sec={ckov_sec}, delta={delta}")

                if abs(delta) < delta_trig:
                    trigger = 1
                    timestamp = delta
                    print(f"[INFO] Det {det+1}, Event {i}: Trigger found with delta={delta}")
                    break

            event_ckovs.append({
                'ckov_id': det + 1,
                'trigger': trigger,
                'timestamp': timestamp,
                'pressure': pressures[det]
            })

        ckovs.append(event_ckovs)

    # Validación de counts (opcional)
    n_ckov1 = sum(1 for ck in ckovs if ck[0]['trigger'] == 1)
    n_ckov2 = sum(1 for ck in ckovs if ck[1]['trigger'] == 1)

    if abs(n_ckov1 - counts[0]) > 0:
        print(f"[WARNING] CKov1 counts mismatch: {n_ckov1} (in spill) vs {counts[0]} (from DB)")

    if abs(n_ckov2 - counts[1]) > 0:
        print(f"[WARNING] CKov2 counts mismatch: {n_ckov2} (in spill) vs {counts[1]} (from DB)")

    return ckovs

def get_tof_vars_values(t0: str, t1: str, tof_var_name: str):

    prefix = "dip/acc/NORTH/NP02/BI/XTOF/"+tof_var_name
    return get_relevant_values(t0, t1, prefix)

def get_ckov_vars_values(t0: str, t1: str, ckov_var_name: str):

    #prefix = "dip/acc/NORTH/NP02/BI/XCET/"+ckov_var_name
    prefix = "dip/acc/NORTH/NP02/BI/TDC/"+ckov_var_name
    
    return get_ckov_relevant_values(t0, t1, prefix)

def get_trigger_values(t0: str, t1: str):

    prefix = "dip/acc/NORTH/NP02/BI/TDC/GeneralTrigger"
    return get_relevant_values(t0, t1, prefix)

def get_ckov_relevant_values(t0: str, t1: str, prefix: str):
    
    seconds_data = get_var_values(t0, t1, prefix+":seconds[]")
    coarse_data  = get_var_values(t0, t1, prefix+":coarse[]")
    frac_data    = get_var_values(t0, t1, prefix+":frac[]")
    count_var    = get_var_values(t0, t1, prefix+":timestampCount")
    pressure    = get_var_values(t0, t1, prefix+":pressure")
    counts    = get_var_values(t0, t1, prefix+":counts")

    return count_var, seconds_data, coarse_data, frac_data, pressure, counts
    
def get_relevant_values(t0: str, t1: str, prefix: str):
    
    seconds_data = get_var_values(t0, t1, prefix+":seconds[]")
    coarse_data  = get_var_values(t0, t1, prefix+":coarse[]")
    frac_data    = get_var_values(t0, t1, prefix+":frac[]")
    count_var    = get_var_values(t0, t1, prefix+":timestampCount")

    return count_var, seconds_data, coarse_data, frac_data
    
def get_var_values(t0: str, t1: str, var_name: str):

    event  = "z,pdune"
    
    # Fetch CSVs
    lines = fetch_ifbeam(var_name, event, t0, t1)

    # Parse numeric values
    data = parse_csv_value(lines)

    return data    

def compute_t(seconds, coarse, frac):
    """Compute T0 in seconds"""
    return 1e9*seconds + 8*coarse + frac/512.0

def compute_tof(t1: int, t2: int):
    return t2-t1

def fetch_ifbeam(var, event, t0, t1):
    """Fetch a single IFBeam variable as CSV"""
    url = (
        "https://dbdata3vm.fnal.gov:9443/ifbeam/data/data"
        f"?e={event}&v={var}&t0={t0}&t1={t1}&f=csv"
    )

    # Disable HTTPS warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    r = requests.get(url, verify=False)
    r.raise_for_status()
    return r.text.splitlines()

def parse_csv_value(lines):
    """Extract numeric values from CSV lines, skip header"""
    values = []
    print ("number of lines: ",len(lines))
    for row in lines[1:]:  # skip header
        parts = row.strip().split(",")
        if len(parts) < 4:
            continue  # skip malformed lines
        try:
            for i in range(5,int(len(parts))):
                val = int(float(parts[i]))
                values.append(val)
        except ValueError:
            continue  # skip non-integer rows
    return values

def main():
    # Define aquí tus tiempos (en formato timestamp o lo que uses)
    t0 = "2025-08-29T00:00:00"
    t1 = "2025-08-29T00:10:00"
    delta_trig = 100.0  # ajusta según tolerancia en nanosegundos
    offset = 0  # ajusta si tienes offset temporal

    print("Iniciando la extracción de CKOVs...")
    ckov_data = get_ckovs(t0, t1, delta_trig, offset)

    print("\nResultado final de CKOVs:")
    for i, event in enumerate(ckov_data):
        print(f"Evento {i}:")
        for det in event:
            print(f"  CKOV {det['ckov_id']} - Trigger: {det['trigger']}, Timestamp: {det['timestamp']}, Pressure: {det['pressure']}")

if __name__ == "__main__":
    main()
