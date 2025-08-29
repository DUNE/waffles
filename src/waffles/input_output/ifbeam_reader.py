import os
import math
from pathlib import Path
import csv
import requests
import sys
import urllib3

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
