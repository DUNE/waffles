import os, io, click, subprocess, stat, math, shlex, sys
from array import array
from tqdm import tqdm
import numpy as np
import pandas as pd
from XRootD import client
import pickle
import gzip

from waffles.np04_analysis.lightyield_vs_energy.scripts.Renan_scripts.function import *
import click
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm

def find_saturation_regions(signal, saturation_threshold):
    # Identify saturated indices
    saturated_indices = np.where(signal >= saturation_threshold)[0]
    
    if len(saturated_indices) == 0:
        return []

    # Compute differences between consecutive indices
    diffs = np.diff(saturated_indices)
    
    # Find breaks in the sequence where difference is greater than 1
    breaks = np.where(diffs > 1)[0]
    
    # Split indices into continuous regions based on breaks
    regions = np.split(saturated_indices, breaks + 1)
    
    # Store the start and end of each continuous region
    region_boundaries = [(np.where(signal >= saturation_threshold)[0][0], region[-1]) for region in regions]
    
    return region_boundaries

def is_saturated(wf):
    saturation_threshold = 8000
    saturated_indices = np.where(wf >= saturation_threshold)[0]
    if len(saturated_indices) > 2 and np.all(np.diff(saturated_indices) == 1):
        return True
    else:
        return False

def saturated_signal(wf):
    saturation_threshold = 8000
    region = find_saturation_regions(wf, saturation_threshold)[0]
    begin = region[0]
    ending = region[1]
    time = np.arange(0, len(wf))

    # Fit first line (from `begin-2` to `begin`)
    y1 = wf[begin-2: begin]
    x1 = time[begin-2: begin]
    c1 = np.polyfit(x1, y1, 1)
    
    # Fit second line (from `ending` to `ending+5`)
    y2 = wf[ending: ending+3]
    x2 = time[ending: ending+3]
    c2 = np.polyfit(x2, y2, 1)
    
    # Create time array for the full range
    t = time[begin:ending]
    
    # Calculate fitted values for the first and second lines
    f1 = np.polyval(c1, t)
    f2 = np.polyval(c2, t)
    
    # Find the index where the first line exceeds the second line
    i = np.where(f1 >= f2)[0][0]
    new_region = np.concatenate((f1[:i], f2[i:]))
    
    # Modify the signal based on the two fitted regions
    wf2 = np.concatenate((wf[:begin], new_region, wf[ending:]))
    return wf2

def PID(energy, tof, c0, c1):
    if energy == '1GeV' or energy == '-1GeV':

        if tof <= 105 and c0 == 1:
            return 'electron'
        elif tof <= 110 and c0 == 0:
            return 'muon'
        elif tof >110 and tof<=160 and c0 == 0:
            return 'proton'
        else:
            return 'unkonwn'

def photon_counter(wf, template, endpoint, tick, debug = False):
    wf = np.array(wf)
    template = np.array(template)
    template = template - np.median(template)
    if is_saturated(wf):
        #plt.plot(wf)
        #plt.show()
        try:
            wf = saturated_signal(wf)
        except:
            return -1, -1

    if (np.max(wf[:150])) < 0:
        return -1, -1

    # Search the delta to understand how to shift the waveform to align with the template
    try:
        template_half_max = np.max(template) / 2
        wf_half_max = np.max(wf[:150]) / 2
        template_idx = np.where(template >= template_half_max)[0][0]
    except:
        return -1, -1
    wf_idx = np.where(wf[:150] >= wf_half_max)[0][0]
    delta = template_idx - wf_idx

    # Alignement of the wf with template (and cut the final part if necessary)
    wf = np.roll(wf, delta)
    wf = wf[:len(template)]
    #wf = wf - np.mean(wf[:80])
    wf = wf[100:]
    template = template[100:]

    if endpoint == 109:
        lim = 500
    else:
        if len(template) > 400:
            lim = 400
        else:
            lim = 200
        
    try:
        params, errors, r_squared = conv_fit_v5(wf, template, 150, lim, factor = 2.5, debug = debug)
        print(params, errors, r_squared)
        if r_squared < 0.5:
            return -1, -1
        else:
        
            A_s, A_t, tau_s, tau_t, offset = params
            eA_s, eA_t,  etau_s, etau_t,  eoffset = errors
        
            area   = A_s*tau_s*(1-np.exp(-16*tick/(tau_s))) + (A_t*tau_t*(1-np.exp(-16*tick/(tau_t))))
            e_area = np.sqrt((A_s*etau_s)**2 + (tau_s*eA_s)**2 + (A_t*etau_t)**2 + (tau_t*eA_t)**2)
        
            return area, e_area
    except:
        return -1, -1
        
def compute_photon(row):
    area, e_area = photon_counter(row['ADC'], row['Template'], row['endpoint'])
    return pd.Series({'Photons': area, 'ePhotons': e_area})

@click.command()
@click.option("--run" , '-r', default = None, help="Insert the run number, ex: 026102")
@click.option("--energy",'-pt', default = '1GeV')

def main(run, energy):
    
    with gzip.open(f"/afs/cern.ch/work/r/rdeaguia/private/files/Photon_Files/{energy}/{run}/{run}_processed_v2.pkl", "rb") as f:
    #with gzip.open(f"/afs/cern.ch/work/r/rdeaguia/private/files/Photon_Files/1GeV/{run}_bkg.pkl.gz", "rb") as f:
        df = pickle.load(f)
    print(df.columns)

    '''
    try:
        df = df[['TC', 'fragment', 'trigger', 'Daphne_trigger_timestamp', 'endpoint',
           'channel', 'timestamp', 'ADC', 't0', 'dev', 'PhotonA', 'ePhotonA', 'PhotonB', 'ePhotonB', 'n_trigger', 'match']]
    except:
        df = df[['TC', 'fragment', 'trigger', 'Daphne_trigger_timestamp', 'endpoint',
           'channel', 'timestamp', 'ADC', 'PhotonA', 'ePhotonA', 'PhotonB', 'ePhotonB']]
    '''

    print(df.head())
    directory_path = '/afs/cern.ch/work/r/rdeaguia/private/files/Templates/v_27xxx_fix_avg'
    template_files = os.listdir(directory_path)
    
    # Filter only files (not directories)
    template_files = [f for f in template_files if os.path.isfile(os.path.join(directory_path, f))]
    edp_list = []
    ch_list  = []
    adc_list = []
    
    for f in template_files:
        try:
            print(f"Loading: {f}")
            parts = f.split()
            if len(parts) < 3 or "_" not in parts[2]:
                print(f"Unexpected filename format: {f}")
                continue

            edp = parts[2].split("_")[0]
            ch = parts[2].split("_")[1].split('.')[0]

            with open(os.path.join(directory_path, f), 'rb') as file:
                temp = pickle.load(file)

            adc_list.append(temp)
            edp_list.append(edp)
            ch_list.append(ch)

        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    df_template = pd.DataFrame({
        'endpoint' : edp_list,
        'channel' : ch_list,
        'Template_avg': adc_list})
    df_template["endpoint"] = df_template["endpoint"].astype(int)
    df_template["channel"]  = df_template["channel"].astype(int)
    print(df_template.head())

    df = pd.merge(df, df_template, 'left', on = ['endpoint', 'channel'])
    df = df.dropna()
    print(df.head())
    tqdm.pandas()
    df[['PhotonA', 'ePhotonA']] = df.progress_apply(
                    lambda row: photon_counter(row['ADC'], row['Template_avg'], row['endpoint'], 256),
                    axis=1, result_type='expand')
    print(df.head())

    with gzip.open(f"/afs/cern.ch/work/r/rdeaguia/private/files/Photon_Files/{energy}/{run}/{run}_processed_v10.pkl.gz", "wb") as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()

