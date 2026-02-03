#!/usr/bin/env python
# coding: utf-8
import waffles
import numpy as np
import json
import shutil 
from tqdm import tqdm
import argparse
from collections import Counter
import pandas as pd
import os


from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.utils.baseline.baseline import SBaseline
from waffles.np02_utils.AutoMap import generate_ChannelMap, dict_uniqch_to_module, dict_module_to_uniqch, ordered_channels_cathode, ordered_channels_membrane, strUch, ordered_modules_cathode, ordered_modules_membrane
from waffles.np02_utils.PlotUtils import process_by_channel
from waffles.np02_utils.load_utils import open_processed, ch_read_calib


def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_strings(arg):
    return arg.split(',')

def get_external(waveform: Waveform, validtimes = [], validchannels = []) -> bool:
    # if np.any((waveform.adcs[0:500] > 16000) | (waveform.adcs[0:500] < 100) ):
    #     return False

    if waveform.channel not in validchannels:
        return False

    if len(validtimes) and waveform.timestamp  not in validtimes:
        return False


    return True

def onlyvalid(waveform: Waveform) -> bool:
    v = waveform.analyses['std'].result['integral']
    if np.isnan(v):
        return True
    return False

def marksaturation(waveform: Waveform) -> bool:
    refval = waveform.adcs[245: 260] - waveform.analyses['std'].result['baseline']
    if np.max(refval) > 15000  or np.min(refval) < -500:
        waveform.issat = True # type: ignore
    else:
        waveform.issat = False # type: ignore
    return True

def upsert_row(LED_int:int, mask:int, module:int, value:float, CSV_PATH:str=f"detector_responses_ch.csv"):
    # 1. Load existing CSV or create an empty DataFrame
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=["LED int", "mask", "module", "value"]) # type: ignore

    # 2. Check if this (LED int, mask, module) already exists
    match = (df["LED int"] == LED_int) & (df["mask"] == mask) & (df["module"] == module)

    if match.any():
        # Update existing row
        df.loc[match, "value"] = value
    else:
        # Insert new row
        df = pd.concat(
            [df if not df.empty else None, pd.DataFrame([{"LED int": LED_int, # type: ignore
                                                          "mask": mask,
                                                          "module": module,
                                                          "value": value}])],
            ignore_index=True
        )

    # 3. Save back to CSV
    df.to_csv(CSV_PATH, index=False)



def main(runs: list[int], LED_int: int, masks: list[int], dettype: str, CSV_PATH: str):
    nwaveforms = 80000

    endpoint = 106 if dettype == "cathode" else 107
    listofch = ordered_channels_cathode if endpoint==106 else ordered_channels_membrane
    validchannels = {}
    validchannels[106] = ordered_channels_cathode 
    validchannels[107] = ordered_channels_membrane


    for run, mask in zip(runs, masks):


        wfset_full = open_processed(run,dettype=dettype, datadir="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/commissioning/", nwaveforms=nwaveforms)

        timestamps = sorted([ wf.timestamp for wf in wfset_full.waveforms ])
        c = Counter(timestamps)
        print(f"Total number of timestamps: {len(c)}")
        matchtimestamps = [ k for k in c if c[k] >= len(list(wfset_full.available_channels[run][endpoint])) ]
        print(f"Remaining timestamps: {len(matchtimestamps)}")
        if len(c) == len(matchtimestamps):
            matchtimestamps = [] # no need for filtering



        wfset= WaveformSet.from_filtered_WaveformSet(wfset_full, get_external, matchtimestamps, validchannels[endpoint], show_progress=True)


        process_by_channel(wfset, show_progress=False, onlyoptimal=False)


        wfset = WaveformSet.from_filtered_WaveformSet(wfset, marksaturation, show_progress=True)


        wfsetch = ChannelWsGrid.clusterize_waveform_set(wfset)
        calib_dict = ch_read_calib()


        modules_npes = {}
        for ep, wfch in wfsetch.items():
            for ch in listofch:
                full_ch_name = dict_uniqch_to_module[strUch(ep,ch)]
                module_name = full_ch_name[:2]
                if module_name not in modules_npes:
                    modules_npes[module_name] = []
                if ch not in wfch.keys() or full_ch_name in ["M2(1)", "C3(2)"]:
                    npes = np.nan
                else:
                    wfs = wfch[ch]
                    specharge = calib_dict[ep][ch]['Gain']
                    charges = np.array([ wf.analyses['std'].result['integral'] for wf in wfs.waveforms ])
                    charges = charges[~np.isnan(charges)]
                    charges = charges[charges>0]

                    nsat = np.array([ wf.issat for wf in wfs.waveforms ])
                    if len( nsat[nsat] ) / len( nsat ) > 0.68: 
                        charges = np.array([specharge*1000])
                    if len(charges):
                        clow, cup = np.quantile(charges, [0.02, 0.98])
                        if len(charges)<5:
                            clow, cup = np.array([np.min(charges), np.max(charges)])
                        npes = charges[ (charges >= clow) & (charges <= cup)].mean()/specharge
                    else: npes = np.nan
                # print(module_name, dict_uniqch_to_module[strUch(ep,ch)], npes)
                modules_npes[module_name] += [npes]

        #CHANNELS PEs
        for m, npe_list in modules_npes.items():
            ch_list = [ch for ch in listofch if dict_uniqch_to_module[strUch(endpoint, ch)][:2] == m]

            for i, npes in enumerate(npe_list):
                full_ch_name = dict_uniqch_to_module[strUch(endpoint, ch_list[i])]
                print(LED_int, mask, full_ch_name, npes if len(npe_list) else np.nan)
                upsert_row(LED_int, mask, full_ch_name, npes if len(npe_list) else np.nan, CSV_PATH=CSV_PATH)


        df = pd.read_csv(CSV_PATH)
        df.sort_values(by=['mask', 'module', 'LED int'], ascending=[True, True, True], inplace=True) 
        df.to_csv(CSV_PATH, index=False)
        print(df)



        df2 = df.pivot(index=["mask","LED int"], columns="module", values="value")
        df2.reset_index()
        print(df2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute detector responses and save to CSV.")
    parser.add_argument("-r", "--runs", type=list_of_ints, help="List of run numbers.")
    parser.add_argument("-m", "--masks", type=list_of_ints, help="List of mask values corresponding to runs.")
    parser.add_argument("-l", "--led_int", type=int, help="LED intensity.")
    parser.add_argument("-d", "--dettypes", type=list_of_strings, help="List of detector types cathode and/or membrane, default C,M", default='C,M')
    parser.add_argument("-o", "--output", type=str, default="detector_responses.csv", help="Output CSV file path. Defaults to detector_responses.csv")
    args = parser.parse_args()
    


    for dt in args.dettypes:
        dettype = "cathode" if dt == 'C' else "membrane"
        main(runs=args.runs, LED_int=args.led_int, masks=args.masks, dettype=dettype, CSV_PATH=args.output)

    # CSV_PATH = f"detector_responses_ch.csv"









