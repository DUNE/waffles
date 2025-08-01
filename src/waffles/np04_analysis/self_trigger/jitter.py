
import waffles.np04_analysis.self_trigger.self_trigger as self_trigger
from waffles.np04_utils.utils import get_np04_channel_mapping
import waffles.input_output.hdf5_structured as reader
import matplotlib.pyplot as plt
from ROOT import TFile, TF1
import mplhep
import numpy as np
import os
import pandas as pd

run_info_file = "./configs/SelfTrigger_RunInfo.csv"
ana_folder  = "/eos/home-f/fegalizz/ProtoDUNE_HD/SelfTrigger/analysis/"
calibration_file = "SelfTrigger_Calibration.csv"
file_folder = "/eos/home-f/fegalizz/ProtoDUNE_HD/SelfTrigger/files/"
files_in_folder = [file_folder+f for f in os.listdir(file_folder) if f.endswith("structured.hdf5")]


SiPM_channel = 11121
perform_selection = True


if __name__ == "__main__":

    df_runs = pd.read_csv(run_info_file, sep=",") 
    out_df_rows = []

    calibration_df = pd.read_csv(ana_folder+calibration_file, sep=",")
    int_low = int(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'IntLow'].values[0])
    int_up = int(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'IntUp'].values[0])
    prepulse_ticks = int(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'PrepulseTicks'].values[0])
    bsl_rms = float(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'BaselineRMS'].values[0])
    spe_charge = float(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'SpeCharge'].values[0])
    snr = float(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'SNR'].values[0])
    spe_ampl = float(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'SpeAmpl'].values[0])

    df_mapping = get_np04_channel_mapping(version="new")
    print(df_mapping.head(5))
    SiPM = df_mapping.loc[((df_mapping['endpoint'] == SiPM_channel//100) & (df_mapping['daphne_ch'] == SiPM_channel%100)), 'sipm'].values[0]


    out_root_file = TFile(ana_folder+f"Ch_{SiPM_channel}_Selection_{perform_selection}.root", "RECREATE")
    ch_folder = ana_folder+f"Ch_{SiPM_channel}_Selection_{perform_selection}/"
    if not os.path.exists(ch_folder):
        os.makedirs(ch_folder)

    thresholds_set = df_runs['Threshold'].unique()

    for threshold in thresholds_set:
        run_with_threshold = df_runs.loc[df_runs['Threshold'] == threshold, 'Run'].values
        filename = [f for f in files_in_folder if str(run_with_threshold[0])+"_ChSiPM_"+str(SiPM_channel) in f]

        if len(filename) == 0:
            print(f"No file found for run {run_with_threshold[0]} and SiPM channel {SiPM_channel}")
            continue
        filename = filename[0]
        print("Reading file ", filename)

        wfset = reader.load_structured_waveformset(filename)

        for run in run_with_threshold[1:]:
            filename = [f for f in files_in_folder if str(run)+"_ChSiPM_"+str(SiPM_channel) in f]
            print(f"Merging file for run {run} with threshold {threshold}")
            wfset_temp = reader.load_structured_waveformset(filename[0])
            wfset.merge(wfset_temp)

        ch_sipm = SiPM_channel
        ch_st = filename[0].split("ChST_")[-1].split("_")[0]
        st = self_trigger.SelfTrigger(ch_sipm=int(ch_sipm),
                                      ch_st=int(ch_st),
                                      wf_set=wfset,
                                      prepulse_ticks=prepulse_ticks,
                                      int_low=int_low,
                                      int_up=int_up,
                                      bsl_rms=bsl_rms,
                                      spe_charge=spe_charge,
                                      spe_ampl=spe_ampl,
                                      snr=snr)
        st.create_wfs()
        
        if perform_selection:
            st.select_waveforms()

         

