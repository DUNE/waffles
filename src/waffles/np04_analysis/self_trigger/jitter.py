# --- IMPORTS -------------------------------------------------------
from re import split
import waffles.np04_analysis.self_trigger.self_trigger as self_trigger
from waffles.np04_utils.utils import get_np04_channel_mapping
import waffles.input_output.hdf5_structured as reader
import matplotlib.pyplot as plt
from ROOT import TFile, TF1
import mplhep
import numpy as np
import os
import pandas as pd
import yaml



# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":

    # --- SETUP -----------------------------------------------------
    with open("params.yml", 'r') as stream:
        user_config = yaml.safe_load(stream)

    run_info_file = user_config.get("run_info_file")
    ana_folder  = user_config.get("ana_folder")
    calibration_file = user_config.get("calibration_file")
    file_folder = user_config.get("file_folder")
    SiPM_channel = user_config.get("SiPM_channel")
    perform_selection = user_config.get("perform_selection", True)
    files_in_folder = [file_folder+f for f in os.listdir(file_folder) if f.endswith("structured.hdf5")]
    
    df_runs = pd.read_csv(run_info_file, sep=",") 
    out_df_rows = []
    out_root_file = TFile(ana_folder+f"Jitter_Ch_{SiPM_channel}_Selection_{perform_selection}.root", "RECREATE")
    out_root_file.cd()

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


    thresholds_set = df_runs['Threshold'].unique()
    print(thresholds_set)


    for threshold in thresholds_set:
        run_with_threshold = df_runs.loc[df_runs['Threshold'] == str(threshold), 'Run'].values
        
        run_files = [f for f in files_in_folder if any(str(run) in f for run in run_with_threshold)]
        files = [f for f in run_files if f"_ChSiPM_{SiPM_channel}" in f]
        if len(files) == 0:
            continue

        out_root_file.mkdir(f"Threshold_{int(threshold,16)}")
        out_root_file.cd(f"Threshold_{int(threshold,16)}")
        
        filename = files[0]
        print("Reading file ", filename)

        wfset = reader.load_structured_waveformset(filename)

        for filename in files[1:]:
            wfset_temp = reader.load_structured_waveformset(filename)
            wfset.merge(wfset_temp)
            del wfset_temp

        ch_sipm = SiPM_channel
        ch_st = files[0].split("ChST_")[-1].split("_")[0]
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

        dict_hSTdisrt = st.trigger_distr_per_nspe()
        del wfset

        for nspe, h_STdisrt in dict_hSTdisrt.items():
            h_STdisrt.SetName(f"h_STdisrt_nspe_{nspe}")
            st.h_st = h_STdisrt
            h_st = st.fit_self_trigger_distribution()
            h_st.Write()

            out_df_rows.append({
                               "Run": run_with_threshold[0],
                               "Threshold": int(threshold, 16),
                               "PE": nspe,
                               "MeanTrgPos": st.f_STpeak.GetParameter(1),
                               "ErrMeanTrgPos": st.f_STpeak.GetParError(1),
                               "SigmaTrg": st.f_STpeak.GetParameter(2),
                               "ErrSigmaTrg": st.f_STpeak.GetParError(2),
                               "IntegralTrg": st.f_STpeak.Integral(st.f_STpeak.GetParameter(1) - 3 * st.f_STpeak.GetParameter(2),
                                                                   st.f_STpeak.GetParameter(1) + 3 * st.f_STpeak.GetParameter(2),
                                                                   1e-4),
            })

    out_df = pd.DataFrame(out_df_rows)
    out_df.to_csv(ana_folder+f"Jitter_Ch_{SiPM_channel}_Selection_{perform_selection}.csv", index=False)
    out_root_file.Close()
