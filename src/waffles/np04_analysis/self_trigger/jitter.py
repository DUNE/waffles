# --- IMPORTS -------------------------------------------------------
import waffles.np04_analysis.self_trigger.self_trigger as self_trigger
from waffles.np04_utils.utils import get_np04_channel_mapping
import waffles.input_output.hdf5_structured as reader
from ROOT import TFile, TF1
import ROOT
import os
import pandas as pd
import yaml



# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    ROOT.gROOT.SetBatch(True)

    # --- SETUP -----------------------------------------------------
    with open("steering.yml", 'r') as stream:
        steering_config = yaml.safe_load(stream)

    params_file_name = steering_config.get("params_file", "params.yml")
    run_info_file    = steering_config.get("run_info_file")
    ana_folder       = steering_config.get("ana_folder")
    if not os.path.exists(ana_folder):
        os.makedirs(ana_folder)
    
    metadata_folder = ana_folder + "metadata/"
    if not os.path.exists(metadata_folder):
        os.makedirs(metadata_folder)
    
    with open(params_file_name, 'r') as stream:
        user_config = yaml.safe_load(stream)
    
    calibration_file = user_config.get("calibration_file")
    SiPM_channel     = user_config.get("SiPM_channel")
    files_in_folder  = [metadata_folder+f for f in os.listdir(metadata_folder) if f.startswith("Merged_")]
    channel_files    = [f for f in files_in_folder if f"_ChSiPM_{SiPM_channel}" in f]
    if channel_files == []:
        raise FileNotFoundError(f"No merged files found for channel {SiPM_channel} in {metadata_folder}\
                \nPlease run ./miscellanea/merger.py first to merge the files for this channel.")
    
    df_runs     = pd.read_csv(run_info_file, sep=",") 
    out_df_rows = []

    calibration_df = pd.read_csv(calibration_file, sep=",")
    print(calibration_df.head(5))
    prepulse_ticks = int(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'PrepulseTicks'].values[0])

    ch_folder = ana_folder+f"Ch_{SiPM_channel}/"
    in_df_filename = ch_folder+f"SelfTrigger_Results_Ch_{SiPM_channel}_merged.csv"
    if not os.path.exists(in_df_filename):
        raise FileNotFoundError(f"Input dataframe file not found for channel {SiPM_channel}")

    threshold_calibration_df = pd.read_csv(in_df_filename, sep=",")
    print(threshold_calibration_df.head(5))
    
    df_mapping = get_np04_channel_mapping(version="new")
    SiPM = df_mapping.loc[((df_mapping['endpoint'] == SiPM_channel//100) & (df_mapping['daphne_ch'] == SiPM_channel%100)), 'sipm'].values[0]

    out_root_file = TFile(ana_folder+f"Jitter_Ch_{SiPM_channel}.root", "RECREATE")
    out_root_file.cd()

    thresholds_set = df_runs['Threshold'].unique()
    print(thresholds_set)

    for exa_threshold in thresholds_set:
        threshold = int(exa_threshold, 16)
        try:
            calibrated_threshold = threshold_calibration_df.loc[threshold_calibration_df['ThresholdSet'] == threshold, 'FiftyCalibrated'].values[0]
        except IndexError:
            print(f"Calibrated threshold not found for threshold set {threshold}, skipping...")
            continue
        print(f"Processing threshold {threshold} ({list(thresholds_set).index(exa_threshold)+1}/{len(thresholds_set)})")
        
        filenames = [f for f in channel_files if str(exa_threshold) in f]
        if len(filenames) == 0:
            continue
        filename = filenames[0]

        out_root_file.mkdir(f"Threshold_{threshold}")
        out_root_file.cd(f"Threshold_{threshold}")
        
        print("Reading file ", filename)


        ch_sipm = SiPM_channel
        ch_st   = filename.split("ChST_")[-1].split(".")[0]
        st = self_trigger.SelfTrigger(ch_sipm=int(ch_sipm),
                                      ch_st=int(ch_st),
                                      prepulse_ticks=prepulse_ticks,
                                      metadata_file=filename)
        st.upload_metadata()
        st.select_events()

        dict_hSTdisrt = st.trigger_distr_per_nspe(calibrated_threshold)
        if dict_hSTdisrt == {}:
            print(f"No triggers found for threshold {threshold}")
            continue

        for nspe, h_STdisrt in dict_hSTdisrt.items():
            h_STdisrt.SetName(f"h_STdisrt_nspe_{nspe}")
            st.h_st = h_STdisrt
            st.h_st.Write()
            h_st, fit_ok = st.fit_self_trigger_distribution(fit_second_peak=True)
            h_st.SetName(f"h_STdisrt_BkgSub_nspe_{nspe}")
            h_st.Write()

            out_df_rows.append({
                               "Threshold": threshold,
                               "PE": nspe,
                               "MeanTrgPos"   : st.f_STpeak.GetParameter(1),
                               "ErrMeanTrgPos": st.f_STpeak.GetParError(1),
                               "SigmaTrg"     : st.f_STpeak.GetParameter(2),
                               "ErrSigmaTrg"  : st.f_STpeak.GetParError(2),
                               "FitOK": fit_ok,
                               "IntegralTrg": st.f_STpeak.Integral(st.f_STpeak.GetParameter(1) - 3 * st.f_STpeak.GetParameter(2),
                                                                   st.f_STpeak.GetParameter(1) + 3 * st.f_STpeak.GetParameter(2),
                                                                   1e-4),
            })

    out_df = pd.DataFrame(out_df_rows)
    out_df.to_csv(ana_folder+f"Jitter_Ch_{SiPM_channel}.csv", index=False)
    out_root_file.Close()
