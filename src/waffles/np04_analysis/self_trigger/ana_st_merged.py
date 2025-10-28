# --- IMPORTS -------------------------------------------------------
import waffles.np04_analysis.self_trigger.self_trigger as self_trigger
from waffles.np04_utils.utils import get_np04_channel_mapping
import waffles.input_output.hdf5_structured as reader
import matplotlib.pyplot as plt
from ROOT import TFile, TF1
import os
import pandas as pd
import numpy as np
import yaml



# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":

    # --- SETUP -----------------------------------------------------
    with open("steering.yml", 'r') as stream:
        steering_config = yaml.safe_load(stream)
    params_file_name = steering_config.get("params_file", "params.yml")
    ana_folder  = steering_config.get("ana_folder")
    if not os.path.exists(ana_folder):
        os.makedirs(ana_folder)

    with open(params_file_name, 'r') as stream:
        user_config = yaml.safe_load(stream)

    run_info_file = user_config.get("run_info_file")
    calibration_file = user_config.get("calibration_file")
    file_folder = user_config.get("file_folder")
    SiPM_channel = user_config.get("SiPM_channel")
    save_pngs = user_config.get("save_pngs", True)
    runs = user_config.get("runs", [])
    files_in_folder = [file_folder+f for f in os.listdir(file_folder) if f.endswith("structured.hdf5")]
    
    df_runs = pd.read_csv(run_info_file, sep=",") 
    out_df_rows = []
    out_root_file = TFile(ana_folder+f"AnaST_Ch_{SiPM_channel}_merged.root", "RECREATE")
    out_root_file.cd()

    calibration_df = pd.read_csv(calibration_file, sep=",")
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

    ch_folder = ana_folder+f"Ch_{SiPM_channel}_merged/"
    if not os.path.exists(ch_folder):
        os.makedirs(ch_folder)

    thresholds_set = df_runs['Threshold'].unique()
    print(thresholds_set)


    for threshold in thresholds_set:
        print(f"Processing threshold {int(threshold,16)} ({list(thresholds_set).index(threshold)+1}/{len(thresholds_set)})")
        run_with_threshold = df_runs.loc[df_runs['Threshold'] == str(threshold), 'Run'].values
        run_with_threshold = [run for run in run_with_threshold if run in runs]

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
            
        h_st_full = st.create_self_trigger_distribution("h_st_full")
        st.find_acceptance_window()
        bkg_trg_rate, unc_bkg_trg_rate = st.get_bkg_trg_rate(h_st_full)
        bkg_trg_rate_preLED, unc_bkg_trg_rate_preLED = st.get_bkg_trg_rate_preLED(h_st_full)

        st.select_waveforms()
        h_st = st.create_self_trigger_distribution()
        
        if save_pngs:
            fig = plt.figure(figsize=(10, 8))
        htotal, hpassed = st.create_efficiency_histos("he_efficiency")
        if save_pngs:
            plt.savefig(ch_folder+f"Efficiency_Thr_{threshold}.png")
            plt.close(fig)
        
        efficiency_fit_ok = st.fit_efficiency()


        out_df_rows.append({
            "SiPMChannel": SiPM_channel,
            "LED": 999,
            "SiPM": SiPM, # FBK or HPK
            "SpeAmpl": spe_ampl,
            "SpeCharge": st.spe_charge,
            "SNR": snr,
            "ThresholdSet": int(threshold, 16),
            "AccWinLow": st.window_low,
            "AccWinUp": st.window_up,
            "BkgTrgRate": bkg_trg_rate,
            "ErrBkgTrgRate": unc_bkg_trg_rate,
            "BkgTrgRatePreLED": bkg_trg_rate_preLED,
            "ErrBkgTrgRatePreLED": unc_bkg_trg_rate_preLED,
            "ThresholdFit": st.f_sigmoid.GetParameter(0) if efficiency_fit_ok else np.nan,
            "ErrThresholdFit": st.f_sigmoid.GetParError(0) if efficiency_fit_ok else np.nan,
            "TauFit": st.f_sigmoid.GetParameter(1) if efficiency_fit_ok else np.nan,
            "ErrTauFit": st.f_sigmoid.GetParError(1) if efficiency_fit_ok else np.nan,
            "MaxEffFit": st.f_sigmoid.GetParameter(2) if efficiency_fit_ok else np.nan,
            "ErrMaxEffFit": st.f_sigmoid.GetParError(2) if efficiency_fit_ok else np.nan,
            "10to90": st.get_10to90_range(),
            "10to90Fit": st.get_10to90_range_fit(),
            "fifty": st.fifty,
            "Errfifty": 0.01
        })
        print(f"\n\n\nfifty: {st.fifty}\n\n")
        h_st.Write()
        st.h_total.Write()
        st.h_passed.Write()
        st.he_STEfficiency.Write()
        st.he_STEfficiency2.Write()
          

    g_st_calib, offset, slope = self_trigger.fit_thrPE_vs_thrSet(pd.DataFrame(out_df_rows))
    g_st_calib.SetTitle(f"Self-Trigger Calibration Ch {SiPM_channel};Threshold Set;Threshold Fit (PE)")
    g_st_calib2, offset2, slope2 = self_trigger.fit_thrPE_vs_thrSet(pd.DataFrame(out_df_rows), "fifty")
    g_st_calib2.SetTitle(f"Self-Trigger Calibration Ch {SiPM_channel};Threshold Set;Threshold Fifty (PE)")
    out_root_file.cd()
    g_st_calib.Write()
    g_st_calib2.Write()
    out_root_file.Close()
    out_df = pd.DataFrame(out_df_rows)
    out_df['ThresholdFitCalibrated'] = slope*out_df['ThresholdSet'] + offset
    out_df['FiftyCalibrated'] = slope2*out_df['ThresholdSet'] + offset2
    out_df.to_csv(ch_folder+f"SelfTrigger_Results_Ch_{SiPM_channel}.csv", index=False)
