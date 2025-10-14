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
    with open("steering.yml", 'r') as stream:
        steering_config = yaml.safe_load(stream)
    params_file_name = steering_config.get("params_file", "params.yml")
    with open(params_file_name, 'r') as stream:
        user_config = yaml.safe_load(stream)

    run_info_file = user_config.get("run_info_file")
    ana_folder  = user_config.get("ana_folder")
    calibration_file = user_config.get("calibration_file")
    file_folder = user_config.get("file_folder")
    SiPM_channel = user_config.get("SiPM_channel")
    save_pngs = user_config.get("save_pngs", True)
    runs = user_config.get("runs", [])
    files_in_folder = [file_folder+f for f in os.listdir(file_folder) if f.endswith("structured.hdf5")]

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


    out_root_file = TFile(ana_folder+f"AnaST_Ch_{SiPM_channel}.root", "RECREATE")
    ch_folder = ana_folder+f"Ch_{SiPM_channel}/"
    if not os.path.exists(ch_folder):
        os.makedirs(ch_folder)

    for run in runs:
        # print the run number and the run index in runs
        print(f"Processing run {run} ({runs.index(run)+1}/{len(runs)})")
        # Load run information ------------------------------------------------
        led  = int(df_runs.loc[df_runs['Run'] == run, 'LED'].values[0])
        exa_threshold = str(df_runs.loc[df_runs['Run'] == run, 'Threshold'].values[0])
        threshold = int(exa_threshold, 16)
        out_root_file.mkdir(f"Run_{run}")
        out_root_file.cd(f"Run_{run}")

        filename = [f for f in files_in_folder if str(run)+"_ChSiPM_"+str(SiPM_channel) in f]
        if len(filename) == 0:
            print(f"No file found for run {run} and SiPM channel {SiPM_channel}")
            continue
        filename = filename[0]
        print("Reading file ", filename)

        wfset = reader.load_structured_waveformset(filename)
        ch_sipm = SiPM_channel
        ch_st = filename.split("ChST_")[-1].split("_")[0]
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
        bkg_trg_rate, unc_bkg_trg_rate = st.get_bkg_trg_rate(h_st_full)
        bkg_trg_rate_preLED, unc_bkg_trg_rate_preLED = st.get_bkg_trg_rate_preLED(h_st_full)

        st.select_waveforms()
        h_st = st.create_self_trigger_distribution()
        st.find_acceptance_window()
        
        if save_pngs:
            fig = plt.figure(figsize=(10, 8))
        htotal, hpassed = st.create_efficiency_histos("he_efficiency")
        if save_pngs:
            plt.savefig(ch_folder+f"Efficiency_Run_{run}_LED_{led}_Thr_{threshold}.png")
            plt.close(fig)
        
        efficiency_fit_ok = st.fit_efficiency()


        out_df_rows.append({
            "Run": run,
            "SiPMChannel": SiPM_channel,
            "LED": led,
            "SiPM": SiPM, # FBK or HPK
            "SpeAmpl": spe_ampl,
            "SpeCharge": st.spe_charge,
            "SNR": snr,
            "ThresholdSet": threshold,
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
            "10to90": st.get_10to90_range()
        })
        h_st.Write()
        st.he_STEfficiency.Write()
          

    out_root_file.Close()
    out_df = pd.DataFrame(out_df_rows)
    out_df.to_csv(ch_folder+f"SelfTrigger_Results_Ch_{SiPM_channel}.csv", index=False)
