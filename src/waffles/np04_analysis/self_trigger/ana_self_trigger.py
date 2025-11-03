import waffles.np04_analysis.self_trigger.self_trigger as self_trigger
from waffles.np04_analysis.self_trigger.utils import get_efficiency_at
from waffles.np04_utils.utils import get_np04_channel_mapping
from ROOT import TFile
import ROOT
import numpy as np
import os
import pandas as pd
import yaml


# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # Set ROOT in batch mode
    ROOT.gROOT.SetBatch(True)
    # Set default minimizer to "Minuit"
    # ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit")

    # --- SETUP -----------------------------------------------------
    with open("steering.yml", 'r') as stream:
        steering_config = yaml.safe_load(stream)
    params_file_name = steering_config.get("params_file", "params.yml")
    run_info_file = steering_config.get("run_info_file")
    ana_folder  = steering_config.get("ana_folder")
    fit_type = steering_config.get("fit_type", "")
    save_fit_pngs = steering_config.get("save_fit_pngs", False)
    verbose = steering_config.get("verbose", False)
    npe_of_interest = int(steering_config.get("npe_of_interest"))
    run_by_run = steering_config.get("run_by_run", False)
    metadata_folder = ana_folder + "metadata/"

    with open(params_file_name, 'r') as stream:
        user_config = yaml.safe_load(stream)
    calibration_file = user_config.get("calibration_file")
    SiPM_channel = user_config.get("SiPM_channel")
    save_pngs = user_config.get("save_pngs", True)
    leds_to_plot = steering_config.get("led_to_plot", [])
    runs = user_config.get("runs", [])
    files_in_folder = [metadata_folder+f for f in os.listdir(metadata_folder) if f.endswith(".root")]
    channel_files = [f for f in files_in_folder if f"_ChSiPM_{SiPM_channel}" in f]

    df_runs = pd.read_csv(run_info_file, sep=",") 
    out_df_rows = []

    calibration_df = pd.read_csv(calibration_file, sep=",")
    print(calibration_df.head(5))
    int_low = int(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'IntLow'].values[0])
    spe_charge = float(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'SpeCharge'].values[0])
    snr = float(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'SNR'].values[0])
    spe_ampl = float(calibration_df.loc[calibration_df['SiPMChannel'] == SiPM_channel, 'SpeAmpl'].values[0])

    df_mapping = get_np04_channel_mapping(version="new")
    SiPM = df_mapping.loc[((df_mapping['endpoint'] == SiPM_channel//100) & (df_mapping['daphne_ch'] == SiPM_channel%100)), 'sipm'].values[0]


    out_root_file_name = ana_folder+f"AnaST_Ch_{SiPM_channel}"
    if not run_by_run:
        out_root_file_name += "_merged"
    out_root_file = TFile(out_root_file_name+".root", "RECREATE")
    ch_folder = ana_folder+f"Ch_{SiPM_channel}/"
    if not os.path.exists(ch_folder):
        os.makedirs(ch_folder)

    datasets = runs
    exa_thresholds = []
    thresholds = []
    if not run_by_run:
        datasets = [f for f in channel_files if "Thr_" in f]
        exa_thresholds = [str(f.split("Thr_")[-1].split("_")[0]) for f in datasets]
        thresholds = [int(thr, 16) for thr in exa_thresholds]
    if not run_by_run:
        datasets = [x for _, x in sorted(zip(thresholds, datasets))]



    for dataset in datasets:
        if run_by_run:
            run = dataset
            print(f"Processing run {run} ({runs.index(run)+1}/{len(runs)})")
            # Load run information ------------------------------------------------
            led  = int(df_runs.loc[df_runs['Run'] == run, 'LED'].values[0])
            exa_threshold = str(df_runs.loc[df_runs['Run'] == run, 'Threshold'].values[0])
            threshold = int(exa_threshold, 16)
            out_root_file.mkdir(f"Run_{run}")
            out_root_file.cd(f"Run_{run}")
            filename = [f for f in channel_files if str(run)+"_" in f][0]

        else:
            print(f"Processing {dataset} {datasets.index(dataset)+1}/{len(datasets)}")
            filename = dataset
            exa_threshold = str(filename.split("Thr_")[-1].split("_")[0])
            threshold = int(exa_threshold, 16)
            out_root_file.mkdir(f"Thr_{threshold}")
            out_root_file.cd(f"Thr_{threshold}")

        print("Reading file ", filename)
        print("Threshold set: ", threshold)
        ch_sipm = SiPM_channel
        ch_st = filename.split("ChST_")[-1].split(".root")[0]
        st = self_trigger.SelfTrigger(ch_sipm=int(ch_sipm),
                                      ch_st=int(ch_st),
                                      int_low=int_low,
                                      spe_charge=spe_charge,
                                      spe_ampl=spe_ampl,
                                      snr=snr,
                                      metadata_file=filename,
                                      run=run if run_by_run else None,
                                      led=led if run_by_run else None,
                                      ana_folder=ana_folder,
                                      fit_type=fit_type,
                                      leds_to_plot=leds_to_plot,
        )
        st.upload_metadata()

            
        h_st_full = st.create_self_trigger_distribution("h_st_full")
        st.find_acceptance_window()
        bkg_trg_rate, unc_bkg_trg_rate = st.get_bkg_trg_rate()

        st.select_events()
        h_st = st.create_self_trigger_distribution()
        
        st.create_efficiency_histos()
        st.fit_efficiency()

        effnpe, errup_eff_npe, errlow_eff_npe = get_efficiency_at(st.he_STEfficiency_quantized, npe_of_interest)
        eff_fit, err_eff_fit, up_err, low_err = st.get_efficiency_at_fit(npe_of_interest)
        effnpep1, errup_effnpep1, errlow_eff_npep1 = get_efficiency_at(st.he_STEfficiency_quantized,npe_of_interest+1)
        eff_fitp1, err_eff_fitp1, up_errp1, low_errp1 = st.get_efficiency_at_fit(npe_of_interest+1)

        out_df_rows.append({
            "Run": run if run_by_run else np.nan,
            "SiPMChannel": SiPM_channel,
            "LED": led if run_by_run else np.nan,
            "SiPM": SiPM, # FBK or HPK
            "SpeAmpl": spe_ampl,
            "SpeCharge": st.spe_charge,
            "SNR": snr,
            "ThresholdSet": threshold,
            "AccWinLow": st.window_low,
            "AccWinUp": st.window_up,
            "BkgTrgRate": bkg_trg_rate,
            "ErrBkgTrgRate": unc_bkg_trg_rate,
            "ThresholdFit": st.f_sigmoid.GetParameter(0) if st.efficiency_fit_ok else np.nan,
            "ErrThresholdFit": st.f_sigmoid.GetParError(0) if st.efficiency_fit_ok else np.nan,
            # "ThresholdFit": st.f_sigmoid.GetX(0.1, st.nspe_min, st.nspe_max) if st.efficiency_fit_ok else np.nan,
            # "ErrThresholdFit": 0.01 if st.efficiency_fit_ok else np.nan,
            "TauFit": st.f_sigmoid.GetParameter(1) if st.efficiency_fit_ok else np.nan,
            "ErrTauFit": st.f_sigmoid.GetParError(1) if st.efficiency_fit_ok else np.nan,
            "MaxEffFit": st.f_sigmoid.GetParameter(2) if st.efficiency_fit_ok else np.nan,
            "ErrMaxEffFit": st.f_sigmoid.GetParError(2) if st.efficiency_fit_ok else np.nan,
            "10to90": st.get_10to90_range(),
            "10to90Fit": st.get_10to90_range_fit(),
            "FiftyEffPoint": st.fifty,
            "ErrFiftyEffPoint": 0.01,
            f"EffAt{npe_of_interest}PE": effnpe,
            f"ErrEffAt{npe_of_interest}PE": (errup_eff_npe + errlow_eff_npe)/2,
            f"UpErrEffAt{npe_of_interest}PE": errup_eff_npe,
            f"LowErrEffAt{npe_of_interest}PE": errlow_eff_npe,
            f"EffAt{npe_of_interest}PEFit": eff_fit,
            f"ErrEffAt{npe_of_interest}PEFit": err_eff_fit,
            f"UpErrEffAt{npe_of_interest}PEFit": up_err,
            f"LowErrEffAt{npe_of_interest}PEFit": low_err,
            f"EffAt{npe_of_interest+1}PE": effnpep1,
            f"ErrEffAt{npe_of_interest+1}PE": (errup_effnpep1 + errlow_eff_npep1)/2,
            f"UpErrEffAt{npe_of_interest+1}PE": errup_effnpep1,
            f"LowErrEffAt{npe_of_interest+1}PE": errlow_eff_npep1,
            f"EffAt{npe_of_interest+1}PEFit": eff_fitp1,
            f"ErrEffAt{npe_of_interest+1}PEFit": err_eff_fitp1,
            f"UpErrEffAt{npe_of_interest+1}PEFit": up_errp1,
            f"LowErrEffAt{npe_of_interest+1}PEFit": low_errp1,
            "WafflesMean": st.wafflesMean,
            "WafflesSNR": st.wafflesSNR
        })
        h_st_full.Write()
        h_st.Write()
        st.h_total.Write()
        st.h_passed.Write()
        st.he_STEfficiency_nofit.Write()
        st.he_STEfficiency.Write()
        st.he_STEfficiency2.Write()
        st.h_total_quantized.Write()
        st.h_passed_quantized.Write()
        st.he_STEfficiency_quantized.Write()
          

    g_st_calib, offset, slope = self_trigger.fit_thrPE_vs_thrSet(pd.DataFrame(out_df_rows))
    g_st_calib.SetTitle(f"Self-Trigger Calibration Ch {SiPM_channel};Threshold Set;Threshold Fit (PE)")
    g_st_calib2, offset2, slope2 = self_trigger.fit_thrPE_vs_thrSet(pd.DataFrame(out_df_rows), "FiftyEffPoint")
    g_st_calib2.SetTitle(f"Self-Trigger Calibration Ch {SiPM_channel};Threshold Set;Threshold Fifty (PE)")
    out_root_file.cd()
    g_st_calib.Write()
    g_st_calib2.Write()
    out_root_file.Close()
    out_df = pd.DataFrame(out_df_rows)
    out_df['ThresholdFitCalibrated'] = slope*out_df['ThresholdSet'] + offset
    out_df['FiftyCalibrated'] = slope2*out_df['ThresholdSet'] + offset2
    out_df_filename = ch_folder+f"SelfTrigger_Results_Ch_{SiPM_channel}"
    if not run_by_run:
        out_df_filename += "_merged"
    out_df.to_csv(out_df_filename+".csv", index=False)
