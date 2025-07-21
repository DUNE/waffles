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
# runs = [31519,
#     31553,
#     31602,
#     31661,
#     31689,
#     31954,
#     31966,
#     31978,
#     31990,
#     31991,
#     31992,
#     31993]

runs = [31519,
    31522,
    31525,
    31527,
    31553,
    31556,
    31558,
    31561,
    31602,
    31605,
    31607,
    31609,
    31632,
    31635,
    31637,
    31639,
    31661,
    31663,
    31665,
    31668,
    31689,
    31691,
    31693,
    31696,
    31954,
    31955,
    31956,
    31957,
    31966,
    31967,
    31968,
    31969,
    31978,
    31979,
    31980,
    31981,
    31990,
    31991,
    31992,
    31993,
    32002,
    32003,
    32004,
    32005,
    32014,
    32015,
    32016,
    32017,
    32026,
    32027,
    32028,
    32029,
    32038,
    32039,
    32040,
    32041,
    32050,
    32051,
    32052,
    32053,
    32062,
    32063,
    32064,
    32065,
    32074,
    32075,
    32076,
    32077,
    32086,
    32087,
    32088,
    32089,
    32098,
    32099,
    32100,
    32101,
    32110,
    32111,
    32112,
    32113,
    32122,
    32123,
    32124,
    32125,
    32135,
    32136,
    32137,
    32138,
    32147,
    32148,
    32149,
    32150,]

SiPM_channel = 11221


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


    out_root_file = TFile(ana_folder+f"Ch_{SiPM_channel}.root", "RECREATE")
    ch_folder = ana_folder+f"Ch_{SiPM_channel}/"
    if not os.path.exists(ch_folder):
        os.makedirs(ch_folder)

    for run in runs:
        print("RUN: ", run)
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

        h_st = st.create_self_trigger_distribution()
        h_st = st.fit_self_trigger_distribution()
        bkg_trg_rate = st.bkg_trg_rate
        
        fig = plt.figure(figsize=(10, 8))
        h_total, h_passed = st.create_efficiency_histos("he_efficiency")
        plt.savefig(ch_folder+f"Efficiency_Run_{run}_LED_{led}_Thr_{threshold}.png")
        
        f_sigmoid = TF1("f_sigmoid", "[2]/(1+exp(([0]-x)/[1]))", -2, 7)
        he_efficiency = st.fit_efficiency(f_sigmoid=f_sigmoid)

        st.get_trigger_rate()
        


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
            "MeanTrgPos": st.f_STpeak.GetParameter(1),
            "ErrMeanTrgPos": st.f_STpeak.GetParError(1),
            "SigmaTrg": st.f_STpeak.GetParameter(2),
            "ErrSigmaTrg": st.f_STpeak.GetParError(2),
            "ThresholdFit": f_sigmoid.GetParameter(0),
            "ErrThresholdFit": f_sigmoid.GetParError(0),
            "TauFit": f_sigmoid.GetParameter(1),
            "ErrTauFit": f_sigmoid.GetParError(1),
            "MaxEffFit": f_sigmoid.GetParameter(2),
            "ErrMaxEffFit": f_sigmoid.GetParError(2),
        })
        h_st.Write()
        he_efficiency.Write()
          

    out_root_file.Close()
    out_df = pd.DataFrame(out_df_rows)
    out_df.to_csv(ch_folder+f"SelfTrigger_Results_Ch_{SiPM_channel}.csv", index=False)




        #
        #
        # # Analysis on selected waveforms ---------------------------------------------
        # st.select_waveforms()
        #
        # h_st = st.create_self_trigger_distribution()
        # x_model, y_model = st.fit_self_trigger_distribution()
        # 
        # fig = plt.figure(figsize=(10, 8))
        # h_total, h_passed = st.create_efficiency_histos("he_efficiency_select")
        # plt.savefig(homedir+"efficiency_select.png")
        # 
        # he_efficiency_select = st.fit_efficiency(f_sigmoid=f_sigmoid)
        #
        #
        # # Outlier studies ---------------------------------------------------------
        # st.select_outliers(f_sigmoid=f_sigmoid)
        # fig = plt.figure(figsize=(10, 8))
        # h_st = st.create_self_trigger_distribution()
        # mplhep.histplot(h_st, label="Self Trigger Distribution", color="blue")
        # plt.xlabel("Ticks")
        # plt.ylabel("Counts")
        # plt.savefig(homedir+"outliers_self_trigger_distribution.png")
        #
        # fig = plt.figure(figsize=(10, 8))
        # h, xedges, yedges = self_trigger.persistence_plot(st.wfs_sipm)
        # x, y = np.meshgrid(xedges, yedges)
        # pcm = plt.pcolormesh(x, y, np.log10(h))
        # plt.xlabel("Ticks")
        # plt.ylabel("Counts")
        # plt.savefig(homedir+"outliers_persistence_plot.png")
        #
        #
