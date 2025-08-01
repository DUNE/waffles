
import waffles.np04_analysis.self_trigger.self_trigger as self_trigger
from waffles.np04_utils.utils import get_np04_channel_mapping
import waffles.input_output.hdf5_structured as reader
from ROOT import TFile, TH1D, TF1, TGraph, gROOT, ROOT
import numpy as np
import os
import pandas as pd

run_info_file = "./configs/SelfTrigger_RunInfo.csv"
ana_folder = "/eos/home-f/fegalizz/ProtoDUNE_HD/SelfTrigger/analysis/"
debug_folder  = "/eos/home-f/fegalizz/ProtoDUNE_HD/SelfTrigger/analysis/debugging/"
homedir =  "/afs/cern.ch/user/f/fegalizz/"
calibration_file = "SelfTrigger_Calibration.csv"
SiPM_channel = 11221
runs = [31553, 31602, 31632, 31661, 31689, 31954, 31966, 31978, 31990]

def create_selected_self_trigger_distribution(wfs_st, window_low, window_up, name="") -> TH1D:
    """
    Take the st waveforms indec where adcs==1
    """
    if name == "":
        name = "h_selftrigger"
    h_selftrigger = TH1D(name, f"{name};Ticks;Counts", 1024, -0.5, 1023.5)

    for wf in wfs_st:
        if wf.adcs[window_low:window_up+1].sum() == 0:
            continue

        st_positions = np.flatnonzero(wf.adcs)
        for st in st_positions:
            h_selftrigger.Fill(st)

    return h_selftrigger


def persistence_plot(wfs_sipm, wfs_st, window_low, window_up):
    oks = np.array([True if (wf.adcs[window_low:window_up+1] > 0).any() else False for wf in wfs_st])
    wfs_sipm = wfs_sipm[oks]
    wvfs = np.array([wf.adcs_float for wf in wfs_sipm])
    times = np.linspace(0, len(wfs_sipm[0].adcs), len(wfs_sipm[0].adcs), endpoint=False)
    times = np.tile(times, (len(wfs_sipm), 1))
    nbinsx = len(wfs_sipm[0].adcs)
    h, yedges, xedges = np.histogram2d(wvfs.flatten(), times.flatten(),
                                       bins=(200, nbinsx),
                                       range=[[-30,50], [0, nbinsx]])
    h[h==0] = np.nan
    return h, xedges, yedges


if __name__ == "__main__":
    ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit");
    df_runs = pd.read_csv(run_info_file, sep=",") 
    out_df_rows = []

    out_root_file = TFile("ST_debug.root", "RECREATE")
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


    ch_folder = debug_folder+f"Ch_{SiPM_channel}/"
    if not os.path.exists(ch_folder):
        os.makedirs(ch_folder)

    g_bkgRate_p0 = TGraph()
    

    for run in runs:

        filename = f"/eos/home-f/fegalizz/ProtoDUNE_HD/SelfTrigger/files/Run_{run}_ChSiPM_11221_ChST_11220_structured.hdf5"
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
        

        #First peak
        st.window_low = 232
        st.window_up  = 247
        st.create_wfs()
        st.select_waveforms()
        h_st = st.create_self_trigger_distribution()
        h_st.SetName(f"h_selftrigger_run_{run}")
        
        x_first_peak = 236.8
        x_second_peak = 242.6
        y_first_peak = h_st.GetBinContent(h_st.FindBin(x_first_peak))
        y_second_peak = h_st.GetBinContent(h_st.FindBin(x_second_peak))
        
        f_STpeak = TF1("f_STpeak", "gaus(0)+gaus(3)", st.window_low, st.window_up)
        f_STpeak.SetParameters(y_first_peak, x_first_peak, 1.5, y_second_peak, x_second_peak, 1.5)
        f_STpeak.SetParLimits(0, y_first_peak * 0.6, y_first_peak * 1.3)
        f_STpeak.SetParLimits(1, x_first_peak - 1., x_first_peak + 1.)
        f_STpeak.SetParLimits(2, 0.9, 2.5)
        f_STpeak.SetParLimits(3, y_second_peak * 0.6, y_second_peak * 1.3)
        f_STpeak.SetParLimits(4, x_second_peak - 1., x_second_peak + 1.)
        f_STpeak.SetParLimits(5, 0.9, 2.5)
        f_STpeak.SetNpx(2000)
        h_st.Fit(f_STpeak, "R")

        f_gaus_first = TF1("f_gaus_first", "gaus", st.window_low, st.window_up)
        f_gaus_first.SetParameters(f_STpeak.GetParameter(0), f_STpeak.GetParameter(1), f_STpeak.GetParameter(2))
        f_gaus_second = TF1("f_gaus_second", "gaus", st.window_low, st.window_up)
        f_gaus_second.SetParameters(f_STpeak.GetParameter(3), f_STpeak.GetParameter(4), f_STpeak.GetParameter(5))
        

        n_bkg_triggers_preLED = h_st.Integral(1, int_low)
        bkg_trg_rate_preLED = (n_bkg_triggers_preLED * 10**9) / (len(st.wfs_st) * int_low * 16.)
        first_fraction = f_gaus_first.Integral(st.window_low, st.window_up) / (f_gaus_first.Integral(st.window_low, st.window_up) + f_gaus_second.Integral(st.window_low, st.window_up))
        g_bkgRate_p0.SetPoint(g_bkgRate_p0.GetN(), bkg_trg_rate_preLED, first_fraction) 

        h_st.Write()



    g_bkgRate_p0.SetName("g_firstFraction_p0")
    g_bkgRate_p0.Write()
    out_root_file.Close()
