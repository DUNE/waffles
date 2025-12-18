import pandas as pd
import numpy as np
from ROOT import TFile, TGraphErrors

channel = 10441
channel = 10903
channel = 10945
channel = 11121
channel = 11221
result_file = f"~/CERN/PDHD/Self_trigger/analysis/Jitter_Ch_{channel}.csv"
out_file_name = f"~/PhD/plotter/projects/NP04_PDS_article/SelfTrigger/Jitter_Results_Ch_{channel}.root"

if __name__ == "__main__":

    df_result = pd.read_csv(result_file, sep=",")
    df_result = df_result[df_result["IntegralTrg"] > 500]
    df_result = df_result[df_result["FitOK"] == True]
    df_result = df_result[df_result["PE"] > 0]
    df_result = df_result[df_result["ErrSigmaTrg2"] < df_result["SigmaTrg2"]]
    out_root_file = TFile(out_file_name, "RECREATE")
    out_root_file.cd()

    thresholds = df_result["Threshold"].to_numpy(dtype=int)
    thresholds = np.unique(thresholds)

    for thr in thresholds:
        # SigmaTrg vs PE
        sigma_trg = np.array(df_result[df_result["Threshold"] == thr]["SigmaTrg"], dtype=float)
        err_sigma_trg = np.array(df_result[df_result["Threshold"] == thr]["ErrSigmaTrg"], dtype=float)
        pe = np.array(df_result[df_result["Threshold"] == thr]["PE"], dtype=float)
        err_zeros = np.zeros_like(pe, dtype=float)

        g_SigmaTrg_PE = TGraphErrors(len(pe), pe, sigma_trg, err_zeros, err_sigma_trg)
        g_SigmaTrg_PE.SetName(f"g_SigmaTrg_PE_Thr_{thr}")
        g_SigmaTrg_PE.SetTitle(f"g_SigmaTrg_PE_Thr_{thr}")
        g_SigmaTrg_PE.GetXaxis().SetTitle("PE")
        g_SigmaTrg_PE.GetYaxis().SetTitle("Sigma Trigger [ticks]")
        g_SigmaTrg_PE.Write()

        # SigmaTrg2 vs PE
        sigma_trg = np.array(df_result[df_result["Threshold"] == thr]["SigmaTrg2"], dtype=float)
        err_sigma_trg = np.array(df_result[df_result["Threshold"] == thr]["ErrSigmaTrg2"], dtype=float)

        g_SigmaTrg2_PE = TGraphErrors(len(pe), pe, sigma_trg, err_zeros, err_sigma_trg)
        g_SigmaTrg2_PE.SetName(f"g_SigmaTrg2_PE_Thr_{thr}")
        g_SigmaTrg2_PE.SetTitle(f"g_SigmaTrg2_PE_Thr_{thr}")
        g_SigmaTrg2_PE.GetXaxis().SetTitle("PE")
        g_SigmaTrg2_PE.GetYaxis().SetTitle("Sigma Trigger2 [ticks]")
        g_SigmaTrg2_PE.Write()

    # Take the "PE" columns and convert it in a numpy array of unique values
    pe = np.array(df_result["PE"], dtype=float)
    pe = np.unique(pe)
    err_zeros = np.zeros_like(pe, dtype=float)
    # For each pe value, take the mean of the SigmaTrg2 and compute the standard deviation
    mean_sigma_trg2 = []
    std_sigma_trg2 = []
    for p in pe:
        sigma_trg2 = np.array(df_result[df_result["PE"] == p]["SigmaTrg2"], dtype=float)
        mean_sigma_trg2.append(np.mean(sigma_trg2))
        std_sigma_trg2.append(np.std(sigma_trg2))

    g_SigmaTrg2_PE_mean = TGraphErrors(len(pe), pe, np.array(mean_sigma_trg2), err_zeros, np.array(std_sigma_trg2))
    g_SigmaTrg2_PE_mean.SetName(f"g_SigmaTrg2_PE_Mean")
    g_SigmaTrg2_PE_mean.SetTitle(f"g_SigmaTrg2_PE_Mean")
    g_SigmaTrg2_PE_mean.GetXaxis().SetTitle("PE")
    g_SigmaTrg2_PE_mean.GetYaxis().SetTitle("Sigma Trigger2 [ticks]")
    g_SigmaTrg2_PE_mean.Write()

    out_root_file.Close()
