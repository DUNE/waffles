import pandas as pd
import numpy as np
import yaml
from ROOT import TFile, TGraphErrors

if __name__ == "__main__":
    with open("steering.yml", 'r') as stream:
        steering_config = yaml.safe_load(stream)
    params_file_name = steering_config.get("params_file", "params.yml")

    with open(params_file_name, 'r') as stream:
        user_config = yaml.safe_load(stream)
    ana_folder       = user_config.get("ana_folder")
    SiPM_channel     = user_config.get("SiPM_channel")
    
    result_file = f"{ana_folder}Jitter_Ch_{SiPM_channel}.csv"
    df_result = pd.read_csv(result_file, sep=",")
    
    out_file_name = f"{ana_folder}Jitter_Results_Ch_{SiPM_channel}.root"

    df_result = pd.read_csv(result_file, sep=",")
    df_result = df_result[df_result["IntegralTrg"] > 500]
    df_result = df_result[df_result["FitOK"] == True]
    df_result = df_result[df_result["PE"] > 0]
    df_result = df_result[df_result["ErrSigmaTrg"] < df_result["SigmaTrg"]]
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

        # MeanTrgPos vs PE
        mean_trg_pos = np.array(df_result[df_result["Threshold"] == thr]["MeanTrgPos"], dtype=float)
        err_mean_trg_pos = np.array(df_result[df_result["Threshold"] == thr]["ErrMeanTrgPos"], dtype=float)
        
        g_MeanTrgPos_PE = TGraphErrors(len(pe), pe, mean_trg_pos, err_zeros, err_mean_trg_pos)
        g_MeanTrgPos_PE.SetName(f"g_MeanTrgPos_PE_Thr_{thr}")
        g_MeanTrgPos_PE.SetTitle(f"g_MeanTrgPos_PE_Thr_{thr}")
        g_MeanTrgPos_PE.GetXaxis().SetTitle("PE")
        g_MeanTrgPos_PE.GetYaxis().SetTitle("Mean Trigger Position [ticks]")
        g_MeanTrgPos_PE.Write()

    # Take the "PE" columns and convert it in a numpy array of unique values
    pe = np.array(df_result["PE"], dtype=float)
    pe = np.unique(pe)
    err_zeros = np.zeros_like(pe, dtype=float)
    # For each pe value, take the mean of the SigmaTrg and MeanTrgPos, then compute their standard deviation
    mean_sigma_trg = []
    std_sigma_trg = []
    mean_mean_trg_pos = []
    std_mean_trg_pos = []
    for p in pe:
        sigma_trg = np.array(df_result[df_result["PE"] == p]["SigmaTrg"], dtype=float)
        mean_sigma_trg.append(np.mean(sigma_trg))
        std_sigma_trg.append(np.std(sigma_trg))

        mean_trg_pos = np.array(df_result[df_result["PE"] == p]["MeanTrgPos"], dtype=float)
        mean_mean_trg_pos.append(np.mean(mean_trg_pos))
        std_mean_trg_pos.append(np.std(mean_trg_pos))
        

    g_SigmaTrg_PE_mean = TGraphErrors(len(pe), pe, np.array(mean_sigma_trg), err_zeros, np.array(std_sigma_trg))
    g_SigmaTrg_PE_mean.SetName(f"g_SigmaTrg_PE_Mean")
    g_SigmaTrg_PE_mean.SetTitle(f"g_SigmaTrg_PE_Mean")
    g_SigmaTrg_PE_mean.GetXaxis().SetTitle("PE")
    g_SigmaTrg_PE_mean.GetYaxis().SetTitle("Sigma Trigger [ticks]")
    g_SigmaTrg_PE_mean.Write()

    g_MeanTrgPos_PE_mean = TGraphErrors(len(pe), pe, np.array(mean_mean_trg_pos), err_zeros, np.array(std_mean_trg_pos))
    g_MeanTrgPos_PE_mean.SetName(f"g_MeanTrgPos_PE_Mean")
    g_MeanTrgPos_PE_mean.SetTitle(f"g_MeanTrgPos_PE_Mean")
    g_MeanTrgPos_PE_mean.GetXaxis().SetTitle("PE")
    g_MeanTrgPos_PE_mean.GetYaxis().SetTitle("Mean Trigger Position [ticks]")
    g_MeanTrgPos_PE_mean.Write()

    out_root_file.Close()
