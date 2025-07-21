import pandas as pd
import numpy as np
from ROOT import TFile, TGraphErrors


result_file = "/eos/home-f/fegalizz/ProtoDUNE_HD/SelfTrigger/analysis/Ch_11221/SelfTrigger_Results_Ch_11221.csv"
out_file_name = "SelfTrigger_Results_Ch_11221.root"

if __name__ == "__main__":

    df_result = pd.read_csv(result_file, sep=",")
    out_root_file = TFile(out_file_name, "RECREATE")

    # Convert "LED" column into numpy array of integers
    leds = df_result["LED"].to_numpy(dtype=int)

    for led in leds:
        # ThresholdFit vs ThresholdSet
        thr_sets = df_result[df_result["LED"] == led]["ThresholdSet"].to_numpy(dtype=float)
        err__thr_sets = np.zeros_like(thr_sets, dtype=float)
        thr_fits = df_result[df_result["LED"] == led]["ThresholdFit"].to_numpy(dtype=float)
        err_thr_fits = df_result[df_result["LED"] == led]["ErrThresholdFit"].to_numpy(dtype=float)

        g_ThrFit_ThrSet = TGraphErrors(len(thr_sets), thr_sets, thr_fits, err__thr_sets, err_thr_fits)
        g_ThrFit_ThrSet.SetName(f"g_ThrFit_ThrSet_LED_{led}")
        g_ThrFit_ThrSet.SetTitle(f"g_ThrFit_ThrSet_LED_{led}")
        g_ThrFit_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_ThrFit_ThrSet.GetYaxis().SetTitle("Threshold Fit [p.e.]")

        g_ThrFit_ThrSet.Write()
        g_ThrFit_ThrSet.Delete()


    out_root_file.Close()
