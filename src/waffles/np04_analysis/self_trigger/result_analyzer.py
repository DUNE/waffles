import pandas as pd
import numpy as np
from ROOT import TFile, TGraphErrors


result_file = "~/CERN/PDHD/Self_trigger/analysis/SelfTrigger_Results_Ch_11221.csv"
out_file_name = "~/PhD/plotter/projects/NP04_PDS_article/SelfTrigger/ch_11221.root"

if __name__ == "__main__":

    df_result = pd.read_csv(result_file, sep=",")
    print("here")
    out_root_file = TFile(out_file_name, "RECREATE")
    print("here")

    # Convert "LED" column into numpy array of integers
    leds = df_result["LED"].to_numpy(dtype=int)

    for led in leds[:1]:
        print(f"Processing LED: {led}")
        # ThresholdFit vs ThresholdSet
        thr_sets = np.array(df_result[df_result["LED"] == led]["ThresholdSet"], dtype=float)
        err__thr_sets = np.zeros_like(thr_sets, dtype=float)
        thr_fits = np.array(df_result[df_result["LED"] == led]["ThresholdFit"], dtype=float)
        err_thr_fits = np.array(df_result[df_result["LED"] == led]["ErrThresholdFit"], dtype=float)

        print("here1")
        g_ThrFit_ThrSet = TGraphErrors(len(thr_sets), thr_sets, thr_fits, err__thr_sets, err_thr_fits)
        print("here2")
        g_ThrFit_ThrSet.SetName(f"g_ThrFit_ThrSet_LED_{led}")
        print("here3")
        g_ThrFit_ThrSet.SetTitle(f"g_ThrFit_ThrSet_LED_{led}")
        print("here4")
        g_ThrFit_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        print("here5")
        g_ThrFit_ThrSet.GetYaxis().SetTitle("Threshold Fit [p.e.]")
        print("here6")

        g_ThrFit_ThrSet.Write()
        print("here7")
        g_ThrFit_ThrSet.Delete()
        print("here8")

        #ThresholdFit vs BkgTrgRate
        # bkg_trg_rates = np.array(df_result[df_result["LED"] == led]["BkgTrgRate"], dtype=float)
        # err_bkg_trg_rates = np.zeros_like(bkg_trg_rates, dtype=float)
        #
        # g_ThrFit_BkgTrgRate = TGraphErrors(len(bkg_trg_rates), bkg_trg_rates, thr_fits, err_bkg_trg_rates, err_thr_fits)
        # g_ThrFit_BkgTrgRate.SetName(f"g_ThrFit_BkgTrgRate_LED_{led}")
        # g_ThrFit_BkgTrgRate.SetTitle(f"g_ThrFit_BkgTrgRate_LED_{led}")
        # g_ThrFit_BkgTrgRate.GetXaxis().SetTitle("Background Trigger Rate [Hz]")
        # g_ThrFit_BkgTrgRate.GetYaxis().SetTitle("Threshold Fit [p.e.]")
        #
        # g_ThrFit_BkgTrgRate.Write()
        # g_ThrFit_BkgTrgRate.Delete()

    # thrs = np.array(df_result["ThresholdSet"], dtype=int)


    out_root_file.Close()
