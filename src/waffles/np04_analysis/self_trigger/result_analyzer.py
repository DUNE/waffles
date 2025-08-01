import pandas as pd
import numpy as np
from ROOT import TFile, TGraphErrors


result_file = "~/CERN/PDHD/Self_trigger/analysis/SelfTrigger_Results_Ch_11221.csv"
out_file_name = "~/PhD/plotter/projects/NP04_PDS_article/SelfTrigger/ch_11221.root"

if __name__ == "__main__":

    df_result = pd.read_csv(result_file, sep=",")
    out_root_file = TFile(out_file_name, "RECREATE")
    out_root_file.cd()

    # Convert "LED" column into numpy array of integers
    leds = df_result["LED"].to_numpy(dtype=int)
    leds = np.unique(leds)

    for led in leds:
        # ThresholdFit vs ThresholdSet
        thr_sets = np.array(df_result[df_result["LED"] == led]["ThresholdSet"], dtype=float)
        err__thr_sets = np.zeros_like(thr_sets, dtype=float)
        thr_fits = np.array(df_result[df_result["LED"] == led]["ThresholdFit"], dtype=float)
        err_thr_fits = np.array(df_result[df_result["LED"] == led]["ErrThresholdFit"], dtype=float)

        g_ThrFit_ThrSet = TGraphErrors(len(thr_sets), thr_sets, thr_fits, err__thr_sets, err_thr_fits)
        g_ThrFit_ThrSet.SetName(f"g_ThrFit_ThrSet_LED_{led}")
        g_ThrFit_ThrSet.SetTitle(f"g_ThrFit_ThrSet_LED_{led}")
        g_ThrFit_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_ThrFit_ThrSet.GetYaxis().SetTitle("Threshold Fit [p.e.]")

        g_ThrFit_ThrSet.Write()

        #ThresholdFit vs BkgTrgRate
        bkg_trg_rates = np.array(df_result[df_result["LED"] == led]["BkgTrgRate"], dtype=float)
        err_bkg_trg_rates = np.zeros_like(bkg_trg_rates, dtype=float)

        g_BkgTrgRate_ThrFit = TGraphErrors(len(bkg_trg_rates), thr_fits, bkg_trg_rates, err_thr_fits, err_bkg_trg_rates)
        g_BkgTrgRate_ThrFit.SetName(f"g_BkgTrgRate_ThrFit_LED_{led}")
        g_BkgTrgRate_ThrFit.SetTitle(f"g_BkgTrgRate_ThrFit_LED_{led}")
        g_BkgTrgRate_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_BkgTrgRate_ThrFit.GetYaxis().SetTitle("Background Trigger Rate [Hz]")

        g_BkgTrgRate_ThrFit.Write()

    thresholds = np.array(df_result["ThresholdSet"], dtype=float)
    thresholds = np.unique(thresholds)

    for threshold in thresholds:
        # SigmaTrg vs LED
        sigmas = np.array(df_result[df_result["ThresholdSet"] == threshold]["SigmaTrg"], dtype=float)
        err_sigmas = np.array(df_result[df_result["ThresholdSet"] == threshold]["ErrSigmaTrg"], dtype=float)
        leds = np.array(df_result[df_result["ThresholdSet"] == threshold]["LED"], dtype=float)
        err_leds = np.zeros_like(leds, dtype=float)
        print()

        g_SigmaTrg_LED = TGraphErrors(len(leds), leds, sigmas, err_leds, err_sigmas)
        g_SigmaTrg_LED.SetName(f"g_SigmaTrg_LED_Thr_{threshold}")
        g_SigmaTrg_LED.SetTitle(f"g_SigmaTrg_LED_Thr_{threshold}")
        g_SigmaTrg_LED.GetXaxis().SetTitle("LED [a.u.]")
        g_SigmaTrg_LED.GetYaxis().SetTitle("Sigma Trigger [ticks]")

        g_SigmaTrg_LED.Write()


    out_root_file.Close()
