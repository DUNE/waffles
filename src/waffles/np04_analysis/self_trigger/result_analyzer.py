import pandas as pd
import numpy as np
from ROOT import TFile, TGraphErrors


result_file = "~/CERN/PDHD/Self_trigger/analysis/SelfTrigger_Results_Ch_11121_Selection_True.csv"
out_file_name = "~/PhD/plotter/projects/PosterINSS/SelfTrigger_Results_Ch_11121_Selection_True.root"

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
        err_thr_sets = np.zeros_like(thr_sets, dtype=float)
        thr_fits = np.array(df_result[df_result["LED"] == led]["ThresholdFit"], dtype=float)
        err_thr_fits = np.array(df_result[df_result["LED"] == led]["ErrThresholdFit"], dtype=float)

        g_ThrFit_ThrSet = TGraphErrors(len(thr_sets), thr_sets, thr_fits, err_thr_sets, err_thr_fits)
        g_ThrFit_ThrSet.SetName(f"g_ThrFit_ThrSet_LED_{led}")
        g_ThrFit_ThrSet.SetTitle(f"g_ThrFit_ThrSet_LED_{led}")
        g_ThrFit_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_ThrFit_ThrSet.GetYaxis().SetTitle("Threshold Fit [p.e.]")
        g_ThrFit_ThrSet.Write()

        #BkgTrgRate vs ThresholdFit
        bkg_trg_rates = np.array(df_result[df_result["LED"] == led]["BkgTrgRate"], dtype=float)
        err_bkg_trg_rates = np.zeros_like(bkg_trg_rates, dtype=float)

        g_BkgTrgRate_ThrFit = TGraphErrors(len(bkg_trg_rates), thr_fits, bkg_trg_rates, err_thr_fits, err_bkg_trg_rates)
        g_BkgTrgRate_ThrFit.SetName(f"g_BkgTrgRate_ThrFit_LED_{led}")
        g_BkgTrgRate_ThrFit.SetTitle(f"g_BkgTrgRate_ThrFit_LED_{led}")
        g_BkgTrgRate_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_BkgTrgRate_ThrFit.GetYaxis().SetTitle("Background Trigger Rate [Hz]")
        g_BkgTrgRate_ThrFit.Write()

        # BkgTrgRate vs ThresholdSet
        g_BkgTrgRate_ThrSet = TGraphErrors(len(bkg_trg_rates), thr_sets, bkg_trg_rates, err_thr_sets, err_bkg_trg_rates)
        g_BkgTrgRate_ThrSet.SetName(f"g_BkgTrgRate_ThrSet_LED_{led}")
        g_BkgTrgRate_ThrSet.SetTitle(f"g_BkgTrgRate_ThrSet_LED_{led}")
        g_BkgTrgRate_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_BkgTrgRate_ThrSet.GetYaxis().SetTitle("Background Trigger Rate [Hz]")
        g_BkgTrgRate_ThrSet.Write()
        
        #BkgTrgRatePreLED vs ThresholdFit
        bkg_trg_rates_pre_led = np.array(df_result[df_result["LED"] == led]["BkgTrgRatePreLED"], dtype=float)
        err_bkg_trg_rates_pre_led = np.zeros_like(bkg_trg_rates_pre_led, dtype=float)
        
        g_BkgTrgRatePreLED_ThrFit = TGraphErrors(len(bkg_trg_rates_pre_led), thr_fits, bkg_trg_rates_pre_led, err_thr_fits, err_bkg_trg_rates_pre_led)
        g_BkgTrgRatePreLED_ThrFit.SetName(f"g_BkgTrgRatePreLED_ThrFit_LED_{led}")
        g_BkgTrgRatePreLED_ThrFit.SetTitle(f"g_BkgTrgRatePreLED_ThrFit_LED_{led}")
        g_BkgTrgRatePreLED_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_BkgTrgRatePreLED_ThrFit.GetYaxis().SetTitle("Background Trigger Rate Pre LED [Hz]")
        g_BkgTrgRatePreLED_ThrFit.Write()

        # BkgTrgRatePreLED vs ThresholdSet
        g_BkgTrgRatePreLED_ThrSet = TGraphErrors(len(bkg_trg_rates_pre_led), thr_sets, bkg_trg_rates_pre_led, err_thr_sets, err_bkg_trg_rates_pre_led)
        g_BkgTrgRatePreLED_ThrSet.SetName(f"g_BkgTrgRatePreLED_ThrSet_LED_{led}")
        g_BkgTrgRatePreLED_ThrSet.SetTitle(f"g_BkgTrgRatePreLED_ThrSet_LED_{led}")
        g_BkgTrgRatePreLED_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_BkgTrgRatePreLED_ThrSet.GetYaxis().SetTitle("Background Trigger Rate Pre LED [Hz]")
        g_BkgTrgRatePreLED_ThrSet.Write()


        # MaxAccuracy vs ThresholdFit
        max_accuracies = np.array(df_result[df_result["LED"] == led]["MaxAccuracy"], dtype=float)*100  # Convert to percentage
        err_max_accuracies = np.zeros_like(max_accuracies, dtype=float)
        
        g_MaxAccuracy_ThrFit = TGraphErrors(len(max_accuracies), thr_fits, max_accuracies, err_thr_fits, err_max_accuracies)
        g_MaxAccuracy_ThrFit.SetName(f"g_MaxAccuracy_ThrFit_LED_{led}")
        g_MaxAccuracy_ThrFit.SetTitle(f"g_MaxAccuracy_ThrFit_LED_{led}")
        g_MaxAccuracy_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_MaxAccuracy_ThrFit.GetYaxis().SetTitle("Max Accuracy [%]")
        g_MaxAccuracy_ThrFit.Write()
        
        # MaxAccuracy vs ThresholdSet
        g_MaxAccuracy_ThrSet = TGraphErrors(len(max_accuracies), thr_sets, max_accuracies, err_thr_sets, err_max_accuracies)
        g_MaxAccuracy_ThrSet.SetName(f"g_MaxAccuracy_ThrSet_LED_{led}")
        g_MaxAccuracy_ThrSet.SetTitle(f"g_MaxAccuracy_ThrSet_LED_{led}")
        g_MaxAccuracy_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_MaxAccuracy_ThrSet.GetYaxis().SetTitle("Max Accuracy [%]")
        g_MaxAccuracy_ThrSet.Write()

        # FalsePositiveRate vs ThresholdFit
        false_positive_rates = np.array(df_result[df_result["LED"] == led]["FalsePositiveRate"], dtype=float)*100
        err_false_positive_rates = np.zeros_like(false_positive_rates, dtype=float)
        
        g_FalsePositiveRate_ThrFit = TGraphErrors(len(false_positive_rates), thr_fits, false_positive_rates, err_thr_fits, err_false_positive_rates)
        g_FalsePositiveRate_ThrFit.SetName(f"g_FalsePositiveRate_ThrFit_LED_{led}")
        g_FalsePositiveRate_ThrFit.SetTitle(f"g_FalsePositiveRate_ThrFit_LED_{led}")
        g_FalsePositiveRate_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_FalsePositiveRate_ThrFit.GetYaxis().SetTitle("False Positive Rate [%]")
        g_FalsePositiveRate_ThrFit.Write()

        # FalsePositiveRate vs ThresholdSet
        g_FalsePositiveRate_ThrSet = TGraphErrors(len(false_positive_rates), thr_sets, false_positive_rates, err_thr_sets, err_false_positive_rates)
        g_FalsePositiveRate_ThrSet.SetName(f"g_FalsePositiveRate_ThrSet_LED_{led}")
        g_FalsePositiveRate_ThrSet.SetTitle(f"g_FalsePositiveRate_ThrSet_LED_{led}")
        g_FalsePositiveRate_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_FalsePositiveRate_ThrSet.GetYaxis().SetTitle("False Positive Rate [%]")
        g_FalsePositiveRate_ThrSet.Write()


        # TruePositiveRate vs ThresholdFit
        true_positive_rates = np.array(df_result[df_result["LED"] == led]["TruePositiveRate"], dtype=float)*100
        err_true_positive_rates = np.zeros_like(true_positive_rates, dtype=float)
        
        g_TruePositiveRate_ThrFit = TGraphErrors(len(true_positive_rates), thr_fits, true_positive_rates, err_thr_fits, err_true_positive_rates)
        g_TruePositiveRate_ThrFit.SetName(f"g_TruePositiveRate_ThrFit_LED_{led}")
        g_TruePositiveRate_ThrFit.SetTitle(f"g_TruePositiveRate_ThrFit_LED_{led}")
        g_TruePositiveRate_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_TruePositiveRate_ThrFit.GetYaxis().SetTitle("True Positive Rate [%]")
        g_TruePositiveRate_ThrFit.Write()
        
        # TruePositiveRate vs ThresholdSet
        g_TruePositiveRate_ThrSet = TGraphErrors(len(true_positive_rates), thr_sets, true_positive_rates, err_thr_sets, err_true_positive_rates)
        g_TruePositiveRate_ThrSet.SetName(f"g_TruePositiveRate_ThrSet_LED_{led}")
        g_TruePositiveRate_ThrSet.SetTitle(f"g_TruePositiveRate_ThrSet_LED_{led}")
        g_TruePositiveRate_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_TruePositiveRate_ThrSet.GetYaxis().SetTitle("True Positive Rate [%]")
        g_TruePositiveRate_ThrSet.Write()

        # TauFit vs ThresholdFit
        tau_fits = np.array(df_result[df_result["LED"] == led]["TauFit"], dtype=float)
        err_tau_fits = np.array(df_result[df_result["LED"] == led]["ErrTauFit"], dtype=float)
        
        g_TauFit_ThrFit = TGraphErrors(len(tau_fits), thr_fits, tau_fits, err_thr_fits, err_tau_fits)
        g_TauFit_ThrFit.SetName(f"g_TauFit_ThrFit_LED_{led}")
        g_TauFit_ThrFit.SetTitle(f"g_TauFit_ThrFit_LED_{led}")
        g_TauFit_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_TauFit_ThrFit.GetYaxis().SetTitle("Tau Fit [p.e.]")
        g_TauFit_ThrFit.Write()
        
        # TauFit vs ThresholdSet
        g_TauFit_ThrSet = TGraphErrors(len(tau_fits), thr_sets, tau_fits, err_thr_sets, err_tau_fits)
        g_TauFit_ThrSet.SetName(f"g_TauFit_ThrSet_LED_{led}")
        g_TauFit_ThrSet.SetTitle(f"g_TauFit_ThrSet_LED_{led}")
        g_TauFit_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_TauFit_ThrSet.GetYaxis().SetTitle("Tau Fit [p.e.]")
        g_TauFit_ThrSet.Write()

        # Accuracy vs ThresholdFit
        accuracies = np.array(df_result[df_result["LED"] == led]["AccuracyThresholdFit"], dtype=float)*100
        err_accuracies = np.zeros_like(accuracies, dtype=float)

        g_Accuracy_ThrFit = TGraphErrors(len(accuracies), thr_fits, accuracies, err_thr_fits, err_accuracies)
        g_Accuracy_ThrFit.SetName(f"g_Accuracy_ThrFit_LED_{led}")
        g_Accuracy_ThrFit.SetTitle(f"g_Accuracy_ThrFit_LED_{led}")
        g_Accuracy_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_Accuracy_ThrFit.GetYaxis().SetTitle("Accuracy [%]")
        g_Accuracy_ThrFit.Write()
        
        # Accuracy vs ThresholdSet
        g_Accuracy_ThrSet = TGraphErrors(len(accuracies), thr_sets, accuracies, err_thr_sets, err_accuracies)
        g_Accuracy_ThrSet.SetName(f"g_Accuracy_ThrSet_LED_{led}")
        g_Accuracy_ThrSet.SetTitle(f"g_Accuracy_ThrSet_LED_{led}")
        g_Accuracy_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_Accuracy_ThrSet.GetYaxis().SetTitle("Accuracy [%]")
        g_Accuracy_ThrSet.Write()
        
        # MaxAccuracy vs ThresholdFit
        max_accuracies = np.array(df_result[df_result["LED"] == led]["MaxAccuracy"], dtype=float)*100
        err_max_accuracies = np.zeros_like(max_accuracies, dtype=float)
        g_MaxAccuracy_ThrFit = TGraphErrors(len(max_accuracies), thr_fits, max_accuracies, err_thr_fits, err_max_accuracies)
        g_MaxAccuracy_ThrFit.SetName(f"g_MaxAccuracy_ThrFit_LED_{led}")
        g_MaxAccuracy_ThrFit.SetTitle(f"g_MaxAccuracy_ThrFit_LED_{led}")
        g_MaxAccuracy_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_MaxAccuracy_ThrFit.GetYaxis().SetTitle("Max Accuracy [%]")
        g_MaxAccuracy_ThrFit.Write()
    
        # MaxAccuracy vs ThresholdSet
        g_MaxAccuracy_ThrSet = TGraphErrors(len(max_accuracies), thr_sets, max_accuracies, err_thr_sets, err_max_accuracies)
        g_MaxAccuracy_ThrSet.SetName(f"g_MaxAccuracy_ThrSet_LED_{led}")
        g_MaxAccuracy_ThrSet.SetTitle(f"g_MaxAccuracy_ThrSet_LED_{led}")
        g_MaxAccuracy_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_MaxAccuracy_ThrSet.GetYaxis().SetTitle("Max Accuracy [%]")
        g_MaxAccuracy_ThrSet.Write()

        # FalsePositiveRateAcc vs ThresholdFit
        false_positive_rates_acc = np.array(df_result[df_result["LED"] == led]["FalsePositiveRateMaxAcc"], dtype=float)*100
        err_false_positive_rates_acc = np.zeros_like(false_positive_rates_acc, dtype=float)
        g_FalsePositiveRateAcc_ThrFit = TGraphErrors(len(false_positive_rates_acc), thr_fits, false_positive_rates_acc, err_thr_fits, err_false_positive_rates_acc)
        g_FalsePositiveRateAcc_ThrFit.SetName(f"g_FalsePositiveRateMaxAcc_ThrFit_LED_{led}")
        g_FalsePositiveRateAcc_ThrFit.SetTitle(f"g_FalsePositiveRateMaxAcc_ThrFit_LED_{led}")
        g_FalsePositiveRateAcc_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_FalsePositiveRateAcc_ThrFit.GetYaxis().SetTitle("False Positive Rate Acc [%]")
        g_FalsePositiveRateAcc_ThrFit.Write()

        # FalsePositiveRateAcc vs ThresholdSet
        g_FalsePositiveRateAcc_ThrSet = TGraphErrors(len(false_positive_rates_acc), thr_sets, false_positive_rates_acc, err_thr_sets, err_false_positive_rates_acc)
        g_FalsePositiveRateAcc_ThrSet.SetName(f"g_FalsePositiveRateMaxAcc_ThrSet_LED_{led}")
        g_FalsePositiveRateAcc_ThrSet.SetTitle(f"g_FalsePositiveRateMaxAcc_ThrSet_LED_{led}")
        g_FalsePositiveRateAcc_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_FalsePositiveRateAcc_ThrSet.GetYaxis().SetTitle("False Positive Rate Acc [%]")
        g_FalsePositiveRateAcc_ThrSet.Write()

        # TruePositiveRateAcc vs ThresholdFit
        true_positive_rates_acc = np.array(df_result[df_result["LED"] == led]["TruePositiveRateMaxAcc"], dtype=float)*100
        err_true_positive_rates_acc = np.zeros_like(true_positive_rates_acc, dtype=float)
        g_TruePositiveRateAcc_ThrFit = TGraphErrors(len(true_positive_rates_acc), thr_fits, true_positive_rates_acc, err_thr_fits, err_true_positive_rates_acc)
        g_TruePositiveRateAcc_ThrFit.SetName(f"g_TruePositiveRateMaxAcc_ThrFit_LED_{led}")
        g_TruePositiveRateAcc_ThrFit.SetTitle(f"g_TruePositiveRateMaxAcc_ThrFit_LED_{led}")
        g_TruePositiveRateAcc_ThrFit.GetXaxis().SetTitle("Threshold Fit [p.e.]")
        g_TruePositiveRateAcc_ThrFit.GetYaxis().SetTitle("True Positive Rate Acc [%]")
        g_TruePositiveRateAcc_ThrFit.Write()

        # TruePositiveRateAcc vs ThresholdSet
        g_TruePositiveRateAcc_ThrSet = TGraphErrors(len(true_positive_rates_acc), thr_sets, true_positive_rates_acc, err_thr_sets, err_true_positive_rates_acc)
        g_TruePositiveRateAcc_ThrSet.SetName(f"g_TruePositiveRateMaxAcc_ThrSet_LED_{led}")
        g_TruePositiveRateAcc_ThrSet.SetTitle(f"g_TruePositiveRateMaxAcc_ThrSet_LED_{led}")
        g_TruePositiveRateAcc_ThrSet.GetXaxis().SetTitle("Threshold Set [a.u.]")
        g_TruePositiveRateAcc_ThrSet.GetYaxis().SetTitle("True Positive Rate Acc [%]")
        g_TruePositiveRateAcc_ThrSet.Write()



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
