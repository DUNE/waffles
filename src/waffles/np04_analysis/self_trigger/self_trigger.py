from math import sqrt
import waffles
import numpy as np
import pandas as pd
from waffles.np04_analysis.time_resolution.utils import create_float_waveforms, sub_baseline_to_wfs
from ROOT import TH1D, TEfficiency, TF1, TSpectrum, TGraphErrors, TFitResultPtr
import ROOT
import waffles.utils.numerical_utils as wun
import waffles.data_classes.CalibrationHistogram as cls
from waffles.utils.fit_peaks.fit_peaks import fit_peaks_of_CalibrationHistogram
from waffles.np04_analysis.self_trigger.utils import *




class SelfTrigger:
    def __init__(self,
                 ch_sipm: int=0,
                 ch_st: int=0,
                 wf_set = None,
                 prepulse_ticks: int=0,
                 int_low: int=0,
                 int_up: int=0,
                 bsl_rms: float=0.0,
                 spe_charge: float=0.0,
                 spe_ampl: float=0.0,
                 snr: float=0.0,
                 metadata_file: str="",
                 run: int=0,
                 led: int=0,
                 ana_folder: str="",
                 fit_type: str="multigauss_iminuit",
                 leds_to_plot: list[int]=[],
                 verbose: bool=False
                 ) -> None:

        """
        This class is used to set the parameters for self-triggering.
        """
        self.ch_sipm = ch_sipm
        self.ch_st = ch_st
        self.wf_set = wf_set

        self.prep = prepulse_ticks
        self.int_low = int_low
        self.int_up = int_up
        self.spe_charge = spe_charge
        self.spe_ampl = spe_ampl
        self.snr = snr
        self.bsl_rms = bsl_rms
        self.metadata_file = metadata_file
        
        self.h_low = -1.5
        self.h_up = 10
        self.trigger_rate = 0.0

        self.window_low = 0
        self.window_up = 0
        self.run = run
        self.led = led
        self.ana_folder = ana_folder
        self.fit_type = fit_type
        self.leds_to_plot = leds_to_plot
        self.verbose = verbose

        self.nticks = 1024
        
        self.f_sigmoid = TF1("f_sigmoid", "[2]/(1+exp(([0]-x)/[1]))", -2, 7)


    
    def create_wfs(self) -> None:
        """
        Create the waveforms for the SiPM and ST channels. Subtract the baseline for SiPM waveforms.
        Check that the number of waveforms in both channels is the same.
        """
        t_wfset = waffles.WaveformSet.from_filtered_WaveformSet(self.wf_set, allow_channel_wfs, self.ch_sipm)
        self.wfs_sipm = np.array(t_wfset.waveforms)
        create_float_waveforms(self.wfs_sipm)
        sub_baseline_to_wfs(self.wfs_sipm, self.prep)
        t_wfset = waffles.WaveformSet.from_filtered_WaveformSet(self.wf_set, allow_channel_wfs, self.ch_st)
        self.wfs_st = np.array(t_wfset.waveforms)
        # check len
        if len(self.wfs_sipm) != len(self.wfs_st):
            print(f"Number of waveforms in SiPM channel {self.ch_sipm}: {len(self.wfs_sipm)}")
            print(f"Number of waveforms in ST channel {self.ch_st}: {len(self.wfs_st)}")
            raise ValueError("The number of waveforms in the SiPM and ST channels do not match. ")

        del t_wfset

    def upload_metadata(self) -> None:
        """
        Upload the metadata from the ROOT file created using dump_raw_to_meta.py
        """
        df = ROOT.RDataFrame("SelfTriggerTree", self.metadata_file)
        arrays = df.AsNumpy()

        self.selection = arrays["selection"].astype(bool)
        self.pe = arrays["pe"].astype(float)
        self.trigger_times_list = list(arrays["trigger_times"])

        return None

    def create_self_trigger_distribution(self, name="") -> TH1D:
        """
        Create the self-trigger time-distribution histogram.
        """
        if name == "":
            name = "h_selftrigger"
        
        flat = np.fromiter((trg_time for trg_times in self.trigger_times_list for trg_time in trg_times), dtype=np.double)
        h_selftrigger = TH1D(name, f"{name};Ticks;Counts", self.nticks, -0.5, self.nticks-0.5)
        h_selftrigger.FillN(len(flat), flat, np.ones_like(flat))
       
        self.h_st = h_selftrigger
        return h_selftrigger

    def get_bkg_trg_rate(self, window_low: int=1, window_up: int=1) -> tuple:
        """
        Get the background trigger rate from the self-trigger time-distribution.
        Consider a window where the distribution is flat and not affected by the LED pulse.
        Typically before the LED pulse (in NP04 data we saw that the background rate
        estimated after the LED peak depended on the LED intensity).
        Parameters:
        window_low: int
            Lower bound of the integration window.
        window_up: int
            Upper bound of the integration window.
        Returns:
        bkg_trg_rate: float
            Background trigger rate in Hz.
        """
        if window_up == 1:
            window_up = self.int_low
        n_bkg_triggers = self.h_st.Integral(int(window_low), int(window_up))
        unc_n_bkg_triggers= sqrt(n_bkg_triggers)
        norm_factor = 10**9 / (len(self.pe) * (window_up-window_low) * 16.)
        bkg_trg_rate= n_bkg_triggers* norm_factor
        unc_bkg_trg_rate= unc_n_bkg_triggers* norm_factor

        return bkg_trg_rate, unc_bkg_trg_rate

    def find_acceptance_window(self) -> None:
        """
        Find the acceptance window, namely the region where the self-trigger distribution
        is above the background level. This region is the one corresponding to the LED pulse.
        """
        x_hST_max = self.h_st.GetMaximumBin()

        f_constant = TF1("f_constant", "pol0", 2, self.prep)
        f_constant.SetParameter(0, self.h_st.GetBinContent(5))
        f_constant.SetNpx(1000)
        fit_option = "QR"
        if self.verbose:
            fit_option = "R"
            print(f"Fitting range for acceptance window: 2 - {self.prep}")
        self.h_st.Fit(f_constant, fit_option)
        
        thr_counts = f_constant.GetParameter(0)+sqrt(f_constant.GetParameter(0))
        pretrg = x_hST_max
        i = x_hST_max

        while i > 1:
            pretrg = i
            if self.h_st.GetBinContent(i - 1) < thr_counts:
                break
            i -= 1

        afttrg = x_hST_max
        i = x_hST_max
        while i < self.nticks - 1:
            afttrg = i
            if self.h_st.GetBinContent(i + 1) < thr_counts:
                break
            i += 1 

        self.window_low = int(pretrg)
        self.window_up  = int(afttrg)
        return self.h_st

    def fit_self_trigger_distribution(self) -> TH1D:
        """
        Fit the self-trigger distribution.
        """
        x_hST_max = self.h_st.GetMaximumBin()
        y_hST_max = self.h_st.GetBinContent(x_hST_max)

        f_constant = TF1("f_constant", "pol0", 2, self.prep)
        f_constant.SetParameter(0, self.h_st.GetBinContent(5))
        f_constant.SetNpx(2000)
        fit_option = "QR"
        if self.verbose:
            fit_option = "R"
        self.h_st.Fit(f_constant, fit_option)
        
        thr_counts = f_constant.GetParameter(0)+sqrt(f_constant.GetParameter(0))
        if thr_counts < 1.0:
            thr_counts = 1.0
        pretrg = x_hST_max
        i = x_hST_max

        while i > 1:
            pretrg = i
            print(i)
            if self.h_st.GetBinContent(i) < thr_counts \
                or self.h_st.GetBinContent(i - 1) > self.h_st.GetBinContent(i):
                pretrg = i-1
                break
            i -= 1

        afttrg = x_hST_max
        i = x_hST_max
        while i < self.nticks - 1:
            afttrg = i
            if self.h_st.GetBinContent(i + 1) < thr_counts \
                or self.h_st.GetBinContent(i + 1) > self.h_st.GetBinContent(i):
                break
            i += 1 

        f_STpeak = TF1("f_STpeak", f"{f_constant.GetParameter(0)}+gaus", pretrg, afttrg)
        estimated_gaus_amp = y_hST_max - f_constant.GetParameter(0)
        f_STpeak.SetParameters(estimated_gaus_amp, x_hST_max, 1.0)
        f_STpeak.SetParLimits(0, estimated_gaus_amp * 0.5, estimated_gaus_amp * 1.5)
        f_STpeak.SetParLimits(1, pretrg, afttrg)
        f_STpeak.SetParLimits(2, 0.05, 5.0)
        f_STpeak.SetNpx(1000)
        self.h_st.Fit(f_STpeak, fit_option)

        self.f_STpeak = f_STpeak
        self.window_low = int(pretrg)
        self.window_up  = int(afttrg)
        return self.h_st

    def fit_self_trigger_distribution2(self, fit_second_peak: bool=False) -> tuple:
        """
        Now use ROOT::TSpectrum
        """
        self.h_st2 = self.h_st.Clone("h_selftrigger_bkgsub")
        h_bkg = TSpectrum()
        background = h_bkg.Background(self.h_st2, 15)
        for i in range(1, self.h_st2.GetNbinsX() + 1):
            self.h_st2.SetBinContent(i, self.h_st2.GetBinContent(i) - background[i - 1])

        # Starting from the maximum, go left and right until we find the first bin with content larger than the previous one
        x_hST_max = self.h_st2.GetMaximumBin()
        y_hST_max = self.h_st2.GetBinContent(x_hST_max)
        left_peak_candidate = False
        x_second_peak = np.nan
        # Left scan
        self.window_low2 = x_hST_max
        for i in range(2, x_hST_max):
            if self.h_st2.GetBinContent(x_hST_max - i) > self.h_st2.GetBinContent(x_hST_max - (i - 1)) \
                    or self.h_st2.GetBinContent(x_hST_max - i) < y_hST_max * 0.005:
                self.window_low2 = x_hST_max - i
                break

        if fit_second_peak:
            x_second_peak = get_xmax_in_range(self.h_st2, max(1, self.window_low2 - 10), self.window_low2)
            y_second_peak = self.h_st2.GetBinContent(x_second_peak)
            if y_second_peak > 0.03 * y_hST_max:
                left_peak_candidate = True
                for i in range (1, x_second_peak):
                    if self.h_st2.GetBinContent(x_second_peak - i) > self.h_st2.GetBinContent(x_second_peak - (i - 1)) \
                            or self.h_st2.GetBinContent(x_second_peak - i) < y_hST_max * 0.02:
                        self.window_low2 = x_second_peak - i
                        break


        right_peak_candidate = False
        # Right scan
        self.window_up2 = x_hST_max
        for i in range(x_hST_max+1, self.h_st2.GetNbinsX()):
            if self.h_st2.GetBinContent(i + 1) > self.h_st2.GetBinContent(i) \
                    or self.h_st2.GetBinContent(i) < y_hST_max * 0.005:
                self.window_up2 = i
                break

        if not left_peak_candidate and fit_second_peak:
            x_second_peak = get_xmax_in_range(self.h_st2, self.window_up2, min(self.h_st2.GetNbinsX(), self.window_up2 + 10))
            y_second_peak = self.h_st2.GetBinContent(x_second_peak)
            if y_second_peak > 0.03 * y_hST_max:
                right_peak_candidate = True
                for i in range(x_second_peak, self.h_st2.GetNbinsX()):
                    if self.h_st2.GetBinContent(i + 1) > self.h_st2.GetBinContent(i) \
                            or self.h_st2.GetBinContent(i) < y_hST_max * 0.02:
                        self.window_up2 = i
                        break

        found_second_peak = left_peak_candidate or right_peak_candidate
        if found_second_peak and self.verbose:
            print(f"\n\nFound second peak at {x_second_peak} with {self.h_st2.GetBinContent(x_second_peak)} counts\n\n")
        

        f_STpeak = TF1("f_STpeak2", "gaus", self.window_low2, self.window_up2)
        if not found_second_peak:
            estimated_gaus_amp = self.h_st2.GetBinContent(x_hST_max)
            f_STpeak.SetParameters(estimated_gaus_amp, x_hST_max, 1.0)
            f_STpeak.SetParLimits(0, estimated_gaus_amp * 0.5, estimated_gaus_amp * 1.5)
            f_STpeak.SetParLimits(1, self.window_low2, self.window_up2)
            f_STpeak.SetParLimits(2, 0.05, 5.0)
            f_STpeak.SetNpx(1000)
            fit_option = "QRS"
            if self.verbose:
                print(f"Fitting range: {self.window_low2} - {self.window_up2}")
                fit_option = "RS"
            fit_result = self.h_st2.Fit(f_STpeak, fit_option)
        
        else:
            f_STpeak = TF1("f_STpeak2", "gaus(0)+gaus(3)", self.window_low2, self.window_up2)
            estimated_gaus_amp1 = self.h_st2.GetBinContent(x_hST_max)
            estimated_gaus_amp2 = self.h_st2.GetBinContent(x_second_peak)
            # Print initial parameters
            f_STpeak.SetParameters(estimated_gaus_amp1, x_hST_max, 0.5,
                                   estimated_gaus_amp2, x_second_peak, 0.5)
            f_STpeak.SetParLimits(0, estimated_gaus_amp1 * 0.5, estimated_gaus_amp1 * 1.5)
            f_STpeak.SetParLimits(1, x_hST_max-3, x_hST_max+3)
            f_STpeak.SetParLimits(2, 0.05, 5.0)
            f_STpeak.SetParLimits(3, estimated_gaus_amp2 * 0.5, estimated_gaus_amp2 * 1.5)
            f_STpeak.SetParLimits(4, x_second_peak-3, x_second_peak+3)
            f_STpeak.SetParLimits(5, 0.05, 5.0)
            f_STpeak.SetNpx(2000)
            fit_option = "QRS"
            if self.verbose:
                print(f"Initial parameters: amp1={estimated_gaus_amp1}, mean1={x_hST_max}, sigma1=1.0, amp2={estimated_gaus_amp2}, mean2={x_second_peak}, sigma2=1.0")
                print(f"Fitting range: {self.window_low2} - {self.window_up2} with second peak")
                fit_option = "RS"
            fit_result = self.h_st2.Fit(f_STpeak, fit_option)

        
        #Check fit convergence
        fit_ok = bool(fit_result.Get() and fit_result.Get().IsValid())
        self.f_STpeak2 = f_STpeak

        return (self.h_st2, fit_ok)




    def create_efficiency_histos(self) -> None:
        """
        Create external-trigger (h_total) and self-trigger (h_passed) histograms,
        and the efficiency histogram (he_STEfficiency).
        h_passed is filled with the PE values of the waveforms that have at least one self-trigger
        in the acceptance window [window_low, window_up].
        """
        n_bins   = int((self.nspe_max - self.nspe_min) * 16)
        n_bins_quantized = int(self.nspe_max) - int(self.nspe_min) + 1

        h_total  = TH1D("h_total",  "h_total;#P.E.;Counts",  n_bins, self.nspe_min, self.nspe_max)
        h_passed = TH1D("h_passed", "h_passed;#P.E.;Counts", n_bins, self.nspe_min, self.nspe_max)
        h_total_quantized  = TH1D("h_total_quantized",  "h_total_quantized;#P.E.;Counts",
                                  n_bins_quantized, int(self.nspe_min)-0.5, int(self.nspe_max)+0.5)
        h_passed_quantized = TH1D("h_passed_quantized", "h_passed_quantized;#P.E.;Counts",
                                  n_bins_quantized, int(self.nspe_min)-0.5, int(self.nspe_max)+0.5)

        for pe, trig_times in zip(self.pe, self.trigger_times_list):
            h_total.Fill(pe)
            h_total_quantized.Fill(pe)
            triggers = [t for t in trig_times if t >= self.window_low and t <= self.window_up]
            if len(triggers) > 0:
                h_passed.Fill(pe)
                h_passed_quantized.Fill(pe)

        for i in range(1, h_total.GetNbinsX()):
            if h_total.GetBinContent(i) < 5:
                h_total.SetBinContent(i, 0)
                h_passed.SetBinContent(i, 0)

        self.h_total = h_total
        self.h_passed = h_passed
        he_name = "he_efficiency"
        self.he_STEfficiency = TEfficiency(h_passed, h_total)
        self.he_STEfficiency.SetName(he_name)
        self.he_STEfficiency.SetTitle(he_name)
        self.he_STEfficiency2 = TEfficiency(h_passed, h_total)
        self.he_STEfficiency2.SetName(he_name+"2")
        self.he_STEfficiency2.SetTitle(he_name+"2")
        self.he_STEfficiency_nofit = TEfficiency(h_passed, h_total)
        self.he_STEfficiency_nofit.SetName(he_name+"_nofit")
        self.he_STEfficiency_nofit.SetTitle(he_name+"_nofit")

        self.h_total_quantized = h_total_quantized
        self.h_passed_quantized = h_passed_quantized
        self.he_STEfficiency_quantized = TEfficiency(h_passed_quantized, h_total_quantized)
        self.he_STEfficiency_quantized.SetName(he_name+"_quantized")
        self.he_STEfficiency_quantized.SetTitle(he_name+"_quantized")

        return None


    def get_efficiency_at_fit(self, pe: int) -> tuple[float, float, float, float]:
        """
        Get the efficiency at a given PE value.
        """
        bins_number = self.h_total.GetNbinsX()
        domain = np.array([self.nspe_min, self.nspe_max])
        edges = np.linspace(domain[0],
                            domain[1],
                            num=bins_number + 1,
                            endpoint=True)
        counts, indices = wun.histogram1d(self.pe, bins_number, domain, True)
        
        charge_histo = cls.CalibrationHistogram(
            bins_number,
            edges,
            counts,
            indices,
        )

        output = True
        
        output *= fit_peaks_of_CalibrationHistogram(
                    charge_histo,
                    int(self.nspe_max-1),
                    std_increment_seed_fallback=0.3,
                    prominence=0.1,
                    fit_type=self.fit_type,
        )
        if len(charge_histo.gaussian_fits_parameters["mean"]) > 1:
            self.wafflesSNR = (charge_histo.gaussian_fits_parameters["mean"][1][0] - charge_histo.gaussian_fits_parameters["mean"][0][0]) / \
                                charge_histo.gaussian_fits_parameters["std"][0][0]
        else:
            self.wafflesSNR = np.nan

        if self.led in self.leds_to_plot:
            plot_charge_histo_fit(self, charge_histo)

        if len(charge_histo.gaussian_fits_parameters["mean"]) < pe + 1:
            print(f"Not enough peaks found to get efficiency at {pe} PE.")
            self.wafflesMean = np.nan
            return np.nan, np.nan, np.nan, np.nan

        scale     = charge_histo.gaussian_fits_parameters["scale"][pe][0]
        err_scale = charge_histo.gaussian_fits_parameters["scale"][pe][1]
        mean      = charge_histo.gaussian_fits_parameters["mean"][pe][0]
        err_mean  = charge_histo.gaussian_fits_parameters["mean"][pe][1]
        std       = charge_histo.gaussian_fits_parameters["std"][pe][0]
        err_std   = charge_histo.gaussian_fits_parameters["std"][pe][1]

        bin_centers = np.array([self.h_total.GetBinCenter(i+1) for i in range(self.h_total.GetNbinsX())])
        effs = np.array([self.he_STEfficiency.GetEfficiency(i+1) for i in range(self.h_total.GetNbinsX())])
        sum   = float(0)
        total = float(0)
        for eff, bin_center in zip(effs,bin_centers):
            sum +=  eff * wun.gaussian(bin_center, scale, mean, std)
            total += wun.gaussian(bin_center, scale, mean, std)

        eff = sum / total
        up_boundary  = TEfficiency.ClopperPearson(total, sum, 0.68, True)
        low_boundary = TEfficiency.ClopperPearson(total, sum, 0.68, False)
        err_eff = 0.5 * (up_boundary - low_boundary)

        self.wafflesMean = mean

        return eff, err_eff, up_boundary-eff, eff-low_boundary
    
    def fit_efficiency(self) -> None:
        """
        Fit the efficiency histogram with a sigmoid function.
        """
        self.efficiency_fit_ok = False
        h_total = self.he_STEfficiency.GetTotalHistogram()
        max_efficiency = 0
        for i in range(h_total.GetNbinsX()):
            if self.he_STEfficiency.GetEfficiency(i+1) > max_efficiency:
                max_efficiency = self.he_STEfficiency.GetEfficiency(i+1)

        bin_contents = np.array([h_total.GetBinContent(i+1) for i in range(h_total.GetNbinsX())])
        effs = np.array([self.he_STEfficiency2.GetEfficiency(i+1) for i in range(h_total.GetNbinsX())])
        meaningful_mask = bin_contents > 10
        effs = effs[meaningful_mask]
        max_efficiency = np.max(effs)
        
        if len(effs[effs<0.05]) < 10 \
                or len(effs[effs>0.95*max_efficiency]) < 10:
            print("No low or high efficiency points found.")
            self.he_STEfficiency.SetTitle("Not ok: no low or high efficiency points found.")
            return

        thr_x = find_50efficiency(self.he_STEfficiency)
        if np.isnan(thr_x):
            print("No 50% efficiency point found.")
            self.he_STEfficiency.SetTitle("Not ok: no 50% efficiency point found.")
            return
        
        populated_bins = np.where(bin_contents > 10)
        xlow = h_total.GetBinCenter(int(populated_bins[0][0]))
        xup  = h_total.GetBinCenter(int(populated_bins[-1][-1]))
       
        self.f_sigmoid.SetRange(xlow, xup)
        self.f_sigmoid.SetParameters(thr_x, 0.15, max_efficiency)
        self.f_sigmoid.SetParLimits(0, thr_x-2, thr_x+2)
        self.f_sigmoid.SetParLimits(1, 0.01, 1)
        self.f_sigmoid.SetParLimits(2, 0.7 * max_efficiency, max_efficiency)
        self.f_sigmoid.SetParNames("threshold", "#tau", "Max_{eff}")
        self.f_sigmoid.SetNpx(3000)

        fit_option = "QR"
        if self.verbose:
            fit_option = "R"
        self.he_STEfficiency.Fit(self.f_sigmoid, fit_option)

        tau = self.f_sigmoid.GetParameter(1)
        thr = self.f_sigmoid.GetParameter(0)

        if xlow > thr - 5*tau or xup < thr + 5*tau:
            print(f"The fit is not well constrained: fit range [{xlow}, {xup}], thr={thr}, tau={tau}")
            self.he_STEfficiency.SetTitle("Not ok: fit not well constrained.")
            return

        self.he_STEfficiency.SetTitle("Fit ok")
        self.efficiency_fit_ok = True
        return

    def get_10to90_range(self) -> float:
        """
        Get the 10% to 90% range of the efficiencies points.
        """
        h_total = self.he_STEfficiency.GetTotalHistogram()
        thr_10 = np.nan
        thr_90 = np.nan

        for i in range(2,h_total.GetNbinsX()-2):
            if self.he_STEfficiency.GetEfficiency(i+1) >= 0.1 \
                and self.he_STEfficiency.GetEfficiency(i) < 0.1 \
                and self.he_STEfficiency.GetEfficiency(i-1) < 0.1 \
                and self.he_STEfficiency.GetEfficiency(i+2) > 0.1 \
                and self.he_STEfficiency.GetEfficiency(i+3) > 0.1:
                thr_10 = h_total.GetBinCenter(i+1)
                break
        
        for i in range(2,h_total.GetNbinsX()-2):
            if self.he_STEfficiency.GetEfficiency(i+1) >= 0.9 \
                and self.he_STEfficiency.GetEfficiency(i) < 0.9 \
                and self.he_STEfficiency.GetEfficiency(i-1) < 0.9 \
                and self.he_STEfficiency.GetEfficiency(i+2) > 0.9 \
                and self.he_STEfficiency.GetEfficiency(i+3) > 0.9:
                thr_90 = h_total.GetBinCenter(i+1)
                break

        return thr_90 - thr_10

    def np04_get_10to90_range_fit(self) -> float:
        """
        Get the 10% to 90% range from a custom fit function
        """
        thr_10 = np.nan
        thr_90 = np.nan
        self.fifty = np.nan
        self.twenty = np.nan
        self.chi2ndf = np.nan
        h_total = self.he_STEfficiency2.GetTotalHistogram()
        
        bin_contents = np.array([h_total.GetBinContent(i+1) for i in range(h_total.GetNbinsX())])
        effs = np.array([self.he_STEfficiency2.GetEfficiency(i+1) for i in range(h_total.GetNbinsX())])
        meaningful_mask = bin_contents > 10
        effs = effs[meaningful_mask]
        max_efficiency = np.max(effs)
        
        if len(effs[effs<0.05]) < 10 \
                or len(effs[effs>0.95*max_efficiency]) < 10:
            print("No low or high efficiency points found.")
            self.he_STEfficiency.SetTitle("Not ok: no low or high efficiency points found.")
            return np.nan

        thr_x = find_50efficiency(self.he_STEfficiency)
        if np.isnan(thr_x):
            print("No 50% efficiency point found.")
            return np.nan
       
        populated_bins = np.where(bin_contents > 10)
        xlow = h_total.GetBinCenter(int(populated_bins[0][0]))
        xup  = h_total.GetBinCenter(int(populated_bins[-1][-1]))
        f_custom = TF1("f_sigmoidss", "[2]/(1+exp(([0]-x)/[1]))+[5]/(1+exp(([0]+[3]-x)/[4]))", xlow, xup)
        f_custom.SetParNames("x0", "#tau", "A", "#Delta", "#tau_{2}", "A_{2}")
        f_custom.SetParLimits(0, max([xlow,thr_x-0.6]), thr_x+0.6)
        f_custom.SetParLimits(1, 0.03, 0.3)
        f_custom.SetParLimits(2, 0.1, 1)
        f_custom.SetParLimits(3, 0.1, 6)
        f_custom.SetParLimits(4, 0.03, 0.6)
        f_custom.SetParLimits(5, 0.01, 1)

        par_init_dict_list = []
        par_init_dict = {"x0": None, "tau": None, "A": None, "Delta": None, "tau2": None, "A2": None, "x0_low": None, "x0_up": None}
        if self.efficiency_fit_ok:
            x0  = self.f_sigmoid.GetParameter(0)
            tau = self.f_sigmoid.GetParameter(1)
            eff_x02tau, _, _, = get_efficiency_at(self.he_STEfficiency, x0+2*tau)
            gap = 1-eff_x02tau
            a1 = self.f_sigmoid.GetParameter(2)-gap
            par_init_dict = {"x0": x0, "tau": tau, "A": a1, "Delta": 3, "tau2": 0.08, "A2": gap, "x0_low": x0-0.2, "x0_up": x0+0.2}
            par_init_dict_list.append(par_init_dict)
            par_init_dict = {"x0": x0, "tau": tau, "A": a1, "Delta": 4, "tau2": 0.3, "A2": gap, "x0_low": x0-0.2, "x0_up": x0+0.2}
            par_init_dict_list.append(par_init_dict)
            eff_x0, _, _, = get_efficiency_at(self.he_STEfficiency, x0)
            par_init_dict = {"x0": x0-tau, "tau": tau/2., "A": eff_x0, "Delta": 2*tau, "tau2": 0.08, "A2": 1-eff_x0, "x0_low": x0-2*tau, "x0_up": x0+2*tau}
            par_init_dict_list.append(par_init_dict)
            par_init_dict = {"x0": x0-2*tau, "tau": tau/2., "A": eff_x0, "Delta": 4*tau, "tau2": 0.08, "A2": 1-eff_x0, "x0_low": x0-4*tau, "x0_up": x0}
            par_init_dict_list.append(par_init_dict)
            par_init_dict = {"x0": x0-0.5, "tau": tau/2., "A": eff_x0, "Delta": 1., "tau2": 0.08, "A2": 1-eff_x0, "x0_low": x0-1., "x0_up": x0}
            par_init_dict_list.append(par_init_dict)
        else:
            par_init_dict = {"x0": thr_x, "tau": 0.08, "A": 0.9, "Delta": 3, "tau2": 0.08, "A2": 0.1, "x0_low": max([xlow,thr_x-0.6]), "x0_up": thr_x+0.6}
            par_init_dict_list.append(par_init_dict)
            par_init_dict = {"x0": thr_x-0.5, "tau": 0.08, "A": 0.7, "Delta": 1, "tau2": 0.08, "A2": 0.3, "x0_low": max([xlow,thr_x-0.6]), "x0_up": thr_x+0.6}
            par_init_dict_list.append(par_init_dict)
       
        fit_option = "QRS"
        if self.verbose:
            fit_option = "RS"
        f_custom.SetNpx(3000)
        
        chi2_list = []
        for par_init_dict in par_init_dict_list:
            f_custom.SetParameter(0, par_init_dict["x0"])
            f_custom.SetParameter(1, par_init_dict["tau"])
            f_custom.SetParameter(2, par_init_dict["A"])
            f_custom.SetParameter(3, par_init_dict["Delta"])
            f_custom.SetParameter(4, par_init_dict["tau2"])
            f_custom.SetParameter(5, par_init_dict["A2"])
            f_custom.SetParLimits(0, par_init_dict["x0_low"], par_init_dict["x0_up"])
            fit_result = TFitResultPtr(self.he_STEfficiency2.Fit(f_custom, fit_option))
            fit_status = int(fit_result)
            if fit_status == 0:
                chi2 = f_custom.GetChisquare() / f_custom.GetNDF()
                chi2_list.append((chi2, par_init_dict.copy()))

        if len(chi2_list) == 0:
            par_init_dict = {"x0": thr_x, "tau": 0.08, "A": 0.9, "Delta": 3, "tau2": 0.08, "A2": 0.1, "x0_low": max([xlow,thr_x-0.6]), "x0_up": thr_x+0.6}
            par_init_dict_list.append(par_init_dict)
            chi2 = f_custom.GetChisquare() / f_custom.GetNDF()
            chi2_list.append((chi2, par_init_dict.copy()))
        else:
            print("\n\n\nSuccessful fits with different initial parameters:\n\n\n")
        chi2_list.sort(key=lambda x: x[0])
        best_pars = chi2_list[0][1]
        f_custom.SetParameter(0, best_pars["x0"])
        f_custom.SetParameter(1, best_pars["tau"])
        f_custom.SetParameter(2, best_pars["A"])
        f_custom.SetParameter(3, best_pars["Delta"])
        f_custom.SetParameter(4, best_pars["tau2"])
        f_custom.SetParameter(5, best_pars["A2"])
        f_custom.SetParLimits(0, best_pars["x0_low"], best_pars["x0_up"])
        self.he_STEfficiency2.Fit(f_custom, fit_option)
        self.chi2ndf = f_custom.GetChisquare() / f_custom.GetNDF()

        if f_custom.GetParameter(2)+f_custom.GetParameter(5) < np.max(effs)-0.1:
            f_custom.SetParameter(2, np.max(effs))
            f_custom.SetParameter(5, 0.01)
            print("\n\n\nRefitting due to low max efficiency")
            self.he_STEfficiency2.Fit(f_custom, fit_option)
            self.chi2ndf = f_custom.GetChisquare() / f_custom.GetNDF()

        self.f_sigmoid2 = f_custom

        x_fifty = f_custom.GetX(0.5, xlow, xup)
        if not np.isnan(x_fifty):
            self.fifty = x_fifty
        x_twenty = f_custom.GetX(0.2, xlow, xup)
        if not np.isnan(x_twenty):
            self.twenty = x_twenty

        # Find 10% and 90%
        for i in range(h_total.GetNbinsX()-2):
            if f_custom.Eval(h_total.GetBinCenter(i+1)) >= 0.1:
                thr_10 = h_total.GetBinCenter(i+1)
                break

        for i in range(h_total.GetNbinsX()-2):
            if f_custom.Eval(h_total.GetBinCenter(i+1)) >= 0.9:
                thr_90 = h_total.GetBinCenter(i+1)
                break

        return thr_90 - thr_10


    def st_selector(self, wf) -> bool:
        """
        Selection criteria for self-triggering events:
        - Baseline: max and min ADC values in the pre-trigger region
          must be within 3*bsl_rms.
        Parameters:
        wf: Waveform
            The waveform to be evaluated. Note that wf.adcs_float is used.
        Returns:
        bool
            True if the waveform passes the selection, False otherwise.
        """
        max_pre = np.max(wf.adcs_float[:self.prep])
        min_pre = np.min(wf.adcs_float[:self.prep])

        if max_pre > 3 * self.bsl_rms or min_pre < -3 * self.bsl_rms:
            return False
        return True


    def select_events(self) -> None:
        """
        Select events based on the self-triggering criteria and store the
        number of photoelectrons and trigger-times for the selected events.
        In addition, compute the nspe_min and nspe_max values that are used
        to create the efficiency histograms.
        """
        self.pe = np.array(self.pe[self.selection])
        self.trigger_times_list = [self.trigger_times_list[i] for i in range(len(self.trigger_times_list)) if self.selection[i]]
        
        sorted_nspes = self.pe.copy()
        sorted_nspes.sort()
        self.nspe_min = sorted_nspes[int(0.001 * len(sorted_nspes))]
        self.nspe_max = sorted_nspes[int(0.96 * len(sorted_nspes))]

        return None

    def from_raw_to_metadata(self):
        """
        From raw waveforms to metadata.
        Prepare metadata arrays ready to be stored in a ROOT TTree. Each entry corresponds to a waveform.
        """
        # Exclude waveform with different timestamps
        timestamp_array = np.array([wf.timestamp for wf in self.wfs_sipm])
        st_timestamp_array = np.array([wf.timestamp for wf in self.wfs_st])
        print(f"\n\nNumber of waveforms before timestamp selection: {len(self.wfs_sipm)}")
        selection_array = timestamp_array == st_timestamp_array
        self.wfs_sipm = np.array(self.wfs_sipm[selection_array])
        self.wfs_st = np.array(self.wfs_st[selection_array])
        print(f"Number of waveforms after timestamp selection: {len(self.wfs_sipm)}\n\n")

        # prepare arrays
        self.selection = np.array([self.st_selector(wf) for wf in self.wfs_sipm], dtype=bool)
        spe_norm = 1./self.spe_charge
        self.pe = np.array([wf.adcs_float[self.int_low:self.int_up+1].sum() * spe_norm for wf in self.wfs_sipm])
        trigger_times = []
        for wf_st in self.wfs_st:
            st_positions = np.flatnonzero(wf_st.adcs)
            trigger_times.append(st_positions.astype(np.int16))

        self.trigger_times = trigger_times

        return

    def trigger_distr_per_nspe(self, calibrated_threshold: float) -> dict:
        """
        Create a dictionary of self-trigger time-distribution histograms,
        one for each integer number of photoelectrons (nspe).
        Returns:
        dict_hists: dict
            Dictionary where keys are integer nspe values and values are
            corresponding TH1D histograms of self-trigger time-distributions.
        """
        npe_arr = self.pe.copy()
        npe_arr.sort()
        if len(npe_arr) == 0:
           return {}
        npe_min = int(max(npe_arr[int(0.005 * len(npe_arr))], calibrated_threshold))
        npe_max = int(npe_arr[int(0.995 * len(npe_arr))]+1)

        dict_hists = {}

        if npe_min > npe_max:
            return dict_hists

        for pe in range(npe_min, npe_max):
            dict_hists[pe] = TH1D(f"h_st_{pe}", f"h_st_{pe};Ticks;Counts", self.nticks, -0.5, self.nticks-0.5)

        for npe, trigger_times in zip(self.pe, self.trigger_times_list):
            if npe < npe_min or npe > npe_max:
                continue
            pe = int(npe+0.5)
            if pe not in dict_hists:
                continue
            for trig_time in trigger_times:
                dict_hists[pe].Fill(trig_time)
        
        return dict_hists
