from math import sqrt
import waffles
import numpy as np
from waffles.np04_analysis.time_resolution.utils import create_float_waveforms, sub_baseline_to_wfs
from hist import Hist
from iminuit import Minuit, cost
from ROOT import TH1D, TEfficiency, TF1


def allow_channel_wfs(waveform: waffles.Waveform, channel: int) -> bool:
    return waveform.endpoint == (channel//100) and waveform.channel == (channel%100)


def gaussian(x, amp, mean, sigma):
    """Gaussian function."""
    return amp * np.exp(-0.5 * (((x - mean) / sigma) ** 2) )

def persistence_plot(wfs):
    wvfs = np.array([wf.adcs_float for wf in wfs])
    times = np.linspace(0, len(wfs[0].adcs), len(wfs[0].adcs), endpoint=False)
    times = np.tile(times, (len(wfs), 1))
    nbinsx = len(wfs[0].adcs)
    h, yedges, xedges = np.histogram2d(wvfs.flatten(), times.flatten(),
                                       bins=(200, nbinsx),
                                       range=[[-30,50], [0, nbinsx]])
    h[h==0] = np.nan
    return h, xedges, yedges

def get_max_accuracy(h_total: TH1D, h_passed: TH1D):
    """
    Calculate the accuracy of the self-triggering based on the total and passed histograms.
    Def: accuracy = true_positive + true_negative / (true_positive + true_negative + false_positive + false_negative)
    Returns the accuracy and the threshold at which it is achieved.
    NOTE: in computing "acc" we omitted the terms that cancel out.
    """
    accuracy = -1e6
    accuracy_thr = -1e6
    nbins = h_total.GetNbinsX()
    itera = 0
    for i in range(3,nbins-1):
        if h_total.GetBinContent(i+1) == 0:
            continue
        acc = ( h_passed.Integral(i,nbins) + h_total.Integral(1,i-1)-h_passed.Integral(1,i-1) ) / \
                ( h_total.Integral(1,nbins) )
        if acc > accuracy:
            accuracy = acc
            accuracy_thr = h_total.GetBinCenter(i+1)
            itera = i

    print(f"Max accuracy: {accuracy} at threshold {accuracy_thr} (bin {itera})")
    return accuracy, accuracy_thr
    
def get_true_positive_rate(h_total: TH1D, h_passed: TH1D, threshold: float):
    """

    """
    thr_bin = h_total.FindBin(threshold)
    true_positive_rate = h_passed.Integral(thr_bin, h_passed.GetNbinsX()) / h_total.Integral(thr_bin, h_total.GetNbinsX())
    
    return true_positive_rate

def get_false_positive_rate(h_total: TH1D, h_passed: TH1D, threshold: float):
    """

    """
    thr_bin = h_total.FindBin(threshold)
    false_positive_rate = h_passed.Integral(1, thr_bin) / h_total.Integral(thr_bin, h_total.GetNbinsX())

    return false_positive_rate

def get_accuracy(h_total: TH1D, h_passed: TH1D, threshold: float):
    """

    """
    thr_bin = h_total.FindBin(threshold)    
    nbins = h_total.GetNbinsX()
    acc = ( h_passed.Integral(thr_bin,nbins) + h_total.Integral(1,thr_bin)-h_passed.Integral(1,thr_bin) ) / \
                ( h_total.Integral(1,nbins) )

    return acc

    


class SelfTrigger:
    def __init__(self,
                 ch_sipm: int,
                 ch_st: int,
                 wf_set: waffles.WaveformSet,
                 prepulse_ticks: int,
                 int_low: int,
                 int_up: int,
                 bsl_rms: float,
                 spe_charge: float,
                 spe_ampl: float,
                 snr: float,
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
        
        self.h_low = -1.5
        self.h_up = 10
        self.h_bins = 140
        self.bkg_trg_win_low = 800
        self.trigger_rate = 0.0

        self.window_low = 0
        self.window_up = 0
        # self.st_selection = np.zeros(len(self.wfset_st.waveforms), dtype=bool)


    
    def create_wfs(self) -> None:
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


    def create_self_trigger_distribution(self, name="") -> TH1D:
        """
        Take the st waveforms indec where adcs==1
        """
        if name == "":
            name = "h_selftrigger"
        h_selftrigger = TH1D(name, f"{name};Ticks;Counts", 1024, -0.5, 1023.5)

        for wf in self.wfs_st:
            st_positions = np.flatnonzero(wf.adcs)
            for st in st_positions:
                h_selftrigger.Fill(st)

        n_bkg_triggers = h_selftrigger.Integral(self.bkg_trg_win_low, h_selftrigger.GetNbinsX())
        self.bkg_trg_rate = (n_bkg_triggers * 10**9) / (len(self.wfs_st) * (len(self.wfs_st[0].adcs)-self.bkg_trg_win_low) * 16.)
        n_bkg_triggers_preLED = h_selftrigger.Integral(1, self.int_low)
        self.bkg_trg_rate_preLED = (n_bkg_triggers_preLED * 10**9) / (len(self.wfs_st) * self.int_low * 16.)

       
        self.h_st = h_selftrigger
        return h_selftrigger


    def fit_self_trigger_distribution(self) -> TH1D:
        """
        Fit the self-trigger distribution.
        """
        x_hST_max = self.h_st.GetMaximumBin()
        y_hST_max = self.h_st.GetBinContent(x_hST_max)

        f_constant = TF1("f_constant", "pol0", 2, self.prep)
        f_constant.SetParameter(0, self.h_st.GetBinContent(5))
        f_constant.SetNpx(1000)
        self.h_st.Fit(f_constant, "R")
        
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
        while i < self.wfs_st[0].adcs.size - 1:
            afttrg = i
            if self.h_st.GetBinContent(i + 1) < thr_counts:
                break
            i += 1 

        f_STpeak = TF1("f_STpeak", f"{f_constant.GetParameter(0)}+gaus", pretrg, afttrg)
        estimated_gaus_amp = y_hST_max - f_constant.GetParameter(0)
        f_STpeak.SetParameters(estimated_gaus_amp, x_hST_max, 1.0)
        f_STpeak.SetParLimits(0, estimated_gaus_amp * 0.5, estimated_gaus_amp * 1.5)
        f_STpeak.SetParLimits(1, pretrg, afttrg)
        f_STpeak.SetParLimits(2, 0.05, 5.0)
        f_STpeak.SetNpx(1000)
        self.h_st.Fit(f_STpeak, "R")

        self.f_STpeak = f_STpeak
        # if (self.window_low==0): self.window_low = int(pretrg)
        # if (self.window_up==0):  self.window_up  = int(afttrg)
        self.window_low = int(pretrg)
        self.window_up  = int(afttrg)
        return self.h_st


    def create_efficiency_histos(self, he_name: str) -> tuple:
        """
        Create efficiency histograms for the self-triggering.
        """
        htotal   = Hist.new.Reg(self.h_bins, self.h_low, self.h_up, label="#P.E.").Double()
        hpassed  = Hist.new.Reg(self.h_bins, self.h_low, self.h_up, label="#P.E.").Double()
        spe_norm = 1./self.spe_charge

        nspes = np.array([wf.adcs_float[self.int_low:self.int_up+1].sum() * spe_norm for wf in self.wfs_sipm])
        sorted_nspes = nspes.copy()
        sorted_nspes.sort()

        nspe_min = sorted_nspes[int(0.001 * len(sorted_nspes))]
        nspe_max = sorted_nspes[int(0.96 * len(sorted_nspes))]

        h_total  = TH1D("h_total",  "h_total;#P.E.;Counts",  self.h_bins, nspe_min, nspe_max)
        h_passed = TH1D("h_passed", "h_passed;#P.E.;Counts", self.h_bins, nspe_min, nspe_max)

        for wf_sipm, wf_st, nspe in zip(self.wfs_sipm, self.wfs_st, nspes):
            if (wf_sipm.timestamp != wf_st.timestamp):
                print(f"Timestamp mismatch: SiPM {wf_sipm.timestamp}, ST {wf_st.timestamp}")
                continue
            htotal.fill(nspe)
            h_total.Fill(nspe)
            triggers = np.flatnonzero(wf_st.adcs[self.window_low:self.window_up+1])
            if len(triggers) > 0:
                hpassed.fill(nspe)
                h_passed.Fill(nspe)

        for i in range(1, h_total.GetNbinsX()):
            if h_total.GetBinContent(i) < 5:
                h_total.SetBinContent(i, 0)
                h_passed.SetBinContent(i, 0)

        self.h_total = h_total
        self.h_passed = h_passed
        self.he_STEfficiency = TEfficiency(h_passed, h_total)
        self.he_STEfficiency.SetName(he_name)
        self.he_STEfficiency.SetTitle(he_name)

        main_ax_artists, sublot_ax_arists = hpassed.plot_ratio(
            htotal,
            rp_num_label="passed",
            rp_denom_label="total",
            rp_uncert_draw_type="line",
            rp_uncertainty_type="efficiency",
        )
        return main_ax_artists, sublot_ax_arists


    def fit_efficiency(self, f_sigmoid) -> TEfficiency:
        """
        Fit the efficiency histogram.
        """
        h_total = self.he_STEfficiency.GetTotalHistogram()
        max_efficiency = 0
        for i in range(h_total.GetNbinsX()):
            if self.he_STEfficiency.GetEfficiency(i+1) > max_efficiency:
                max_efficiency = self.he_STEfficiency.GetEfficiency(i+1)

        half_max_efficiency = 0.5 * max_efficiency
        effs = np.array([self.he_STEfficiency.GetEfficiency(i+1) for i in range(h_total.GetNbinsX())])
        thr_x = 0
        for i in range(len(effs)-4):
            effs_short = effs[i:i+5]
            if effs_short[-1] < half_max_efficiency:
                continue
            grad = np.gradient(effs_short)
            if np.all(grad > 0):
                thr_x = h_total.GetBinCenter(i+3)
                break
        
        print(f"\n--------------------\nThreshold x: {thr_x}, Max efficiency: {max_efficiency}\n--------------------\n")
        # f_sigmoid.SetParameters(thr_x, 0.15, max_efficiency)
        # f_sigmoid.SetParLimits(0, thr_x-2, thr_x+2)
        # f_sigmoid.SetParLimits(1, 0.01, 2)
        # f_sigmoid.SetParLimits(2, 0.7 * max_efficiency, max_efficiency)
        # f_sigmoid.SetParNames("threshold", "#tau", "Max_{eff}")
        # f_sigmoid.SetNpx(1000)
        #
        # self.he_STEfficiency.Fit(f_sigmoid, "R")

        return self.he_STEfficiency


    def st_selector(self, wf) -> bool:
        """
        
        """
        max_pre = np.max(wf.adcs_float[:self.prep])
        min_pre = np.min(wf.adcs_float[:self.prep])

        if max_pre > 3 * self.bsl_rms or min_pre < -3 * self.bsl_rms:
            return False
        return True


    def select_waveforms(self) -> None:
        """
        Select waveforms based on the self-triggering criteria.
        """
        self.st_selection = np.array([self.st_selector(wf) for wf in self.wfs_sipm], dtype=bool)
        self.wfs_sipm = np.array(self.wfs_sipm[self.st_selection])
        self.wfs_st = np.array(self.wfs_st[self.st_selection])


    def outlier_selector(self, wf_sipm, wf_st, spe_norm, outlier_threshold) -> bool:
        """

        """
        nspe = wf_sipm.adcs_float[self.int_low:self.int_up].sum() * spe_norm
        if nspe < outlier_threshold:
            return False
        # elif (len(np.flatnonzero(wf_st.adcs[self.window_low:self.window_up+1])) == 0):
        elif (len(np.flatnonzero(wf_st.adcs)) == 0):
            return True
        else:
            return False


    def select_outliers(self, f_sigmoid) -> None:
        """

        """
        self.create_wfs()
        self.select_waveforms()
        spe_norm = 1./self.spe_charge
        # outlier_threshold = f_sigmoid.GetParameter(0) + 4 * f_sigmoid.GetParameter(1)
        outlier_threshold = 3.
        print(f"Outlier threshold: {outlier_threshold}")
        self.outlier_selection = np.array([self.outlier_selector(wf_sipm, wf_st, spe_norm, outlier_threshold)
                                           for wf_sipm, wf_st in zip(self.wfs_sipm, self.wfs_st)], dtype=bool)
        self.wfs_sipm = np.array(self.wfs_sipm[self.outlier_selection])
        self.wfs_st = np.array(self.wfs_st[self.outlier_selection])

    def get_trigger_rate(self) -> float:
        """
        Calculate the trigger rate based on the self-triggering waveforms.
        """
        n_triggers = 0
        n_wfs = len(self.wfs_st)
        win_low = 800

        for wf in self.wfs_st:
            n_triggers += np.sum(wf.adcs[win_low+1:])

        ntrig_hist = self.h_st.Integral(win_low, self.h_st.GetNbinsX())

        print(f"Number of triggers: {n_triggers} or {ntrig_hist}")

        full_trigger_rate = (n_triggers * 10**9) / (n_wfs * len(self.wfs_st[0].adcs) * 16.)
        return full_trigger_rate

    def trigger_distr_per_nspe(self) -> dict:
        """

        """
        for wf in self.wfs_sipm:
            wf.nspe = wf.adcs_float[self.int_low:self.int_up].sum() * (1./self.spe_charge)

        nspe_arr = np.array([wf.nspe for wf in self.wfs_sipm])
        nspe_arr.sort()
        nspe_min = nspe_arr[int(0.005 * len(nspe_arr))]
        nspe_max = nspe_arr[int(0.995 * len(nspe_arr))]

        dict_hists = {}

        for spe in range(int(nspe_min), int(nspe_max)+1):
            dict_hists[spe] = TH1D(f"h_st_{spe}", f"h_st_{spe};Ticks;Counts", 1024, -0.5, 1023.5)

        for wf_sipm, wf_st in zip(self.wfs_sipm, self.wfs_st):
            if wf_sipm.nspe < nspe_min or wf_sipm.nspe > nspe_max:
                continue
            spe = int(wf_sipm.nspe+0.5)
            if spe not in dict_hists:
                continue
            st_arr = np.flatnonzero(wf_st.adcs)
            for st in st_arr:
                dict_hists[spe].Fill(st)
        
        return dict_hists
