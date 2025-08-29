import os
import pandas as pd
import yaml
from pandas.core import base
import ROOT # The best!! :D
from ROOT import TFile, TH2F, TGraph, TTree

from waffles.np02_utils.AutoMap import dict_module_to_uniqch, dict_uniqch_to_module, ordered_channels_cathode, ordered_channels_membrane
from waffles.np02_utils.load_utils import open_processed
from waffles.np04_analysis.time_resolution.time_resolution import TimeResolution
from waffles.np04_analysis.time_resolution.utils import *

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the configs/time_resolution_config.yml file
    with open("./configs/time_resolution_configs.yml", 'r') as config_stream:
        config_variables = yaml.safe_load(config_stream)

    data_folder = config_variables.get("data_folder")
    ana_folder = config_variables.get("ana_folder")
    raw_ana_folder = ana_folder+config_variables.get("raw_ana_folder")
    
    # Setup variables according to the params.yml file
    with open("./params.yml", 'r') as params_stream:
        params_variables = yaml.safe_load(params_stream)

    runs = params_variables.get("runs")
    noise_results_file = params_variables.get("noise_results_file")
    noise_df = pd.read_csv(noise_results_file, sep=",")
    # print(noise_df.head(5))

    calibration_file = params_variables.get("calibration_file")
    calibration_df = pd.read_csv(calibration_file, sep=",")
    # print(calibration_df.head(5))


    channels = params_variables.get("channels")    
    dettype = params_variables.get("dettype")
    nwaveforms = params_variables.get("nwaveforms")
    endpoint = 106 if dettype=="cathode" else 107
    listofchannels = ordered_channels_cathode if dettype=="cathode" else ordered_channels_membrane

    global_prepulse_ticks = params_variables.get("prepulse_ticks")
    global_int_low = params_variables.get("int_low")
    global_int_up = params_variables.get("int_up")
    global_postpulse_ticks = params_variables.get("postpulse_ticks")
    min_pes = params_variables.get("min_pes")
    global_baseline_rms = params_variables.get("baseline_rms")
    methods = params_variables.get("methods")
    relative_thrs = params_variables.get("relative_thrs")
    filt_levels = params_variables.get("filt_levels")
    h2_nbins = params_variables.get("h2_nbins")
    stat_lost = params_variables.get("stat_lost")

    min_selec_wfs = params_variables.get("min_selec_wfs", 500)

    invert = False # Hard coded because all channels are positive 
    # PS.: in some runs inverters on daphne were off... :/

    # --- EXTRA VARIABLES -------------------------------------------
    min_min_pe = []
    max_max_pe = []
    min_min_t0 = []
    max_max_t0 = []
    hp_t0_pes = []
    os.makedirs(ana_folder, exist_ok=True)
    os.makedirs(raw_ana_folder, exist_ok=True)
    
    # --- LOOP OVER RUNS --------------------------------------------
    for run in runs:
        print("Reading run ", run)
        wfset_run = open_processed(run, dettype, data_folder, nwaveforms=nwaveforms, verbose=True, mergefiles=True)
            
        timeRes = TimeResolution(wf_set=wfset_run) 
           
        # --- LOOP OVER CHANNELS ------------------------------------------
        if (channels == []):
            print("No channels specified. Using all channels.")
            channels = ordered_channels_cathode if dettype=="cathode" else ordered_channels_membrane
        for ch in channels:
            if ch not in listofchannels:
                print(f"Channel {ch} is not valid...")
                continue
            if ch not in wfset_run.available_channels[run][endpoint]:
                print(f"Channel {ch} not in run {run}")
                print(f"Available channels: {wfset_run.available_channels[run]}")
                continue

            epchcombo = (calibration_df['channel'] == ch) & (calibration_df['endpoint'] == endpoint)

            if global_prepulse_ticks != 0:
                prepulse_ticks = global_prepulse_ticks
            else:
                prepulse_ticks = int(calibration_df.loc[epchcombo, 'prepulse_ticks'].values[0])

            if global_int_low != 0:
                int_low = global_int_low
            else:
                int_low = int(calibration_df.loc[epchcombo, 'int_low'].values[0])

            if global_int_up != 0:
                int_up = global_int_up
            else:
                int_up = int(calibration_df.loc[epchcombo, 'int_up'].values[0])

            if global_postpulse_ticks != 0:
                postpulse_ticks = global_postpulse_ticks
            else:
                postpulse_ticks = int(calibration_df.loc[epchcombo, 'postpulse_ticks'].values[0])

                
            spe_charge = float(calibration_df.loc[epchcombo, 'Gain'].values[0])
            spe_ampl = float(calibration_df.loc[epchcombo, 'SpeAmpl'].values[0])
            baseline_rms = float(noise_df.loc[(noise_df['channel'] == ch) & (noise_df['endpoint'] == endpoint), 'rms'].values[0])

           
            try:
                timeRes.set_analysis_parameters(ch=endpoint*100 + ch, prepulse_ticks=prepulse_ticks,
                                                postpulse_ticks=postpulse_ticks, int_low=int_low,
                                                int_up=int_up, spe_charge=spe_charge,
                                                spe_ampl=spe_ampl, min_pes=min_pes,
                                                baseline_rms=baseline_rms, invert=invert)
            except ValueError as e:
                print(f"Error: {e}")
                continue

            timeRes.create_wfs()
            timeRes.select_time_resolution_wfs()
            print(timeRes.debug_counter)
   
            out_root_file_name = raw_ana_folder+f"Run_{run}_Ep_{endpoint}_Ch_{ch}_time_resolution.root"
            root_file = TFile(out_root_file_name, "RECREATE")
        
            if timeRes.n_select_wfs > min_selec_wfs:
                n_pwfs = min(9000, timeRes.n_select_wfs)
                n_selected = [ wf.time_resolution_selection for wf in timeRes.wfs].count(True)
                all_wfs = np.array([wf.adcs_float for wf in timeRes.wfs if wf.time_resolution_selection])[:n_pwfs].flatten()
                all_tikcs = np.array([np.arange(len(wf.adcs_float)) for wf in timeRes.wfs if wf.time_resolution_selection])[:n_pwfs].flatten()

                histmin = float(np.quantile(all_wfs,0.02))
                histmax = float(np.quantile(all_wfs,0.98))
                counts, xedges, yedges = np.histogram2d(all_tikcs, all_wfs, bins=(len(timeRes.wfs[0].adcs_float),h2_nbins),
                                                        range=[[0, len(timeRes.wfs[0].adcs_float)],
                                                               [histmin, histmax]])
                                                        

                # Histogram 2D of t0 vs pe
                h2_persistence = TH2F("persistence", ";time [ticks]; Amplitude [ADC]",
                                len(timeRes.wfs[0].adcs_float), 0, len(timeRes.wfs[0].adcs_float),
                                h2_nbins, yedges[0], yedges[-1])

                for i in range(len(timeRes.wfs[0].adcs_float)):
                    for j in range(h2_nbins):
                        h2_persistence.SetBinContent(i+1, j+1, counts[i, j])
                
                h2_persistence.Write()

                print("Channel ", ch)
                for method in methods:
                        if method == "denoise":
                            loop_filt_levels = filt_levels
                        else:
                            loop_filt_levels = [0]

                        for relative_thr in relative_thrs:

                            for filt_level in loop_filt_levels:
                                rel_thr = str(relative_thr).replace(".", "p")
                                root_file.mkdir(f"{method}_filt_{filt_level}_thr_{rel_thr}")
                                root_file.cd(f"{method}_filt_{filt_level}_thr_{rel_thr}")

                                
                                if (method == "denoise" and filt_level > 0):
                                    timeRes.create_denoised_wfs(filt_level=filt_level)
                               
                                t0s, pes, tss = timeRes.set_wfs_t0(method=method, relative_thr=relative_thr)
                                
                                t = TTree("time_resolution", "time_resolution")
                                t0 = np.zeros(1, dtype=np.float64)
                                pe = np.zeros(1, dtype=np.float64)
                                ts = np.zeros(1, dtype=np.float64)
                                t.Branch("t0", t0, "t0/D")
                                t.Branch("pe", pe, "pe/D")
                                t.Branch("timestamp", ts, "timestamp/D")

                                for i in range(len(t0s)):
                                    t0[0] = t0s[i]
                                    pe[0] = pes[i]
                                    ts[0] = tss[i]
                                    t.Fill()

                                t.Write()
            else:
                print(f"Not enough selected waveforms for channel {ch} in run {run}. Needed > {min_selec_wfs}, got {timeRes.n_select_wfs}.")
    
            root_file.Close()
        del timeRes, wfset_run
