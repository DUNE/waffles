import waffles.np04_analysis.self_trigger.self_trigger as self_trigger
from waffles.np04_utils.utils import get_np04_channel_mapping
import waffles.input_output.hdf5_structured as reader
from ROOT import TFile, TTree
import ROOT
from array import array
import os
import pandas as pd
import yaml


# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":

    # --- SETUP -----------------------------------------------------
    with open("steering.yml", 'r') as stream:
        steering_config = yaml.safe_load(stream)
    params_file_name = steering_config.get("params_file", "params.yml")
    ana_folder  = steering_config.get("ana_folder")
    if not os.path.exists(ana_folder):
        os.makedirs(ana_folder)
    metadata_folder = ana_folder + "metadata/"
    if not os.path.exists(metadata_folder):
        os.makedirs(metadata_folder)

    with open(params_file_name, 'r') as stream:
        user_config = yaml.safe_load(stream)

    run_info_file = user_config.get("run_info_file")
    calibration_file = user_config.get("calibration_file")
    file_folder = user_config.get("file_folder")
    SiPM_channel = user_config.get("SiPM_channel")
    save_pngs = user_config.get("save_pngs", True)
    runs = user_config.get("runs", [])
    files_in_folder = [file_folder+f for f in os.listdir(file_folder) if f.endswith("structured.hdf5")]

    df_runs = pd.read_csv(run_info_file, sep=",") 
    out_df_rows = []

    calibration_df = pd.read_csv(calibration_file, sep=",")
    print(calibration_df.head(5))
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

    for run in runs:
        # print the run number and the run index in runs
        print(f"Processing run {run} ({runs.index(run)+1}/{len(runs)})")

        filename = [f for f in files_in_folder if str(run)+"_ChSiPM_"+str(SiPM_channel) in f]
        if len(filename) == 0:
            print(f"No file found for run {run} and SiPM channel {SiPM_channel}")
            continue
        filename = filename[0]
        print("Reading file ", filename)

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
        st.create_wfs()
        st.from_raw_to_metadata()

        # Prepare output
        out_root_file_name = metadata_folder + f"Run_{run}_ChSiPM_{SiPM_channel}_ChST_{ch_st}.root"
        out_root_file_meta = TFile(out_root_file_name, "RECREATE")
        out_root_file_meta.cd()
        selftrigger_tree = TTree("SelfTriggerTree", "SelfTriggerTree")
        
        selection = array('b', [0])
        pe = array('f', [0.])
        trigger_times = ROOT.std.vector('int')()
        selftrigger_tree.Branch("selection", selection, "selection/O")
        selftrigger_tree.Branch("pe", pe, "pe/F")
        selftrigger_tree.Branch("trigger_times", trigger_times)

        for sel, npe, trg_time in zip(st.selection, st.pe, st.trigger_times):
            selection[0] = sel
            pe[0] = npe
            trigger_times.clear()
            for t in trg_time:
                trigger_times.push_back(int(t))
            selftrigger_tree.Fill()

        out_root_file_meta.Write()
        out_root_file_meta.Close()
