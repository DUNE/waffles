# --- IMPORTS -------------------------------------------------------
import yaml
import pandas as pd
import os
from ROOT import TFileMerger


# Merge the TTree of each threshold into a single ROOT file per channel
# Loop over the metadata files of each run of a given threshold and merge them

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    
    # --- SETUP -----------------------------------------------------
    with open("./steering.yml", 'r') as stream:
        steering_config = yaml.safe_load(stream)
    params_file_name = steering_config.get("params_file", "params.yml")
    run_info_file    = steering_config.get("run_info_file")
    ana_folder       = steering_config.get("ana_folder")
    metadata_folder  = ana_folder + "metadata/"
    df_runs          = pd.read_csv(run_info_file, sep=",") 
    thresholds       = df_runs['Threshold'].unique()

    with open(params_file_name, 'r') as stream:
        user_config = yaml.safe_load(stream)
    SiPM_channel = user_config.get("SiPM_channel")
    files_in_folder = [metadata_folder+f for f in os.listdir(metadata_folder) if f.endswith(".root")]

    for threshold in thresholds:
        run_with_threshold = df_runs.loc[df_runs['Threshold'] == str(threshold), 'Run'].values
        
        run_files = [f for f in files_in_folder if any(str(run) in f for run in run_with_threshold)]
        files = [f for f in run_files if f"_ChSiPM_{SiPM_channel}" in f]
        if len(files) == 0:
            continue
        ch_st = files[0].split("ChST_")[-1].split(".root")[0]

        merger = TFileMerger()
        for f in files:
            merger.AddFile(f)

        out_file_name = metadata_folder + f"Merged_Thr_{threshold}_ChSiPM_{SiPM_channel}_ChST_{ch_st}.root"
        merger.OutputFile(out_file_name)
        merger.Merge()
        print(f"Merged {len(files)} files into {out_file_name}")
        del merger
