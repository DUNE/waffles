# --- IMPORTS -------------------------------------------------------
import yaml
import os
import pandas as pd
import numpy as np
import uproot

from waffles.np04_analysis.time_resolution.utils import time_of_flight
from waffles.np04_utils.utils import (get_np04_daphne_to_offline_channel_dict, 
                                      get_np04_channel_position_dataframe)

# --- HARD-CODED VARIABLES ------------------------------------------
golden_run = 30666
reference_channel = 11221
led_x = 0.
led_y = 305.5
led_z = 231.3
# -------------------------------------------------------------------

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the configs/time_resolution_config.yml file
    with open("./configs/time_resolution_configs.yml", 'r') as config_stream:
        config_variables = yaml.safe_load(config_stream)

    ana_folder        = config_variables.get("ana_folder")
    raw_ana_folder    = ana_folder+config_variables.get("raw_ana_folder")
    single_ana_folder = ana_folder+config_variables.get("single_ana_folder")
    
    new_daphne_to_offline        = get_np04_daphne_to_offline_channel_dict(version="new")
    offline_channel_positions_df = get_np04_channel_position_dataframe()
    offline_reference_channel    = new_daphne_to_offline[reference_channel]
    ref_channel_position         = offline_channel_positions_df.loc[offline_channel_positions_df['offline_ch'] == offline_reference_channel, \
                                                            ['det_center_x', 'det_center_y', 'ch_center_z']].values[0]
    led_position                 = np.array([led_x, led_y, led_z])
    ref_ch_tof                   = time_of_flight(led_position, ref_channel_position)

    # --- EXTRA VARIABLES -------------------------------------------
    os.makedirs(single_ana_folder, exist_ok=True)
    
    # --- LOOP OVER RUNS --------------------------------------------
    times_resultion_files = [raw_ana_folder+f for f in os.listdir(raw_ana_folder) if f.endswith("time_resolution.root")]
    files = [f for f in times_resultion_files if f"{golden_run}" in f]
    channel_avgt0_dict = {}
    for file in files:
        daphne_ch = int(file.split("DaphneCh_")[1].split("_")[0])
        root_file = uproot.open(file)
        root_dirs = root_file.keys()

        for root_dir in root_dirs:
            try:
                if "integral" in root_dir and "/" not in root_dir:
                    directory = root_file[root_dir]
                else:
                    continue
            except:
                continue
            tree     = directory["time_resolution"]
            branches = tree.keys()
            arrays   = tree.arrays(branches, library="np")
            
            t0s = arrays["t0"]
            channel_avgt0_dict[daphne_ch] = np.mean(t0s)

    reference_t0 = channel_avgt0_dict[reference_channel]

    # create a dataframe with daphne_ch, offline_ch, avg_t0, t0_offset
    out_df_rows = []
    for daphne_ch, avg_t0 in channel_avgt0_dict.items():
        offline_ch = new_daphne_to_offline[daphne_ch]
        t0_offset  = avg_t0 - reference_t0
        channel_position = offline_channel_positions_df.loc[offline_channel_positions_df['offline_ch'] == offline_ch, \
                                                            ['det_center_x', 'det_center_y', 'ch_center_z']].values[0]
        tof_correction = time_of_flight(led_position, channel_position) - ref_ch_tof

        out_df_rows.append({
            "OfflineChannel"        : offline_ch,
            "DaphneChannel"         : daphne_ch,
            "AverageT0"             : avg_t0,
            "T0Offset [ticks]"      : t0_offset,
            "T0Offset [ns]"         : (t0_offset*16.0),
            "TOFCorrection [ns]"    : tof_correction,
            "CorrectedT0Offset [ns]": (t0_offset*16.0) - tof_correction
        })
   
    aligned_df = pd.DataFrame(out_df_rows)
    aligned_df = aligned_df.sort_values(by="OfflineChannel")
    aligned_df.to_csv("channel_time_offsets.csv", index=False)
