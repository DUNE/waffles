# --- IMPORTS -------------------------------------------------------
from pandas._libs.hashtable import mode
import matplotlib.pyplot as plt
import waffles
import waffles.Exceptions as exceptions
import os
import yaml
import numpy as np
import pandas as pd
import waffles.np04_analysis.noise_studies.noisy_function as nf
from waffles.np04_utils.utils import get_np04_channel_mapping
from waffles.coldboxVD.utils.utils import get_cb25_channel_mapping
from waffles.coldboxVD.utils.spybuffer_reader import create_waveform_set_from_spybuffer

# --- MAIN ----------------------------------------------------------
if __name__ == "__main__":
    print("Imports done")
    # --- SETUP -----------------------------------------------------
    # Setup variables according to the noise_run_info.yaml file
    with open("./configs/led_run_info.yml", 'r') as stream:
        run_info = yaml.safe_load(stream)

    filepath_folder  = run_info.get("filepath_folder")
    led_folder = run_info.get("led_folder")

    # Setup variables according to the user_config.yaml file
    with open("params.yml", 'r') as stream:
        user_config = yaml.safe_load(stream)
    
    memorydepth = user_config.get("memorydepth")
    custom_filepath_folder = user_config.get("custom_filepath_folder")
    if (custom_filepath_folder != ""):
        filepath_folder = custom_filepath_folder
    debug_mode = user_config.get("debug_mode")
    ana_path = user_config.get("ana_path")
    out_writing_mode = user_config.get("out_writing_mode")
    channels      = user_config.get("user_vgains", [])
    if (len(vgains) == 0):
        print("Analyzing led runs")
        if (len(vgains) == 0):
        print("Analyzing all noise runs")
        # list all the directories in filepath_folder
        channel_dirs = [d for d in os.listdir(filepath_folder) if os.path.isdir(os.path.join(filepath_folder, d))]
        channels = [int(d.replace("vgain_", "")) for d in channel_dirs if d.startswith("channel_")]
        channel_dirs = [filepath_folder+d for d in os.listdir(filepath_folder) if os.path.isdir(os.path.join(filepath_folder, d))]
        if (len(channels) == 0):
            print("No runs to analyze")
            exit()
    else:
        vgain_dirs = [filepath_folder+"channel_"+str(vg) for ch in channels]

    