import pandas as pd
import glob
import re
import os

# Path to your CSV files
path  = "/eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/analysis/FFT_sets/"
files = glob.glob(os.path.join(path, "OfflineCh_RMS_Config_*.csv"))

dfs = []
for file in files:
    df = pd.read_csv(file)

    # Extract config number
    match = re.search(r"Config_(\d+)", file)
    config_number = int(match.group(1)) if match else None

    df.insert(0, "ConfigNumber", config_number)
    dfs.append(df)

# Merge into one DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Save to CSV
merged_df.to_csv(path+"OfflineCh_RMS_Config_all.csv", index=False)

print("Merged CSV saved as merged.csv")

