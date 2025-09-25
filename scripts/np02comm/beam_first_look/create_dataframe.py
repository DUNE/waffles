#import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import mplhep
from glob import glob
import argparse
mplhep.style.use(mplhep.style.ROOT)
plt.rcParams.update({'font.size': 16,
                        'grid.linestyle': '--',
                        'axes.grid': True,
                        'figure.autolayout': True,
                        'figure.figsize': [14,6]
                        })


def create_df(runs=[], datadir="/afs/cern.ch/work/f/fraleman/public/np02_light_response_data", blacklist=[]):
    if not runs:
        foundfiles = glob(f"{datadir}/pe_info_*_run*.csv")
        runs = [int(os.path.basename(f).split('_run')[1].split('.csv')[0]) for f in foundfiles]

    #load the csv file from public directory and create the pandas DataFrame
    dfall = pd.DataFrame()
    for run in runs:
        if run in blacklist:
            continue
        saturation_file_cathode = f"{datadir}/pe_info_cathode_run{run:06d}.csv"
        saturation_file_membrane = f"{datadir}/pe_info_membrane_run{run:06d}.csv"
        dfc = pd.read_csv(saturation_file_cathode, sep=",")
        dfm = pd.read_csv(saturation_file_membrane, sep=",")

        dfsum = pd.merge(dfc, dfm, on=["Run", "CH"], suffixes=('_C', '_M'))
        dfsum["SUM"] = dfsum["SUM_C"] + dfsum["SUM_M"]
        if dfall.empty:
            dfall = dfsum
        else:
            dfall = pd.concat([dfall, dfsum], ignore_index=True)
    dfmeta = pd.read_csv(f"{datadir}/pds_beam_run_infos.csv", sep=",")
    dfall = pd.merge(dfmeta, dfall, on="Run", how="inner")
    dfall.sort_values(by=["Run", "CH"], inplace=True, ignore_index=True, ascending=[True, False])
    return dfall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, nargs='+', default=[], help="List of run numbers to process")
    parser.add_argument("--datadir", type=str, default="/afs/cern.ch/work/f/fraleman/public/np02_light_response_data", help="Directory containing the data files")
    parser.add_argument("--blacklist", type=int, nargs='+', default=[], help="List of run numbers to exclude")
    args = parser.parse_args()
    runs = args.runs
    datadir = args.datadir
    blacklist = args.blacklist
    create_df(runs, datadir, blacklist)

