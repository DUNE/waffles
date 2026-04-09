import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

import dunestyle.matplotlib as dunestyle
import mplhep
mplhep.style.use(mplhep.style.ROOT)
plt.rcParams.update({
                     'font.size': 14,
                     'grid.linestyle': '--',
                     'axes.grid': True,
                     'figure.autolayout': True,
                     'figure.figsize': [14,6],
})


def linear_fit(input_file:str, offset:str, time_fit_min:str, time_fit_max:str):
    data = pd.read_csv(input_file, skiprows=2, header=None, usecols=[0, 1])
    data.columns = ["Time", "Weight"]
    data = data.dropna(subset=["Time", "Weight"])  
    data["Time"] = pd.to_datetime(data["Time"], format="%Y/%m/%d %H:%M:%S.%f")
    
    if offset is not None:
        t0 = pd.to_datetime(offset, format="%Y/%m/%d %H:%M:%S.%f")
    else:
        t0 = data["Time"].iloc[0]  
        
    data["Hours"] = (data["Time"] - t0).dt.total_seconds() / 3600.0
    data["Weight"] = data["Weight"].astype(float)
    data_fit = data.copy()
    
    if time_fit_min is not None:
        data_start = (pd.to_datetime(time_fit_min, format="%Y/%m/%d %H:%M:%S.%f") - t0).total_seconds() / 3600.0
        data_fit = data_fit[data_fit["Hours"] >= data_start]

    if time_fit_max is not None:
        data_end = (pd.to_datetime(time_fit_max, format="%Y/%m/%d %H:%M:%S.%f") - t0).total_seconds() / 3600.0
        data_fit = data_fit[data_fit["Hours"] <= data_end]  

    # Linear fit
    coeffs, cov = np.polyfit(data_fit["Hours"], data_fit["Weight"], 1, cov=True)
    slope = coeffs[0]
    intercept = coeffs[1]
    slope_err = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])

    x_fit = np.linspace(data_fit["Hours"].min(), data_fit["Hours"].max(), 300)
    y_fit = slope * x_fit + intercept

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax.scatter(data["Hours"], data["Weight"], s=[14]*len(data["Hours"]), color='black', zorder=1)

    label = (
        f"slope = {slope:.4f} ± {slope_err:.4f} kg/h\n"
        f"intercept = {intercept:.4f} ± {intercept_err:.4f} kg\n"
    )

    ax.plot(x_fit, y_fit, color="red", linewidth=1.5, label=label)
    ax.set_xlabel("Elapsed time [h]")
    ax.set_ylabel("Xenon bottle weight [kg]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"fit_{input_file}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="File with data")
    parser.add_argument("--offset", type=str, default=None, help="Start time (format: '2026/03/24 15:03:00.000')")
    parser.add_argument("--fit-min", type=str, default=None, help="Remove points under a threshold (format: '2026/03/24 15:03:00.000')")
    parser.add_argument("--fit-max", type=str, default=None, help="Remove points over a threshold (format: '2026/03/24 15:03:00.000')")
    args = parser.parse_args()

    input_file = Path(args.file)
    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found.")

    print(f"Reading: {input_file}")
    linear_fit(args.file, args.offset, args.fit_min, args.fit_max)

if __name__ == "__main__":
    main()
