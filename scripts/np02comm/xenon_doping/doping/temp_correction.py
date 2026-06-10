import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from glob import glob 
import argparse
from typing import Optional

import dunestyle.matplotlib as dunestyle
import mplhep
mplhep.style.use(mplhep.style.ROOT)
plt.rcParams.update({
                     'font.size': 14,
                     'grid.linestyle': '--',
                     'axes.grid': False,
                     'figure.autolayout': True,
                     'figure.figsize': [14,6],
})

TEMP_OFFSET = 18.0
MAX_TIME_DELTA_SECONDS = 600
WEIGHT_TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"
TEMP_TIME_FORMAT = "%Y/%m/%d %H:%M:%S"


def read_weight(weight_filepath: str) -> pd.DataFrame:
    """Read the weight CSV file and return a DataFrame with columns time and weight."""
    df = pd.read_csv(weight_filepath, skiprows=2, header=None, usecols=[0, 1])
    df.columns = ["time", "weight"]
    df = df.dropna(subset=["time", "weight"])
    df["time"] = pd.to_datetime(df["time"], format=WEIGHT_TIME_FORMAT)
    df["weight"] = df["weight"].astype(float)
    return df


def read_temperature() -> Optional[pd.DataFrame]:
    """
    Read all weather_*.csv files and return a unified DataFrame
    with columns time and temperature, sorted by time.
    Returns None if no files are found.
    """
    temp_filepaths = glob("weather_*.csv")
    if not temp_filepaths:
        return None
    frames = []
    for filepath in temp_filepaths:
        date_str = Path(filepath).stem.split("_", maxsplit=1)[1].replace("-", "/")
        df = pd.read_csv(filepath, skiprows=1, header=None, usecols=[0, 1])
        df.columns = ["time", "temperature"]
        df = df.dropna(subset=["time", "temperature"])
        df["temperature"] -= TEMP_OFFSET
        df["time"] = pd.to_datetime(date_str + " " + df["time"], format=TEMP_TIME_FORMAT)
        frames.append(df)
    return pd.concat(frames).sort_values("time").reset_index(drop=True)
    

def match_weight_to_temperature(weight_df: pd.DataFrame, temp_df: pd.DataFrame,) -> pd.DataFrame:
    """
    For each weight sample, find the closest temperature measurement in time.
    Pairs with a time gap greater than MAX_TIME_DELTA_SECONDS are discarded.

    Returns a DataFrame with columns: time, weight, temperature.
    """
    weight_timestamps = weight_df["time"].values.astype("datetime64[s]").astype("int64")
    temp_timestamps = temp_df["time"].values.astype("datetime64[s]").astype("int64")

    closest_temp_indices = np.argmin(np.abs(weight_timestamps[:, None] - temp_timestamps[None, :]), axis=1)
    time_deltas = np.abs(weight_timestamps - temp_timestamps[closest_temp_indices])

    valid_mask = time_deltas <= MAX_TIME_DELTA_SECONDS

    if not valid_mask.all():
        n_discarded = (~valid_mask).sum()
        print(f"No match for {n_discarded} values (Δt > {MAX_TIME_DELTA_SECONDS}s)")

    matched_df = pd.DataFrame({
        "time": weight_df["time"].values[valid_mask],
        "weight": weight_df["weight"].values[valid_mask],
        "temperature": temp_df["temperature"].values[closest_temp_indices[valid_mask]],
    })
    return matched_df 
    
def plot(weight_filepath: str, temp_df: pd.DataFrame) -> None:
    weight_df = read_weight(weight_filepath)
    matched_df = match_weight_to_temperature(weight_df, temp_df)

    t0 = matched_df["time"].iloc[0]
    matched_hours = (matched_df["time"] - t0).dt.total_seconds() / 3600.0
    all_hours = (weight_df["time"] - t0).dt.total_seconds() / 3600.0

    ORANGE = "#F96B2D"
    BLUE_LIGHT = "#7CAED5"
    BLUE_DARK = "#2e86c1"

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#f9f9f9")
    ax1.set_facecolor("#f9f9f9")

    # Temperature
    ax1.plot(matched_hours, matched_df["temperature"], color=ORANGE, linewidth=1.5, label="Temperature")
    ax1.set_ylabel("Temperature [°C]", fontsize=12)
    ax1.set_xlabel("Elapsed time [h]", fontsize=12)
    ax1.tick_params(axis="y", color=ORANGE, which="both", labelcolor=ORANGE)
    ax1.yaxis.label.set_color(ORANGE)
    ax1.spines["left"].set_color(ORANGE)
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="x", color="black", labelcolor="black", top=False)
    ax1.spines["bottom"].set_color("black")

    # Weight
    ax2 = ax1.twinx()
    ax2.plot(all_hours, weight_df["weight"], ls="--", color=BLUE_DARK, linewidth=1.0, alpha=0.5, label="Weight (all)")
    ax2.plot(matched_hours, matched_df["weight"], color=BLUE_LIGHT, linewidth=1.5, label="Weight (matched)")
    ax2.set_ylabel("Weight [kg]", fontsize=12)
    ax2.tick_params(axis="y", color=BLUE_LIGHT, which="both", labelcolor=BLUE_LIGHT)
    ax2.yaxis.label.set_color(BLUE_LIGHT)
    ax2.spines["right"].set_color(BLUE_LIGHT)
    ax2.spines["right"].set_linewidth(1.5)
    ax2.spines["left"].set_visible(False)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(
    lines1 + lines2, labels1 + labels2,
    frameon=False,
    fontsize=10,
    loc="upper left",
    bbox_to_anchor=(0.1, 0.95),
)

    ax1.grid(axis="x", color="#dddddd", linewidth=0.7, linestyle="--")

    plt.tight_layout()
    plt.savefig("temp_weight_vs_time.png", dpi=150, bbox_inches="tight")

    # --- Plot 2: weight vs temperature with linear fit ---
    temp = matched_df["temperature"].values
    weight = matched_df["weight"].values

    coeffs, cov = np.polyfit(temp, weight, 1, cov=True)
    slope, intercept = coeffs
    slope_err = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])
    print(f"slope = {slope:.4f} ± {slope_err:.4f} kg/°C")

    x_fit = np.linspace(temp.min(), temp.max(), 300)
    y_fit = slope * x_fit + intercept

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(temp, weight, s=14, color="black", zorder=1)
    label = (
        f"slope = {slope:.4f} ± {slope_err:.4f} kg/°C\n"
        f"intercept = {intercept:.4f} ± {intercept_err:.4f} kg"
    )
    ax.plot(x_fit, y_fit, color="red", linewidth=1.5, label=label)
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Xenon bottle weight [kg]")
    ax.legend()
    plt.tight_layout()
    plt.savefig("weight_vs_temperature.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_file", type=str, help="File with weight data")
    args = parser.parse_args()

    weight_filepath = Path(args.weight_file)
    if not weight_filepath.exists():
        raise FileNotFoundError(f"{weight_filepath} not found.")

    print(f"Reading: {weight_filepath}")
    temp_df = read_temperature()
    if temp_df is not None:
        plot(args.weight_file, temp_df)
    else:
        print("No temperature data found.")


if __name__ == "__main__":
    main()
