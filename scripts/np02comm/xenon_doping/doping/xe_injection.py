import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional
import argparse
from temp_correction import read_temperature, read_weight, match_weight_to_temperature

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


TEMP_CORRECTION_COEFF = 0.042
WEIGHT_TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


def linear_fit(
    weight_df: pd.DataFrame,
    offset: Optional[str],
    time_fit_min: Optional[str],
    time_fit_max: Optional[str],
    input_filepath: str,
) -> None:
    
    """Fit a linear model to the weight data and save the resulting plot."""

    # Determine time origin
    t0 = pd.to_datetime(offset, format=WEIGHT_TIME_FORMAT) if offset is not None else weight_df["time"].iloc[0]

    weight_df = weight_df.copy()
    weight_df["hours"] = (weight_df["time"] - t0).dt.total_seconds() / 3600.0

    # Apply temperature correction if the column is present
    if "temperature" in weight_df.columns:
        weight_df["weight"] -= TEMP_CORRECTION_COEFF * weight_df["temperature"]

    # Select the subset used for fitting
    fit_df = weight_df.copy()
    if time_fit_min is not None:
        fit_start = (pd.to_datetime(time_fit_min, format=WEIGHT_TIME_FORMAT) - t0).total_seconds() / 3600.0
        fit_df = fit_df[fit_df["hours"] >= fit_start]
    if time_fit_max is not None:
        fit_end = (pd.to_datetime(time_fit_max, format=WEIGHT_TIME_FORMAT) - t0).total_seconds() / 3600.0
        fit_df = fit_df[fit_df["hours"] <= fit_end]

    # Linear fit with covariance estimate
    coeffs, cov = np.polyfit(fit_df["hours"], fit_df["weight"], 1, cov=True)
    slope, intercept = coeffs
    slope_err = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])

    x_fit = np.linspace(fit_df["hours"].min(), fit_df["hours"].max(), 300)
    y_fit = slope * x_fit + intercept

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(weight_df["hours"], weight_df["weight"], s=14, color="black", zorder=1)
    label = (
        f"slope = {slope:.4f} ± {slope_err:.4f} kg/h\n"
        f"intercept = {intercept:.4f} ± {intercept_err:.4f} kg"
    )
    ax.plot(x_fit, y_fit, color="red", linewidth=1.5, label=label)
    ax.set_xlabel("Elapsed time [h]")
    ax.set_ylabel("Xenon bottle weight [kg]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"fit_{Path(input_filepath).stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="File with weight data")
    parser.add_argument("--offset", type=str, default=None, help="Start time (format: '2026/03/24 15:03:00.000')")
    parser.add_argument("--fit-min", type=str, default=None, help="Lower time bound for fit (format: '2026/03/24 15:03:00.000')")
    parser.add_argument("--fit-max", type=str, default=None, help="Upper time bound for fit (format: '2026/03/24 15:03:00.000')")
    parser.add_argument("--skip-temp", action="store_true", help="Skip temperature correction")
    args = parser.parse_args()

    input_filepath = Path(args.file)
    if not input_filepath.exists():
        raise FileNotFoundError(f"{input_filepath} not found.")

    print(f"Reading: {input_filepath}")
    weight_df = read_weight(args.file)

    if not args.skip_temp:
        temp_df = read_temperature()
    else:
        temp_df = None
    if temp_df is not None:
        matched_df = match_weight_to_temperature(weight_df, temp_df)
        if len(matched_df) > 1:
            weight_df = matched_df  # already contains time, weight, temperature
    else:
        print("No temperature data found.")

    linear_fit(weight_df, args.offset, args.fit_min, args.fit_max, args.file)


if __name__ == "__main__":
    main()
