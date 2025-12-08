import pandas as pd
import numpy as np
import argparse


# === Load CSV for selected AFE ===
def load_data(afe_num):
    base_path = "/eos/experiment/neutplatform/protodune/experiments/ColdBoxVD/November2025run/spy_buffer/commissioning/DAPHNE_bias"
    csv_path = f"{base_path}/AFE{afe_num}_bias.csv"
    return pd.read_csv(csv_path)


# === Weighted linear fit ===
def linear_fit(x, y, sigma_y):
    params, cov = np.polyfit(x, y, 1, w=1 / sigma_y, cov=True)
    slope, intercept = params
    return slope, intercept


# === Convert using slope and intercept ===
def forward(slope, intercept, x):
    return slope * x + intercept


def inverse(slope, intercept, y):
    return (y - intercept) / slope


# === Main program ===
def main():

    parser = argparse.ArgumentParser(
        description="Universal DAPHNE bias conversion tool (fully symmetric)"
    )

    parser.add_argument("--afe", type=int, required=True, choices=[0, 2, 3, 4],
                        help="AFE number (0–4)")

    parser.add_argument("--input_var", type=str, required=True,
                        choices=["Vbias_mult", "Vbias_monitor", "Vbias_DAC", "Vbias_daphne"],
                        help="Known input variable")

    parser.add_argument("--value", type=float, required=True,
                        help="Numerical value of the input variable")

    parser.add_argument("--target_var", type=str, required=True,
                        choices=["Vbias_mult", "Vbias_monitor", "Vbias_DAC", "Vbias_daphne"],
                        help="Variable you want to compute")

    args = parser.parse_args()

    # === Load data ===
    df = load_data(args.afe)
    df.columns = df.columns.str.strip() #it automatically removes trailing spaces and leading spaces

    # Extract columns
    DAC = df["Vbias_DAC"].values
    Daphne = df["Vbias_daphne"].values
    Mult = df["Vbias_mult"].values
    Monitor = df["Vbias_monitor"].values

    # Errors
    sigma_mult = np.full_like(Mult, 0.01)
    sigma_monitor = np.full_like(Monitor, 0.005)

    # === Perform all fits ===
    fits = {
        "DAC->mult": linear_fit(DAC, Mult, sigma_mult),
        "Daphne->mult": linear_fit(Daphne, Mult, sigma_mult),

        "DAC->monitor": linear_fit(DAC, Monitor, sigma_monitor),
        "Daphne->monitor": linear_fit(Daphne, Monitor, sigma_monitor),
    }

    print("\n=== Fit Results ===")
    for key, (s, i) in fits.items():
        print(f"{key}: y = {s:.4f} * x + {i:.4f}")

    print("\n=== Computing conversions ===")

    # === Determine all variables from the input ===
    known_var = args.input_var
    v = args.value

    # We'll store results here
    result = {
        "Vbias_mult": None,
        "Vbias_monitor": None,
        "Vbias_DAC": None,
        "Vbias_daphne": None
    }

    # Put input directly
    result[known_var] = v

    # Helper: propagate through fits
    def ensure(var, compute_func):
        if result[var] is None:
            result[var] = compute_func()

    # ---- CASES ----

    # If we know mult:
    if known_var == "Vbias_mult":
        ensure("Vbias_DAC", lambda: inverse(*fits["DAC->mult"], result["Vbias_mult"]))
        ensure("Vbias_daphne", lambda: inverse(*fits["Daphne->mult"], result["Vbias_mult"]))
        ensure("Vbias_monitor", lambda:
               forward(*fits["DAC->monitor"], result["Vbias_DAC"]))

    # If we know monitor:
    elif known_var == "Vbias_monitor":
        ensure("Vbias_DAC", lambda: inverse(*fits["DAC->monitor"], result["Vbias_monitor"]))
        ensure("Vbias_daphne", lambda: inverse(*fits["Daphne->monitor"], result["Vbias_monitor"]))
        ensure("Vbias_mult", lambda:
               forward(*fits["DAC->mult"], result["Vbias_DAC"]))

    # If we know DAC:
    elif known_var == "Vbias_DAC":
        ensure("Vbias_mult", lambda: forward(*fits["DAC->mult"], result["Vbias_DAC"]))
        ensure("Vbias_monitor", lambda: forward(*fits["DAC->monitor"], result["Vbias_DAC"]))
        ensure("Vbias_daphne", lambda:
               inverse(*fits["Daphne->mult"], result["Vbias_mult"]))

    # If we know daphne:
    elif known_var == "Vbias_daphne":
        ensure("Vbias_mult", lambda: forward(*fits["Daphne->mult"], result["Vbias_daphne"]))
        ensure("Vbias_monitor", lambda: forward(*fits["Daphne->monitor"], result["Vbias_daphne"]))
        ensure("Vbias_DAC", lambda:
               inverse(*fits["DAC->mult"], result["Vbias_mult"]))

    # === Output ===
    print("\n=== Final Results ===")
    for key, val in result.items():
        tag = "<-- target" if key == args.target_var else ""
        print(f"{key:15s} = {val:.4f}  {tag}")

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
