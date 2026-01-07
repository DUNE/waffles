import pandas as pd
import numpy as np
import argparse

# === Load CSV for selected AFE ===
def load_data(afe_num):
    """
    Fully robust loader for DAPHNE bias CSV files.
    Handles:
      - CSVs with all columns in one string
      - Extra spaces or invisible characters
      - Missing optional columns (Vbias_monitor)
    """
    import os
    import pandas as pd

    base_path = "/eos/experiment/neutplatform/protodune/experiments/ColdBoxVD/November2025run/spy_buffer/commissioning/DAPHNE_bias"
    csv_path = os.path.join(base_path, f"AFE{afe_num}_bias.csv")

    # Read file line by line
    with open(csv_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise RuntimeError(f"CSV file {csv_path} is empty")

    # Split header manually
    header = lines[0].replace("\ufeff", "").split(",")  # remove BOM if present
    header = [h.strip() for h in header]

    # Split data manually
    data = [line.split(",") for line in lines[1:]]

    # Create DataFrame
    df = pd.DataFrame(data, columns=header)

    # Convert numeric columns to float
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass  # leave as string if conversion fails

    # Check mandatory columns (ignore optional monitor)
    required_cols = ["Vbias_daphne", "Vbias_DAC", "Vbias_mult"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # Print first row for debug
        print("DEBUG: header read from CSV:", header)
        print("DEBUG: first row:", df.iloc[0].tolist() if len(df) > 0 else "empty")
        raise KeyError(f"Missing required columns in CSV {csv_path}: {missing}")

    return df



# === Weighted linear fit ===
def linear_fit(x, y, sigma_y):
    """
    Perform a weighted linear fit y = slope*x + intercept
    using weights 1/sigma_y.
    Returns slope and intercept.
    """
    params, cov = np.polyfit(x, y, 1, w=1 / sigma_y, cov=True)
    slope, intercept = params
    return slope, intercept

# === Forward and inverse conversions ===
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

    # Extract mandatory columns
    DAC = df["Vbias_DAC"].values
    Daphne = df["Vbias_daphne"].values
    Mult = df["Vbias_mult"].values

    # Extract optional Monitor column
    if "Vbias_monitor" in df.columns:
        Monitor = df["Vbias_monitor"].values
        sigma_monitor = np.full_like(Monitor, 0.005)
        has_monitor = True
    else:
        has_monitor = False

    # Define errors
    sigma_mult = np.full_like(Mult, 0.01)

    # === Perform all fits ===
    fits = {
        "DAC->mult": linear_fit(DAC, Mult, sigma_mult),
        "Daphne->mult": linear_fit(Daphne, Mult, sigma_mult),
    }

    if has_monitor:
        fits.update({
            "DAC->monitor": linear_fit(DAC, Monitor, sigma_monitor),
            "Daphne->monitor": linear_fit(Daphne, Monitor, sigma_monitor),
        })

    print("\n=== Fit Results ===")
    for key, (s, i) in fits.items():
        print(f"{key}: y = {s:.4f} * x + {i:.4f}")

    print("\n=== Computing conversions ===")

    # === Initialize results dictionary ===
    known_var = args.input_var
    v = args.value

    result = {
        "Vbias_mult": None,
        "Vbias_monitor": None if has_monitor else None,
        "Vbias_DAC": None,
        "Vbias_daphne": None
    }

    # Put input directly
    result[known_var] = v

    # Helper function to compute only if not already set
    def ensure(var, compute_func):
        if result[var] is None:
            try:
                result[var] = compute_func()
            except KeyError:
                # Skip if the required fit does not exist (e.g., monitor missing)
                result[var] = None

    # ---- Conversion cases ----

    if known_var == "Vbias_mult":
        ensure("Vbias_DAC", lambda: inverse(*fits["DAC->mult"], result["Vbias_mult"]))
        ensure("Vbias_daphne", lambda: inverse(*fits["Daphne->mult"], result["Vbias_mult"]))
        if has_monitor:
            ensure("Vbias_monitor", lambda: forward(*fits["DAC->monitor"], result["Vbias_DAC"]))

    elif known_var == "Vbias_monitor" and has_monitor:
        ensure("Vbias_DAC", lambda: inverse(*fits["DAC->monitor"], result["Vbias_monitor"]))
        ensure("Vbias_daphne", lambda: inverse(*fits["Daphne->monitor"], result["Vbias_monitor"]))
        ensure("Vbias_mult", lambda: forward(*fits["DAC->mult"], result["Vbias_DAC"]))

    elif known_var == "Vbias_DAC":
        ensure("Vbias_mult", lambda: forward(*fits["DAC->mult"], result["Vbias_DAC"]))
        if has_monitor:
            ensure("Vbias_monitor", lambda: forward(*fits["DAC->monitor"], result["Vbias_DAC"]))
        ensure("Vbias_daphne", lambda: inverse(*fits["Daphne->mult"], result["Vbias_mult"]))

    elif known_var == "Vbias_daphne":
        ensure("Vbias_mult", lambda: forward(*fits["Daphne->mult"], result["Vbias_daphne"]))
        if has_monitor:
            ensure("Vbias_monitor", lambda: forward(*fits["Daphne->monitor"], result["Vbias_daphne"]))
        ensure("Vbias_DAC", lambda: inverse(*fits["DAC->mult"], result["Vbias_mult"]))

    # === Output ===
    print("\n=== Final Results ===")
    for key, val in result.items():
        tag = "<-- target" if key == args.target_var else ""
        if val is not None:
            print(f"{key:15s} = {val:.4f}  {tag}")
        else:
            print(f"{key:15s} = N/A       {tag}")

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
