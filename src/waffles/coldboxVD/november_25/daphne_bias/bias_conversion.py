import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import ODR, Model, RealData
import json
import waffles

# === Data loading: there is one csv file for AFE === 
csv_path = "/eos/experiment/neutplatform/protodune/experiments/ColdBoxVD/November2025run/spy_buffer/commissioning/DAPHNE_bias/AFE0_bias.csv"

# === Reading csv file ===
df = pd.read_csv(
    csv_path,
    sep=',',
    decimal='.'
)

# === Extract columns ===
x_DAC = df["Vbias_DAC"].values
x_daphne = df["Vbias_daphne"].values
y = df["Vbias_mult"].values
sigma_y = np.full_like(y, 0.01) #multimeter error

# === Linear Fit 1: DAC → multimeter ===
params1, cov1 = np.polyfit(x_DAC, y, 1, w=1/sigma_y, cov=True)
slope1, intercept1 = params1
err_slope1 = np.sqrt(cov1[0, 0])
err_intercept1 = np.sqrt(cov1[1, 1])

# === Linear Fit 2: Daphne → multimeter ===
params2, cov2 = np.polyfit(x_daphne, y, 1, w=1/sigma_y, cov=True)
slope2, intercept2 = params2
err_slope2 = np.sqrt(cov2[0, 0])
err_intercept2 = np.sqrt(cov2[1, 1])

# === Save CSV with fit results ===
results = {
    "Fit": ["DAC → mult", "DAC → mult", "Daphne → mult", "Daphne → mult"],
    "Parameter": ["Slope", "Intercept", "Slope", "Intercept"],
    "Value": [slope1, intercept1, slope2, intercept2],
    #"Error": [err_slope1, err_intercept1, err_slope2, err_intercept2]
}

df_results = pd.DataFrame(results) 
#remember to change name of the file (suggestion: change AFEnumber) to not overwrite it 
output_csv = "/eos/home-v/vtrabatt/waffles/src/waffles/coldboxVD//november_25/daphne_bias/AFE0_bias_conversions.csv"

df_results.to_csv(output_csv, index=False)

print(f"\nResults saved to: {output_csv}")

