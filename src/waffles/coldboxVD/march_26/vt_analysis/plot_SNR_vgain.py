import pandas as pd
import matplotlib.pyplot as plt
import os

# ======================
# === PARAMETRI BASE ===
# ======================
modulo = "M2"
channel = "3"
led = "755"

# ======================
# === FILE INPUT =======
# ======================
files = [
    (f"/eos/home-v/vtrabatt/waffles/src/waffles/coldboxVD/march_26/vt_analysis/output/membrane_{modulo}/channel_{channel}/bias_1143/membrane{modulo}_channel{channel}_led{led}_bias1143_vgain_scan.csv", "+3 OV"),
    (f"/eos/home-v/vtrabatt/waffles/src/waffles/coldboxVD/march_26/vt_analysis/output/membrane_{modulo}/channel_{channel}/bias_1169/membrane{modulo}_channel{channel}_led{led}_bias1169_vgain_scan.csv", "+4 OV"),
    (f"/eos/home-v/vtrabatt/waffles/src/waffles/coldboxVD/march_26/vt_analysis/output/membrane_{modulo}/channel_{channel}/bias_1195/membrane{modulo}_channel{channel}_led{led}_bias1195_vgain_scan.csv", "+5 OV")
]

colors = ["blue", "green", "red"]

# ======================
# === FILTRI ===========
# ======================
exclusions = [
    # ("+5 OV", 2200),  # esclude vgain=2200 solo per +5 OV
    # ("+4 OV", 1700),
]

# ======================
# === PLOT SNR vs SPE ==
# ======================
plt.figure(figsize=(8,5))

for (file_path, label), color in zip(files, colors):

    df = pd.read_csv(file_path)

    # ======================
    # === APPLY FILTERS ===
    # ======================
    for ex_label, ex_vgain in exclusions:
        if label == ex_label:
            df = df[df["vgain"] != ex_vgain]

    # ======================
    # === PLOT ============
    # ======================
    plt.errorbar(
        df["spe_amplitude"],
        df["snr"],
        yerr=df["e_snr"],
        fmt='o',
        capsize=3,
        label=label,
        color=color
    )

# ======================
# === STYLING ==========
# ======================
plt.xlabel("SPE Amplitude")
plt.ylabel("SNR")
plt.title(f"Membrane module {modulo}, DAPHNE Channel {channel}: SNR vs SPE amplitude")
plt.grid(True, alpha=0.3)
plt.legend()

# ======================
# === SAVE PNG =========
# ======================
output_name = f"snr_vs_spe_{modulo}_ch{channel}.png"
output_path = os.path.join(os.getcwd(), output_name)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")