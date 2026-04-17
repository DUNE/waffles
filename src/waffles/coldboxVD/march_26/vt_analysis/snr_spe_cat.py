import pandas as pd
import matplotlib.pyplot as plt
import os

# ======================
# === PARAMETRI BASE ===
# ======================
modulo = "M6"
led = "920"

# directory bias (filesystem)
bias_dir = "0000"

# bias dentro il filename
bias_file = "0"

# ======================
# === CANALI ===========
# ======================
channels = ["19", "21"]
colors = ["blue", "green"]

# base directory
base_path = f"/eos/home-v/vtrabatt/waffles/src/waffles/coldboxVD/march_26/vt_analysis/output/membrane_{modulo}"

# ======================
# === FILTRI (OPZIONALI)
# ======================
exclusions = [
    # ("16", 2200),  # esempio: escludi vgain 2200 solo per channel 16
]

# ======================
# === PLOT SETUP =======
# ======================
plt.figure(figsize=(8,5))

# ======================
# === LOOP CANALI ======
# ======================
for ch, color in zip(channels, colors):

    file_path = os.path.join(
        base_path,
        f"channel_{ch}",
        f"bias_{bias_dir}",
        f"membrane{modulo}_channel{ch}_led{led}_bias{bias_file}_vgain_scan.csv"
    )

    print(f"Loading: {file_path}")

    # safety check
    if not os.path.exists(file_path):
        print(f"❌ Missing file: {file_path}")
        continue

    df = pd.read_csv(file_path)

    # ======================
    # === APPLY FILTERS ===
    # ======================
    if "vgain" in df.columns:
        for ex_ch, ex_vgain in exclusions:
            if ch == ex_ch:
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
        label=f"Channel {ch}",
        color=color
    )

# ======================
# === STYLING ==========
# ======================
plt.xlabel("SPE Amplitude")
plt.ylabel("SNR")
plt.title(f"Cathode module {modulo}: SNR vs SPE amplitude")
plt.grid(True, alpha=0.3)
plt.legend()

# ======================
# === SAVE PNG =========
# ======================
output_name = f"snr_vs_spe_{modulo}_bias{bias_file}_ch19.png"
output_path = os.path.join(os.getcwd(), output_name)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")