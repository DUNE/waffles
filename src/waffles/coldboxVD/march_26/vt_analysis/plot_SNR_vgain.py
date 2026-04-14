import pandas as pd
import matplotlib.pyplot as plt
import os

# === Parametri generali ===
modulo = "M1"
channel = "0"
led = "725"

# Lista dei file e legenda corrispondente
files = [
    (f"/eos/home-v/vtrabatt/waffles/src/waffles/coldboxVD/march_26/vt_analysis/output/membrane_{modulo}/channel_{channel}/bias_1143/membrane{modulo}_channel{channel}_led{led}_bias1143_vgain_scan.csv", "+3 OV"),
    (f"/eos/home-v/vtrabatt/waffles/src/waffles/coldboxVD/march_26/vt_analysis/output/membrane_{modulo}/channel_{channel}/bias_1169/membrane{modulo}_channel{channel}_led{led}_bias1169_vgain_scan.csv", "+4 OV"),
    (f"/eos/home-v/vtrabatt/waffles/src/waffles/coldboxVD/march_26/vt_analysis/output/membrane_{modulo}/channel_{channel}/bias_1195/membrane{modulo}_channel{channel}_led{led}_bias1195_vgain_scan.csv", "+5 OV")
]

colors = ["blue", "green", "red"]

# Lista di filtri: (legenda, vgain da escludere)
exclusions = [
    #("+5 OV", 2600),  # esclude vgain=2200 solo per bias 1195
    # puoi aggiungere altri filtri qui
    # ("+4 OV", 1700),
]

plt.figure(figsize=(8,5))

for (file_path, label), color in zip(files, colors):
    df = pd.read_csv(file_path)

    # Applica eventuali filtri
    for ex_label, ex_vgain in exclusions:
        if label == ex_label:
            df = df[df["vgain"] != ex_vgain]

    plt.errorbar(
        df["vgain"],
        df["snr"],
        yerr=df["e_snr"],
        fmt='o',
        capsize=3,
        label=label,
        color=color
    )

# === Abbellimenti ===
plt.xlabel("VGain")
plt.ylabel("SNR")
plt.title(f"Membrane {modulo}, DAPHNE Channel {channel}: SNR vs VGain")
plt.grid(True, alpha=0.3)
plt.legend()

# === Salva PDF nella cartella corrente ===
output_name = f"snr_vs_vgain_{modulo}_channel{channel}.pdf"
output_path = os.path.join(os.getcwd(), output_name)
plt.savefig(output_path)
plt.close()

print(f"Plot salvato in: {output_path}")


# =========================
# === PLOT 2: SNR vs SPE
# =========================
plt.figure(figsize=(8,5))

for (file_path, label), color in zip(files, colors):
    df = pd.read_csv(file_path)

    # stessi filtri
    for ex_label, ex_vgain in exclusions:
        if label == ex_label:
            df = df[df["vgain"] != ex_vgain]

    plt.errorbar(
        df["spe_amplitude"],        # <-- asse x = SPE
        df["snr"],
        yerr=df["e_snr"],
        fmt='o',
        capsize=3,
        label=label,
        color=color
    )

plt.xlabel("SPE Amplitude")
plt.ylabel("SNR")
plt.title(f"Membrane {modulo}, DAPHNE Channel {channel}: SNR vs SPE")
plt.grid(True, alpha=0.3)
plt.legend()

output_name = f"snr_vs_spe_{modulo}_channel{channel}.pdf"
output_path = os.path.join(os.getcwd(), output_name)
plt.savefig(output_path)
plt.close()

print(f"Plot salvato in: {output_path}")