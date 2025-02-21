import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import matplotlib.colors as mcolors

# Percorso del file JSON (Modifica con il tuo vero percorso)
json_file_path = "/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/channel_saturation/output/set_A_self_200files109_channel_saturation.json"

# Carica il file JSON
with open(json_file_path, "r") as file:
    data = json.load(file)

# Verifica la struttura del JSON
if isinstance(data, str):
    data = json.loads(data)  # Decodifica se è una stringa JSON

# Se il JSON è un dizionario con una chiave principale
if isinstance(data, dict):
    data = data[list(data.keys())[0]]  # Estrai la lista se è nidificata

# Inizializza correttamente i dizionari
channel_data = {}
saturation_values = {1: [], 2: [], 3: [], 5: [], 7: []}
print(data)

# Elenco dei canali disconnessi
disconnected_channels = ["109-17", "109-10", "109-11", "109-13", "109-14", "109-16"]

# Esamina ogni "entry" nel dataset
for channel_key, channel_info in data["109"].items():
    print(f"  Canale {channel_key}: {channel_info}")
    channel_number = str(channel_info["channel"])
    
    for energy_key, energy_info in channel_info["Beam saturation"].items():
        energy = int(energy_key)

        # Verifica che energy_info sia un dizionario valido
        if isinstance(energy_info, dict):
            if "Saturated event fraction" in energy_info and "Saturated event fraction error" in energy_info:
                fraction = energy_info["Saturated event fraction"]
                error = energy_info["Saturated event fraction error"]
                print(f"    Energia {energy} -> frazione: {fraction}, errore: {error}")
            else:
                fraction = 0
                error = 0
                print(f"    Energia {energy} -> Dati di saturazione mancanti o invalidi.")
        else:
            fraction = 0
            error = 0
            print(f"    Energia {energy} -> Dati di saturazione invalidi (NoneType).")

        channel_name = f"109-{channel_number}"
        # Salva i dati per ciascuna energia e canale
        if channel_name not in channel_data:
            channel_data[channel_name] = {}
        channel_data[channel_name][energy] = (fraction, error)

        # Verifica che l'energia sia valida e aggiungi la frazione ai valori di saturazione
        if energy in saturation_values:
            saturation_values[energy].append(fraction)
        else:
            print(f"Attenzione! Energia {energy} non trovata in saturation_values.")

# Calcolare il valore massimo di frazione di saturazione (escludendo i canali disconnessi)
max_fraction = max(
    [max(fractions) for channel, fractions in saturation_values.items() if fractions and channel not in disconnected_channels],
    default=0
)
print(f"Valore massimo di saturazione (escludendo i canali disconnessi): {max_fraction}")

# Debug per vedere se i dizionari sono stati riempiti
print("Dati dei canali:", channel_data)
print("Valori di saturazione:", saturation_values)

# Definisci la disposizione della griglia
channel_order = [
    ["109-27", "109-25", "109-22", "109-20"],
    ["109-21", "109-23", "109-24", "109-26"],
    ["109-37", "109-35", "109-32", "109-30"],
    ["109-31", "109-33", "109-34", "109-36"],
    ["109-7", "109-5", "109-2", "109-0"],
    ["109-1", "109-3", "109-4", "109-6"],
    ["109-17", "109-15", "109-12", "109-10"],
    ["109-11", "109-13", "109-14", "109-16"],
    ["109-47", "109-45", "109-42", "109-40"],
    ["109-41", "109-43", "109-44", "109-46"]
]

# Creazione della griglia per ogni energia
for energy in [1, 2, 3, 5, 7]:
    fig, ax = plt.subplots(figsize=(6, 12))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Normalizzazione per avere il massimo valore di fraction come riferimento
    norm = mcolors.Normalize(vmin=0, vmax=max_fraction)
    cmap = plt.cm.Reds  # Usa la mappa di colori 'Reds' (solo rosso)

    # Disegna la griglia
    for y, row in enumerate(channel_order):
        for x, channel in enumerate(row):
            # Se il canale è disconnesso, coloralo di giallo e scrivi "Disconnected"
            if channel in disconnected_channels:
                color = "yellow"  # Colore giallo
                rect = patches.Rectangle((x, 9 - y), 1, 1, edgecolor="black", facecolor=color)
                ax.add_patch(rect)
                ax.text(x + 0.5, 9 - y + 0.7, channel, ha="center", va="center", fontsize=8, fontweight='bold', color="black")
                ax.text(x + 0.5, 9 - y + 0.3, "Disconnected", ha="center", va="center", fontsize=8, color="black")
            else:
                # Verifica i dati di saturazione per il canale ed energia corrente
                fraction, error = channel_data.get(channel, {}).get(energy, (None, None))

                # Se non ci sono dati validi, mostra il colore bianco
                if fraction is None or error is None:
                    fraction, error = 0, 0

                # Usa il colore rosso con intensità variabile
                color = cmap(norm(fraction)) if fraction > 0 else "white"
                rect = patches.Rectangle((x, 9 - y), 1, 1, edgecolor="black", facecolor=color)
                ax.add_patch(rect)
                # Mostra la frazione in percentuale senza l'errore
                ax.text(x + 0.5, 9 - y + 0.7, channel, ha="center", va="center", fontsize=8, fontweight='bold')
                ax.text(x + 0.5, 9 - y + 0.3, f"{fraction*100:.3f} %", ha="center", va="center", fontsize=8)

    # Salva la figura per l'energia corrente
    output_path = os.path.join(os.getcwd(), f"heatmap_saturation_energy{energy}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Heatmap per energia {energy} salvata in: {output_path}")
