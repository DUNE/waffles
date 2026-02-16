import uproot
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def daphne_to_offline_channel(apa, endpoint, daq_channel, map_path = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft/PDHD_PDS_ChannelMap.csv', maritza_template_folder = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft'):
    df = pd.read_csv(map_path, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    daphne_to_offline = dict(zip(daphne_channels, df['offline_ch']))    
    return daphne_to_offline[daq_channel + 100*endpoint]

def offline_to_daphne_channel(offline_ch, map_path = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft/PDHD_PDS_ChannelMap.csv', maritza_template_folder = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft'):
    df = pd.read_csv(map_path, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    offline_to_daphne = dict(zip(df['offline_ch'],daphne_channels))  
    daphne_ch = offline_to_daphne[offline_ch]
    end = int(str(daphne_ch)[:3])
    daq_ch = int(str(daphne_ch)[3:])
    return end, daq_ch


''' To combine the visibility information with the x-arapuca position'''

f = uproot.open("/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/vis_Anna.root")

treeVis = f["visAnna/treeVis"]
treePos = f["visAnna/treePos"]

df_vis = treeVis.arrays(library="pd")
df_pos = treePos.arrays(library="pd")

# --- PUNTO DELLA BEAM WINDOW ---
end104_ch13 = {'x': -356.445655, 'y': 396.65925000000004, 'z': 84.61124999999998}

'''
x --> direction of E field
z --> direction of beam
y --> height

From Luis Gustavo: 
physics.producers.generator.X0: [-52.4]         # Starting position (cm)
physics.producers.generator.Y0: [399.09] 
physics.producers.generator.Z0: [0.0]
physics.producers.generator.Theta0XZ: [-10.7985] # Starting angles (degrees)
physics.producers.generator.Theta0YZ: [-11.63333] # based on dir (-0.1836, -0.1982, 0.9628)

From Dante proceedings:
The EM shower develops immediately after the beam entry point in the LAr volume at 3m
distance in the x (drift) direction in front of the ARAPUCA module and nearly parallel to it.
The beam in fact is oriented 11.7 downwards and 10.5 towards the anode plane where the PD
modules are located.
'''

beam_window = {'x': -52.4, 'y': 399.09, 'z': 0.0, 'theta0XZ': -10.7985, 'theta0YZ': -11.63333, 'direction': np.array([-0.1836, -0.1982, 0.9628])}
beam_window['r0'] = np.array([beam_window['x'], beam_window['y'], beam_window['z']]) # Beam entrance window position


## From Dante's proceedings the EM develops nerly 3m in x direction
# delta_x = -300.0  # cm
# ## Solve for parameter t such that: x = x0 + t * vx = x0 + delta_x
# t = delta_x / beam_window['direction'][0]
# r_shower = beam_window['r0'] + t * beam_window['direction']

## If the dispacement in 3m in the beam direction 
t = 60.0 
r_shower = beam_window['r0'] + t * beam_window['direction']

print("Parameter t =", t)
print("Shower position (cm):")
print("x =", r_shower[0])
print("y =", r_shower[1])
print("z =", r_shower[2])



# --- CREA COLONNA VIS ---
df_pos = df_pos.copy()
df_pos["vis"] = np.nan

for ch in df_pos["ch"].unique():
    # seleziona solo i punti con quel canale
    vis_ch = df_vis[df_vis["ch"] == ch]
    if vis_ch.empty:
        continue

    # coord x,y,z dei punti in df_vis
    coords_vis = vis_ch[["x","y","z"]].values

    tree = cKDTree(coords_vis)
    dist, idx = tree.query(r_shower)

    vis_value = vis_ch.iloc[idx]["vis"]


    # assegna vis a tutte le righe di df_pos con quel canale
    df_pos.loc[df_pos["ch"] == ch, "vis"] = vis_value


df_pos[["endpoint", "DAQ_ch"]] = (
    df_pos["ch"]
    .apply(lambda ch: offline_to_daphne_channel(ch))
    .apply(pd.Series)
)

df_pos.to_csv("/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/visibility_arapuca.csv", index=False)


