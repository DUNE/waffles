# It return a json file with the number of files that were read in the old format (before May 2025)
# Before May 2025, the files were named self_<index>.pkl and stored in /afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles
# The index is an integer and it is referred to the nummber of the line of the 012345.txt file


import os
import json
import re


pickle_folder = "/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles"
output_folder = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output"

output_filename = "read_pkl_files_OLD"

with open(f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/beam_run_info.json", "r") as file:
    run_set_dic = json.load(file)
    print(run_set_dic)

data = {}

for set, run_set in run_set_dic.items():
    set_dic = {}
    print(f"\n\n ----------- SET {run_set['Name']} ----------- \n")
    
    for energy, run in run_set['Runs'].items():
        print(f"\n--- RUN {run} ({energy} GeV) ---")

        # Path alla cartella dei pickle
        pickle_path = os.path.join(pickle_folder, f"set_{run_set['Name']}", f"run_{run}")

        try:
            files = os.listdir(pickle_path)
        except FileNotFoundError:
            print("Cartella non trovata.")
            set_dic[run] = {'Files found': 0, 'Indices present': []}
            continue

        # Cerca file che matchano self_<index>.pkl
        pattern = re.compile(r"self_(\d+)\.pkl")
        indices = sorted(int(pattern.match(f).group(1)) for f in files if pattern.match(f))

        if indices:
            run_dic = {
                'Files found': len(indices),
                'Indices present': indices
            }
            for key, value in run_dic.items():
                print(f"{key} : {value}")
        else:
            run_dic = {
                'Files found': 0,
                'Indices present': []
            }
            print("Nessun file self_<index>.pkl trovato.")

        set_dic[run] = run_dic

    data[run_set['Name']] = set_dic

# Salva tutto in un file JSON
with open(os.path.join(output_folder, f"{output_filename}.json"), "w") as file:
    json.dump(data, file, indent=4)