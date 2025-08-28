# The script create a modifiied 012345.txt file with only the not read files, taking into account both OLD pkl files and NEW hdf5 files read

import os
import json

original_rucio_txt_folder = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths"

run_set_dic_path = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/beam_run_info.json"

pickle_folder = "/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles"
read_pkl_path = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/read_pkl_files_OLD.json"

hdf5_folder = "/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW"

output_folder = "/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/rucio_filepath"

with open(run_set_dic_path, "r") as file:
    run_set_dic = json.load(file)

with open(read_pkl_path, "r") as file:
    pkl_file_read = json.load(file)

for set, run_set in run_set_dic.items():
    if set == 'B':
        continue
    set_dic = {}
    print(f"\n\n ----------- SET {run_set['Name']} ----------- \n")
    for energy, run in run_set['Runs'].items():
        print(f"\n--- RUN {run} ({energy} GeV) ---")

        run_file = f"0{run}.txt"

        # All rucio files   
        rucio_run_filepath = os.path.join(original_rucio_txt_folder, run_file)
        if os.path.isfile(rucio_run_filepath):
            with open(rucio_run_filepath, 'r') as f:
                rucio_file_path_list_original = [line.strip() for line in f]
                rucio_file_path_list = [ os.path.basename(line) for line in rucio_file_path_list_original]
            rucio_file_index_list = list(range(len(rucio_file_path_list)))
            rucio_file_path_index_dic = {i: line.strip() for i, line in enumerate(rucio_file_path_list)}
        else:
            rucio_file_path_list = []
            rucio_file_index_list = []
            rucio_file_path_index_dic = {}


        # hdf5 read (NEW)
        hdf5_run_folder_filepath = os.path.join(hdf5_folder, f"run0{run}")
        if os.path.isdir(hdf5_run_folder_filepath):
            hdf5_read_file_path_list = [f.replace("processed_", "", 1).removesuffix("_structured.hdf5") for f in os.listdir(hdf5_run_folder_filepath) if os.path.isfile(os.path.join(hdf5_run_folder_filepath, f))]
            hdf5_read_file_path_index_dic = {k: v for k, v in rucio_file_path_index_dic.items() if v in hdf5_read_file_path_list}
            hdf5_read_file_index_list =list(hdf5_read_file_path_index_dic.keys())

        else:
            hdf5_read_file_path_list = []

        
        # Old pkl file read
        pkl_read_file_index_list = pkl_file_read[set][str(run)]['Indices present']
        pkl_read_file_path_index_dic = {i: rucio_file_path_index_dic[i] for i in pkl_read_file_index_list if i in rucio_file_path_index_dic}

        rucio_not_read_dic = {i: rucio_file_path_index_dic[i] for i in rucio_file_index_list if i not in hdf5_read_file_index_list and i not in pkl_read_file_index_list}
        rucio_not_read_dic_original = {i: rucio_file_path_list_original[i] for i in rucio_not_read_dic}


        # with open(os.path.join(output_folder, f"0{run}.txt"), 'w') as out_file:
        #     for filepath in rucio_not_read_dic_original.values():
        #         out_file.write(filepath + '\n')
        
        print(f"Rucio files: {len(rucio_file_index_list)}")
        print(f"HDF5 files read: {len(hdf5_read_file_path_list)}")
        print(f"Old pkl files read: {len(pkl_read_file_index_list)}")
        print(f"Rucio files not read: {len(rucio_not_read_dic)}")
