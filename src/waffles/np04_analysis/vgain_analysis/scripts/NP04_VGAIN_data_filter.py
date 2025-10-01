# THis script creates pickles or binaries from the VGAIN SCAN or Selftrigger
# run list. The idea is to take the 12 runs that a VGAIN point comprises and
# generate 4 pickle files or 4 sets of binary files for the corresponding VGAIN point

print('importing waffles: waffles.input.raw_hdf5_reader')
import argparse
import time
from waffles.input_output.raw_hdf5_reader import *
import waffles.input_output.raw_hdf5_reader as reader
import numpy as np
import os
import sys
import traceback
import pickle
import pandas as pd
import ast
from itertools import chain
from collections import defaultdict
from waffles.np04_analysis.vgain_analysis.scripts.rucioHandler import *
import pdb
import json

print('finished importing waffles: waffles.input.raw_hdf5_reader')

def getRunsFromCSV(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()
        content = content.split('\n')
        content = content[0].split(';')
    return content


#################################################################
### HARD CODE HERE ##############################################

parser = argparse.ArgumentParser(description="NP04 Filter app.")
parser.add_argument("-vgain", type=int, required=True, help="Vgain of the dataset to retrieve.")
parser.add_argument("-destination_folder", type=str, required=True, help="Location to save procesed data.")
parser.add_argument("-error_file_name", type=str, required=True, help="Error file name.")
parser.add_argument("-temp_data_folder", type=str, required=True, help="Directory location for temporal files.")
#parser.add_argument("-local_temp_hdf5_folder", type=str, required=True, help="Temporal storage location of downloaded HDF5 files.")
# # Parse arguments
args = parser.parse_args()


save_pickle = True
save_binary = True

# folder_with_file_locations = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/"
# destination_folder = "/afs/cern.ch/work/e/ecristal/"
destination_folder = args.destination_folder
csv_database = "../configs/vgain_top_level.csv"
run_error_database = "../configs/run_numbers_with_errors.csv"
csv_channels_per_run = "../configs/vgain_channels.csv"
run_numbers_error_list = f"../output/{args.error_file_name}_error.txt"
#run_numbers_error_list = args.run_numbers_error_list
local_rucio_txt_files_folder = f'{args.temp_data_folder}/rucio_txt_files' 
local_temp_hdf5_files = f'{args.temp_data_folder}/temp_hdf5_files_{args.vgain}'
#local_rucio_txt_files_folder = args.local_rucio_txt_files
#local_temp_hdf5_files = args.local_temp_hdf5_files
#Rucio handler for downloads
rh = RucioHandler(data_folder=local_temp_hdf5_files,txt_folder=local_rucio_txt_files_folder,max_files=10)
#runs_to_convert = getRunsFromCSV(csv_database)
# runs_to_convert must be changed to the column runs of the table vgain_top_level.csv.
rh.setup_rucio_1()
rh.setup_rucio_2()

run_database = pd.read_csv(csv_database)
ch_per_run_database = pd.read_csv(csv_channels_per_run)

run_error_database = pd.read_csv(run_error_database)

runs_to_convert = run_database["run"]
runs_with_errors = run_error_database["run"]
led_intensity = run_database["intensity"]
vgain = run_database["vgain"]
ov = run_database["ov"]

vgain_skip_list = []
run_number_skip_list = runs_to_convert[~runs_to_convert.isin(runs_with_errors)].tolist()

runs_ch_list = ch_per_run_database["run"]
channels_list_per_run = ch_per_run_database["channels"]

#channels_to_save = [0, 1, 2, 3, 4, 5, 6, 7,
#                    10, 11, 12, 13, 14, 15, 16, 17,
#                     20, 21, 22, 23, 24, 25, 26, 27,
#                     30, 31, 32, 33, 34, 35, 36, 37,
#                     40, 41, 42, 43, 44, 45, 46, 47]
#channels_to_save must be retrieved from the vgain_channels.csv table
# rucio_filepaths = [folder_with_file_locations + "0" + str(run) + ".txt" for run in
#                    runs_to_convert]

# filepaths_dict = defaultdict(dict)
# for path_index, path in enumerate(rucio_filepaths):
#     filepaths_dict[runs_to_convert[path_index]] = path

channels_by_run_dict = defaultdict(dict)
for run_index, run_value in enumerate(runs_ch_list):
    run_value = int(run_value)
    raw_list = channels_list_per_run[run_index]
    if isinstance(raw_list,str):
        raw_list = ast.literal_eval(raw_list)
    channels_by_run_dict[run_value] = [int(ch) for ch in raw_list]

ov_string_dict = {
    "[2.0, 3.5]": "40p",
    "[2.5, 4.5]": "45p",
    "[3.0, 7.0]": "50p"
}
#################################################################

# Here I will create the list for which I will iterate
# The main iterator should be VGAIN. Then for each VGAIN, I should
# have other 3 iterators that are the overvoltages.
# the for each overvoltage is endpoint

unique_vgain = np.unique(vgain)
unique_vgain = unique_vgain[unique_vgain == args.vgain]
unique_ov = np.unique(ov)
iteration_dict = defaultdict(dict)
for vgain_value in unique_vgain:
    for ov_value in unique_ov:
        runs_ = runs_to_convert[(vgain_value == vgain) & (ov_value == ov)]
        iteration_dict[vgain_value][ov_value] = runs_.tolist()

channels_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17,
                 16: 20, 17: 21, 18: 22, 19: 23, 20: 24, 21: 25, 22: 26, 23: 27,
                 24: 30, 25: 31, 26: 32, 27: 33, 28: 34, 29: 35, 30: 36, 31: 37,
                 32: 40, 33: 41, 34: 42, 35: 43, 36: 44, 37: 45, 38: 46, 39: 47,
                 }
channels_dict_inv = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
                     10: 8, 11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15,
                     20: 16, 21: 17, 22: 18, 23: 19, 24: 20, 25: 21, 26: 22, 27: 23,
                     30: 24, 31: 25, 32: 26, 33: 27, 34: 28, 35: 29, 36: 30, 37: 31,
                     40: 32, 41: 33, 42: 34, 43: 35, 44: 36, 45: 37, 46: 38, 47: 39,
                     }
def sanitize_data(obj):
    if isinstance(obj, dict):
        return {key: sanitize_data(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_data(element) for element in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj


def getAlignedData(wfset, user_selected_endpoint, channel_list):
    try:

        endpoint_waveforms = [waveform for waveform in wfset.waveforms if
                              waveform.endpoint == user_selected_endpoint]

        timestamps = []
        for waveform in endpoint_waveforms:
            timestamps.append(waveform.timestamp)

        zipped_lists = list(zip(timestamps, endpoint_waveforms))
        zipped_lists.sort(key=lambda x: x[0])
        sorted_timestamps, sorted_waveforms = zip(*zipped_lists)
        sorted_timestamps = list(sorted_timestamps)
        sorted_waveforms = list(sorted_waveforms)
        # Get the unique timestamps
        unique_selected_timestamps = np.unique(sorted_timestamps)

        complete_data = []
        print('Finding complete datasets with the same timestamp')
        data_length = len(unique_selected_timestamps)

        # n_divide_list = 40
        # divided_endpoint_waveforms = [endpoint_waveforms[i:i+n_divide_list] for i in range(0, len(sorted_waveforms), n_divide_list)]
    except Exception as e:
        print(f"Error en procesamiento de los datos:\n {e}")
        print(traceback.format_exc())

    try:
        shuffled_index = 0
        for data_index, unique_timestamp in tqdm(enumerate(unique_selected_timestamps), total=data_length):

            complete_40_channels_waveforms = []
            for index_internal, waveform in enumerate(sorted_waveforms[shuffled_index:shuffled_index + 40]):
                if (waveform.timestamp == unique_timestamp):
                    complete_40_channels_waveforms.append(waveform)
                else:
                    break
            shuffled_index = shuffled_index + len(complete_40_channels_waveforms)
            # sorted_waveforms = sorted_waveforms[len(complete_40_channels_waveforms):]

            # print(len(complete_40_channels_waveforms))
            # old slow method
            # timestamp_waveforms = [waveform for waveform in endpoint_waveforms if waveform.timestamp == unique_timestamp]

            if (len(complete_40_channels_waveforms) == 40):
                complete_data.append(complete_40_channels_waveforms)

        data_length = len(complete_data)
        data_aux = complete_data[0]
        length_waveforms = len(data_aux[0].adcs)
        for data in complete_data:
            local_timestamp = data[0].timestamp
            assert (len(data) == 40), 'Error: incomplete dataset'
            for waveform in data:
                assert (waveform.timestamp == local_timestamp), 'Error: timestamp missmatch'

        assert (data_length != 0), 'Error: complete datalength is 0'
        print('Success:\nNumber of complete datasets: ', len(complete_data))
        print(f'Waveforms length is {length_waveforms}')
        print('Total number of waveforms in endpoint ' + str(user_selected_endpoint) + ': ' + str(
            len(endpoint_waveforms)))
        print('Total number of complete waveforms in endpoint ' + str(user_selected_endpoint) + ': ' + str(
            40 * len(complete_data)))

        selected_complete_data = []
        if not channel_list:
            return None
        else:
            for complete_data_sublist in complete_data:
                sub_ch_list = []
                for ch in channel_list:
                    wf_ch = [waveform for waveform in complete_data_sublist if waveform.channel == ch]
                    sub_ch_list.append(wf_ch[0])
                selected_complete_data.append(sub_ch_list)
            return selected_complete_data

    except Exception as e:
        print(f"Error en procesamiento de los datos:\n {e}")
        print(traceback.format_exc())
def sortWaveformsByTimestamp(wflist):
    timestamps = []
    for waveform in wflist:
        timestamps.append(waveform.timestamp)

    #print(len(wflist))
    #print(len(timestamps))
    zipped_lists = list(zip(timestamps, wflist))
    zipped_lists.sort(key=lambda x: x[0])
    sorted_timestamps, sorted_waveforms = zip(*zipped_lists)
    sorted_timestamps = list(sorted_timestamps)
    sorted_waveforms = list(sorted_waveforms)
    # Get the unique timestamps
    unique_selected_timestamps = np.unique(sorted_timestamps)
    return sorted_waveforms, sorted_timestamps, unique_selected_timestamps
def reorderWaveformDict(wf_dict, channels_dict):
    for wf_list_key in wf_dict.keys():
        flattened_datatset = list(chain(*wf_dict[wf_list_key]))
        sorted_dataset = sortWaveformsByTimestamp(flattened_datatset)
        wf_dict[wf_list_key] = unflatten_list(sorted_dataset, len(channels_dict[wf_list_key]))
    return wf_dict
def saveWaveDict(wf_dict,save_pickle, save_binary, folder_location, channels_to_save_dict , channels_dict_inv, channels_dict):
    for wf_list_key in wf_dict.keys():
        if (save_pickle == True and channels_to_save_dict[wf_list_key]):
            try:
                pickle_file_name = 'data_endpoint_' + str(wf_list_key) + '.pickle'
                print('Saving saving data to a pickle file named: ' + folder_location + '/' + pickle_file_name)
                with open(folder_location + '/' + pickle_file_name, 'wb') as pickle_file:
                    # for waveforms in tqdm(complete_data):
                    pickle.dump(wf_dict[wf_list_key], pickle_file)

            except Exception as e:
                print(f"Error en procesamiento de los datos:\n {e}")
                print(traceback.format_exc())

        if (save_binary == True and channels_to_save_dict[wf_list_key]):
            channels_to_save = channels_to_save_dict[wf_list_key][0]
            selected_channels_inv = [channels_dict_inv[ch] for ch in channels_to_save]
            selected_channels_dict = {}
            dic_increment = 0
            try:
                print('Saving files as binaries files')
                save_files = []
                endpoint_save_folder = folder_location + '/binary_endpoint_' + str(wf_list_key)
                os.makedirs(endpoint_save_folder, exist_ok=True)
                for i in range(0, 40):
                    if i in selected_channels_inv:
                        save_file = open(endpoint_save_folder + '/channel_' + str(channels_dict[i]) + '.dat', 'ab')
                        save_files.append(save_file)
                        selected_channels_dict[channels_dict[i]] = dic_increment
                        dic_increment = dic_increment + 1
                print(selected_channels_dict)

                for data_index, data in tqdm(enumerate(wf_dict[wf_list_key]), total=len(wf_dict[wf_list_key])):
                    for channel_index, waveform in enumerate(data):
                        if waveform.channel in channels_to_save:
                            waveform.adcs.tofile(save_files[selected_channels_dict[waveform.channel]])

                for i in range(0, len(channels_to_save)):
                    save_files[i].close()

            except Exception as e:
                print(f"Error en procesamiento de los datos:\n {e}")
                print(traceback.format_exc())
        with open(folder_location + '/ok_compression.txt', 'w') as ok_compress_file:
            ok_compress_file.write('OK to compress files.')
        ok_compress_file.close()

def unflatten_list(flat_list, n):
    return [flat_list[i:i+n] for i in range(0, len(flat_list), n)]

def load_waveform(file):
    print(f'Reading file: ' + file)
    return WaveformSet_from_hdf5_file(file)

def populateEnpointsWfDict(wfset, run_endpoints, channels_to_save, endpoint_dict, channels_to_save_dict, endpoint_wf_dict, appendChannelsToSave):
    for user_selected_endpoint in run_endpoints:
        channel_list = [ch for ch in channels_to_save if ch // 100 == user_selected_endpoint]
        channel_list = [ch % 100 for ch in channel_list]
        print(f'Endpoint: {user_selected_endpoint}')
        print(f'Retrieving channels: ')
        for ch in channel_list:
            print(f'Channel: {ch}')
        if appendChannelsToSave:
            endpoint_dict[user_selected_endpoint]['CH_LIST'] = channel_list
            channels_to_save_dict[user_selected_endpoint].append(channel_list)
        waveforms_to_append = getAlignedData(wfset, user_selected_endpoint, channel_list)
        if waveforms_to_append is not None:
            for channel_set in waveforms_to_append:
                endpoint_wf_dict[user_selected_endpoint].append(channel_set)

def getWfsetFromFile(file):
        try:
            wfset = WaveformSet_from_hdf5_file(file)
            return wfset
        except Exception as e:
            print(f"Error loading HDF5 file: {file}")
            with open(run_numbers_error_list,'a') as error_file:
                error_file.write(f"Error loading HDF5 file: {file}" + '\n')
                error_file.close()
            print(f"Exception: {e}")


for vgain_index, vgain_value in enumerate(iteration_dict.keys()):
    if vgain_value in vgain_skip_list:
        print(f"Skipping vgain: {vgain_value}")
        continue
    vgain_folder_name = destination_folder + 'vgain_' + str(vgain_value)
    os.makedirs(vgain_folder_name, exist_ok=True)
    print(f'Getting runs with VGIAN: {vgain_value}')
    export_dict = defaultdict(dict)
    export_dict['VGAIN'] = vgain_value
    ov_dict = defaultdict(dict)
    for ov_index, ov_value in enumerate(unique_ov):
        print(f'Getting runs with overvoltage: {ov_value}')
        ov_folder = vgain_folder_name + '/' + str(ov_string_dict[ov_value])
        os.makedirs(ov_folder, exist_ok=True)
        run_list = iteration_dict[vgain_value][ov_value]
        ov_dict[ov_string_dict[ov_value]]['RUNLIST'] = run_list
        # Folder are create in a structure:
        # vgain_{vgain_value}
        #     |
        #     40p
        #     45p
        #     50p
        
        for run_index, run_value in enumerate(run_list):
            endpoint_wf_dict = {
            104: [],
            109: [],
            111: [],
            112: [],
            113: []
            }
            channels_to_save_dict = {
                104: [],
                109: [],
                111: [],
                112: [],
                113: []
            }
            if run_value in run_number_skip_list:
                print(f"Skipping run: {run_value}")
                continue
            print(f'Retrieving files for RUN: {run_value} - VGIAN: {vgain_value}')
            try:
                run_folder = ov_folder + '/run_' + str(run_value)
                os.makedirs(run_folder, exist_ok=True)
                #rucio_path = filepaths_dict[run_value]
                #print(rucio_path)
                #file_to_read = reader.get_filepaths_from_rucio(rucio_path)
                file_to_read = rh.download_data_from_rucio(run_number=run_value)
                wfset = getWfsetFromFile(file_to_read[0])
                run_endpoints = wfset.get_set_of_endpoints()
                channels_to_save = channels_by_run_dict[run_value]
                # Here channels_to_save already has the endpoint information.
                # The information needs to be further parsed, because the channels
                # are ch + 100*endpoint format.
            except Exception as e:
                print(f"Error en procesamiento de los datos:\n {e}")
                print(traceback.format_exc())
                with open(run_numbers_error_list, 'a') as error_file:
                    error_file.write(f'Error in run number: {run_value}')
                error_file.close()
                continue
            endpoint_dict = defaultdict(dict)
            populateEnpointsWfDict(wfset, run_endpoints, channels_to_save, endpoint_dict, channels_to_save_dict, endpoint_wf_dict, True)
            for file in file_to_read[1:-1]:
                wfset = getWfsetFromFile(file)
                populateEnpointsWfDict(wfset, run_endpoints, channels_to_save, endpoint_dict, channels_to_save_dict, endpoint_wf_dict, False)
        #endpoint_wf_dict = reorderWaveformDict(endpoint_wf_dict)
            ov_dict[ov_string_dict[ov_value]][run_value] = endpoint_dict
            saveWaveDict(endpoint_wf_dict, save_pickle, save_binary, run_folder, channels_to_save_dict, channels_dict_inv,
                         channels_dict)
            rh.clean_downloads()
    export_dict['METADATA'] = ov_dict
    with open(vgain_folder_name + '/export.json', 'w') as export_file:
        json.dump(sanitize_data(export_dict), export_file, indent=4)
    export_file.close

