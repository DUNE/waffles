# THis script creates pickles or binaries from the VGAIN SCAN or Selftrigger
# run list. The idea is to take the 12 runs that a VGAIN point comprises and
# generate 4 pickle files or 4 sets of binary files for the corresponding VGAIN point
from waffles.np04_analysis.vgain_analysis.scripts.hdf5_2_pickle_binary import endpoint_waveforms

print('importing waffles: waffles.input.raw_hdf5_reader')
import time
from waffles.input_output.raw_hdf5_reader import *
import waffles.input_output.raw_hdf5_reader as reader
import numpy as np
import os
import sys
import traceback
import pickle
import pandas as pd
from itertools import chain

print('finished importing waffles: waffles.input.raw_hdf5_reader')


def getRunsFromCSV(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        content = file.read()
        content = content.split('\n')
        content = content[0].split(';')
    return content


#################################################################
### HARD CODE HERE ##############################################

save_pickle = True
save_binary = False

folder_with_file_locations = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/"
destination_folder = "/afs/cern.ch/work/e/ecristal/"
csv_database = "configs/vgain_top_level.csv"
csv_channels_per_run = "configs/vgain_channels.csv"
run_numbers_error_list = "output/vgain_scans_mapping_runs_error_list.txt"
#runs_to_convert = getRunsFromCSV(csv_database)
# runs_to_convert must be changed to the column runs of the table vgain_top_level.csv.

run_database = pd.read_csv(csv_database)
ch_per_run_database = pd.read_csv(csv_channels_per_run)

runs_to_convert = run_database["run"]
led_intensity = run_database["intensity"]
vgain = run_database["vgain"]
ov = run_database["ov"]

runs_ch_list = ch_per_run_database["run"]
channels_list_per_run = ch_per_run_database["channels"]

#channels_to_save = [0, 1, 2, 3, 4, 5, 6, 7,
#                    10, 11, 12, 13, 14, 15, 16, 17,
#                     20, 21, 22, 23, 24, 25, 26, 27,
#                     30, 31, 32, 33, 34, 35, 36, 37,
#                     40, 41, 42, 43, 44, 45, 46, 47]
#channels_to_save must be retrieved from the vgain_channels.csv table
rucio_filepaths = [folder_with_file_locations + "0" + str(run) + ".txt" for run in
                   runs_to_convert]

filepaths_dict = {}
for path_index, path in enumerate(rucio_filepaths):
    filepaths_dict[runs_to_convert[path_index]] = path

channels_by_run_dict = {}
for run_index, run_value in enumerate():
    channels_by_run_dict[run_value] = channels_list_per_run[run_index]

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
unique_ov = np.unique(ov)
iteration_dict = {}
for vgain_value in unique_vgain:
    for ov_value in unique_ov:
        runs_ = runs_to_convert[vgain_value == vgain & ov_value == ov]
        iteration_dict[vgain_value][ov_value] = {runs_}

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


def getAlignedData(wfset, user_selected_endpoint, channel_list):
    try:

        endpoint_waveforms = [waveform for waveform in wfset.waveforms if
                              waveform.endpoint == user_selected_endpoint]

        print(endpoint_waveforms[0].endpoint)
        timestamps = []
        for waveform in endpoint_waveforms:
            timestamps.append(waveform.timestamp)

        print(len(endpoint_waveforms))
        print(len(timestamps))
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
        for ch in channel_list:
            filtered_data = [waveform for waveform in complete_data if
                              waveform.channel == ch]
            selected_complete_data.append(filtered_data)
        return selected_complete_data

    except Exception as e:
        print(f"Error en procesamiento de los datos:\n {e}")
        print(traceback.format_exc())
def sortWaveformsByTimestamp(wflist):
    timestamps = []
    for waveform in wflist:
        timestamps.append(waveform.timestamp)

    print(len(wflist))
    print(len(timestamps))
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
        if (save_pickle == True):
            try:
                pickle_file_name = 'data_endpoint_' + str(wf_list_key) + '.pickle'
                print('Saving saving data to a pickle file named: ' + folder_location + '/' + pickle_file_name)
                with open(folder_location + '/' + pickle_file_name, 'wb') as pickle_file:
                    # for waveforms in tqdm(complete_data):
                    pickle.dump(wf_dict[wf_list_key], pickle_file)

            except Exception as e:
                print(f"Error en procesamiento de los datos:\n {e}")
                print(traceback.format_exc())

        if (save_binary == True):
            channels_to_save = channels_to_save_dict[wf_list_key]
            selected_channels_inv = [channels_dict_inv[channels_to_save[i]] for i in range(len(channels_to_save))]
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
def unflatten_list(flat_list, n):
    return [flat_list[i:i+n] for i in range(0, len(flat_list), n)]
for vgain_index, vgain_value in enumerate(iteration_dict.keys()):
    vgain_folder_name = destination_folder + 'vgain_' + str(list(vgain_value))
    os.makedirs(vgain_folder_name, exist_ok=True)
    for ov_index, ov_value in enumerate(unique_ov):
        ov_folder = vgain_folder_name + '/' + str(ov_string_dict[ov_value])
        os.makedirs(ov_folder, exist_ok=True)
        run_list = iteration_dict[vgain_value][ov_value]
        # Folder are create in a structure:
        # vgain_{vgain_value}
        #     |
        #     40p
        #     45p
        #     50p
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
        for run_index, run_value in enumerate(run_list):
            try:
                rucio_path = filepaths_dict[run_value]
                print(rucio_path)
                file_to_read = reader.get_filepaths_from_rucio(rucio_path)
                print('Starting reading file: ' + file_to_read[0])

                wfset = WaveformSet_from_hdf5_file(file_to_read[0])
                wfsets = [WaveformSet_from_hdf5_file(file) for file in file_to_read[1:-1]]
                for wf_set in wfsets:
                    wfset.merge(wf_set)
                run_endpoints = wfset.get_set_of_endpoints()
                channels_to_save = channels_by_run_dict[run_value]
                # Here channels_to_save already has the endpoint information.
                # The information needs to be further parsed, because the channels
                # are ch + 100*endpoint format.
            except Exception as e:
                print(f"Error en procesamiento de los datos:\n {e}")
                print(traceback.format_exc())
                with open(run_numbers_error_list, 'a') as error_file:
                    error_file.write(rucio_path + '\n')
                error_file.close()
                continue
            for user_selected_endpoint in run_endpoints:
                channel_list = [ch for ch in channels_to_save if ch // 100 == user_selected_endpoint]
                channel_list = [ch % 100 for ch in channel_list]
                channels_to_save_dict[user_selected_endpoint].append(channel_list)
                endpoint_wf_dict[user_selected_endpoint].append(getAlignedData(wfset, user_selected_endpoint, channel_list))
        endpoint_wf_dict = reorderWaveformDict(endpoint_wf_dict)
        saveWaveDict(endpoint_wf_dict, save_pickle, save_binary, ov_folder, channels_to_save_dict, channels_dict_inv,
                         channels_dict)
    with open(vgain_folder_name + '/export.txt', 'w') as export_file:
        export_file.write(f'file read: {file_to_read[0]}')
    export_file.close;