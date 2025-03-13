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
print('finished importing waffles: waffles.input.raw_hdf5_reader')

def getRunsFromCSV(file_path):
    with open(file_path, 'r', encoding ='utf-8-sig') as file:
        content = file.read()
        content = content.split('\n')
        content = content[0].split(';')
    return content



#################################################################
### HARD CODE HERE ##############################################

save_pickle = False
save_binary = True

folder_with_file_locations = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/"
#folder_with_file_locations = "/eos/experiment/neutplatform/protodune/experiments/ColdBoxVD/December2024run/files_location_cb/"
#folder_with_file_locations = "/eos/home-e/ecristal/NP04/"
destination_folder = "/afs/cern.ch/work/e/ecristal/"
csv_run_numbers = "/eos/home-e/ecristal/NP04/data/np04_csv_run_list/vgain_scans_mapping_runs.csv"
run_numbers_error_list = "/eos/home-e/ecristal/NP04/data/np04_csv_run_list/vgain_scans_mapping_runs_error_list.txt"
runs_to_convert = getRunsFromCSV(csv_run_numbers)
channels_to_save = [0,  1, 2, 3, 4, 5, 6, 7,
                    10,11,12,13,14,15,16,17,
                    20,21,22,23,24,25,26,27,
                    30,31,32,33,34,35,36,37,
                    40,41,42,43,44,45,46,47]
rucio_filepaths = [folder_with_file_locations+"0"+str(run)+".txt" for run in
    runs_to_convert]

#################################################################


channels_dict = {0: 0,    1: 1,    2: 2,    3: 3,    4: 4,    5: 5,    6: 6,    7: 7,
                 8: 10,   9: 11,   10: 12,  11: 13,  12: 14,  13: 15,  14: 16,  15: 17,
                 16: 20,  17: 21,  18: 22,  19: 23,  20: 24,  21: 25,  22: 26,  23: 27,
                 24: 30,  25: 31,  26: 32,  27: 33,  28: 34,  29: 35,  30: 36,  31: 37,
                 32: 40,  33: 41,  34: 42,  35: 43,  36: 44,  37: 45,  38: 46,  39: 47,
                 }
channels_dict_inv = {0: 0,    1: 1,    2: 2,    3: 3,    4: 4,    5: 5,    6: 6,    7: 7,
                     10: 8,   11: 9,   12: 10,  13: 11,  14: 12,  15: 13,  16: 14,  17: 15,
                     20: 16,  21: 17,  22: 18,  23: 19,  24: 20,  25: 21,  26: 22,  27: 23,
                     30: 24,  31: 25,  32: 26,  33: 27,  34: 28,  35: 29,  36: 30,  37: 31,
                     40: 32,  41: 33,  42: 34,  43: 35,  44: 36,  45: 37,  46: 38,  47: 39,
                     }

for run_index, rucio_path in enumerate(rucio_filepaths):
    try:
        print(rucio_path)
        file_to_read = reader.get_filepaths_from_rucio(rucio_path)
        #file_to_read = ["/eos/home-e/ecristal/NP04/np02vdcoldbox_raw_run033026_0000_dataflow0_datawriter_0_20241209T085511.hdf5"]
        print('Starting reading file: ' + file_to_read[0])

    
        wfset = WaveformSet_from_hdf5_file(file_to_read[0])
        wfsets = [WaveformSet_from_hdf5_file(file) for file in file_to_read[1:-1]]
        for wf_set in wfsets:
            wfset.merge(wf_set)
        run_endpoints = wfset.get_set_of_endpoints()
    except Exception as e:
        print(f"Error en procesamiento de los datos:\n {e}")
        print(traceback.format_exc())
        with open(run_numbers_error_list,'a') as error_file:
            error_file.write(rucio_path + '\n')
        error_file.close()
        continue


    for user_selected_endpoint in run_endpoints:
        try:
            parent_folder_name = destination_folder + 'run_' + str(list(wfset.runs)[0])
            endpoint_folder = parent_folder_name + '/' + str(user_selected_endpoint)
            os.makedirs(parent_folder_name, exist_ok=True)
            os.makedirs(endpoint_folder, exist_ok=True)

            endpoint_waveforms = [waveform for waveform in wfset.waveforms if waveform.endpoint == user_selected_endpoint]
            
            print(endpoint_waveforms[0].endpoint)
            timestamps = []
            for waveform in endpoint_waveforms:
                timestamps.append(waveform.timestamp)

            print(len(endpoint_waveforms))
            print(len(timestamps))
            zipped_lists = list(zip(timestamps, endpoint_waveforms))
            zipped_lists.sort(key=lambda x: x[0])
            sorted_timestamps, shuffled_waveforms = zip(*zipped_lists)
            sorted_timestamps = list(sorted_timestamps)
            shuffled_waveforms = list(shuffled_waveforms)
            #Get the unique timestamps
            unique_selected_timestamps = np.unique(sorted_timestamps)

            complete_data = []
            print('Finding complete datasets with the same timestamp')
            data_length = len(unique_selected_timestamps)

            #n_divide_list = 40
            #divided_endpoint_waveforms = [endpoint_waveforms[i:i+n_divide_list] for i in range(0, len(shuffled_waveforms), n_divide_list)]
        except Exception as e:
            print(f"Error en procesamiento de los datos:\n {e}")
            print(traceback.format_exc())


        try:
            shuffled_index = 0
            for data_index, unique_timestamp in tqdm(enumerate(unique_selected_timestamps), total = data_length):

                complete_40_channels_waveforms = []
                for index_internal, waveform in enumerate(shuffled_waveforms[shuffled_index:shuffled_index+40]):
                    if(waveform.timestamp == unique_timestamp):
                        complete_40_channels_waveforms.append(waveform)
                    else:
                        break
                shuffled_index = shuffled_index + len(complete_40_channels_waveforms)
                #shuffled_waveforms = shuffled_waveforms[len(complete_40_channels_waveforms):]
                
                #print(len(complete_40_channels_waveforms))
                #old slow method 
                #timestamp_waveforms = [waveform for waveform in endpoint_waveforms if waveform.timestamp == unique_timestamp]    
                    
                if(len(complete_40_channels_waveforms) == 40):
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
            print('Success:\nNumber of complete datasets: ',len(complete_data))
            print(f'Waveforms length is {length_waveforms}')
            print('Total number of waveforms in endpoint ' + str(user_selected_endpoint) + ': ' + str(len(endpoint_waveforms)))
            print('Total number of complete waveforms in endpoint ' + str(user_selected_endpoint) + ': ' + str(40*len(complete_data)))

        except Exception as e:
            print(f"Error en procesamiento de los datos:\n {e}")
            print(traceback.format_exc())

        if(save_pickle == True):
            try:
                pickle_file_name = 'data_endpoint_' + str(user_selected_endpoint) + '.pickle'
                print('Saving saving data to a pickle file named: '+ endpoint_folder + '/' + pickle_file_name)
                with open(endpoint_folder + '/' + pickle_file_name, 'wb') as pickle_file:
                    #for waveforms in tqdm(complete_data):
                    pickle.dump(complete_data,pickle_file)

            except Exception as e:
                print(f"Error en procesamiento de los datos:\n {e}")
                print(traceback.format_exc())


        if(save_binary == True):
            selected_channels_inv = [channels_dict_inv[channels_to_save[i]] for i in range(len(channels_to_save))]
            selected_channels_dict = {}
            dic_increment = 0
            try:
                start = time.time()
                print('Saving files as binaries files')
                save_files = [] 
                for i in range(0,40):
                    if i in selected_channels_inv:
                        save_file = open(endpoint_folder + '/channel_' + str(channels_dict[i]) + '.dat', 'ab')
                        save_files.append(save_file)
                        selected_channels_dict[channels_dict[i]] = dic_increment
                        dic_increment = dic_increment + 1
                print(selected_channels_dict)
                
                # Old inneficient way
                for data_index, data in tqdm(enumerate(complete_data),total = len(complete_data)):
                    for channel_index, waveform in enumerate(data):
                        if waveform.channel in channels_to_save:
                            waveform.adcs.tofile(save_files[selected_channels_dict[waveform.channel]])
            
                for i in range(0,len(channels_to_save)):
                    save_files[i].close()

                end = time.time()
                print("The time of execution of above program is :", (end-start) * 10**3, "ms")
            except Exception as e:
                print(f"Error en procesamiento de los datos:\n {e}")
                print(traceback.format_exc())
    del wfset            
    with open(parent_folder_name + '/export.txt','w') as export_file:
        export_file.write(f'file read: {file_to_read[0]}')
    export_file.close;