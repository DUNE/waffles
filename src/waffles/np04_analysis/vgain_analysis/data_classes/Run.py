from typing import List, Any, Tuple, Dict

from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelMap import ChannelMap
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.ChannelWsGrid import ChannelWs

import numpy as np

class Run:
    def __init__(self,
                 run_number: int,
                 channel_map: ChannelMap,
                 waveform_set: WaveformSet,
                 configuration: List[Any]):

        self.__run_number = run_number
        self.__channel_ws_grid = ChannelWsGrid(channel_map, waveform_set)
        self.__configuration = configuration

        

    @property
    def run_number(self):
        return self.__run_number

    @property
    def channel_ws_grid(self):
        return self.__channel_ws_grid

    @property
    def configuration(self):
        return self.__configuration

    def find_channel(self, channel: UniqueChannel) -> Tuple[bool, Tuple[int, int]]:
        map_ = self.__channel_ws_grid.ch_map
        return map_.find_channel(channel)

    def get_channel_waveform_set(self, channel: UniqueChannel) -> WaveformSet:
        channel_pos = self.find_channel(channel)[1]
        waveform_set_ = self.__channel_ws_grid.get_channel_ws_by_ij_position_in_map(channel_pos[0],channel_pos[1])
        return waveform_set_

    def get_endpoints_in_run(self) -> List:
        return self.__channel_ws_grid.ch_wf_sets.keys()

    def syncronize_ws_grid(self):
        endpoints = self.get_endpoints_in_run()
        sync_dict = {}
        #waveformSet_ = WaveformSet() # This cannot be done
        k = False
        for endpoint in endpoints:
            sync_dict.update(self.synchronize_endpoint_waveform_set(endpoint))
        for i, endpoint in enumerate(sync_dict):
            for channel in endpoint.keys():
                if(k == True):
                    waveformSet_.merge(*sync_dict[endpoint.keys()[i]][channel])
                else:
                    waveformSet_ = WaveformSet(*sync_dict[endpoint][channel])
                k = True
        channel_map = self.__channel_ws_grid.ch_map()
        self.__channel_ws_grid = ChannelWsGrid(channel_map, waveformSet_)

    def synchronize_endpoint_waveform_set(self, endpoint) -> Dict[int, Dict[int, ChannelWs]]:
        map_ = self.__channel_ws_grid.ch_map
        map_data = map_.data
        map_rows = map_.rows
        map_columns = map_.columns
        endpoint_waveform_list = []
        endpoint_waveform_timestamp_list = []
        channels_list = []
        for i in range(map_rows):
            for j in range(map_columns):
                if(map_data[i][j].endpoint == endpoint):
                    waveform_ = self.get_channel_waveform_set(map_data[i][j]).waveforms
                    timestamps_ = []
                    for waveform in waveform_:
                        timestamps_.append(waveform.timestamp)
                    endpoint_waveform_list.append(waveform_)
                    endpoint_waveform_timestamp_list.append(timestamps_)
                    channels_list.append(map_data[i][j])
                    print(map_data[i][j])
        # I have to order the timestamps
        assert (len(endpoint_waveform_list) == len(endpoint_waveform_timestamp_list)), 'Error: missmatch between waveform list and etimestamp list'
        waveforms = [wf for wfs in endpoint_waveform_list for wf in wfs]
        timestamps = [tstamps for ts in endpoint_waveform_timestamp_list for tstamps in ts]
        del endpoint_waveform_list, endpoint_waveform_timestamp_list
        #[item for sublist in nested_list for item in sublist]
        
        zipped_lists = list(zip(timestamps, waveforms))
        zipped_lists.sort(key=lambda x: x[0])
        sorted_timestamps, shuffled_waveforms = zip(*zipped_lists)
        sorted_timestamps = list(sorted_timestamps)
        shuffled_waveforms = list(shuffled_waveforms)
        unique_selected_timestamps = np.unique(sorted_timestamps)
        complete_data = []
        number_of_channels_in_endpoint = len(channels_list)
        print('Finding complete datasets with the same timestamp')
        data_length = len(unique_selected_timestamps)
        del timestamps, zipped_lists
        try:
            shuffled_index = 0
            for data_index, unique_timestamp in enumerate(unique_selected_timestamps):
    
                sync_channels_waveform = []
                for index_internal, waveform in enumerate(shuffled_waveforms[shuffled_index:shuffled_index+number_of_channels_in_endpoint]):
                    if(waveform.timestamp == unique_timestamp):
                        sync_channels_waveform.append(waveform)
                    else:
                        break
                shuffled_index = shuffled_index + len(sync_channels_waveform)
                #shuffled_waveforms = shuffled_waveforms[len(sync_channels_waveform):]
                
                #print(len(sync_channels_waveform))
                #old slow method 
                #timestamp_waveforms = [waveform for waveform in endpoint_waveforms if waveform.timestamp == unique_timestamp]    
                    
                if(len(sync_channels_waveform) == number_of_channels_in_endpoint):
                    complete_data.append(sync_channels_waveform)
                    
            data_length = len(complete_data)
            data_aux = complete_data[0]
            length_waveforms = len(data_aux[0].adcs)
            for data in complete_data:
                local_timestamp = data[0].timestamp
                assert (len(data) == number_of_channels_in_endpoint), 'Error: incomplete dataset'
                for waveform in data:
                    assert (waveform.timestamp == local_timestamp), 'Error: timestamp missmatch'
                
            assert (data_length != 0), 'Error: complete datalength is 0'
            print('Success:\nNumber of complete datasets: ',len(complete_data))
            print(f'Waveforms length is {length_waveforms}')
            print('Total number of waveforms in endpoint ' + str(endpoint) + ': ' + str(len(waveforms)))
            print('Total number of complete waveforms in endpoint ' + str(endpoint) + ': ' + str(number_of_channels_in_endpoint*len(complete_data))) 
            del sorted_timestamps, shuffled_waveforms, waveforms
            waveforms = [wf for wfs in complete_data for wf in wfs]
            channel_sync_waveforms_dict = {}
            channel_sync_waveforms_dict[endpoint] = {}
            for channel in channels_list:
                endpoint_waveforms = [waveform for waveform in waveforms if waveform.channel == channel.channel]
                channel_sync_waveforms_dict[endpoint][channel.channel] = ChannelWs(*endpoint_waveforms)
            return channel_sync_waveforms_dict
        except Exception as e:
            print(f"Error en procesamiento de los datos:\n {e}")
            print(traceback.format_exc())









        
            