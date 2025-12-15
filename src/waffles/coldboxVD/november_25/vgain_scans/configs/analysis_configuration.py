from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap

endpoint = 104
folder_with_file_locations = "/eos/experiment/neutplatform/protodune/experiments/ColdBoxVD/December2024run/files_location_cb/"
run_error_list = "./run_error_list.txt"

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

# Coldbox channel map

HD_style_signal_map_list = [UniqueChannel(endpoint,0),
                            UniqueChannel(endpoint,1),
                            UniqueChannel(endpoint,2),
                            UniqueChannel(endpoint,3),
                            UniqueChannel(endpoint,4),
                            UniqueChannel(endpoint,5),
                            UniqueChannel(endpoint,6),
                            UniqueChannel(endpoint,7)]

VD_style_signal_map_list = [UniqueChannel(endpoint,20),
                            UniqueChannel(endpoint,21),
                            UniqueChannel(endpoint,22),
                            UniqueChannel(endpoint,23),
                            UniqueChannel(endpoint,24),
                            UniqueChannel(endpoint,25),
                            UniqueChannel(endpoint,26),
                            UniqueChannel(endpoint,27)]

SoF_signal_map_1_list    = [UniqueChannel(endpoint,30),
                            UniqueChannel(endpoint,31),
                            UniqueChannel(endpoint,32),
                            UniqueChannel(endpoint,33),
                            UniqueChannel(endpoint,34),
                            UniqueChannel(endpoint,35),
                            UniqueChannel(endpoint,36),
                            UniqueChannel(endpoint,37)]

SoF_signal_map_2_list    = [UniqueChannel(endpoint,40),
                            UniqueChannel(endpoint,41),
                            UniqueChannel(endpoint,42),
                            UniqueChannel(endpoint,43),
                            UniqueChannel(endpoint,44),
                            UniqueChannel(endpoint,45),
                            UniqueChannel(endpoint,46),
                            UniqueChannel(endpoint,47)]

coldbox_map_list = [HD_style_signal_map_list, VD_style_signal_map_list, SoF_signal_map_1_list, SoF_signal_map_2_list]
coldbox_map = ChannelMap(4,8,coldbox_map_list)
