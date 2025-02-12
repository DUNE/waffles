from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap
from waffles.np04_data_classes.MEMMap import MEMMap

import waffles.utils.wf_maps_utils as wuw

mem_1_data = [  [UniqueChannel(109, 47),  UniqueChannel(109, 45) ],
                [UniqueChannel(109, 40),  UniqueChannel(109, 42) ],
                [UniqueChannel(109,  0),  UniqueChannel(109,  7) ],
                [UniqueChannel(109, 20),  UniqueChannel(109, 27) ]]

mem_1 = MEMMap(mem_1_data)

mem_2_data = [  [UniqueChannel(109, 46  ),  UniqueChannel(109, 44)],
                [UniqueChannel(109, 43  ),  UniqueChannel(109, 41)],
                [UniqueChannel(109, 30  ),  UniqueChannel(109, 37)],
                [UniqueChannel(109, 10  ),  UniqueChannel(109, 17)]]

mem_2 = MEMMap(mem_2_data)


mem_map = { 1 : mem_1, 
            2 : mem_2}

flat_MEM_map = {1 : ChannelMap(1, 8, [ wuw.flatten_2D_list(mem_map[1].data) ]), 
                2 : ChannelMap(1, 8, [ wuw.flatten_2D_list(mem_map[2].data) ])}