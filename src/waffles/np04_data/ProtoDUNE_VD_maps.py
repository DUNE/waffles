from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap
from waffles.np04_data_classes.MEMMap import MEMMap

import waffles.utils.wf_maps_utils as wuw

mem_notco_data = [  [UniqueChannel(107, 47),  UniqueChannel(107, 45) ],
                    [UniqueChannel(107, 40),  UniqueChannel(107, 42) ],
                    [UniqueChannel(107,  0),  UniqueChannel(107,  7) ],
                    [UniqueChannel(107, 20),  UniqueChannel(107, 27) ]]

mem_notco = MEMMap(mem_notco_data)

mem_tco_data = [    [UniqueChannel(107, 46  ),  UniqueChannel(107, 44)],
                    [UniqueChannel(107, 43  ),  UniqueChannel(107, 41)],
                    [UniqueChannel(107, 30  ),  UniqueChannel(107, 37)],
                    [UniqueChannel(107, 10  ),  UniqueChannel(107, 17)]]

mem_tco = MEMMap(mem_tco_data)


mem_map = { 1 : mem_notco, 
            2 : mem_tco}

flat_MEM_map = {1 : ChannelMap(1, 8, [ wuw.flatten_2D_list(mem_map[1].data) ]), 
                2 : ChannelMap(1, 8, [ wuw.flatten_2D_list(mem_map[2].data) ])}