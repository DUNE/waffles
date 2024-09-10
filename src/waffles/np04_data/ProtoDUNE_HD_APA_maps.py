from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap
from waffles.np04_data_classes.APAMap import APAMap

import waffles.utils.wf_maps_utils as wuw

apa_1_data = [  [UniqueChannel(104, 7   ),  UniqueChannel(104, 5 ), UniqueChannel(104, 2 ), UniqueChannel(104, 0    )],
                [UniqueChannel(104, 1   ),  UniqueChannel(104, 3 ), UniqueChannel(104, 4 ), UniqueChannel(104, 6    )],
                [UniqueChannel(104, 17  ),  UniqueChannel(104, 15), UniqueChannel(104, 12), UniqueChannel(104, 10   )],
                [UniqueChannel(104, 11  ),  UniqueChannel(104, 13), UniqueChannel(104, 14), UniqueChannel(104, 16   )],
                [UniqueChannel(105, 7   ),  UniqueChannel(105, 5 ), UniqueChannel(105, 2 ), UniqueChannel(105, 0    )],
                [UniqueChannel(105, 1   ),  UniqueChannel(105, 3 ), UniqueChannel(105, 4 ), UniqueChannel(105, 6    )],
                [UniqueChannel(105, 26  ),  UniqueChannel(105, 24), UniqueChannel(105, 23), UniqueChannel(105, 21   )],
                [UniqueChannel(105, 10  ),  UniqueChannel(105, 12), UniqueChannel(105, 15), UniqueChannel(105, 17   )],
                [UniqueChannel(107, 17  ),  UniqueChannel(107, 15), UniqueChannel(107, 12), UniqueChannel(107, 10   )],
                [UniqueChannel(107, 0   ),  UniqueChannel(107, 2 ), UniqueChannel(107, 5 ), UniqueChannel(107, 7    )]]

apa_1 = APAMap(apa_1_data)

apa_2_data = [  [UniqueChannel(109, 27  ),  UniqueChannel(109, 25), UniqueChannel(109, 22), UniqueChannel(109, 20   )],
                [UniqueChannel(109, 21  ),  UniqueChannel(109, 23), UniqueChannel(109, 24), UniqueChannel(109, 26   )],
                [UniqueChannel(109, 37  ),  UniqueChannel(109, 35), UniqueChannel(109, 32), UniqueChannel(109, 30   )],
                [UniqueChannel(109, 31  ),  UniqueChannel(109, 33), UniqueChannel(109, 34), UniqueChannel(109, 36   )],
                [UniqueChannel(109, 7   ),  UniqueChannel(109, 5 ), UniqueChannel(109, 2 ), UniqueChannel(109, 0    )],
                [UniqueChannel(109, 1   ),  UniqueChannel(109, 3 ), UniqueChannel(109, 4 ), UniqueChannel(109, 6    )],
                [UniqueChannel(109, 17  ),  UniqueChannel(109, 15), UniqueChannel(109, 12), UniqueChannel(109, 10   )],
                [UniqueChannel(109, 11  ),  UniqueChannel(109, 13), UniqueChannel(109, 14), UniqueChannel(109, 16   )],
                [UniqueChannel(109, 47  ),  UniqueChannel(109, 45), UniqueChannel(109, 42), UniqueChannel(109, 40   )],
                [UniqueChannel(109, 41  ),  UniqueChannel(109, 43), UniqueChannel(109, 44), UniqueChannel(109, 46   )]]

apa_2 = APAMap(apa_2_data)

apa_3_data = [  [UniqueChannel(111, 1   ),  UniqueChannel(111, 3 ), UniqueChannel(111, 4 ), UniqueChannel(111, 6    )],
                [UniqueChannel(111, 36  ),  UniqueChannel(111, 34), UniqueChannel(111, 33), UniqueChannel(111, 31   )],
                [UniqueChannel(111, 0   ),  UniqueChannel(111, 2 ), UniqueChannel(111, 5 ), UniqueChannel(111, 7    )],
                [UniqueChannel(111, 37  ),  UniqueChannel(111, 35), UniqueChannel(111, 32), UniqueChannel(111, 30   )],
                [UniqueChannel(111, 41  ),  UniqueChannel(111, 43), UniqueChannel(111, 44), UniqueChannel(111, 46   )],
                [UniqueChannel(111, 16  ),  UniqueChannel(111, 14), UniqueChannel(111, 13), UniqueChannel(111, 11   )],
                [UniqueChannel(111, 10  ),  UniqueChannel(111, 12), UniqueChannel(111, 15), UniqueChannel(111, 17   )],
                [UniqueChannel(111, 26  ),  UniqueChannel(111, 24), UniqueChannel(111, 23), UniqueChannel(111, 21   )],
                [UniqueChannel(111, 40  ),  UniqueChannel(111, 42), UniqueChannel(111, 45), UniqueChannel(111, 47   )],
                [UniqueChannel(111, 27  ),  UniqueChannel(111, 25), UniqueChannel(111, 22), UniqueChannel(111, 20   )]]

apa_3 = APAMap(apa_3_data)

apa_4_data = [  [UniqueChannel(112, 0   ),  UniqueChannel(112, 2 ), UniqueChannel(112, 5 ), UniqueChannel(112, 7    )],
                [UniqueChannel(112, 6   ),  UniqueChannel(112, 4 ), UniqueChannel(112, 3 ), UniqueChannel(112, 1    )],
                [UniqueChannel(112, 10  ),  UniqueChannel(112, 12), UniqueChannel(112, 15), UniqueChannel(112, 17   )],
                [UniqueChannel(112, 16  ),  UniqueChannel(112, 14), UniqueChannel(112, 13), UniqueChannel(112, 11   )],
                [UniqueChannel(113, 0   ),  UniqueChannel(113, 2 ), UniqueChannel(113, 5 ), UniqueChannel(113, 7    )],
                [UniqueChannel(112, 27  ),  UniqueChannel(112, 25), UniqueChannel(112, 22), UniqueChannel(112, 20   )],
                [UniqueChannel(112, 21  ),  UniqueChannel(112, 23), UniqueChannel(112, 24), UniqueChannel(112, 26   )],
                [UniqueChannel(112, 37  ),  UniqueChannel(112, 35), UniqueChannel(112, 32), UniqueChannel(112, 30   )],
                [UniqueChannel(112, 31  ),  UniqueChannel(112, 33), UniqueChannel(112, 34), UniqueChannel(112, 36   )],
                [UniqueChannel(112, 47  ),  UniqueChannel(112, 45), UniqueChannel(112, 42), UniqueChannel(112, 40   )]]

apa_4 = APAMap(apa_4_data)

APA_map = { 1 : apa_1, 
            2 : apa_2, 
            3 : apa_3, 
            4 : apa_4}

flat_APA_map = {1 : ChannelMap(1, 40, [ wuw.flatten_2D_list(APA_map[1].data) ]), 
                2 : ChannelMap(1, 40, [ wuw.flatten_2D_list(APA_map[2].data) ]),
                3 : ChannelMap(1, 40, [ wuw.flatten_2D_list(APA_map[3].data) ]), 
                4 : ChannelMap(1, 40, [ wuw.flatten_2D_list(APA_map[4].data) ])}