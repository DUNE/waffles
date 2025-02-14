from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap

from waffles.np04_data_classes.MEMMap import MEMMap_geo
from waffles.np04_data_classes.MEMMap import MEMMap_ind

from waffles.np04_data_classes.CATMap import CATMap_geo
from waffles.np04_data_classes.CATMap import CATMap_ind

import waffles.utils.wf_maps_utils as wuw

# ---------------------------MEMBRANES MAPPING---------------------------------

# Membranes geometrical mapping

#         M1(1)|M1(2)    M3(1)|M3(2)
# NO-TCO  M2(1)|M2(2)    M4(1)|M4(2)  TCO
#         M5(1)|M5(2)    M7(1)|M7(2)
#         M6(1)|M6(2)    M8(1)|M8(2)

# * M1(1)|M1(2) representes the first (1) and second (2) channels of membrane M1.

# --------------------------- Membrane NO-TCO ---------------------------------

mem_geometry_notco_titles = ["M1(1)","M1(2)", "M2(1)","M2(2)", "M5(1)","M5(2)", "M6(1)","M6(2)"]
        
mem_geometry_notco_data = [  [UniqueChannel(107, 47),  UniqueChannel(107, 45) ],
                             [UniqueChannel(107, 40),  UniqueChannel(107, 42) ],
                             [UniqueChannel(107,  0),  UniqueChannel(107,  7) ],
                             [UniqueChannel(107, 20),  UniqueChannel(107, 27) ]]

mem_geometry_notco = MEMMap_geo(mem_geometry_notco_data)

# --------------------------- Membrane TCO ---------------------------------

mem_geometry_tco_titles = ["M3(1)","M3(2)", "M4(1)","M4(2)", "M7(1)","M7(2)", "M8(1)","M8(2)"]

mem_geometry_tco_data = [    [UniqueChannel(107, 46  ),  UniqueChannel(107, 44)],
                             [UniqueChannel(107, 43  ),  UniqueChannel(107, 41)],
                             [UniqueChannel(107, 30  ),  UniqueChannel(107, 37)],
                             [UniqueChannel(107, 10  ),  UniqueChannel(107, 17)]]

mem_geometry_tco = MEMMap_geo(mem_geometry_tco_data)


mem_geometry_map = { 1 : mem_geometry_notco, 
                     2 : mem_geometry_tco}

flat_MEM_geometry_map = {1 : ChannelMap(1, 8, [ wuw.flatten_2D_list(mem_geometry_map[1].data) ]), 
                         2 : ChannelMap(1, 8, [ wuw.flatten_2D_list(mem_geometry_map[2].data) ])}


# Membranes index mapping

#         M1(1)|M1(2)    M5(1)|M3(2)
#         M2(1)|M2(2)    M6(1)|M4(2)  
#         M3(1)|M5(2)    M7(1)|M7(2)
#         M4(1)|M6(2)    M8(1)|M8(2)

mem_index_titles = ["M1(1)","M1(2)", "M5(1)","M5(2)", "M2(1)","M2(2)", "M6(1)","M6(2)","M3(1)","M3(2)", "M7(1)","M7(2)", "M4(1)", "M4(2)","M8(1)","M8(2)"]
        
mem_index_data = [  [UniqueChannel(107, 47),  UniqueChannel(107, 45), UniqueChannel(107,  0),  UniqueChannel(107,  7)],
                    [UniqueChannel(107, 40),  UniqueChannel(107, 42), UniqueChannel(107, 20),  UniqueChannel(107, 27)],
                    [UniqueChannel(107, 46),  UniqueChannel(107, 44), UniqueChannel(107, 30),  UniqueChannel(107, 37)],
                    [UniqueChannel(107, 43),  UniqueChannel(107, 41), UniqueChannel(107, 10),  UniqueChannel(107, 17)]]
                      

mem_index = MEMMap_ind(mem_index_data)

mem_index_map = { 1 : mem_index}

flat_MEM_geometry_map = {1 : ChannelMap(1, 16, [ wuw.flatten_2D_list(mem_index_map[1].data) ])}

# -----------------------------CATHODE MAPPING---------------------------------

# Cathode geometrical mapping

# First channel from each cathode
        
#              C4(1)           |      C8(1)
#                        C3(1) |           C7(1)
# NO-TCO  C1(1)                | C5(1)                 TCO
#                   C2(1)      |           C6(1)

# Second channel from each cathode

#              C4(2)           |      C8(2)
#                        C3(2) |           C7(2)
# NO-TCO  C1(2)                | C5(2)                 TCO
#                   C2(2)      |           C6(2)

# * C1(1) representes the first channel (1) of cathode C1.

# ------------------------- Cathode NO-TCO ----------------------------

cat_geometry_notco_1_titles = [None, "C4(1)", None, None,
                              None, None, None, "C3(1)",
                              "C1(1)", None, None, None, 
                              None, None, "C2(1)", None]

cat_geometry_notco_1_data = [[UniqueChannel(101, 45), UniqueChannel(106, 45), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                       [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(106, 42)],
                       [UniqueChannel(106, 0), UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                       [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(106, 20), UniqueChannel(101, 0)]]

cat_geometry_notco_1 = CATMap_geo(cat_geometry_notco_1_data)


cat_geometry_notco_2_titles = [None, "C4(2)", None, None,
                              None, None, None, "C3(2)",
                              "C1(2)", None, None, None, 
                              None, None, "C2(2)", None]


cat_geometry_notco_2_data = [[UniqueChannel(101, 45), UniqueChannel(106, 45), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                       [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(106, 42)],
                       [UniqueChannel(106, 0), UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                       [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(106, 20), UniqueChannel(101, 0)]]

cat_geometry_notco_2 = CATMap_geo(cat_geometry_notco_2_data)

cat_geometry_notco_map = { 1 : cat_geometry_notco_1, 
                           2 : cat_geometry_notco_2}

flat_CAT_notco_geometry_map = {1 : ChannelMap(1, 16, [ wuw.flatten_2D_list(cat_geometry_notco_map[1].data) ]), 
                             2 : ChannelMap(1, 16, [ wuw.flatten_2D_list(cat_geometry_notco_map[2].data) ])}


# ------------------------- Cathode TCO ----------------------------

cat_geometry_tco_1_titles = [ None, "C8(1)", None, None,
                            None, None, "C7(1)", None,
                            "C5(1)", None, None, None,
                            None, None, "C6(1)", None]

cat_geometry_tco_1_data = [[UniqueChannel(101, 0), UniqueChannel(106, 45), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                       [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(106, 47), UniqueChannel(101, 0)],
                       [UniqueChannel(106, 7), UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                       [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(106, 27), UniqueChannel(101, 0)]]

cat_geometry_tco_1 = CATMap_geo(cat_geometry_tco_1_data)


cat_geometry_tco_2_titles = [ None, "C8(2)", None, None,
                            None, None, "C7(2)", None,
                            "C5(2)", None, None, None,
                            None, None, "C6(2)", None]

cat_geometry_tco_2_data = [[UniqueChannel(101, 0), UniqueChannel(106, 45), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                       [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(106, 47), UniqueChannel(101, 0)],
                       [UniqueChannel(106, 7), UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                       [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(106, 27), UniqueChannel(101, 0)]]

cat_geometry_tco_2 = CATMap_geo(cat_geometry_tco_2_data)


cat_geometry_tco_map = { 1 : cat_geometry_tco_1, 
                     2 : cat_geometry_tco_2}

flat_CAT_tco_geometry_map = {1 : ChannelMap(1, 16, [ wuw.flatten_2D_list(cat_geometry_tco_map[1].data) ]), 
                             2 : ChannelMap(1, 16, [ wuw.flatten_2D_list(cat_geometry_tco_map[2].data) ])}

# Cathode index mapping

#         C1(1)|C1(2)    C5(1)|C3(2)
#         C2(1)|C2(2)    C6(1)|C4(2)  
#         C3(1)|C5(2)    C7(1)|C7(2)
#         C4(1)|C6(2)    C8(1)|C8(2)

cat_index_titles = ["C1(1)","C1(2)", "C5(1)","C5(2)", "C2(1)","C2(2)", "C6(1)","C6(2)","C3(1)","C3(2)", "C7(1)","C7(2)", "C4(1)", "C4(2)","C8(1)","C8(2)"]
        
cat_index_data = [  [UniqueChannel(107, 47),  UniqueChannel(107, 45), UniqueChannel(107,  0),  UniqueChannel(107,  7)],
                    [UniqueChannel(107, 40),  UniqueChannel(107, 42), UniqueChannel(107, 20),  UniqueChannel(107, 27)],
                    [UniqueChannel(107, 46),  UniqueChannel(107, 44), UniqueChannel(107, 30),  UniqueChannel(107, 37)],
                    [UniqueChannel(107, 43),  UniqueChannel(107, 41), UniqueChannel(107, 10),  UniqueChannel(107, 17)]]
                      

cat_index = CATMap_ind(cat_index_data)

cat_index_map = { 1 : cat_index}

flat_MEM_geometry_map = {1 : ChannelMap(1, 16, [ wuw.flatten_2D_list(mem_index_map[1].data) ])}