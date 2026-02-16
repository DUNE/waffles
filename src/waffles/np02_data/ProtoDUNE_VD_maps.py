from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap

from waffles.np02_data_classes.MEMMap import MEMMap_geo
from waffles.np02_data_classes.MEMMap import MEMMap_ind

from waffles.np02_data_classes.CATMap import CATMap_geo
from waffles.np02_data_classes.CATMap import CATMap_ind

import waffles.utils.wf_maps_utils as wuw

cathode_endpoint:int = 106
membrane_endpoint:int = 107
pmt_endpoint:int = 110

# ---------------------------MEMBRANES MAPPING---------------------------------

# Membranes geometrical mapping

#         M1(1)|M1(2)    M3(1)|M3(2)
# NO-TCO  M2(1)|M2(2)    M4(1)|M4(2)  TCO
#         M5(1)|M5(2)    M7(1)|M7(2)
#         M6(1)|M6(2)    M8(1)|M8(2)

# * M1(1)|M1(2) representes the first (1) and second (2) channels of membrane M1.

# --------------------------- Membrane NO-TCO ---------------------------------

mem_geometry_nontco_titles = ["M1(1)","M1(2)", "M2(1)","M2(2)", "M5(1)","M5(2)", "M6(1)","M6(2)"]
        
mem_geometry_nontco_data = [  [UniqueChannel(membrane_endpoint, 47),  UniqueChannel(membrane_endpoint, 45) ],
                             [UniqueChannel(membrane_endpoint, 40),  UniqueChannel(membrane_endpoint, 42) ],
                             [UniqueChannel(membrane_endpoint,  7),  UniqueChannel(membrane_endpoint,  0) ],
                             [UniqueChannel(membrane_endpoint, 27),  UniqueChannel(membrane_endpoint, 20) ]]

mem_geometry_nontco = MEMMap_geo(mem_geometry_nontco_data)
mem_geometry_nontco.titles = mem_geometry_nontco_titles

# --------------------------- Membrane TCO ---------------------------------

mem_geometry_tco_titles = ["M3(1)","M3(2)", "M4(1)","M4(2)", "M7(1)","M7(2)", "M8(1)","M8(2)"]

mem_geometry_tco_data = [    [UniqueChannel(membrane_endpoint, 46  ),  UniqueChannel(membrane_endpoint, 44)],
                             [UniqueChannel(membrane_endpoint, 43  ),  UniqueChannel(membrane_endpoint, 41)],
                             [UniqueChannel(membrane_endpoint, 37  ),  UniqueChannel(membrane_endpoint, 30)],
                             [UniqueChannel(membrane_endpoint, 17  ),  UniqueChannel(membrane_endpoint, 10)]]

mem_geometry_tco = MEMMap_geo(mem_geometry_tco_data)
mem_geometry_tco.  titles = mem_geometry_tco_titles


mem_geometry_map = { 1 : mem_geometry_nontco, 
                     2 : mem_geometry_tco}

flat_MEM_geometry_map = {1 : ChannelMap(1, 8, [ wuw.flatten_2D_list(mem_geometry_map[1].data) ]), 
                         2 : ChannelMap(1, 8, [ wuw.flatten_2D_list(mem_geometry_map[2].data) ])}


# Membranes index mapping

#         M1(1)|M1(2)    M5(1)|M3(2)
#         M2(1)|M2(2)    M6(1)|M4(2)  
#         M3(1)|M5(2)    M7(1)|M7(2)
#         M4(1)|M6(2)    M8(1)|M8(2)

mem_index_titles = ["M1(1)","M1(2)", "M5(1)","M5(2)", "M2(1)","M2(2)", "M6(1)","M6(2)","M3(1)","M3(2)", "M7(1)","M7(2)", "M4(1)", "M4(2)","M8(1)","M8(2)"]
        
mem_index_data = [  [UniqueChannel(membrane_endpoint, 47),  UniqueChannel(membrane_endpoint, 45), UniqueChannel(membrane_endpoint,  7),  UniqueChannel(membrane_endpoint,  0)],
                    [UniqueChannel(membrane_endpoint, 40),  UniqueChannel(membrane_endpoint, 42), UniqueChannel(membrane_endpoint, 27),  UniqueChannel(membrane_endpoint, 20)],
                    [UniqueChannel(membrane_endpoint, 46),  UniqueChannel(membrane_endpoint, 44), UniqueChannel(membrane_endpoint, 37),  UniqueChannel(membrane_endpoint, 30)],
                    [UniqueChannel(membrane_endpoint, 43),  UniqueChannel(membrane_endpoint, 41), UniqueChannel(membrane_endpoint, 17),  UniqueChannel(membrane_endpoint, 10)]]
                      

mem_index = MEMMap_ind(mem_index_data)

mem_index_map = { 1 : mem_index}

flat_MEM_geometry_map = {1 : ChannelMap(1, 16, [ wuw.flatten_2D_list(mem_index_map[1].data) ])}

# -----------------------------CATHODE MAPPING---------------------------------

# Cathode geometrical mapping
        
#               C4(1)           |      C8(1)
#               C4(2)           |      C8(1)
#                         C3(1) |           C7(1)           
#                         C3(2) |           C7(1)
# NO-TCO   C1(1)                | C5(1)                 TCO
#          C1(2)                | C5(1)          
#                    C2(1)      |           C6(1)
#                    C2(2)      |           C6(1)

# * C1(1) representes the first channel (1) of cathode C1.

# ------------------------- Cathode NO-TCO ----------------------------

cat_geometry_nontco_titles = [None, "C4(1)", None, None,
                             None, "C4(2)", None, None,
                             None, None, None, "C3(1)", 
                             None, None, None, "C3(2)",
                             "C1(1)", None, None, None, 
                             "C1(2)", None, None, None, 
                             None, None, "C2(1)", None,
                             None, None, "C2(2)", None]

cat_geometry_nontco_data = [[UniqueChannel(101, 45), UniqueChannel(cathode_endpoint, 36), UniqueChannel(101, 0),  UniqueChannel(101, 0)],
                            [UniqueChannel(101, 45), UniqueChannel(cathode_endpoint, 37), UniqueChannel(101, 0),  UniqueChannel(101, 0)],
                            [UniqueChannel(101, 0),  UniqueChannel(101, 0),  UniqueChannel(101, 0),  UniqueChannel(cathode_endpoint, 34)],
                            [UniqueChannel(101, 0),  UniqueChannel(101, 0),  UniqueChannel(101, 0),  UniqueChannel(cathode_endpoint, 35)],
                            [UniqueChannel(cathode_endpoint, 32), UniqueChannel(101, 0),  UniqueChannel(101, 0),  UniqueChannel(101, 0)],
                            [UniqueChannel(cathode_endpoint, 33), UniqueChannel(101, 0),  UniqueChannel(101, 0),  UniqueChannel(101, 0)],
                            [UniqueChannel(101, 0),  UniqueChannel(101, 0),  UniqueChannel(cathode_endpoint, 30), UniqueChannel(101, 0)],
                            [UniqueChannel(101, 0),  UniqueChannel(101, 0),  UniqueChannel(cathode_endpoint, 31), UniqueChannel(101, 0)]]

cat_geometry_nontco = CATMap_geo(cat_geometry_nontco_data)
cat_geometry_nontco.titles = cat_geometry_nontco_titles

# ------------------------- Cathode TCO ----------------------------

cat_geometry_tco_titles = [None, "C8(1)", None, None,
                           None, "C8(2)", None, None,
                           None, None, "C7(1)", None,
                           None, None, "C7(2)", None,
                           "C5(1)", None, None, None,
                           "C5(2)", None, None, None,
                           None, None, "C6(1)", None,
                           None, None, "C6(2)", None]

cat_geometry_tco_data = [[UniqueChannel(101, 0), UniqueChannel(cathode_endpoint, 4), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                         [UniqueChannel(101, 0), UniqueChannel(cathode_endpoint, 6), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                         [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(cathode_endpoint, 5), UniqueChannel(101, 0)],
                         [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(cathode_endpoint, 7), UniqueChannel(101, 0)],
                         [UniqueChannel(cathode_endpoint, 0), UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                         [UniqueChannel(cathode_endpoint, 2), UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(101, 0)],
                         [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(cathode_endpoint, 1), UniqueChannel(101, 0)],
                         [UniqueChannel(101, 0), UniqueChannel(101, 0), UniqueChannel(cathode_endpoint, 3), UniqueChannel(101, 0)]]

cat_geometry_tco = CATMap_geo(cat_geometry_tco_data)
cat_geometry_tco.titles = cat_geometry_tco_titles


cat_geometry_map = { 1 : cat_geometry_nontco, 
                     2 : cat_geometry_tco}

flat_CAT_tco_geometry_map = {1 : ChannelMap(1, 32, [ wuw.flatten_2D_list(cat_geometry_map[1].data) ]), 
                             2 : ChannelMap(1, 32, [ wuw.flatten_2D_list(cat_geometry_map[2].data) ])}

# Cathode index mapping

#         C1(1)|C1(2)    C5(1)|C5(2)
#         C2(1)|C2(2)    C6(1)|C6(2)  
#         C3(1)|C3(2)    C7(1)|C7(2)
#         C4(1)|C4(2)    C8(1)|C8(2)

cat_index_titles = ["C1(1)","C1(2)", "C5(1)","C5(2)", "C2(1)","C2(2)", "C6(1)","C6(2)","C3(1)","C3(2)", "C7(1)","C7(2)", "C4(1)", "C4(2)","C8(1)","C8(2)"]
        
cat_index_data = [  [UniqueChannel(cathode_endpoint, 32),  UniqueChannel(cathode_endpoint, 33), UniqueChannel(cathode_endpoint,  0),  UniqueChannel(cathode_endpoint,  2)],
                    [UniqueChannel(cathode_endpoint, 30),  UniqueChannel(cathode_endpoint, 31), UniqueChannel(cathode_endpoint,  1),  UniqueChannel(cathode_endpoint,  3)],
                    [UniqueChannel(cathode_endpoint, 34),  UniqueChannel(cathode_endpoint, 35), UniqueChannel(cathode_endpoint,  5),  UniqueChannel(cathode_endpoint,  7)],
                    [UniqueChannel(cathode_endpoint, 36),  UniqueChannel(cathode_endpoint, 37), UniqueChannel(cathode_endpoint,  4),  UniqueChannel(cathode_endpoint,  6)]]
                      

cat_index = CATMap_ind(cat_index_data)

cat_index_map = { 1 : cat_index}

flat_MEM_geometry_map = {1 : ChannelMap(1, 16, [ wuw.flatten_2D_list(mem_index_map[1].data) ])}


pmt_geometry_data = [UniqueChannel(pmt_endpoint, 0),  UniqueChannel(pmt_endpoint, 2),
                     UniqueChannel(pmt_endpoint, 6),  UniqueChannel(pmt_endpoint, 7),
                     UniqueChannel(pmt_endpoint, 10), UniqueChannel(pmt_endpoint, 12),
                     UniqueChannel(pmt_endpoint, 14), UniqueChannel(pmt_endpoint, 16),
                     UniqueChannel(pmt_endpoint, 17), UniqueChannel(pmt_endpoint, 20),
                     UniqueChannel(pmt_endpoint, 22), UniqueChannel(pmt_endpoint, 24),
                     UniqueChannel(pmt_endpoint, 26), UniqueChannel(pmt_endpoint, 30),
                     UniqueChannel(pmt_endpoint, 32), UniqueChannel(pmt_endpoint, 34),
                     UniqueChannel(pmt_endpoint, 36), UniqueChannel(pmt_endpoint, 37),
                     UniqueChannel(pmt_endpoint, 40), UniqueChannel(pmt_endpoint, 42),
                     UniqueChannel(pmt_endpoint, 44), UniqueChannel(pmt_endpoint, 46)]

pmt_geometry_titles = ["P5",  "P37",
                       "P7 ", "P29",
                       "P31", "P41",
                       "P15", "P35",
                       "P38", "P21",
                       "P14", "P25",
                       "P6",  "P20",
                       "P26", "P32",
                       "P28", "P12",
                       "P34", "P16",
                       "P19", "P17"]


pmt_geometry = MEMMap_ind([pmt_geometry_data], r=1, c=len(pmt_geometry_data))
pmt_geometry.titles = pmt_geometry_titles

