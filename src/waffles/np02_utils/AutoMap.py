import numpy as np
from waffles.np02_data.ProtoDUNE_VD_maps import cathode_endpoint, membrane_endpoint, pmt_endpoint
from waffles.np02_data.ProtoDUNE_VD_maps import cat_geometry_nontco_data, cat_geometry_tco_data, cat_geometry_nontco_titles, cat_geometry_tco_titles
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_nontco_data, mem_geometry_tco_data, mem_geometry_nontco_titles, mem_geometry_tco_titles
from waffles.np02_data.ProtoDUNE_VD_maps import pmt_geometry_data, pmt_geometry_titles
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap
from typing import List, cast, Union

import math

# This code creates a general mapping of the unique channels matching the
# modules and the inverse map

def _setup_dicts(map_data, map_title):
    map_data = np.array([map_data]).flatten()
    map_title = np.array([map_title]).flatten()
    dict_uniqch_to_module = {str(k): v for k, v in zip(map_data, map_title) if v}
    dict_module_to_uniqch = {v: k for k, v in zip(map_data, map_title) if v}
    return dict_uniqch_to_module, dict_module_to_uniqch

def _merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict

dict_uniqch_to_module = _merge_dicts(_merge_dicts(_setup_dicts(cat_geometry_nontco_data, cat_geometry_nontco_titles)[0],
                                                  _setup_dicts(cat_geometry_tco_data, cat_geometry_tco_titles)[0]),
                                     _merge_dicts(_setup_dicts(mem_geometry_nontco_data, mem_geometry_nontco_titles)[0],
                                                  _setup_dicts(mem_geometry_tco_data, mem_geometry_tco_titles)[0]),
                                     )

dict_uniqch_to_module = _merge_dicts(dict_uniqch_to_module, _setup_dicts(pmt_geometry_data, pmt_geometry_titles)[0])

dict_module_to_uniqch = _merge_dicts(_merge_dicts(_setup_dicts(cat_geometry_nontco_data, cat_geometry_nontco_titles)[1],
                                                  _setup_dicts(cat_geometry_tco_data, cat_geometry_tco_titles)[1]),
                                     _merge_dicts(_setup_dicts(mem_geometry_nontco_data, mem_geometry_nontco_titles)[1],
                                                  _setup_dicts(mem_geometry_tco_data, mem_geometry_tco_titles)[1])
                                     )

dict_module_to_uniqch = _merge_dicts(dict_module_to_uniqch, _setup_dicts(pmt_geometry_data, pmt_geometry_titles)[1])
dict_module_to_uniqch = { k: v for k, v in sorted(dict_module_to_uniqch.items(), key=lambda x: x[0]) }

ordered_modules_membrane = sorted( [ m for m in dict_module_to_uniqch.keys() if m and m.startswith("M") ] )
ordered_modules_cathode = sorted( [ m for m in dict_module_to_uniqch.keys() if m and m.startswith("C") ] )
ordered_modules_pmt = sorted( [ m for m in dict_module_to_uniqch.keys() if m and m.startswith("P") ], key=lambda x: int(x[1:]) )

ordered_channels_membrane:list[int] = [ dict_module_to_uniqch[m].channel for m in ordered_modules_membrane ]
ordered_channels_cathode:list[int] = [ dict_module_to_uniqch[m].channel for m in ordered_modules_cathode ]
ordered_channels_pmt:list[int] = [ dict_module_to_uniqch[m].channel for m in ordered_modules_pmt ]

dict_endpoints_channels_list: dict[int, list[int]]= { membrane_endpoint : ordered_channels_membrane, cathode_endpoint : ordered_channels_cathode, pmt_endpoint : ordered_channels_pmt }

def expand_modules(modules: list[str], available: list[str]) -> list[str]:
    if modules == [""] or len(modules) == 0:
        modules = ["C", "M", "P"]
    expanded = []
    for m in modules:
        if m == "C":
            expanded += [x for x in available if x.startswith("C")]
        elif m == "M":
            expanded += [x for x in available if x.startswith("M") ]
        elif m == "P":
            expanded += [x for x in available if x.startswith("P") ]
        elif m.startswith("C") and ")" not in m:
            # e.g. "C1" -> matches "C1(1)", "C1(2)"
            expanded += [x for x in available if x.startswith(m)  ]
        elif m.startswith("M") and ")" not in m:
            # e.g. "M3" -> matches "M3(1)", "M3(2)"
            expanded += [x for x in available if x.startswith(m)  ]
        else:
            expanded += [m]  # exact match like "C1(1)"
    return list(expanded)

def generate_ChannelMap(channels: Union[List[UniqueChannel], List[str], List[Union[UniqueChannel, str]]], rows:int = 0, cols:int = 0) -> ChannelMap:
    """
    Generates a ChannelMap from a list of UniqueChannel objects.
    If the number of channels is odd, a dummy channel (UniqueChannel(101, 0)) is added to make it even.
    The rows and columns can be specified, but if they do not match the number of channels,
    they will be adjusted to fit all channels.
    If no rows or columns are specified, they will be calculated based on the number of channels.
    The titles for the channels are derived from a predefined mapping.

    Parameters
    ----------
    channels: List[UniqueChannel]
    rows: int, optional
    cols: int, optional

    Returns
    -------
    ChannelMap
    """
    unch: List[UniqueChannel] 
    if all(isinstance(ch, str) for ch in channels):
       channels = expand_modules(cast(List[str], channels), list(dict_module_to_uniqch.keys())) 
    unch = [ channel if isinstance(channel, UniqueChannel) else (dict_module_to_uniqch.get(channel, UniqueChannel(101, 0)) ) for channel in channels]

    titles = [ dict_uniqch_to_module[str(channel)] if channel and str(channel) in dict_uniqch_to_module else f"{channel.endpoint}-{channel.channel}" for channel in unch ]

    if len(unch)%2 != 0 and len(unch) != 1:
        unch.append(UniqueChannel(101, 0))

    n = len(unch)
    if rows and cols:
        if rows*cols != n:
            print("Warning: The specified rows and columns do not match the number of channels. Adjusting to fit all channels.")
            rows = 0
            cols = 0

    if rows:
        cols = math.ceil(n / rows)
    elif cols:
        rows = math.ceil(n / cols)
    else:
        rows = math.isqrt(n)
        cols = rows
        while cols * rows < n:
            cols += 1
    if cols * rows > n:
        unch.extend( [ UniqueChannel(101,0) ] * (cols*rows - n) )
    channels_shaped = np.array(unch).reshape(rows, cols)
    channelsmap:List[List[UniqueChannel]] = [ list(row) for row in channels_shaped ]
    output = ChannelMap(rows, cols, channelsmap)
    output.titles = titles
    return output

def strUch(endpoint:int, channel: int):
    return str(UniqueChannel(endpoint, channel))

def getModuleName(ep: int, ch: int) -> str:
    return dict_uniqch_to_module.get(strUch(ep,ch), f"{ep}-{ch}")

def getEndpointChannelFromModule(module_name: str) -> tuple[int, int]:
    uch:UniqueChannel = dict_module_to_uniqch.get(module_name, UniqueChannel(-1, -1))
    return uch.endpoint, uch.channel


