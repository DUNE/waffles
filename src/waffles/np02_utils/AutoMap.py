import numpy as np
from waffles.np02_data.ProtoDUNE_VD_maps import cat_geometry_nontco_data, cat_geometry_tco_data, cat_geometry_nontco_titles, cat_geometry_tco_titles
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_nontco_data, mem_geometry_tco_data, mem_geometry_nontco_titles, mem_geometry_tco_titles
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
                                                  _setup_dicts(mem_geometry_tco_data, mem_geometry_tco_titles)[0])
                                     )

dict_module_to_uniqch = _merge_dicts(_merge_dicts(_setup_dicts(cat_geometry_nontco_data, cat_geometry_nontco_titles)[1],
                                                  _setup_dicts(cat_geometry_tco_data, cat_geometry_tco_titles)[1]),
                                     _merge_dicts(_setup_dicts(mem_geometry_nontco_data, mem_geometry_nontco_titles)[1],
                                                  _setup_dicts(mem_geometry_tco_data, mem_geometry_tco_titles)[1])
                                     )

ordered_modules_membrane = sorted( [ m for m in dict_module_to_uniqch.keys() if m and m.startswith("M") ] )
ordered_modules_cathode = sorted( [ m for m in dict_module_to_uniqch.keys() if m and m.startswith("C") ] )
ordered_channels_membrane = [ dict_module_to_uniqch[m].channel for m in ordered_modules_membrane ]
ordered_channels_cathode = [ dict_module_to_uniqch[m].channel for m in ordered_modules_cathode ]


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
    unch = [ channel if isinstance(channel, UniqueChannel) else (dict_module_to_uniqch[channel] if channel[0] in ["M", "C"] else dict_uniqch_to_module[channel]) for channel in channels]

    titles = [ dict_uniqch_to_module[str(channel)] for channel in unch ]

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
        while n % rows != 0:
            rows -= 1
        cols = n // rows
    channels_shaped = np.array(unch).reshape(rows, cols)
    channelsmap:List[List[UniqueChannel]] = [ list(row) for row in channels_shaped ]
    output = ChannelMap(rows, cols, channelsmap)
    output.titles = titles
    return output

def strUch(endpoint:int, channel: int):
    return str(UniqueChannel(endpoint, channel))


