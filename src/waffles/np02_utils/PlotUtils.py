import numpy as np
from pathlib import Path
import plotly.subplots as psu
import logging
from typing import List, Union, Set

from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.Map import Map
from waffles.np02_data_classes.CATMap import CATMap_geo
from waffles.np02_data_classes.MEMMap import MEMMap_geo
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map
from waffles.np02_data.ProtoDUNE_VD_maps import cat_geometry_map
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.np02_utils.AutoMap import generate_ChannelMap

def np02_resolve_detectors(wfset, detectors: List[str] | List[UniqueChannel] | List[UniqueChannel | str ], rows=0, cols=1) -> dict[str, List[Union[ChannelWsGrid, Map]]]:

    detmap = generate_ChannelMap(channels=detectors, rows=rows, cols=cols)
    return dict( 
        Custom=[ChannelWsGrid(detmap, wfset), detmap]
    )


def np02_gen_grids(wfset, detector:str | List[str] | List[UniqueChannel] | List[UniqueChannel | str ] = "", rows=0, cols=0) -> dict[str, List[Union[ChannelWsGrid, Map]]]:

    if isinstance(detector, str):
        if detector == 'VD_Membrane_PDS':
            return dict(
                TCO=[
                    ChannelWsGrid(mem_geometry_map[2], wfset,
                                  bins_number=115,
                                  domain=np.array([-1e4, 5e4]),
                                  variable="integral"),
                    mem_geometry_map[2]
                ],
                nTCO=[
                    ChannelWsGrid(mem_geometry_map[1], wfset,
                                  bins_number=115,
                                  domain=np.array([-1e4, 5e4]),
                                  variable="integral"),
                    mem_geometry_map[1]
                ]
            )
        elif detector == 'VD_Cathode_PDS':
            return dict(
                TCO=[
                    ChannelWsGrid(cat_geometry_map[2], wfset,
                                  bins_number=115,
                                  domain=np.array([-1e4, 5e4]),
                                  variable="integral"),
                    cat_geometry_map[2]
                ],
                nTCO=[
                    ChannelWsGrid(cat_geometry_map[1], wfset,
                                  bins_number=115,
                                  domain=np.array([-1e4, 5e4]),
                                  variable="integral"),
                    cat_geometry_map[1]
                ]
            )
        else:
            detectors = [detector]
    else:
        detectors = detector
    if isinstance(detectors, list):
        return np02_resolve_detectors(wfset, detectors, rows, cols)

    raise ValueError(f"Could not resolve detector: {detector} or {detectors}")


def plot_grid(chgrid: ChannelWsGrid, detmap:Map, title:str = "", html: Path | None = None, detector:str | List[str] = ""):

    rows, cols= detmap.rows, detmap.columns

    subtitles = chgrid.titles

    fig = psu.make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subtitles,
        shared_xaxes=True,
        shared_yaxes=True
    )
    
    plot_ChannelWsGrid( chgrid, figure=fig, share_x_scale=True,
                       share_y_scale=True, mode="overlay", wfs_per_axes=50)
    fig.update_layout(title=title, template="plotly_white",
                      width=1000, height=800, showlegend=True)
    if html:
        fig.write_html(html.as_posix())
        logging.info("💾 %s", html)
    else:
        fig.show()
# ╰────────────────────────────────────────────────────────────────────────────╯

