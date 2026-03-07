from typing import Callable, Optional, Union
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import plotly.graph_objects as go
import matplotlib.image as mpimg
import re
import os

from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.np02_utils.AutoMap import getModuleName, dict_module_to_uniqch


def expand_modules(modules: list[str], available: list[str]) -> list[str]:
    expanded = set()
    for m in modules:
        if m == "C":
            expanded.update(x for x in available if x.startswith("C"))
        elif m == "M":
            expanded.update(x for x in available if x.startswith("M"))
        elif m.startswith("C") and ")" not in m:
            # e.g. "C1" -> matches "C1(1)", "C1(2)"
            expanded.update(x for x in available if x.startswith(m) )
        elif m.startswith("M") and ")" not in m:
            # e.g. "M3" -> matches "M3(1)", "M3(2)"
            expanded.update(x for x in available if x.startswith(m) )
        else:
            expanded.add(m)  # exact match like "C1(1)"
    return list(expanded)


def load_database():
    ## Loading xenon data base
    url = "https://docs.google.com/spreadsheets/d/1WtGkVkxM_4X4zdqMdCCIVsJ2cng8doSG2NC95fXmv-k/export?format=xlsx"
    df = pd.read_excel(url, sheet_name="HV_studies")
    xedb = df.drop(columns=['date', 'time'], axis=1)
    xedb['Efield'] = xedb['HV'] / 350.  # Assuming a drift distance of 3500 cm.
    return xedb


def load_fit_results(path_to_data:str, sufix:str = 'conv', optional_filelist = []) -> pd.DataFrame:
    if optional_filelist:
        files = optional_filelist.copy()
    else:
        files = glob(f"{path_to_data}/{sufix}fit_output_*.csv")
    if not files:
        raise ValueError(f"No files found in {path_to_data} with sufix {sufix}. Please check the path and sufix.")

    listdf = []
    for file in tqdm(files, desc="Loading fit results"):
        dftmp = pd.read_csv(file)
        listdf.append(dftmp)

    print("All files loaded, concatenating into a single DataFrame...")
    df = pd.concat(listdf, ignore_index=True)
    df["timestamp"] = df['timestamp[ticks]'] * 16.e-9
    df["time"] = pd.to_datetime(df["timestamp"], unit='s')
    df['module'] = df.apply( lambda x: getModuleName(int(x['ep']), int(x['ch'])), axis = 1)
    df = df.sort_values(by='time')


    return df.reset_index(drop=True)

def getDataFrameCh(df:pd.DataFrame, endpoint:int, channel:Union[int,list[int]]) -> pd.DataFrame:
    if isinstance(channel, int):
        dfret = df.loc[(df['ep'] == endpoint) & (df['ch'] == channel)]
    else:
        dfret = df.loc[(df['ep'] == endpoint) & (df['ch'].isin(channel))]
    return dfret

def getDataFrameModule(df:pd.DataFrame, module:Union[str,list[str]]) -> pd.DataFrame:
    if isinstance(module, str):
        dfret = df.loc[df['module'] == module]
    else:
        dfret = df.loc[df['module'].isin(module)]
    return dfret


def plot_vs_time_per_channel(df:pd.DataFrame,
                             endpoint:int=0,
                             channel:int=0,
                             module = '',
                             y='t3[ns]',
                             selection:Optional[Callable]=None,
                             ylabel=r'$\tau_\text{slow}$ [ns]',
                             label:str = '',
                             showhours = False,
                             xlim = None,
                             ):
    """
    Plots a given variable vs time for a specific endpoint and channel, or for a specific module if provided.
    Parameters:
        - df: DataFrame containing the data to plot.
        - endpoint: Endpoint number to filter the data.
        - channel: Channel number to filter the data.
        - module: Module name to filter the data. If provided, endpoint and channel filters will be ignored.
        - y: Column name of the variable to plot on the y-axis.
        - selection: Optional function to further filter the DataFrame before plotting.
        - ylabel: Label for the y-axis.
        - label: Label for the plot legend.
        - xlim: Tuple specifying the limits for the x-axis (start, end) in datetime format.
          Example: xlim = (pd.to_datetime('2024-07-29 10:00'), pd.to_datetime('2024-07-29 12:00'))
                or xlim = (dt.datetime(2024,7,29,10), dt.datetime(2024,7,29,12))
    """

    if selection:
        df = selection(df)
    if module:
        if module not in df['module'].unique():
            print(f"Module {module} not found in the data. Skipping plot.")
            return
        df_ch = df[df['module'] == module]
        unch = dict_module_to_uniqch.get(module, UniqueChannel(endpoint, channel))
        endpoint, channel = unch.endpoint, unch.channel
    else:
        df_ch = getDataFrameCh(df, endpoint, channel)

    if df_ch.empty:
        print(f"No data found for endpoint {endpoint} and channel {channel}. Skipping plot.")
        return



    plt.errorbar(df_ch['time'], df_ch[y], fmt='o', markersize=5, linewidth=1.5, label=label)

    plt.ylabel(ylabel)
    plt.title(f'{getModuleName(endpoint, channel)}: {endpoint}-{channel}')
    plt.grid(True, linestyle='--', alpha=0.5)

    ax = plt.gca()
    timefmt = '%Y-%m-%d %H:%M' if showhours else '%Y-%m-%d'
    xaxis = getattr(ax, 'xaxis', None)
    if xaxis:
        xaxis.set_major_formatter(mdate.DateFormatter(timefmt))
    plt.xticks(rotation=25)

    # Example xlim:
    # # xlim = (dt.datetime(2024,7,29,10), dt.datetime(2024,7,29,12))
    if xlim:
        plt.xlim(xlim)

    if label:
        plt.legend()
    plt.tight_layout()

def iplot_vs_time(df:pd.DataFrame, fig:go.Figure, x='time', y='tau_s', name='', yaxis_range =[None,None], selection=None):
    if selection:
        df = selection(df)
    if not fig:
        fig = go.Figure()
    list_of_modules = df['module'].unique()
    if len(list_of_modules) == 1 and not name:
        unch = dict_module_to_uniqch[list_of_modules[0]]
        ep, ch = unch.endpoint, unch.channel
        name = f"{list_of_modules[0]}: ({ep}-{ch})"
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='markers', name=name))
    return fig


def execute_by_module(df:pd.DataFrame, func_ch:Callable, modules = None, **kwargs):
    available = df['module'].unique()

    if not modules:
        modules_to_run = available
    else:
        if isinstance(modules, str):
            modules = [modules]
        modules_to_run = expand_modules(list(modules), list(available))
    
    for module in sorted(modules_to_run):
        if module not in available:
            continue
        df_ch = df[df['module'] == module]
        func_ch(df_ch, **kwargs)

#Loader for conv/deconv plots


#Loader for conv/deconv plots
def load_module_images(path_to_data: str, run=None, modules:Union[None, list, str]=None):
    images = {}

    if modules == 'grid' or modules == ['grid']:
        png_files = glob(f"{path_to_data}/*_grid_run0{run}_*.png")
        if len(png_files) == 0:
            print(f"No grid images found for run {run} in {path_to_data}. Please check the path and run number.")
            return {}
    elif modules is None or modules == [None]:
        all_png = glob(f"{path_to_data}/*_plot_run0{run}*.png")
        png_files = all_png
    else:
        if isinstance(modules, str):
            modules = [modules]
        all_png = glob(f"{path_to_data}/*_plot_run0{run}*.png")
        available = set()
        pattern = re.compile(r"(C\d+_\d+|M\d+_\d+)")

        for file in all_png:
            match = pattern.search(os.path.basename(file))
            if match:
                mod = match.group(1)           
                mod = mod.replace("_", "(") + ")" 
                available.add(mod)

        expanded_modules = expand_modules(modules, list(available))

        expanded_filename_format = [
            m.replace("(", "_").replace(")", "")
            for m in expanded_modules
        ]

        png_files = [
            file for file in all_png
            if any(mod in file for mod in expanded_filename_format)
        ]
    if len(png_files) == 0:
        print(f"No images found for modules {modules} in run {run} at {path_to_data}. Please check the path, run number, and module names.")
        return {}

    for file in png_files:
        images[file] = mpimg.imread(file)

    return images

def show_images_grid(image_dict, forced_figsize=None):
    if not image_dict:  
        return

    for path, image in image_dict.items():
        figsize = (32, 32) if 'grid' in path else (14, 6)
        if forced_figsize is not None:
            figsize = forced_figsize
        plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.gca().axis("off")
        plt.tight_layout()
        plt.show()
