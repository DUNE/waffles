from typing import Callable, Optional, Union
import pandas as pd
from pandas._typing import MergeHow
from glob import glob
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import plotly.graph_objects as go
import matplotlib.image as mpimg
import re
import os
import subprocess
from io import StringIO

from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.np02_utils.AutoMap import getModuleName, dict_module_to_uniqch, expand_modules
from waffles.np02_utils.PlotUtils import endpoint_channel_colormap, modules_colormap

dict_axes_label_to_data = {
        'time': r'Time CET',
        't1[ns]': r'$\tau_\text{fast}\, \text{[ns]}$',
        't3[ns]': r'$\tau_\text{slow}\, \text{[ns]}$',
        'td[ns]': r'$\tau_\text{delay}\, \text{[ns]}$',
        'fp' : r'$A_\text{fast}$',
        'fs' : r'$A_\text{slow}$',
        't0[ns]' : r'$t_0\, \text{[ns]}$',
        'sigma' : r'$\sigma\, \text{[ns]}$',
        'HV' : 'Voltage [kV]',
        'Efield' : 'Electric field [kV/cm]',
        'ppm' : 'Concentration [ppm]',
        'chi2' : r'$\chi^2$',
        'nselected' : 'Number of waveforms selected',
}

endpoint_channel_symbol = {}
for ep, chc in endpoint_channel_colormap.items():
    endpoint_channel_symbol[ep] = endpoint_channel_symbol.get(ep, {})
    for ch, _ in chc.items():
        if ep != 110:
            endpoint_channel_symbol[ep][ch] = 'cross' if getModuleName(ep, ch)[3]=='1' else 'circle'
        else:
            endpoint_channel_symbol[ep][ch] = 'diamond'

style_map_matplotlib = {
    "1": {"marker": "+" },
    "2": {"marker": "." }
}

def get_style_module(modulename:str) -> dict:
    ch_num = modulename[3:-1] # "1" or "2"
    return {
        "color": modules_colormap[modulename],
        **style_map_matplotlib[ch_num]
    }
def define_labels(x, y):
    return dict_axes_label_to_data.get(x, x), dict_axes_label_to_data.get(y, y)

def make_relative_cols(df:pd.DataFrame, relative_to = "ppm", ref_value = 0, cols:list[str] = ['t1[ns]', 't3[ns]', 'td[ns]', 'fp', 'fs', 't0[ns]', 'sigma', 'chi2']) -> pd.DataFrame:
    for col in cols:
        baseline = df[df[relative_to] == ref_value].groupby('module')[col].mean()
        df[f'{col} relative'] = df[col] / df['module'].map(baseline) # type: ignore
    return df


def load_results(analysisname:str, path_with_analysis:str, types:list[str] = ['convolution', 'deconvolution'], load_hv:bool = False, how:MergeHow = 'inner') -> dict[str, pd.DataFrame]:
    ret = {}
    xedb = load_database(load_hv=load_hv)
    for typeofana in types:
        print(f"Loading {typeofana.upper()} ...")
        sufix = 'conv' if typeofana == "convolution" else 'deconv'
        dfres = load_fit_results(path_to_data=f"{path_with_analysis}/{analysisname}/{typeofana}", sufix=sufix, exists_ok=True)
        if dfres.empty:
            print(f"No fit results found for {typeofana} in {path_with_analysis}/{analysisname}/{typeofana}. Skipping this type.")
            continue
        df = pd.merge(dfres, xedb, on='run', how=how)
        c1 = set(dfres['run'].unique())
        c2 = set(xedb['run'].unique())

        if c1 - c2:
            print("Runs not in the database:", ','.join(list(map(str, c1 - c2))))
        if c2 - c1:
            print("Missing runs to process:", ','.join(list(map(str, c2 - c1))))
        ret[sufix] = df.copy()
        print()

    runsdeconv = set(ret['deconv']['run'].unique()) if 'deconv' in ret else set()
    runsconv = set(ret['conv']['run'].unique()) if 'conv' in ret else set()
    if runsdeconv - runsconv:
        print("Runs with deconv results but no conv results:", ','.join(list(map(str, runsdeconv - runsconv))))
    if runsconv - runsdeconv:
        print("Runs with conv results but no deconv results:", ','.join(list(map(str, runsconv - runsdeconv))))
    return ret

def load_database(load_hv:bool = False) -> pd.DataFrame:
    ## Loading xenon data base
    if load_hv:
        url = "https://docs.google.com/spreadsheets/d/1WtGkVkxM_4X4zdqMdCCIVsJ2cng8doSG2NC95fXmv-k/export?format=xlsx"
        df = pd.read_excel(url, sheet_name="HV_studies")
    else:
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXuwLxht-9hnNHvNCYqQRQulTMk7ymE1gRTpo1LGHfey4HGaGqz0CqIFo8IBYevtVXsqj_9aitUD5t/pub?output=xlsx"
        df = pd.read_excel(url, sheet_name="Xe-PDS-only")
    xedb = df.drop(columns=['date', 'time'], axis=1)
    xedb['Efield'] = xedb['HV'] / 350.  # Assuming a drift distance of 3500 cm.
    return xedb

def load_injections():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTXuwLxht-9hnNHvNCYqQRQulTMk7ymE1gRTpo1LGHfey4HGaGqz0CqIFo8IBYevtVXsqj_9aitUD5t/pub?output=xlsx"
    df = pd.read_excel(url, sheet_name="Injections")
    df['start_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['start'].astype(str))
    df['end_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['end'].astype(str))
    return df
    


def load_fit_results(path_to_data:str, sufix:str = 'conv', optional_filelist = [], exists_ok:bool = False) -> pd.DataFrame:
    # TODO: Add command bash to concaternate all files first.
    if not Path(path_to_data).is_dir():
        if exists_ok:
            print(f"Provided path {path_to_data} is not a valid directory. Returning empty DataFrame.")
            return pd.DataFrame()
        raise ValueError(f"Provided path {path_to_data} is not a valid directory. Please check the path and try again.")
    if optional_filelist:
        files = optional_filelist.copy()
        listdf = []
        for file in tqdm(files, desc="Loading fit results"):
            dftmp = pd.read_csv(file)
            listdf.append(dftmp)
        df = pd.concat(listdf, ignore_index=True)
        print("All files loaded, concatenating into a single DataFrame...")
    else:
        print("Running awk command to concatenate CSV files while skipping headers...")
        result = subprocess.run("awk 'FNR==1 && NR!=1 {next} {print}'"+ f" {path_to_data}/{sufix}fit_output_*.csv", 
                                shell=True, capture_output=True, text=True)
        # print("Command executed, loading output into DataFrame...")
        df = pd.read_csv(StringIO(result.stdout))
        print("Loaded DataFrame with shape:", df.shape)

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

def draw_injections_matplotplib(df:pd.DataFrame,):
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    # matplotlib stores times as float internally, so convert
    xmin = mdate.num2date(xmin).replace(tzinfo=None)
    xmax = mdate.num2date(xmax).replace(tzinfo=None)
    for _, row in df.iterrows():
        if row['end_time'] < xmin or row['start_time'] > xmax:
            continue  # outside the plot range, skip
        plt.axvspan(row.loc['start_time'], row.loc['end_time'], alpha=0.1, color='grey')
        plt.text(
            row['end_time']+pd.Timedelta("00:05:00"), plt.gca().get_ylim()[1]*0.995,
            f"{row['ppm']} ppm",
            fontsize=12, color='black', va='top'
        )

def draw_injections_plotly(df: pd.DataFrame, fig: go.Figure):
   # Try to get range from layout, otherwise infer from the data

    if 'axis' in fig.layout and fig.layout['axis']['range']:
        xmin, xmax = fig.layout['axis']['range']
    else:
        x_values = []
        for trace in fig.data:
            if (v:= getattr(trace, 'x', None)) is not None:
                x_values.extend(v)
        if not x_values:
            return  # no data, nothing to do
        xmin, xmax = min(x_values), max(x_values)
    for _, row in df.iterrows():
        if row['end_time'] < xmin or row['start_time'] > xmax:
            continue  # outside the plot range, skip
        fig.add_vrect(
            x0=row['start_time'], x1=row['end_time'],
            fillcolor='grey', opacity=0.1, line_width=0,
        )
        fig.add_annotation(
            x=row['end_time'] + pd.Timedelta("00:05:00"),
            y=1,  # in paper coordinates (top of plot)
            yref='paper',
            text=f"{row['ppm']} ppm",
            showarrow=False,
            font=dict(size=12, color='black'),
            yanchor='top'
        )

def plot_vs_time_per_channel(df:pd.DataFrame,
                             endpoint:int=0,
                             channel:int=0,
                             module = '',
                             y='t3[ns]',
                             selection:Optional[Callable]=None,
                             label:str = '',
                             showhours = False,
                             xlim = None,
                             dotitle=False,
                             autolabel = False,
                             legendoutside = False,
                             df_injections:Optional[pd.DataFrame] = None
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
    elif endpoint != 0:
        df_ch = getDataFrameCh(df, endpoint, channel)
        module = getModuleName(endpoint, channel)
    else:
        if len(df['module'].unique()) == 1:
            module = df['module'].unique()[0]
            df_ch = df.copy()
        else:
            raise ValueError("Multiple modules found in the data. Please specify a module or endpoint/channel to plot.")
    unch = dict_module_to_uniqch.get(module, UniqueChannel(endpoint, channel))
    endpoint, channel = unch.endpoint, unch.channel

    if df_ch.empty:
        print(f"No data found for endpoint {endpoint} and channel {channel}. Skipping plot.")
        return

    modulename = getModuleName(endpoint, channel)
    if autolabel:
        label = f"{modulename}: {endpoint}-{channel}"
    plt.errorbar(df_ch['time'], df_ch[y], ls='' , markersize=10, **get_style_module(modulename), label=label)

    xlabel, ylabel = define_labels('time', y)
    plt.ylabel(ylabel)
    if dotitle:
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
        if not legendoutside:
            plt.legend()
        else:
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncols=2)

    if df_injections is not None:
        draw_injections_matplotplib(df_injections)
    plt.tight_layout()

def iplot(df:pd.DataFrame, fig:go.Figure, x='time', y='tau_s', name='', yaxis_range =[None,None], selection=None, df_injections = None, width=1200, heigth=600):
    if selection:
        df = selection(df)
    if not fig:
        fig = go.Figure()
    list_of_modules = df['module'].unique()
    unch = dict_module_to_uniqch[list_of_modules[0]]
    ep, ch = unch.endpoint, unch.channel
    if len(list_of_modules) == 1 and not name:
        name = f"{list_of_modules[0]}: ({ep}-{ch})"

    dictmarker = dict(size=8)
    if ep in endpoint_channel_symbol and ch in endpoint_channel_symbol[ep]:
        dictmarker['symbol'] = endpoint_channel_symbol[ep][ch]
    if ep in endpoint_channel_colormap and ch in endpoint_channel_colormap[ep]:
        dictmarker['color'] = endpoint_channel_colormap[ep][ch]
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='markers', name=name, marker=dictmarker))
    labelx, labely = define_labels(x, y)
    fig.update_layout(template="plotly_white",
                  width=width, height=heigth, showlegend=True,
                      xaxis_title=labelx,
                      yaxis_title=labely,
                 )

    if yaxis_range != [None, None]:
        fig.update_yaxes(range=yaxis_range)

    if df_injections is not None:
        draw_injections_plotly(df_injections, fig)

    return fig


def execute_by_module(df:pd.DataFrame, func_ch:Callable, modules = None, **kwargs):
    available = df['module'].unique()
    df_injections = kwargs.pop('df_injections', None)

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

    if 'x' in kwargs and  kwargs['x'] in ['time', 'timestamp[ticks]', 'timestamp']:
        if df_injections is not None:
            if 'fig' in kwargs:
                draw_injections_plotly(df_injections, kwargs['fig'])
            else:
                draw_injections_matplotplib(df_injections)


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
        pattern = re.compile(r"(C\d+_\d+|M\d+_\d+|P\d+)")  # Matches "C1_1", "M3_2", "P02", etc.

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
