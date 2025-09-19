import inspect
import importlib
from math import sqrt
import pathlib
from typing import List

import numpy as np
import plotly.graph_objs as go
from plotly import graph_objects as pgo
import plotly.io as pio

import waffles.utils.wf_maps_utils as wmu
from waffles.plotting.plot import *
import waffles.input_output.raw_root_reader as root_reader
import waffles.input_output.pickle_file_reader as pickle_reader
import waffles.input_output.hdf5_structured as hdf5_reader
from waffles.utils.fit_peaks import fit_peaks as fp
import waffles.utils.numerical_utils as wun

from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.ChannelWs import ChannelWs
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.Event import Event
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map
from waffles.np04_data.ProtoDUNE_HD_APA_maps_APA1_104 import APA_map as APA_map_2
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
import waffles.utils.event_utils as evt_utils
from scipy import signal
from scipy.fft import fft, fftshift
import pickle
from typing import Union
import warnings 

# Global plotting settings
line_color = 'black'

templates = []

###########################
def help(cls: str = None):
    """Print available commands or specific help for a command."""
    funcs = [
        ['plot', 'plot waveforms for a single waveform, list of waveforms or WaveformSet'],
        ['plot_hm', 'plot heat map for a WaveformSet'],
        ['plot_charge', 'plot charge histogram for a WaveformSet'],
        ['plot_charge_peaks', 'plot charge histogram peaks given a charge histogram'],
        ['plot_avg', 'plot average waveform for a WaveformSet'],
        ['plot_to', 'plot time offset (timestamp-daq_timestamp) for a WaveformSet'],
        ['plot_spe_mean_vs_var', 'plot s.p.e. mean vs variable for various WaveformSets'],
        ['plot_sn_vs_var', 'plot signal to noise ratio vs variable for various WaveformSets'],
        ['plot_gain_vs_var', 'plot gain vs variable for various WaveformSets'],
        ['plot_spe_mean_vs_channel', 'plot s.p.e. mean vs channel for endpoint']
    ]
    
    if cls is None:
        print("List of commands. Type draw.help(draw.X) to see the arguments of command X")
        for func in funcs:
            print(f"{func[0]:32} {func[1]}")
    else:
        for func in funcs:
            if cls.__qualname__ == func[0]:
                print(f"{func[0]:32} {func[1]}")
        print(inspect.signature(cls))
    
###########################
def read(filename, start_fraction: float = 0, stop_fraction: float = 1,
         read_full_streaming_data: bool = False, truncate_wfs_to_minimum: bool = False,
         set_offset_wrt_daq_window: bool = False,
         nwfs: int = None) -> WaveformSet:
    """Read waveform data from file."""
    print(f"Reading file {filename}...")
    
    file_extension = pathlib.Path(filename).suffix

    if  file_extension == ".root":
        wset = root_reader.WaveformSet_from_root_file(
            filename,
            library='pyroot',
            start_fraction=start_fraction,
            stop_fraction=stop_fraction,
            read_full_streaming_data=read_full_streaming_data,
            truncate_wfs_to_minimum=truncate_wfs_to_minimum,
            set_offset_wrt_daq_window=set_offset_wrt_daq_window
        )
    elif file_extension == ".pkl":
        wset = pickle_reader.WaveformSet_from_pickle_file(filename) 
    elif file_extension == ".hdf5":
#        wset = hdf5_reader.WaveformSet_from_hdf5_file(filename)
        wset = hdf5_reader.load_structured_waveformset(filename,max_waveforms=nwfs) 
   
        
    print("Done!")
    return wset

###########################
def eread(filename, nevents: int = 1000000000) -> List[Event]:
    """Read waveform data from file."""
    print(f"Reading file {filename}...")
        
    wfset = read(filename)
    events = evt_utils.events_from_wfset(wfset, nevents=nevents) 

    print("Done!")
    return events


###########################
def tsort_wfset(wfset0: WaveformSet) -> WaveformSet:

    waveforms = sorted(wfset0.waveforms, key=lambda Waveform: Waveform.timestamp)
    wfset = WaveformSet(*waveforms)

    return wfset


###########################
def get_grid(wfs: list,                
             apa: int = -1,
             run: int = -1):

    if run < 29927:
        grid_apa = ChannelWsGrid(APA_map[apa], WaveformSet(*wfs))
    else:
        grid_apa = ChannelWsGrid(APA_map_2[apa], WaveformSet(*wfs))        
        
    return grid_apa
            
###########################
def get_grid_index(wf: Waveform):

    # get the apa for that waveform      
    if    wf.endpoint <  109: gi = 1
    elif  wf.endpoint == 109: gi = 2 
    elif  wf.endpoint == 111: gi = 3
    elif  wf.endpoint >  111: gi = 4  

    return gi 

###########################
def get_endpoints(apa: int):

    eps=[]

    if    apa == 1: eps =[104,105,107]
    elif  apa == 2: eps =[109]
    elif  apa == 3: eps =[111]
    elif  apa == 4: eps =[112,113]

    return eps
        
###########################        
def read_avg(filename):

    with open(filename, 'rb') as file:
        output = pickle.load(file)

    return output


###########################
def get_wfs(wfs: list,                
            ep: Union[int, list]=-1,
            ch: Union[int, list]=-1,
            nwfs: int = -1,
            tmin: int = -1,
            tmax: int = -1,
            rec: list = [-1]):

    # plot all waveforms in a given endpoint and channel

    if type(ch) == int:
        ch = [ch]

    if type(ep) == int:
        ep = [ep]
        
    waveforms = []
    n=0
    for wf in wfs:
        t = np.float32(np.int64(wf.timestamp)-np.int64(wf.daq_window_timestamp))
        if (wf.endpoint      in ep  or  ep[0]==-1) and \
           (wf.channel       in ch  or  ch[0]==-1) and \
           (wf.record_number in rec or rec[0]==-1) and \
           ((t > tmin and t< tmax) or (tmin==-1 and tmax==-1)):
            n=n+1
            waveforms.append(wf)
        if n>=nwfs and nwfs!=-1:
            break

    return waveforms

###########################
def get_wfs_interval(wfs: list,                
            tmin: int = -1,
            tmax: int = -1,
            nwfs: int = -1):
        
    waveforms = []
    n=0
    for wf in wfs:
        t = np.float32(np.int64(wf.timestamp)-np.int64(wf.daq_window_timestamp))
        if ((t > tmin and t< tmax) or (tmin==-1 and tmax==-1)):
            n=n+1
            waveforms.append(wf)
        if n>=nwfs and nwfs!=-1:
            break

    return waveforms
    
###########################
def compute_charge_histogram(wset: WaveformSet,
            ep: int = -1, 
            ch: int = -1,            
            int_ll: int = 128,
            int_ul: int = 170,
            nb: int = 300,
            hl: int = -5000,
            hu: int = 50000,
            b_ll: int = 0,
            b_ul: int = 100,
            nwfs: int = -1, 
            variable: str = 'integral',
            op: str = ''):        

    # get wfs in specific channel
    wset2 = get_wfs_in_channel(wset,ep,ch)

    # Compute the charge (amplitude and integral) 
    compute_charge(wset2,int_ll,int_ul,b_ll,b_ul,nwfs,op)

    # Compute the calibration histogram for the channel
    ch_wfs = ChannelWs(*wset2.waveforms,compute_calib_histo=True,bins_number=nb,domain=np.array([hl,hu]),variable=variable)

    if has_option(op,'peaks'):
        compute_peaks(ch_wfs.calib_histo,op=op)

    return ch_wfs.calib_histo

#########################
def compute_charge(wset: WaveformSet,        
            int_ll: int = 135,
            int_ul: int = 165,
            b_ll: int = 0,
            b_ul: int = 100,
            nwfs: int = -1, 
            op: str = ''):        
    
    # baseline limits
    bl = [b_ll, b_ul, 900, 1000]

    peak_finding_kwargs = dict( prominence = 20,rel_height=0.5,width=[0,75])
    ip = IPDict(baseline_limits=bl,
                int_ll=int_ll,int_ul=int_ul,amp_ll=int_ll,amp_ul=int_ul,
                points_no=10,
                peak_finding_kwargs=peak_finding_kwargs)
    analysis_kwargs = dict(  return_peaks_properties = False)
    checks_kwargs   = dict( points_no = wset.points_per_wf)
    #if wset.waveforms[0].has_analysis('standard') == False:

    # analyse the waveforms
    a=wset.analyse('standard',BasicWfAna,ip,checks_kwargs = checks_kwargs,overwrite=True)

###########################
def compute_peaks(calibh: CalibrationHistogram,
                    npeaks: int=2, 
                    prominence: float=0.2,
                    half_points_to_fit: int =10,
                    op: str = ''):        


    # fit the peaks of the calibration histogram
    fp.fit_peaks_of_CalibrationHistogram(calibration_histogram=calibh,
                                        max_peaks = npeaks,
                                        prominence = prominence,
                                        half_points_to_fit = half_points_to_fit,
                                        initial_percentage = 0.1,
                                        percentage_step = 0.1)


    # print the gain and the S/N
    if op and op.find('print') !=-1:
        if len(calibh.gaussian_fits_parameters['mean']) < 2:
            print ('<2 peaks found. S/N and gain cannot be computed')
        else:
            gain = (calibh.gaussian_fits_parameters['mean'][1][0]-calibh.gaussian_fits_parameters['mean'][0][0])
            signal_to_noise = gain/sqrt(calibh.gaussian_fits_parameters['std'][1][0]**2+calibh.gaussian_fits_parameters['std'][0][0]**2)        
            print ('S/N =  ', signal_to_noise)
            print ('gain = ', gain)
            print ('s.p.e. mean charge = ', calibh.gaussian_fits_parameters['mean'][1][0], 
                ' +- ', 
                calibh.gaussian_fits_parameters['mean'][1][1])

    
###########################
def get_wfs_with_variable_in_range(wset:WaveformSet,
                                 vmin: float=-10000,
                                 vmax: float=1000000,
                                 variable: str = 'integral'):
    
    # three options: timeoffset, integral and amplitude

    wfs = []
    for w in wset.waveforms:
        if variable=='timeoffset':
            var = w._Waveform__timestamp-w._Waveform__daq_window_timestamp
        elif  variable == 'integral' or variable =='amplitude':
            var=w.get_analysis('standard').result[variable]
        else:
            print ('variable ', variable, ' not supported!!!')
            break 

        # select waveforms in range
        if var>vmin and var<vmax:
            wfs.append(w)

    return WaveformSet(*wfs)

###########################
def get_wfs_with_timeoffset_in_range(wset:WaveformSet,
                                 imin: float=-10000,
                                 imax: float=1000000):
    
    return get_wfs_with_variable_in_range(wset,imin,imax,'timeoffset')

###########################
def get_wfs_with_amplitude_in_range(wset:WaveformSet,
                                 imin: float=-10000,
                                 imax: float=1000000):
    
    return get_wfs_with_variable_in_range(wset,imin,imax,'amplitude')

###########################
def get_wfs_with_integral_in_range(wset:WaveformSet,
                                 imin: float=-10000,
                                 imax: float=1000000):
    
    return get_wfs_with_variable_in_range(wset,imin,imax,'integral')
     
###########################
def get_wfs_with_adcs_in_range(wset:WaveformSet,
                                amin: float=-10000,
                                amax: float=1000000):
    
    wfs = []
    for w in wset.waveforms:
        if min(w.adcs)>amin and max(w.adcs)<amax:
            wfs.append(w)

    return WaveformSet(*wfs)

###########################
def get_wfs_in_channel( wset : WaveformSet,    
                        ep : int = -1,
                        ch : int = -1):
    
    wfs = []
    for w in wset.waveforms:
        if (w.endpoint == ep or ep==-1) and (w.channel == ch or ch==-1):
            wfs.append(w)
    return WaveformSet(*wfs)

###########################
def get_wfs_in_channels( wset : WaveformSet,
                        ep : int = -1,
                        chs : list = None):

    wfs = []
    for w in wset.waveforms:
        if (w.endpoint == ep or ep==-1):
            if chs:
                # loop over channels
                for ch in chs: 
                    if w.channel == ch:
                        wfs.append(w)
    return WaveformSet(*wfs)

###########################
def subplot_heatmap_ans(waveform_set : WaveformSet, 
                        figure : pgo.Figure,
                        name : str,
                        time_bins : int,
                        adc_bins : int,
                        ranges : np.ndarray,
                        show_color_bar : bool = False) -> pgo.Figure:
    

    figure_ = figure

    time_step   = (ranges[0,1] - ranges[0,0]) / time_bins
    adc_step    = (ranges[1,1] - ranges[1,0]) / adc_bins

    aux_x = np.hstack([np.arange(   0,
                                    waveform_set.points_per_wf,
                                    dtype = np.float32) + waveform_set.waveforms[idx].time_offset for idx in range(len(waveform_set.waveforms))])

    aux_y = np.hstack([waveform_set.waveforms[idx].adcs  for idx in range(len(waveform_set.waveforms))])


    aux = wun.histogram2d(  np.vstack((aux_x, aux_y)), 
                            np.array((time_bins, adc_bins)),
                            ranges)

    heatmap =   pgo.Heatmap(z = aux,
                            x0 = ranges[0,0],
                            dx = time_step,
                            y0 = ranges[1,0],
                            dy = adc_step,
                            name = name,
                            transpose = True,
                            showscale = show_color_bar)

    figure_.add_trace(heatmap)
                        
    return figure_

##########################
def compute_charge_histogram_params(calibh: CalibrationHistogram):
    if len(calibh.gaussian_fits_parameters['mean']) > 1:
        gain = (calibh.gaussian_fits_parameters['mean'][1][0]-calibh.gaussian_fits_parameters['mean'][0][0])
        sn = gain/sqrt(calibh.gaussian_fits_parameters['std'][1][0]**2+calibh.gaussian_fits_parameters['std'][0][0]**2)
        spe_mean = calibh.gaussian_fits_parameters['mean'][1][0]
    else:
        gain=sn=spe_mean=0

    return gain,sn,spe_mean

##########################
def has_option(ops: str, op: str):
 
    if ops.find(op) == -1:
        return False
    else:
        return True  
    
###########################
def get_histogram(values: list,
                   nbins: int = 100,
                   xmin: float = None,
                   xmax: float = None,
                   line_color: str = 'black',
                   line_width: float = 2):
    if not values:  
        raise ValueError("'values' is empty, the histogram can't be computed.")
    
    # Histogram limits
    tmin = min(values)
    tmax = max(values)

    if xmin is None:
        xmin = tmin - (tmax - tmin) * 0.1
    if xmax is None:
        xmax = tmax + (tmax - tmin) * 0.1
    
    domain = [xmin, xmax]
    
    # Create the histogram
    counts, edges = np.histogram(values, bins=nbins, range=domain)
    
    return counts, edges

###########################
def get_histogram_trace(values: list,
                        nbins: int = 100,
                        xmin: float = None,
                        xmax: float = None,
                        line_color: str = 'black',
                        line_width: float = 2):


    counts, edges = get_histogram(values,nbins,xmin,xmax)
    
    histogram_trace = go.Scatter(
        x=edges[:-1],  
        y=counts,
        mode='lines',
        line=dict(
            color=line_color,
            width=line_width,
            shape='hv'
        )
    )
    
    return histogram_trace




##########################
def create_figure(rows: int=1,
                  cols: int=1):

    if rows==1 and cols==1:
        fig = go.Figure()
    else:
        fig = psu.make_subplots(rows=rows, cols=cols)

    return fig
