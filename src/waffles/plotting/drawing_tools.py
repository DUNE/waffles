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
from waffles.utils.fit_peaks import fit_peaks as fp
import waffles.utils.numerical_utils as wun

from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.ChannelWs import ChannelWs
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.Event import Event
import waffles.utils.event_utils as evt_utils

# Global plotting settings
fig = go.Figure()
line_color = 'black'
html_file_path = 'temp_plot.html'
plotting_mode='html'

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
         set_offset_wrt_daq_window: bool = False) -> WaveformSet:
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
def plot_event(evt: Event, apa: int):
    fig = plot_ChannelWsGrid(evt.channel_wfs[apa-1])
    write_image(fig)


###########################
def plot_evt_nch(events: List[Event], 
            nbins: int = 100, xmin: np.uint64 = None,
            xmax: np.uint64 = None, op: str = ''):
    """Plot histogram fwith number of channels firing per event"""
    
    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()
    
    # get number of channels with wfs 
    nchs = [ev.get_nchannels() for ev in events]

    # build an histogram with those times
    histogram_trace = get_histogram(nchs, nbins, xmin, xmax)
    
    fig.add_trace(histogram_trace)
    fig.update_layout(xaxis_title="# channels", yaxis_title="entries")
    
    
    write_image(fig)


###########################
def plot_evt_time(events: List[Event], type: str = 'ref',
            nbins: int = 100, xmin: np.uint64 = None,
            xmax: np.uint64 = None, op: str = ''):
    """Plot histogram fwith number of channels firing per event"""
    
    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()
    
    # get number of channels with wfs 
    if type == 'ref':
        times = [ev.ref_timestamp*1e-9*16 for ev in events]
    elif type == 'first':
        times = [ev.first_timestamp*1e-9*16 for ev in events]
    if type == 'last':
        times = [ev.last_timestamp*1e-9*16 for ev in events]

    # build an histogram with those times
    histogram_trace = get_histogram(times, nbins, xmin, xmax)
    
    fig.add_trace(histogram_trace)
    fig.update_layout(xaxis_title=f"{type}_timestamp", yaxis_title="entries")
    
    
    write_image(fig)

###########################
def plot_to(wset: WaveformSet, ep: int = -1, ch: int = -1, nwfs: int = -1,
            op: str = '', nbins: int = 100, xmin: np.uint64 = None,
            xmax: np.uint64 = None):
    """Plot time offset histogram for a WaveformSet."""
    
    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()
    
    # get the time offset for all wfs in the specific ep and channel
    times = [wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp
             for wf in wset.waveforms
             if (wf.endpoint == ep or ep == -1) and (wf.channel == ch or ch == -1)]
    
    # build an histogram with those times
    histogram_trace = get_histogram(times, nbins, xmin, xmax)
    
    fig.add_trace(histogram_trace)
    fig.update_layout(xaxis_title="time offset", yaxis_title="entries")
    
    
    write_image(fig)

###########################
def plot_hm(object, ep: int = -1, ch: int = -1, nx: int = 100, xmin: int = 0,
            xmax: int = 1024, ny: int = 100, ymin: int = 0, ymax: int = 15000,
            nwfs: int = -1, variable='integral', op: str = '', vmin: float = None,
            vmax: float = None, show: bool = True, bar: bool = False):
    """Plot heatmap for waveforms in a specified range."""
    
    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()

    # get all wfs in a specific ep and channel
    wset = get_wfs_in_channel(object, ep, ch)

    # filter with appropriate variables limits
    if vmin is not None:        
        wset = get_wfs_with_variable_in_range(wset, vmin, vmax, variable)

    # build the plot
    ranges = np.array([[xmin, xmax], [ymin, ymax]])
    fig = __subplot_heatmap_ans(wset, fig, "name", nx, ny, ranges, show_color_bar=bar)
    fig.update_layout(xaxis_title="time tick", yaxis_title="adcs")
    write_image(fig)

###########################
def write_image(fig: go.Figure) -> None:
    """Save or display the figure based on plotting mode."""
    if plotting_mode == 'html':
        pio.write_html(fig, file=html_file_path, auto_open=True)
    elif plotting_mode == 'png':
        pio.write_image(fig, file=png_file_path, format='png')
    else:
        print(f"Unknown plotting mode '{plotting_mode}', should be 'png' or 'html'!")

###########################
def plot(object,                   
         ep: int = -1, 
         ch: int = -1,
         nwfs: int = -1,
         xmin: int = -1,
         xmax: int = -1,
         offset: bool = False,
         rec: int = -1,
         op: str = '',
         show: bool = True):

    # Case when the input object is a Waveform
    if type(object)==Waveform:    
        plot_wfs(list([object]),ep,ch,nwfs,xmin,xmax,offset,rec,op)
    
    # Case when the input object is a list of Waveforms
    if type(object)==list and type(object[0])==Waveform:
        plot_wfs(object,ep,ch,nwfs,xmin,xmax,offset,rec,op)

    # Case when the input object is a WaveformSet                    
    if type(object)==WaveformSet:
        plot_wfs(object.waveforms,ep,ch,nwfs,xmin,xmax,offset,rec,op)
    
###########################
def plot_wfs(wfs: list,                
                ep: int = -1, 
                ch: int = -1,
                nwfs: int = -1,
                xmin: int = -1,
                xmax: int = -1,
                offset: bool = False,
                rec: int = -1,
                op: str = ''):
        
    global fig
    if not has_option(op,'same'):
        fig=go.Figure()

    # plot all waveforms in a given endpoint and channel
    n=0
    for wf in wfs:
        if (wf.endpoint==ep or ep==-1) and (wf.channel==ch or ch==-1) and (wf.record_number==rec or rec==-1):
            n=n+1
            plot_WaveformAdcs2(wf,fig, offset,xmin,xmax)
        if n>=nwfs and nwfs!=-1:
            break

    # add axes titles
    fig.update_layout(xaxis_title="time tick", yaxis_title="adcs")

    write_image(fig)     
        
###########################
def plot_charge(wset: WaveformSet,
            ep: int = -1, 
            ch: int = -1,            
            int_ll: int = 135,
            int_ul: int = 165,
            nb: int = 200,
            hl: int = -5000,
            hu: int = 50000,
            b_ll: int = 0,
            b_ul: int = 100,
            nwfs: int = -1, 
            variable: str = 'integral',
            op: str = ''):        

    global fig
    if not has_option(op,'same'):
        fig=go.Figure()

    chist = compute_charge_histogram(wset,ep,ch,int_ll,int_ul,nb,hl,hu,b_ll,b_ul,nwfs,variable,op+' print')

    plot_CalibrationHistogram(chist,fig,'hola',None,None,True,200)

    # add axes titles
    fig.update_layout(xaxis_title=variable, yaxis_title="entries")

    write_image(fig)

    return chist

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

###########################
def plot_charge_peaks(calibh: CalibrationHistogram,
                    npeaks: int=2, 
                    prominence: float=0.2,
                    half_points_to_fit: int =10,
                    op: str = ''):        

    global fig
    if not has_option(op,'same'):
        fig=go.Figure()

    # find and fit
    compute_peaks(calibh,npeaks,prominence,half_points_to_fit,op)

    #plot the calibration histogram
    plot_CalibrationHistogram(calibh,fig,'hola',None,None,True,200)
    
    write_image(fig)

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
                                        initial_percentage = 0.1,
                                        percentage_step = 0.1,
                                        half_points_to_fit = half_points_to_fit)


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
def plot_avg(wset: WaveformSet,
            ep: int = -1, 
            ch: int = -1,            
            nwfs: int = -1,
            imin: float = None,
            imax: float = None, 
            op: str = ''):        

    global fig
    if not has_option(op,'same'):
        fig=go.Figure()

    # get wfs in specific channel
    wset2 = get_wfs_in_channel(wset,ep,ch)

    # select an integral range
    if imin != None:
        wset2=get_wfs_with_integral_in_range(wset2,imin,imax)

    # Create the Channel WaveformSet needed to compute the mean waveform
    ch_ws = ChannelWs(*wset2.waveforms)

    # compute the mean waveform 
    aux = ch_ws.compute_mean_waveform()

    # plot the mean waveform
    plot_WaveformAdcs2(aux,fig)

    # add axes titles
    fig.update_layout(xaxis_title='time tick', yaxis_title='average adcs')

    write_image(fig)
    
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
def zoom(xmin: float = -999,
         xmax: float = -999,
         ymin: float = -999,
         ymax: float = -999):

    if xmin!=-999 and xmax!=-999:
        fig.update_layout(xaxis_range=[xmin,xmax])
    if ymin!=-999 and ymax!=-999:
        fig.update_layout(yaxis_range=[ymin,ymax])
    write_image(fig)



###########################
def __subplot_heatmap_ans(  waveform_set : WaveformSet, 
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

###########################
# variant of plot_WaveformAdcs with option to consider offsets or not
def plot_WaveformAdcs2( waveform_adcs : Waveform,  
                        figure : pgo.Figure,
                        offset: bool = False,
                        xmin: int = -1,
                        xmax: int = -1,
                        name : Optional[str] = None,
                        row : Optional[int] = None,
                        col : Optional[int] = None,
                        plot_analysis_markers : bool = False,
                        show_baseline_limits : bool = False, 
                        show_baseline : bool = True,
                        show_general_integration_limits : bool = False,
                        show_general_amplitude_limits : bool = False,
                        show_spotted_peaks : bool = True,
                        show_peaks_integration_limits : bool = False,
                        analysis_label : Optional[str] = None,
                        verbose : bool = False) -> None:

    if xmin!=-1 and xmax!=-1:
        x0 = np.arange(  xmin, xmax,
                        dtype = np.float32)
        y0 = waveform_adcs.adcs[xmin:xmax]    
    else:
        x0 = np.arange(  len(waveform_adcs.adcs),
                        dtype = np.float32)
        y0 = waveform_adcs.adcs

    names=waveform_adcs.channel

#    wf_trace = pgo.Scatter( x = x + waveform_adcs.time_offsetn,   ## If at some point we think x might match for
    if offset:
        wf_trace = pgo.Scatter( x = x0 + np.float32(waveform_adcs.timestamp-waveform_adcs.daq_window_timestamp),   
                                                                ## If at some point we think x might match for
                                                                ## every waveform, in a certain WaveformSet 
                                                                ## object, it might be more efficient to let
                                                                ## the caller define it, so as not to recompute
                                                                ## this array for each waveform.
                            y = y0,
                            mode = 'lines',
                            line=dict(  color=line_color, 
                                        width=0.5),
                            name = names)
    else:
        wf_trace = pgo.Scatter( x = x0,   ## If at some point we think x might match for
                                                                ## every waveform, in a certain WaveformSet 
                                                                ## object, it might be more efficient to let
                                                                ## the caller define it, so as not to recompute
                                                                ## this array for each waveform.
                            y = y0,
                            mode = 'lines',
                            line=dict(  color=line_color, 
                                        width=0.5),
                            name = names)




    figure.add_trace(   wf_trace,
                        row = row,
                        col = col)

###########################
def get_histogram(values: [],
                    nbins: int = 100,
                    xmin: np.uint64 = None,
                    xmax: np.uint64 = None):


    # compute the histogram edges
    tmin = min(values)
    tmax = max(values)

    if xmin == None:
        xmin = tmin-(tmax-tmin)*0.1
    if xmax == None:
        xmax = tmax+(tmax-tmin)*0.1
    
    domain=[xmin,xmax]

    # create the histogram
    counts, indices = wun.histogram1d(  np.array(values),
                                        nbins,
                                        domain,
                                        keep_track_of_idcs = True)
    
    # plot the histogram
    edges = np.linspace(domain[0],
                        domain[1], 
                        num = nbins + 1,
                        endpoint = True)

    histogram_trace = pgo.Scatter(  x = edges,
                                    y = counts,
                                    mode = 'lines',
                                    line=dict(  color = line_color, 
                                                width = 0.5,
                                                shape = 'hv'),
                                    name = "Hola")
    
    return histogram_trace

##########################
def plot_spe_mean_vs_var(wset_map, ep: int = -1, ch: int = -1, var: str = None, op: str = ''):       
    plot_chist_param_vs_var(wset_map,ep,ch,'spe_mean',var,op)

##########################
def plot_sn_vs_var(wset_map, ep: int = -1, ch: int = -1, var: str = None, op: str = ''):       
    plot_chist_param_vs_var(wset_map,ep,ch,'sn',var,op)

##########################
def plot_gain_vs_var(wset_map, ep: int = -1, ch: int = -1, var: str = None, op: str = ''):       
    plot_chist_param_vs_var(wset_map,ep,ch,'gain',var,op)

##########################
def plot_spe_mean_vs_channel(wset_map, ep: int = -1, chs: list = None, op: str = ''):       
    plot_param_vs_channel(wset_map,ep,chs,'spe_mean',op)

##########################
def plot_sn_vs_channel(wset_map, ep: int = -1, chs: list = None, op: str = ''):       
    plot_param_vs_channel(wset_map,ep,chs,'sn',op)

##########################
def plot_gain_vs_channel(wset_map, ep: int = -1, chs: list = None, op: str = ''):       
    plot_param_vs_channel(wset_map,ep,chs,'gain',op)


###########################
def plot_chist_param_vs_var(wset_map, 
                     ep: int = -1,
                     ch: int = -1,
                     param: str = None,
                     var: str = None,
                     op: str = ''):
       
    global fig
    if not has_option(op,'same'):
        fig=go.Figure()    

    par_values = []
    var_values = []
    # loop over pairs [WaveformSet, var]
    for wset in wset_map:
        # compute the charge/amplitude histogram for this wset and find/fit the peaks
        calibh = compute_charge_histogram(wset[0],ep,ch,128,170,300,-5000,40000,op="peaks")
        # get the parameters from the fitted peaks
        gain,sn,spe_mean = compute_charge_histogram_params(calibh)
        # add var values to the list
        var_values.append(wset[1])
        # add param values to the list, depending on the chosen param
        if param == 'gain':
            par_values.append(gain)
        elif param == 'sn':
            par_values.append(sn)
        elif param == 'spe_mean':
            par_values.append(spe_mean)


    # get the trace 
    trace = pgo.Scatter(x = var_values,
                        y = par_values)
    # add it to the figure
    fig.add_trace(trace)

    # add axes titles
    fig.update_layout(xaxis_title=var, yaxis_title=param)    

    write_image(fig)

###########################
def plot_param_vs_channel(wset: WaveformSet, 
                        ep: int = -1,
                        chs: list = None,
                        param: str = None,
                        op: str = ''):
       
    global fig
    if not has_option(op,'same'):
        fig=go.Figure()    

    ch_values = []
    par_values = []
    # loop over channels
    for ch in chs:
        # compute the charge/amplitude histogram for this wset and find/fit the peaks
        calibh = compute_charge_histogram(wset,ep,ch,135,165,200,-5000,20000,op=op+' peaks')
        # get the parameters from the fitted peaks
        gain,sn,spe_mean = compute_charge_histogram_params(calibh)
        # add var values to the list
        ch_values.append(ch)
        # add param values to the list, depending on the chosen param
        if param == 'gain':
            par_values.append(gain)
        elif param == 'sn':
            par_values.append(sn)
        elif param == 'spe_mean':
            par_values.append(spe_mean)


    # get the trace 
    trace = pgo.Scatter(x = ch_values,
                        y = par_values,
                        mode = 'markers')
    # add it to the figure
    fig.add_trace(trace)

    # add axes titles
    fig.update_layout(xaxis_title="channel", yaxis_title=param)

    write_image(fig)

###########################
def plot_integral_vs_amplitude(wset: WaveformSet, 
                        ep: int = -1,
                        ch: int = -1,
                        int_ll: int = 135,
                        int_ul: int = 165,
                        b_ll: int = 0,
                        b_ul: int = 100,                       
                        op: str = ''):
       
    global fig
    if not has_option(op,'same'):
        fig=go.Figure()    

    # get the waveforms in the specific ep and ch
    wset2 = get_wfs_in_channel(wset,ep,ch)

    # Compute the charge (amplitude and integral) 
    compute_charge(wset2,int_ll,int_ul,b_ll,b_ul,op)

    amp_values = []
    int_values = []
    # loop over waveforms
    for w in wset2.waveforms:                
        amp_values.append(w.get_analysis('standard').Result['amplitude'])
        int_values.append(w.get_analysis('standard').Result['integral'])
        
    # get the trace 
    trace = pgo.Scatter(x = amp_values,
                        y = int_values,
                        mode = 'markers')

    # add it to the figure
    fig.add_trace(trace)

    # add axes titles
    fig.update_layout(xaxis_title="amplitude", yaxis_title="integral")

    write_image(fig)


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