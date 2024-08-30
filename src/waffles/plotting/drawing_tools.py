import plotly.graph_objs as go
from plotly import graph_objects as pgo
import plotly.io as pio
import numpy as np



from math import sqrt

import waffles.utils.wf_maps_utils as wmu
from waffles.plotting.plot import *
import waffles.input.raw_ROOT_reader as reader
from waffles.utils.fit_peaks import fit_peaks as fp
import waffles.utils.numerical_utils as wun

# import data classes
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.ChannelWS import ChannelWS
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.Waveform import Waveform


# Define a file path to save the HTML plot
file_path = 'temp_plot.html'
png_file_path = 'temp_plot.png'

#plotting_mode = 'html'
plotting_mode = 'png'


# define a global figure
fig=go.Figure()

# the ploted calib histo
#chist = CalibrationHistogram()


###########################
def help():

    funcs=[
        ['plot','plot waveforms for a single waveform, list of waveforms or WaveformSet',],
        ['plot_hm','plot heat map for a WaveformSet'],
        ['plot_charge','plot charge histogram for a WaveformSet'],
        ['plot_avg', 'plot average waveform for a WaveformSet'],
        ['plot_to','plot time offset (timestamp-daq_timestamp) for a WaveformSet',]
    ]

    for i in funcs:
        print (f'{i[0]:15}', i[1])

###########################
def read(filename, 
         start_fraction: float = 0, 
         stop_fraction: float = 1,
         read_full_streaming_data: bool = False,
         truncate_wfs_to_minimum: bool = False,
         set_offset_wrt_daq_window : bool = True):
    
    print ('reading file ', filename, '...')

    wset=reader.WaveformSet_from_ROOT_file(filename,library='pyroot',start_fraction=start_fraction, stop_fraction=stop_fraction,
                                           read_full_streaming_data=read_full_streaming_data,
                                           truncate_wfs_to_minimum=truncate_wfs_to_minimum,
                                           set_offset_wrt_daq_window=set_offset_wrt_daq_window)

    print ('done !!!')

    return wset

###########################
def plot_to(wset: WaveformSet,
            ep: int = -1, 
            ch: int = -1,
            nwfs: int = -1,
            op: str = None,
            nbins: int = 100,
            xmin: np.uint64 = None,
            xmax: np.uint64 = None):

    global fig
    if op != 'same':
        fig=go.Figure()

    # get the array of time stamps
    n=0
    times = []
    for wf in wset.Waveforms:
        if (wf.Endpoint==ep or ep==-1) and (wf.Channel==ch or ch==-1):
            n=n+1
            times.append(wf._WaveformAdcs__time_offset2)    
        if n>=nwfs and nwfs!=-1:
            break

    # compute the histogram edges
    tmin = min(times)
    tmax = max(times)

    if xmin == None:
        xmin = tmin-(tmax-tmin)*0.1
    if xmax == None:
        xmax = tmax+(tmax-tmin)*0.1
    
    domain=[xmin,xmax]

    # create the histogram
    counts, indices = wun.histogram1d(  np.array(times),
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
                                    line=dict(  color = 'black', 
                                                width = 0.5,
                                                shape = 'hv'),
                                    name = "Hola")
    
    fig.add_trace(histogram_trace)
    
    #Update
    fig.update_layout(
                        title='Time-offset histogram',
                        xaxis=dict(
                            title='Time-offset ticks',
                            tickmode='linear',  # per avere tick lineari
                            dtick= int ((domain[1]- domain[0])/10),  # intervallo tra i tick sull'asse x
                        ),
                        yaxis=dict(
                            title='Counts',
                        )
                    )   

    write_image(fig)
    fig.show()

###########################
def plot_hm(object,
            ep: int = -1, 
            ch: int = -1,
            nx:   int = 100, 
            xmin: int = 0, 
            xmax: int = 1024, 
            ny:   int = 100, 
            ymin: int = 0, 
            ymax: int = 15000, 
            nwfs: int = -1,
            variable = 'integral', 
            op: str = None,
            imin: float = None,
            imax: float = None,
            show: bool = True,             
            bar : bool = False):
 
    global fig

    if op != 'same':
        fig=go.Figure()    

    ranges = np.array ([[xmin,xmax],[ymin,ymax]])
    wset = get_wfs_in_channel(object,ep,ch)
    if imin != None:
        wset=get_wfs_with_variable_in_range(wset,imin,imax,variable)

#    fig = wpu.__subplot_heatmap_ans(wset,fig,"name",nx,ny,ranges,show_color_bar=bar)
    fig = __subplot_heatmap_ans(wset,fig,"name",nx,ny,ranges,show_color_bar=bar)
    write_image(fig)


        
##########################
def plot(object,
        min_int : int = 0,
        max_int : int = 0,
        ep: int = -1, 
        ch: int = -1,
        nwfs: int = -1,
        offset: bool = False,
        op: str = None,
        show: bool = True
        ):


# Case when the input object is a Waveform
    if type(object)==Waveform:    
        plot_wfs(list([object]),ep,ch,nwfs,offset,op,min_int,max_int)

    # Case when the input object is a list of Waveforms
    if type(object)==list and type(object[0])==Waveform:
        plot_wfs(object,ep,ch,nwfs,offset,op,min_int,max_int)

    # Case when the input object is a WaveformSet                    
    if type(object)==WaveformSet:
        plot_wfs(object.Waveforms,ep,ch,nwfs,offset,op,min_int,max_int)       


#########################
 
def plot_wfs(wfs: list,                
                ep: int = -1, 
                ch: int = -1,
                nwfs: int = -1,
                offset: bool = False, 
                op: str = None,
                min_int : int = 0,
                max_int : int = 0
                ):
        
    global fig

    if op != 'same':
        fig=go.Figure()

    n=0
    for wf in wfs:
        if (wf.Endpoint==ep or ep==-1) and (wf.Channel==ch or ch==-1):
            n=n+1
            plot_WaveformAdcs(wf,fig, offset)
        if n>=nwfs and nwfs!=-1:
            break

    if (min_int != 0) and (max_int !=0):
        # Aggiunta delle linee verticali
        fig.add_shape(
            type='line',
            x0=min_int, y0=0,
            x1=min_int, y1=1,
            xref='x', yref='paper',
            line=dict(color='red', width=2, dash='dash'),
            )

        fig.add_shape(
            type='line',
            x0=max_int, y0=0,
            x1=max_int, y1=1,
            xref='x', yref='paper',
            line=dict(color='red', width=2, dash='dash'),
            )
        
        min_x = 0 if min_int < 100 else min_int - 100
        max_x = max_int + 100
    else:
        min_x = 0
        max_x = len(wf.Adcs)

    # Aggiunta di un layout (opzionale)
    fig.update_layout(
                    title='Waveforms',
                    xaxis=dict(
                        range=[min_x, max_x],
                        title='Time ticks'#, tickmode='linear',  # per avere tick lineari # dtick= int(len(wf.Adcs)/20),  # intervallo tra i tick sull'asse x
                    ),
                    yaxis=dict(
                        title='ADC',
                    )
                    )   
            
    write_image(fig)  
    fig.show()     
        
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
            op: str = None):        

    global fig

    if op != 'same':
        fig=go.Figure()

    # get wfs in specific channel
    wset2 = get_wfs_in_channel(wset,ep,ch)
    
    # baseline limits
    bl = [b_ll, b_ul, 900, 1000]
    peak_finding_kwargs = dict( prominence = 20,rel_height=0.5,width=[0,75])
    ip = IPDict(baseline_limits=bl,
                int_ll=int_ll,int_ul=int_ul,amp_ll=int_ll,amp_ul=int_ul,
                points_no=10,
                peak_finding_kwargs=peak_finding_kwargs)
    analysis_kwargs = dict(  return_peaks_properties = False)
    checks_kwargs   = dict( points_no = wset.PointsPerWf )
    #if wset.Waveforms[0].has_analysis('standard') == False:

    # analyse the waveforms
    a=wset2.analyse('standard',BasicWfAna,ip,checks_kwargs = checks_kwargs,overwrite=True)
    
    n= len(wset2.Waveforms)
    integral_list = []
    for i in range(n):
        integral_list.append(wset2.Waveforms[i].Analyses['standard'].Result['integral'])
    mean = np.mean(integral_list)
    std = np.std(integral_list)
    print(f'We have {n:.0f} beam waveforms and their average integral value is: {mean:.0f} +/- {std:.0f} \n')
    
    # Compute the calibration histogram for the channel
    ch_wfs = ChannelWS(*wset2.Waveforms,compute_calib_histo=True,bins_number=nb,domain=np.array([hl,hu]),variable=variable)

    if (op!='peaks'):
       #plot the calibration histogram
        plot_CalibrationHistogram(ch_wfs.CalibHisto,fig,'Charge_histogram',None,None,True,200)
        write_image(fig)

    else:
        plot_charge_peaks(ch_wfs.CalibHisto)


    fig.update_layout(
                        title='Charge histogram',
                        xaxis=dict(
                            title='Charge',
                        ),
                        yaxis=dict(
                            title='Counts',
                        )
                      )  

    return ch_wfs.CalibHisto


###########################
def plot_charge_3(wset: WaveformSet,
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
            op: str = None):        

    global fig

    if op != 'same':
        fig=go.Figure()

    # get wfs in specific channel
    wset2 = get_wfs_in_channel(wset,ep,ch)
    
    # baseline limits
    bl = [b_ll, b_ul, 900, 1000]
    peak_finding_kwargs = dict( prominence = 20,rel_height=0.5,width=[0,75])
    ip = IPDict(baseline_limits=bl,
                int_ll=int_ll,int_ul=int_ul,amp_ll=int_ll,amp_ul=int_ul,
                points_no=10,
                peak_finding_kwargs=peak_finding_kwargs)
    analysis_kwargs = dict(  return_peaks_properties = False)
    checks_kwargs   = dict( points_no = wset.PointsPerWf )
    #if wset.Waveforms[0].has_analysis('standard') == False:

    # analyse the waveforms
    a=wset2.analyse('standard',BasicWfAna,ip,checks_kwargs = checks_kwargs,overwrite=True)
    
    n = len(wset2.Waveforms)
    integral_list = []
    for i in range(n):
        integral_list.append(wset2.Waveforms[i].Analyses['standard'].Result['integral'])
    mean = np.mean(integral_list)
    std = np.std(integral_list)
    print(f'We have {n:.0f} beam waveforms and their average integral value is: {mean:.0f} +/- {std:.0f} \n')

    return integral_list


###########################
def plot_charge_peaks(calibh: CalibrationHistogram,
                    npeaks: int=2, 
                    prominence: float=0.2,
                    half_points_to_fit: int =10,
                    op: str = None):        

    global fig

    if op != 'same':
        fig=go.Figure()


    # fit the peaks of the calibration histogram
    fp.fit_peaks_of_CalibrationHistogram(calibration_histogram=calibh,
                                        max_peaks = npeaks,
                                        prominence = prominence,
                                        half_points_to_fit = half_points_to_fit,
                                        initial_percentage = 0.1,
                                        percentage_step = 0.1)

    #plot the calibration histogram
    plot_CalibrationHistogram(calibh,fig,'hola',None,None,True,200)
    write_image(fig)

    # print the gain and the S/N
    if len(calibh.GaussianFitsParameters['mean']) < 2:
        print ('<2 peaks found. S/N and gain cannot be computed')
    else:
        gain = (calibh.GaussianFitsParameters['mean'][1][0]-calibh.GaussianFitsParameters['mean'][0][0])
        signal_to_noise = gain/sqrt(calibh.GaussianFitsParameters['std'][1][0]**2+calibh.GaussianFitsParameters['std'][0][0]**2)        
        print ('S/N =  ', signal_to_noise)
        print ('gain = ', gain)


###########################
def plot_avg(wset: WaveformSet,
            ep: int = -1, 
            ch: int = -1,            
            nwfs: int = -1,
            imin: float = None,
            imax: float = None, 
            op: str = None):        

    global fig

    if op != 'same':
        fig=go.Figure()

    # get wfs in specific channel
    wset2 = get_wfs_in_channel(wset,ep,ch)

    if imin != None:
        wset2=get_wfs_with_integral_in_range(wset2,imin,imax)

    ch_ws = ChannelWS(*wset2.Waveforms)
    aux = ch_ws.compute_mean_waveform()

    #aux = wset2.compute_mean_waveform() 
    plot_WaveformAdcs(aux,fig)

    write_image(fig)


    


###########################
def compute_charge(wset: WaveformSet,
            ep: int = -1, 
            ch: int = -1,            
            int_ll: int = 135,
            int_ul: int = 165,
            b_ll: int = 0,
            b_ul: int = 100,
            nwfs: int = -1, 
            op: str = None):        


    # get wfs in specific channel
    wset2 = get_wfs_in_channel(wset,ep,ch)
    
    # baseline limits
    bl = [b_ll, b_ul, 900, 1000]
    peak_finding_kwargs = dict( prominence = 20,rel_height=0.5,width=[0,75])
    ip = IPDict(baseline_limits=bl,
                int_ll=int_ll,int_ul=int_ul,amp_ll=int_ll,amp_ul=int_ul,
                points_no=10,
                peak_finding_kwargs=peak_finding_kwargs)
    analysis_kwargs = dict(  return_peaks_properties = False)
    checks_kwargs   = dict( points_no = wset.PointsPerWf )
    #if wset.Waveforms[0].has_analysis('standard') == False:

    # analyse the waveforms
    a=wset2.analyse('standard',BasicWfAna,ip,checks_kwargs = checks_kwargs,overwrite=True)

    return wset2

###########################
def get_wfs_with_variable_in_range(wset:WaveformSet,
                                 imin: float=-10000,
                                 imax: float=1000000,
                                 variable: str = 'integral'):
    
    wfs = []
    for w in wset.Waveforms:
        if variable=='timeoffset':
            var = w._WaveformAdcs__time_offset2
        elif  variable == 'integral' or variable =='amplitude':
            var=w.get_analysis('standard').Result[variable]
        else:
            print ('variable ', variable, ' not supported!!!')
            break 

        if var>imin and var<imax:
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
    for w in wset.Waveforms:
        if min(w.Adcs)>amin and max(w.Adcs)<amax:
            wfs.append(w)

    return WaveformSet(*wfs)

###########################
def get_wfs_in_channel( wset : WaveformSet,    
                        ep : int = -1,
                        ch : int = -1):
    
    wfs = []
    for w in wset.Waveforms:
        if (w.Endpoint == ep or ep==-1) and (w.Channel == ch or ch==-1):
            wfs.append(w)
    return WaveformSet(*wfs)

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
                                    waveform_set.PointsPerWf,
                                    dtype = np.float32) + waveform_set.Waveforms[idx].TimeOffset for idx in range(len(waveform_set.Waveforms))])

    aux_y = np.hstack([waveform_set.Waveforms[idx].Adcs  for idx in range(len(waveform_set.Waveforms))])


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
def write_image(fig: go.Figure()):                    
    if plotting_mode == 'html':        
        pio.write_html(fig, file=file_path, auto_open=True)
    elif plotting_mode == 'png':  
        pio.write_image(fig, file=png_file_path, format='png')  
    else:
        print ('unknown plotting mode ', plotting_mode, ' it should be png or html !!!')


###########################
# variant of plot_WaveformAdcs with option to consider offsets or not + integration limit ANNA
def plot_WaveformAdcs( waveform_adcs : WaveformAdcs,  
                        figure : pgo.Figure,
                        offset: bool = False,
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

    x = np.arange(  len(waveform_adcs.Adcs),
                    dtype = np.float32)

    #    wf_trace = pgo.Scatter( x = x + waveform_adcs.TimeOffset,   ## If at some point we think x might match for
    if offset:
        wf_trace = pgo.Scatter( x = x + waveform_adcs.TimeOffset2,   ## If at some point we think x might match for
                                                                ## every waveform, in a certain WaveformSet 
                                                                ## object, it might be more efficient to let
                                                                ## the caller define it, so as not to recompute
                                                                ## this array for each waveform.
                            y = waveform_adcs.Adcs,
                            mode = 'lines',
                            line=dict(  color='black', 
                                        width=0.5),
                            name = name)
    else:
        wf_trace = pgo.Scatter( x = x,   ## If at some point we think x might match for
                                                                ## every waveform, in a certain WaveformSet 
                                                                ## object, it might be more efficient to let
                                                                ## the caller define it, so as not to recompute
                                                                ## this array for each waveform.
                            y = waveform_adcs.Adcs,
                            mode = 'lines',
                            line=dict(  color='black', 
                                        width=0.5),
                            name = name)

    
    figure.add_trace(   wf_trace,
                        row = row,
                        col = col)
    
