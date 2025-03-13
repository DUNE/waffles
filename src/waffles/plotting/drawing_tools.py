from waffles.plotting.drawing_tools_utils import *
from typing import Union
import warnings

# Global plotting settings
fig = go.Figure()
html_file_path = 'temp_plot.html'
png_file_path = 'temp_plot.png'
plotting_mode='html'
line_color = 'black'

templates = []


###########################
def plot(object,                   
         ep: int = -1, 
         ch: Union[int, list]=-1,
         nwfs: int = -1,
         xmin: int = -1,
         xmax: int = -1,
         tmin: int = -1,
         tmax: int = -1,
         offset: bool = False,
         rec: list = [-1],
         op: str = ''):

    """
    Plot a single or many waveforms
    """

    
    # Case when the input object is a Waveform
    if type(object)==Waveform:    
        plot_wfs(list([object]),ep,ch,nwfs,xmin,xmax,tmin,tmax,offset,rec,op)
    
    # Case when the input object is a WaveformAdcs
    elif type(object)==WaveformAdcs:    
        plot_wfs(list([object]),-100,-100,nwfs,xmin,xmax,tmin,tmax,offset,rec,op)
    
    # Case when the input object is a list of Waveforms
    elif type(object)==list and type(object[0])==Waveform:
        plot_wfs(object,ep,ch,nwfs,xmin,xmax,tmin,tmax,offset,rec,op)

    # Case when the input object is a WaveformSet                    
    elif type(object)==WaveformSet:
        plot_wfs(object.waveforms,ep,ch,nwfs,xmin,xmax,tmin,tmax,offset,rec,op)

###########################
def plot_wfs(wfs: list,                
             ep: int = -1, 
             ch: Union[int, list]=-1,
             nwfs: int = -1,
             xmin: int = -1,
             xmax: int = -1,
             tmin: int = -1,
             tmax: int = -1,
             offset: bool = False,
             rec: list = [-1],
             op: str = ''):

    """
    Plot a list of waveforms
    """

    
    global fig
    if not has_option(op,'same'):
        fig=go.Figure()

    # get all waveforms in the specified endpoint, channels,  time offset range and record
    selected_wfs= get_wfs(wfs,[ep],ch,nwfs,tmin,tmax,rec)

    # plot nwfs waveforms
    n=0        
    for wf in selected_wfs:
        n=n+1
        # plot the single waveform
        plot_wf(wf,fig, offset,xmin,xmax)
        if n>=nwfs and nwfs!=-1:
            break

    # add axes titles
    fig.update_layout(xaxis_title="time tick", yaxis_title="adcs")

    write_image(fig)     


###########################
def plot_wf( waveform_adcs : WaveformAdcs,  
             figure : pgo.Figure,
             offset: bool = False,
             xmin: int = -1,
             xmax: int = -1,
             name : Optional[str] = None
             ) -> None:

    """
    Plot a single waveform
    """
    
    if xmin!=-1 and xmax!=-1:
        x0 = np.arange(  xmin, xmax,
                        dtype = np.float32)
        y0 = waveform_adcs.adcs[xmin:xmax]    
    else:
        x0 = np.arange(  len(waveform_adcs.adcs),
                        dtype = np.float32)
        y0 = waveform_adcs.adcs

    names=""#waveform_adcs.channel

    if offset:        
        dt = np.float32(np.int64(waveform_adcs.timestamp)-np.int64(waveform_adcs.daq_window_timestamp))
    else:
        dt = 0

    wf_trace = pgo.Scatter(x = x0 + dt,   
                           y = y0,
                           mode = 'lines',
                           line=dict(  color=line_color, width=0.5)
                           )
    # name = names)

    figure.add_trace(wf_trace)

    
###########################
def plot_grid(wfset: WaveformSet,                
              apa: int = -1, 
              ch: Union[int, list]=-1,
              nwfs: int = -1,
              xmin: int = -1,
              xmax: int = -1,
              tmin: int = -1,
              tmax: int = -1,
              offset: bool = False,
              rec: list = [-1],
              mode: str = 'overlay'):

    """
    Plot a WaveformSet in grid mode
    """
    
    # get the endpoints corresponding to a given APA
    eps= get_endpoints(apa)

    # get all waveforms in the specified endpoint, channels,  time offset range and record
    selected_wfs= get_wfs(wfset.waveforms,eps,ch,nwfs,tmin,tmax,rec)

    run = wfset.waveforms[0].run_number
    
    # get the ChannelWsGrid for that subset of wafeforms and that APA
    grid = get_grid(selected_wfs,apa,run)

    # plot the grid
    fig = plot_ChannelWsGrid(grid, wfs_per_axes=1000,mode=mode)

    write_image(fig,800,1200)

    
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
def plot_to(wset: WaveformSet,
            ep: int = -1,
            ch: int = -1,            
            nwfs: int = -1,
            op: str = '',
            nbins: int = 100,
            xmin: np.uint64 = None,
            xmax: np.uint64 = None):
    """Plot time offset histogram for a WaveformSet."""
    
    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()

        
    # get the time offset for all wfs in the specific ep and channel
    times = [wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp
             for wf in wset.waveforms
             if (wf.endpoint == ep or ep == -1) and (wf.channel==ch or ch == -1)]
    
    # build an histogram with those times
    histogram_trace = get_histogram(times, nbins, xmin, xmax)
    
    fig.add_trace(histogram_trace)
    fig.update_layout(xaxis_title="time offset", yaxis_title="entries")
    
    
    write_image(fig)

###########################
def plot_hm(object,
            ep: int = -1,
            ch: Union[int, list]=-1,
            nx: int = 100,
            xmin: int = 0,
            xmax: int = 1024,
            ny: int = 100,
            ymin: int = 0,
            ymax: int = 15000,
            nwfs: int = -1,
            variable='integral',
            op: str = '',
            vmin: float = None,
            vmax: float = None,
            bar: bool = False):
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
    fig = subplot_heatmap_ans(wset, fig, "name", nx, ny, ranges, show_color_bar=bar)
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
    plot_wf(aux,fig)

    # add axes titles
    fig.update_layout(xaxis_title='time tick', yaxis_title='average adcs')

    write_image(fig)
    
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


    ###########################        
def plot_fft(w: Waveform, xmin: int = -1, xmax: int =-1, op: str=''):


    global fig
    if not has_option(op,'same'):
        fig=go.Figure()

    
    w_fft = np.abs((np.fft.fft(w.adcs)).real)


    if xmin!=-1 and xmax!=-1:
        x0 = np.arange(  xmin, xmax,
                        dtype = np.float32)*16
        y0 = w_fft[xmin:xmax]    
    else:
        x0 = np.arange(  len(w_fft),
                        dtype = np.float32)*16
        y0 = w_fft


    freq = np.fft.fftfreq(x0.shape[-1])
    
    wf_trace = pgo.Scatter( x = freq,
                            y = y0,
                            mode = 'lines',
                            line=dict(color=line_color, width=0.5))
                            #name = names)

                            
    fig.add_trace(   wf_trace,
                     row = None,
                     col = None)


        # add axes titles
    fig.update_layout(xaxis_title="time tick", yaxis_title="adcs")

    write_image(fig)     


###########################        
def deconv_wf(w: Waveform, template: Waveform) -> Waveform:
    
    signal_fft = np.fft.fft(w.adcs)
    template_menos_fft = np.fft.fft(template.adcs, n=len(w.adcs))  # Match signal length
    deconvolved_fft = signal_fft/ (template_menos_fft )     # Division in frequency domain
    deconvolved_wf_adcs = np.real(np.fft.ifft(deconvolved_fft))      # Transform back to time domain

    
    deconvolved_wf = Waveform(w.timestamp,
                              w.time_step_ns,
                              w.daq_window_timestamp,
                              deconvolved_wf_adcs,
                              w.run_number,
                              w.record_number,
                              w.endpoint,
                              w.channel)
    
    return deconvolved_wf
        

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
def write_image(fig: go.Figure, width=None, height=None) -> None:
    """Save or display the figure based on plotting mode."""
    if plotting_mode == 'html':
        pio.write_html(fig, file=html_file_path, auto_open=True)
    elif plotting_mode == 'png':
        pio.write_image(fig, file=png_file_path, format='png', width=width, height=height)
    else:
        print(f"Unknown plotting mode '{plotting_mode}', should be 'png' or 'html'!")
        
        
#---------- Time offset histogram for [1,2,3,4] APAs ---------

def plot_to_interval(wset, 
                     apas: Union[int, list] = -1, 
                     ch: Union[int, list] = -1, 
                     nwfs: int = -1, 
                     op: str = '', 
                     nbins: int = 125, 
                     tmin: int = None, 
                     tmax: int = None, 
                     rec: list = [-1]):
    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()

    if isinstance(apas, list):
        eps_list = [get_endpoints(apa) for apa in apas]
    else:
        eps_list = [get_endpoints(apas)]

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for idx, eps in enumerate(eps_list):
        selected_wfs = get_wfs(wset.waveforms, eps, ch, nwfs, tmin, tmax, rec)
        
        times = [
            wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp
            for wf in selected_wfs
            if (
                (eps == -1 or wf.endpoint in (eps if isinstance(eps, list) else [eps])) and
                (ch == -1 or wf.channel in (ch if isinstance(ch, list) else [ch]))
            )
        ]

        color = colors[idx % len(colors)]
        histogram_trace = get_histogram(times, nbins, color)
        histogram_trace.name = f"APA {apas[idx] if isinstance(apas, list) else apas}"
        
        print(f"\nAPA {apas[idx] if isinstance(apas, list) else apas}: {len(selected_wfs)} waveforms ")
        
        fig.add_trace(histogram_trace)

    fig.update_layout(
        xaxis_title=dict(
            text="Time offset",
            font=dict(size=20)
        ),
        yaxis_title=dict(
            text="Entries",
            font=dict(size=20)
        ),
        legend=dict(
            font=dict(size=15)
        ),
        title=dict(
            text=f"Time offset histogram for all chanels in each APA",
            font=dict(size=25)
        )
    )
    
    write_image(fig)

    
#-------------- Time offset histograms in an APA grid -----------

def plot_to_function(channel_ws, idx, figure, row, col, nbins, total_rows, total_cols):

    # Compute the time offset
    times = [wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp for wf in channel_ws.waveforms]

    if not times:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return

    # Generaate the histogram
    histogram = get_histogram(times, nbins, line_width=0.5)

    # Add the histogram to the corresponding channel
    figure.add_trace(histogram, row=row, col=col)


def plot_grid_to_interval(wfset: WaveformSet,                
                          apa: int = -1, 
                          ch: Union[int, list] = -1,
                          nbins: int = 100,
                          nwfs: int = -1,
                          op: str = '',
                          tmin: int = -1,
                          tmax: int = -1,
                          rec: list = [-1]):
    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()
        
    # Obtain the endpoints from the APA
    eps = get_endpoints(apa)
    
    # Select the waveforms in a specific time interval of the DAQ window
    selected_wfs = get_wfs(wfset.waveforms, eps, ch, nwfs, tmin, tmax, rec)
    
    print(f"Number of selected waveforms: {len(selected_wfs)}")

    if not selected_wfs:
        print(f"No waveforms found for APA={apa}, Channel={ch}, Time range=({tmin}, {tmax})")
        return  

    # Obtain the channels grid
    run = wfset.waveforms[0].run_number
    grid = get_grid(selected_wfs, apa, run)

    total_rows = grid.ch_map.rows  
    total_cols = grid.ch_map.columns  

    # Plot a specific function in the APA grid
    fig = plot_CustomChannelGrid(
        grid, 
        plot_function=lambda channel_ws, idx, figure_, row, col: plot_to_function(
            channel_ws, idx, figure_, row, col, nbins, total_rows, total_cols
        ),
        x_axis_title='Time offset',  
        y_axis_title='Entries',  
        figure_title=f'Time offset histograms for APA {apa}',
        share_x_scale=True,
        share_y_scale=True,
        show_ticks_only_on_edges=True
)
    write_image(fig, 800, 1200)
    

# --------------- Sigma vs timestamp in an APA grid --------------

def plot_sigma_vs_ts_function(channel_ws, idx, figure, row, col, total_rows, total_cols):

    timestamps = []
    sigmas = []

    # Iterate over each waveform in the channel
    for wf in channel_ws.waveforms:
        # Calculate the timestamp for the waveform
        timestamp = wf._Waveform__timestamp
        timestamps.append(timestamp)

        # Calculate the standard deviation (sigma) of the ADC values
        sigma = np.std(wf.adcs)
        sigmas.append(sigma)

    # Add the histogram to the corresponding channel
    figure.add_trace(go.Scatter(
        x=timestamps,
        y=sigmas,
        mode='markers',
        marker=dict(color='black', size=2.5)  
    ), row=row, col=col)

def plot_grid_sigma_vs_ts(wfset: WaveformSet,                
                          apa: int = -1, 
                          ch: Union[int, list] = -1,
                          nwfs: int = -1,
                          op: str = '',
                          tmin: int = -1,
                          tmax: int = -1,
                          rec: list = [-1]):

    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()
        
    # Obtain the endpoints from the APA
    eps = get_endpoints(apa)
    
    # Select the waveforms in a specific time interval of the DAQ window
    selected_wfs = get_wfs(wfset.waveforms, eps, ch, nwfs, tmin, tmax, rec)
    
    print(f"Number of selected waveforms: {len(selected_wfs)}")

    if not selected_wfs:
        print(f"No waveforms found for APA={apa}, Channel={ch}, Time range=({tmin}, {tmax})")
        return  

    # Obtain the channels grid
    run = wfset.waveforms[0].run_number
    grid = get_grid(selected_wfs, apa, run)

    total_rows = grid.ch_map.rows  
    total_cols = grid.ch_map.columns  

    # Plot a specific function in the APA grid: plot_sigma_vs_ts_function
    fig = plot_CustomChannelGrid(
        grid, 
        plot_function=lambda channel_ws, idx, figure_, row, col: plot_sigma_vs_ts_function(
            channel_ws, idx, figure_, row, col, total_rows, total_cols
        ),
        x_axis_title='Timestamp',  
        y_axis_title='Sigma',  
        figure_title=f'Sigma vs timestamp for APA {apa}',
        share_x_scale=True,
        share_y_scale=True,
        show_ticks_only_on_edges=True
)
    write_image(fig, 800, 1200)
    
    
# --------------- Sigma histograms in an APA grid --------------

def plot_sigma_function(channel_ws, idx, figure, row, col, nbins, total_rows, total_cols):

    # Compute the sigmas
    sigmas = [np.std(wf.adcs) for wf in channel_ws.waveforms]

    if not sigmas:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return

    # Generate the histogram
    histogram = get_histogram(sigmas, nbins,line_width=0.5)

    # Add the histogram to the corresponding channel
    figure.add_trace(histogram, row=row, col=col)
    
def plot_grid_sigma(wfset: WaveformSet,                
                    apa: int = -1, 
                    ch: Union[int, list] = -1,
                    nbins: int = 100,
                    nwfs: int = -1,
                    op: str = '',
                    tmin: int = -1,
                    tmax: int = -1,
                    rec: list = [-1]):

    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()
        
    # Obtain the endpoints from the APA
    eps = get_endpoints(apa)
    
    # Select the waveforms in a specific time interval of the DAQ window
    selected_wfs = get_wfs(wfset.waveforms, eps, ch, nwfs, tmin, tmax, rec)
    
    print(f"Number of selected waveforms: {len(selected_wfs)}")

    if not selected_wfs:
        print(f"No waveforms found for APA={apa}, Channel={ch}, Time range=({tmin}, {tmax})")
        return  

    # Obtain the channels grid
    run = wfset.waveforms[0].run_number
    grid = get_grid(selected_wfs, apa, run)

    total_rows = grid.ch_map.rows  
    total_cols = grid.ch_map.columns  

    # Plot a specific function in the APA grid: plot_sigma_function
    fig = plot_CustomChannelGrid(
        grid, 
        plot_function=lambda channel_ws, idx, figure_, row, col: plot_sigma_function(
            channel_ws, idx, figure_, row, col, nbins, total_rows, total_cols
        ),
        x_axis_title='Sigma',  
        y_axis_title='Entries',  
        figure_title=f'Sigma histograms for APA {apa}',
        share_x_scale=True,
        share_y_scale=True,
        show_ticks_only_on_edges=True
)
    write_image(fig, 800, 1200)
    
# ------------------ fft from manuel ---------------
    
def fft(
    channel_ws,idx,dt, figure, row, col, total_rows, total_cols):
    """
    Computes the FFT of a waveform and optionally plots it.

    Parameters
    ----------
    wf : np.ndarray
        The waveform signal to transform.
    dt : float, optional
        Time step between samples (default is 16e-9 s).
    plot : bool, optional
        If True, plots the FFT using Plotly.
    figure : go.Figure, optional
        If provided, the FFT plot will be added to this figure.
    row : int, optional
        Row index for subplot placement (used with figure).
    col : int, optional
        Column index for subplot placement (used with figure).
    """
    dt=16e-9
    np.seterr(divide='ignore')  # Ignore division warnings
    freqAxisPos_list=[]
    sigFFTPos_list=[]
    
    for wf in channel_ws:
        if wf.shape[0] % 2 != 0:
            warnings.warn("Signal preferred to be even in size, auto-fixing it...")
            wf = wf[:-1]

        t = np.arange(0, wf.shape[0]) * dt
        sigFFT = np.fft.fft(wf.adcs) / wf.shape[0]
        freq = np.fft.fftfreq(wf.shape[0], d=dt)

        firstNegInd = np.argmax(freq < 0)
        freqAxisPos = freq[:firstNegInd]
        sigFFTPos = 20 * np.log10(2 * np.abs(sigFFT[:firstNegInd]) / 2**14)  # Convert to dB scale
        freqAxisPos_list.append(sigFFTPos)
        sigFFTPos_list.append(sigFFTPos)
        
        # Add FFT plot to the figure
    figure.add_trace(go.Scatter(
        x=freqAxisPos / 1e5,  # Convert to MHz
        y=sigFFTPos,
        mode='lines',
        line=dict(color='black', width=1.5),
        name="FFT"
        ), row=row, col=col)


def mean_fft(data, label, figure, row, col, total_rows, total_cols):
    """
    Calcula la media de las FFTs de una lista de señales y la grafica en un objeto Plotly Figure.

    Parámetros
    ----------
    data : list o np.ndarray
        Lista de waveforms a procesar.
    label : str
        Etiqueta para la gráfica.
    figure : plotly.graph_objects.Figure
        Figura en la que se añadirá el gráfico.
    row : int
        Fila del subplot donde se añadirá la gráfica.
    col : int
        Columna del subplot donde se añadirá la gráfica.
    total_rows : int
        Número total de filas en la figura.
    total_cols : int
        Número total de columnas en la figura.

    Retorna
    -------
    x : np.ndarray
        Eje de frecuencia promedio.
    y : np.ndarray
        Magnitud promedio de la FFT.
    stdx : float
        Desviación estándar media de las señales.
    """

    np.seterr(divide='ignore')

    fft_list_x = []
    fft_list_y = []
    std_list = []

    for k in range(len(data)):
        fft_x = fft(data[k]).x  # Usa la función fft()
        fft_y = fft(data[k]).y  
        fft_list_x.append(fft_x)
        fft_list_y.append(fft_y)
        std_list.append(np.std(data[k], axis=0))

    x = np.mean(fft_list_x, axis=0)
    y = np.mean(fft_list_y, axis=0)
    stdx = np.round(np.mean(std_list, axis=0), 3)

    # Añadir la gráfica a la figura en la posición correspondiente
    figure.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=f'{label} (rms={np.round(stdx, 2)})'
    ), row=row, col=col)

def plot_grid_fft(wfset: WaveformSet,                
                    apa: int = -1, 
                    ch: Union[int, list] = -1,
                    nwfs: int = -1,
                    op: str = '',
                    tmin: int = -1,
                    tmax: int = -1,
                    rec: list = [-1]):

    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()
        
    # Obtain the endpoints from the APA
    eps = get_endpoints(apa)
    
    # Select the waveforms in a specific time interval of the DAQ window
    selected_wfs = get_wfs(wfset.waveforms, eps, ch, nwfs, tmin, tmax, rec)
    
    print(f"Number of selected waveforms: {len(selected_wfs)}")

    if not selected_wfs:
        print(f"No waveforms found for APA={apa}, Channel={ch}, Time range=({tmin}, {tmax})")
        return  

    # Obtain the channels grid
    run = wfset.waveforms[0].run_number
    grid = get_grid(selected_wfs, apa, run)

    total_rows = grid.ch_map.rows  
    total_cols = grid.ch_map.columns  

    # Plot a specific function in the APA grid: plot_sigma_function
    fig = plot_CustomChannelGrid(
        grid, 
        plot_function=lambda channel_ws, idx, figure_, row, col: fft(
            channel_ws, idx, figure_, row, col, total_rows, total_cols
        ),
        x_axis_title='Frequency',  
        y_axis_title='fft',  
        figure_title=f'Fft plots for APA {apa}',
        share_x_scale=True,
        share_y_scale=True,
        show_ticks_only_on_edges=True
)
    write_image(fig, 800, 1200)





