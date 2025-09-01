from waffles.plotting.drawing_tools_utils import *
from typing import Union
from plotly import subplots as psu

# Global plotting settings
fig = go.Figure()
html_file_path = 'temp_plot.html'
png_file_path = 'temp_plot.png'
plotting_mode='html'
line_color = 'black'
ncols=1
nrows=1
current_col=None
current_row=None

templates = []


###########################
def plot(object,                   
         ep: Union[int, list]=-1, 
         ch: Union[int, list]=-1,
         nwfs: int = -1,
         xmin: int = -1,
         xmax: int = -1,
         ymin: int = -1,
         ymax: int = -1,
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
        plot_wfs(list([object]),ep,ch,nwfs,xmin,xmax,ymin,ymax,tmin,tmax,offset,rec,op)
    
    # Case when the input object is a WaveformAdcs
    elif type(object)==WaveformAdcs:    
        plot_wfs(list([object]),-100,-100,nwfs,xmin,xmax,ymin,ymax,tmin,tmax,offset,rec,op)
    
    # Case when the input object is a list of Waveforms
    elif type(object)==list and type(object[0])==Waveform:
        plot_wfs(object,ep,ch,nwfs,xmin,xmax,ymin,ymax,tmin,tmax,offset,rec,op)

    # Case when the input object is a WaveformSet                    
    elif type(object)==WaveformSet:
        plot_wfs(object.waveforms,ep,ch,nwfs,xmin,xmax,ymin,ymax,tmin,tmax,offset,rec,op)

###########################
def plot_wfs(wfs: list,                
             ep: Union[int, list]=-1,
             ch: Union[int, list]=-1,
             nwfs: int = -1,
             xmin: int = -1,
             xmax: int = -1,
             ymin: int = -1,
             ymax: int = -1,
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
        fig=create_figure(nrows,ncols)

    # don't consider time intervals that will not appear in the plot
    if tmin == -1 and tmax == -1:
        tmin=xmin-1024    # harcoded number
        tmax=xmax        
        
    # get all waveforms in the specified endpoint, channels,  time offset range and record
    selected_wfs= get_wfs(wfs,ep,ch,nwfs,tmin,tmax,rec)

    # plot nwfs waveforms
    n=0        
    for wf in selected_wfs:
        n=n+1
        # plot the single waveform
        plot_wf(wf,fig, offset)
        if n>=nwfs and nwfs!=-1:
            break

    # add axes titles
    fig.update_layout(xaxis_title="time tick", yaxis_title="adcs")

    if xmin != -1 and xmax != -1:
        fig.update_xaxes(range = [xmin,xmax])
    
    if ymin != -1 and ymax != -1:
        fig.update_yaxes(range = [ymin,ymax])

    
    write_image(fig)     


###########################
def plot_wf( waveform_adcs : WaveformAdcs,  
             figure : pgo.Figure,
             offset: bool = False,
             name : Optional[str] = None
             ) -> None:

    """
    Plot a single waveform
    """
    
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

    figure.add_trace(wf_trace,current_row,current_col)

###########################
def plot_grid(wfset: WaveformSet,                
              apa: int = -1, 
              ch: Union[int, list]=-1,
              nwfs: int = -1,
              tmin: int = -1,
              tmax: int = -1,
              xmin: int = -1,
              xmax: int = -1,
              ymin: int = -1,
              ymax: int = -1,
              offset: bool = False,
              rec: list = [-1],
              mode: str = 'overlay'):

    """
    Plot a WaveformSet in grid mode
    """

    # don't consider time intervals that will not appear in the plot
    if tmin == -1 and tmax == -1:
        tmin=xmin-1024    # harcoded number
        tmax=xmax        
    
    # get the endpoints corresponding to a given APA
    eps= get_endpoints(apa)

    # get all waveforms in the specified endpoint, channels,  time offset range and record
    selected_wfs= get_wfs(wfset.waveforms,eps,ch,nwfs,tmin,tmax,rec)

    run = wfset.waveforms[0].run_number
    
    # get the ChannelWsGrid for that subset of wafeforms and that APA
    grid = get_grid(selected_wfs,apa,run)

    # plot the grid
    global fig
    fig = plot_ChannelWsGrid(grid, wfs_per_axes=1000,mode=mode,offset=offset)

    if xmin != -1 and xmax != -1:
        fig.update_xaxes(range = [xmin,xmax])
    
    if ymin != -1 and ymax != -1:
        fig.update_yaxes(range = [ymin,ymax])
    
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
        fig = create_figure(nrows,ncols)
    
    # get number of channels with wfs 
    nchs = [ev.get_nchannels() for ev in events]

    # build an histogram with those times
    histogram_trace = get_histogram_trace(nchs, nbins, xmin, xmax)
    
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
        fig = create_figure(nrows,ncols)
    
    # get number of channels with wfs 
    if type == 'ref':
        times = [ev.ref_timestamp*1e-9*16 for ev in events]
    elif type == 'first':
        times = [ev.first_timestamp*1e-9*16 for ev in events]
    if type == 'last':
        times = [ev.last_timestamp*1e-9*16 for ev in events]

    # build an histogram with those times
    histogram_trace = get_histogram_trace(times, nbins, xmin, xmax)
    
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
        fig = create_figure(nrows,ncols)

        
    # get the time offset for all wfs in the specific ep and channel
    times = [wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp
             for wf in wset.waveforms
             if (wf.endpoint == ep or ep == -1) and (wf.channel==ch or ch == -1)]
    
    # build an histogram with those times
    histogram_trace = get_histogram_trace(times, nbins, xmin, xmax)
    
    fig.add_trace(histogram_trace)
    fig.update_layout(xaxis_title="time offset", yaxis_title="entries")
    
    
    write_image(fig)


###########################
def plot_abs_to(wset: WaveformSet,
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
        fig = create_figure(nrows,ncols)

    # sort waveforms by time stamp
    tsort_wfset(wset)

    t0 = wset.waveforms[0].timestamp
    
    # get the timestamp for all wfs in the specific ep and channel
    times = [(wf.timestamp-t0)*16e-9
             for wf in wset.waveforms
             if (wf.endpoint == ep or ep == -1) and (wf.channel==ch or ch == -1)]

    # build an histogram with those times
    histogram_trace = get_histogram_trace(times, nbins, xmin, xmax)

    fig.add_trace(histogram_trace,current_row,current_col)
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
        fig = create_figure(nrows,ncols)

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
        fig=create_figure(nrows,ncols)

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
        fig=create_figure(nrows,ncols)

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
        fig=create_figure(nrows,ncols)

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
        fig=create_figure(nrows,ncols)    

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
        fig=create_figure(nrows,ncols)    

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
        fig=create_figure(nrows,ncols)    

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
        fig=create_figure(nrows,ncols)

    
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
        

###########################
def plot_grid_histogram(wfset: WaveformSet,          
                        wf_func: Callable, 
                        apa: int = -1,                          
                        ch: Union[int, list] = -1,
                        nbins: int = 100,
                        nwfs: int = -1,
                        op: str = '',
                        xmin: np.int64 = None,
                        xmax: np.int64 = None,
                        tmin: int = -1,
                        tmax: int = -1,
                        rec: list = [-1]):
    """
    Plot a WaveformSet in grid mode, generating a histogram per channel.
    """
    global fig
    if not has_option(op, 'same'):
        fig = create_figure(nrows,ncols)

    # don't consider time intervals that will not appear in the plot
    if tmin == -1 and tmax == -1:
        tmin=xmin
        tmax=xmax        
        
    # Obtener los endpoints para el APA
    eps = get_endpoints(apa)
    
    # Obtener solo las waveforms que cumplen las condiciones
    selected_wfs = get_wfs(wfset.waveforms, eps, ch, nwfs, tmin, tmax, rec)
    
    print(f"Number of selected waveforms: {len(selected_wfs)}")

    # Si no hay waveforms, detener la ejecución
    if not selected_wfs:
        print(f"No waveforms found for APA={apa}, Channel={ch}, Time range=({tmin}, {tmax})")
        return  

    # Obtener la cuadrícula de canales
    run = wfset.waveforms[0].run_number
    grid = get_grid(selected_wfs, apa, run)
    
    # Pasar la función correcta para graficar histogramas, con filtrado por canal
    fig = plot_CustomChannelGrid(
        grid,
        plot_function=lambda channel_wfs, idx, figure_, row, col, func, *args, **kwargs: plot_histogram_function_user(
            wf_func,channel_wfs, idx, figure_, row, col, nbins, xmin, xmax),
        x_axis_title='Time offset',  # Se configura después en función de la posición
        y_axis_title='Entries',  # Se configura después en función de la posición
        figure_title=f'Time offset histogram for APA {apa}',
        share_x_scale=True,
        share_y_scale=True,
        wf_func=wf_func

    )
    write_image(fig, 800, 1200)

###########################    
def plot_histogram_function_user(wf_func: Callable, channel_ws, idx, figure, row, col, nbins, xmin, xmax):
    """
    Función para generar el histograma de un canal específico.
    """
    # Extraer los tiempos de las waveforms del canal específico
    values = [wf_func(wf) for wf in channel_ws.waveforms]
    
    # Si no hay datos, no graficar nada
    if not values:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return

    # Generar el histograma
    histogram_trace = get_histogram_trace(values, nbins, xmin, xmax, line_width=0.5)

    # Añadir el histograma al subplot correspondiente
    if row ==0 and col==0:
        figure.add_trace(histogram_trace)
    else:
        figure.add_trace(histogram_trace, row=row, col=col)


###########################
def plot_histogram(wfset: WaveformSet,          
                   wf_func: Callable, 
                   eps: Union[int, list] = -1,                          
                   ch: Union[int, list] = -1,
                   nbins: int = 100,
                   nwfs: int = -1,
                   op: str = '',
                   xmin: np.int64 = None,
                   xmax: np.int64 = None,
                   tmin: int = -1,
                   tmax: int = -1,
                   rec: list = [-1]):
    """
    Plot a WaveformSet histogram
    """
    global fig
    if not has_option(op, 'same'):
        fig = create_figure(nrows,ncols)

    # don't consider time intervals that will not appear in the plot
    #if tmin == -1 and tmax == -1:
    #    tmin=xmin
    #    tmax=xmax        
        
    # Obtener solo las waveforms que cumplen las condiciones
    selected_wfs = get_wfs(wfset.waveforms, eps, ch, nwfs, tmin, tmax, rec)
    channel_ws = ChannelWs(*selected_wfs)

    print(f"Number of selected waveforms: {len(selected_wfs)}")

    # Si no hay waveforms, detener la ejecución
    if not selected_wfs:
        print(f"No waveforms found for ep={eps}, Channel={ch}, Time range=({tmin}, {tmax})")
        return  

    # Pasar la función correcta para graficar histogramas, con filtrado por canal
    plot_histogram_function_user(wf_func,channel_ws, 1, fig, 0, 0, nbins, xmin, xmax)


    write_image(fig, 800, 1200)


###########################
def plot_event_histogram(wfset: WaveformSet,          
                         wf_func: Callable,
                         ev_func: Callable, 
                         eps: Union[int, list] = -1,                          
                         ch: Union[int, list] = -1,
                         nbins: int = 100,
                         nwfs: int = -1,
                         op: str = '',
                         xmin: np.int64 = None,
                         xmax: np.int64 = None,
                         tmin: int = -1,
                         tmax: int = -1,
                         rec: list = [-1]):
    """
    Plot a WaveformSet histogram
    """
    global fig
    if not has_option(op, 'same'):
        fig = create_figure(nrows,ncols)

    # don't consider time intervals that will not appear in the plot
    #if tmin == -1 and tmax == -1:
    #    tmin=xmin
    #    tmax=xmax        

    values = []

    selected_wfs = get_wfs(wfset.waveforms, eps, ch, nwfs, tmin, tmax)
    
    for r in rec:
    
        # Get the selected waveforms
        selected_wfs_rec = get_wfs(selected_wfs, rec=[r])

        wf_values = []
        for wf in selected_wfs_rec:
            f = wf_func(wf)
            if f != -10000000:
                wf_values.append(f)
        
        value = ev_func(wf_values)

        values.append(value)
        
    # if there is no data return
    if not values:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return

    # generate the histogram
    histogram_trace = get_histogram_trace(values, nbins, xmin, xmax, line_width=0.5)

    # Aadd the histogram
    fig.add_trace(histogram_trace)

    write_image(fig, 800, 1200)
    
    

############################
def plot_to_interval(wset, 
                     apa: Union[int, list] = -1, 
                     ch: Union[int, list] = -1, 
                     nwfs: int = -1, 
                     op: str = '', 
                     nbins: int = 125, 
                     tmin: int = None, 
                     tmax: int = None, 
                     xmin: np.uint64 = None, 
                     xmax: np.uint64 = None, 
                     rec: list = [-1]):
    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()

    if isinstance(apa, list):
        eps_list = [get_endpoints(apa_value) for apa_value in apa]
    else:
        eps_list = [get_endpoints(apa)]

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
        histogram_trace = get_histogram_trace(times, nbins, tmin, tmax, color)
        histogram_trace.name = f"APA {apa[idx] if isinstance(apa, list) else apa}"
        
        print(f"\nAPA {apa[idx] if isinstance(apa, list) else apa}: {len(selected_wfs)} waveforms ")
        
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

    
###########################

#-------------- Time offset histograms -----------

def plot_to_function(channel_ws, apa,idx, figure, row, col, nbins):

    # Compute the time offset
    times = [wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp for wf in channel_ws.waveforms]

    if not times:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return

    # Generaate the histogram
    histogram_trace = get_histogram_trace(times, nbins, line_width=0.5)

    # Return the axis titles and figure title along with the figure
    x_axis_title = "Time offset"
    y_axis_title = "Entries"
    figure_title = f"Time offset histograms for APA {apa}"
    
    if figure is None:
        return x_axis_title, y_axis_title, figure_title
    
    # Add the histogram to the corresponding channel
    figure.add_trace(histogram_trace, row=row, col=col)
    
    return figure


# --------------- Sigma vs timestamp  --------------

def plot_sigma_vs_ts_function(channel_ws, apa,idx, figure, row, col,nbins):

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
    
    # Return the axis titles and figure title along with the figure
    x_axis_title = "Timestamp"
    y_axis_title = "Sigma"
    figure_title = f"Sigma vs timestamp for APA {apa}"
    
    if figure is None:
        return x_axis_title, y_axis_title, figure_title
    
    # Add the histogram to the corresponding channel
    figure.add_trace(go.Scatter(
        x=timestamps,
        y=sigmas,
        mode='markers',
        marker=dict(color='black', size=2.5)  
    ), row=row, col=col)
    
    return figure


# --------------- Sigma histograms  --------------
 
def plot_sigma_function(channel_ws, apa, idx, figure, row, col, nbins):
    
    # Compute the sigmas
    
    sigmas = [np.std(wf.adcs) for wf in channel_ws.waveforms]

    if not sigmas:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return None, None, None, None  # Return None if no data
    
        
    # Generate the histogram
    histogram_trace = get_histogram_trace(sigmas, nbins, line_width=0.5)

    # Return the axis titles and figure title along with the figure
    x_axis_title = "Sigma"
    y_axis_title = "Entries"
    figure_title = f"Sigma histograms for APA {apa}"
    
    if figure is None:
        return x_axis_title, y_axis_title, figure_title
    
    # Add the histogram to the corresponding channel
    figure.add_trace(histogram_trace, row=row, col=col)
    
    return figure

# -------------------- Mean FFT --------------------

def plot_meanfft_function(channel_ws, apa, idx, figure, row, col, nbins):

    waveform_sets = {
        "[-1000, -500]": get_wfs_interval(channel_ws.waveforms, -1000, -500),
        "[-450, -300]": get_wfs_interval(channel_ws.waveforms, -450, -300),
        "[0, 300]": get_wfs_interval(channel_ws.waveforms,0, 300),
        "[600, 1000]": get_wfs_interval(channel_ws.waveforms, 600, 1000),
        "[2000, 5000]": get_wfs_interval(channel_ws.waveforms, 2000, 5000)
    }
    
    np.seterr(divide='ignore')  
    
    # Different colors for each range
    colors = ['blue', 'red', 'green', 'purple', 'orange']  
    
    # Return the axis titles and figure title along with the figure
    x_axis_title = "Frequency [MHz]"
    y_axis_title = "Power [dB]"
    figure_title = f"Superimposed FFT of Selected Waveforms for APA {apa}"
    
    if figure is None:
        return x_axis_title, y_axis_title, figure_title


    for i, (label, selected_wfs) in enumerate(waveform_sets.items()):
        if not selected_wfs:
            print(f"No waveforms found for range {label}")
            continue

        fft_list_x = []
        fft_list_y = []

        # Compute the FFT
        for wf in selected_wfs:
            tmpx, tmpy = fft(wf.adcs) 
            fft_list_x.append(tmpx)
            fft_list_y.append(tmpy)

        # Compute the mean FFT
        freq = np.mean(fft_list_x, axis=0)
        power = np.mean(fft_list_y, axis=0)

        figure.add_trace(go.Scatter(
            x=freq,
            y=power,
            mode='lines',
            name=f"FFT {label}",
            line=dict(color=colors[i % len(colors)], width=1)
        ), row=row, col=col)  
    
    return figure  

# ----------- Plot a specific function in an APA grid ---------

def plot_function_grid(wfset: WaveformSet,                
                    apa: int = -1, 
                    ch: Union[int, list] = -1,
                    nbins: int = 120,
                    nwfs: int = -1,
                    op: str = '',
                    tmin: int = -1,
                    tmax: int = -1,
                    rec: list = [-1],
                    plot_function: Callable = None,
                    x_axis_title: str = None,
                    y_axis_title: str = None,
                    figure_title: str = None,
                    share_x_scale=True,
                    share_y_scale=True,
                    show_ticks_only_on_edges=True,
                    func: Callable=None):  

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
    
    # Get the x_axis_title, y_axis_title and figure_title
    x_axis_title, y_axis_title, figure_title = plot_function(wfset, apa, 0, None, 1, 1, nbins)
    
    grid = get_grid(selected_wfs, apa, run)

    # Ensure plot_function is provided
    if plot_function is None:
        raise ValueError("plot_function must be provided")
    
    # Plot using the provided function
    fig= plot_CustomChannelGrid(
        grid, 
        plot_function=lambda channel_ws, idx, figure_, row, col, func: plot_function(
            channel_ws, apa, idx, figure_, row, col, nbins),
        x_axis_title=x_axis_title,  
        y_axis_title=y_axis_title,  
        figure_title=figure_title,
        share_x_scale=share_x_scale,
        share_y_scale=share_y_scale,
        show_ticks_only_on_edges=show_ticks_only_on_edges
    )

    # Return the final figure and axis titles
    write_image(fig, 800, 1200)
    
    
# -------------------- FFT plots ----------------------

def fft(sig, dt=16e-9):
    np.seterr(divide = 'ignore')
    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
    else:
        t = np.arange(0, sig.shape[-1]) * dt
    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[:-1]
        sig = sig[:-1]
    sigFFT = np.fft.fft(sig) / t.shape[0]
    freq = np.fft.fftfreq(t.shape[0], d=dt)
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[:firstNegInd]
    sigFFTPos = 2 * sigFFT[:firstNegInd]
    x = freqAxisPos /1e6
    y = 20*np.log10(np.abs(sigFFTPos)/2**14)
    return x,y


def plot_fft(wf: Waveform, dt: int=16e-9,  op: str=''):
    sig=wf.adcs
    global fig
    if not has_option(op,'same'):
        fig=go.Figure()

    np.seterr(divide = 'ignore')
    if dt is None:
        dt = 1
        t = np.arange(0, sig.shape[-1])
    else:
        t = np.arange(0, sig.shape[-1]) * dt
    if sig.shape[0] % 2 != 0:
        warnings.warn("signal preferred to be even in size, autoFixing it...")
        t = t[:-1]
        sig = sig[:-1]
    sigFFT = np.fft.fft(sig) / t.shape[0]
    freq = np.fft.fftfreq(t.shape[0], d=dt)
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[:firstNegInd]
    sigFFTPos = 2 * sigFFT[:firstNegInd]
    
    wf_trace = pgo.Scatter( x =  freqAxisPos /1e6,
                            y = 20*np.log10(np.abs(sigFFTPos)/2**14),
                            mode = 'lines',
                            line=dict(color=line_color, width=0.5))
                            
    fig.add_trace(   wf_trace,
                     row = None,
                     col = None)

    fig.update_layout(xaxis_title="samples", yaxis_title="freq [Hz]",xaxis_type='log')

    write_image(fig)    
    
def plot_meanfft(wfs: list,                
                 ep: int = -1, 
                 ch: Union[int, list] = -1,
                 nwfs: int = -1,
                 rec: list = [-1],
                 op: str = ''):
    
    # Define waveform sets with different time ranges
    waveform_sets = {
        "[-1000, -500]": get_wfs(wfs.waveforms, [ep], ch, nwfs, -1000, -500, rec),
        "[-450, -300]": get_wfs(wfs.waveforms, [ep], ch, nwfs, -450, -300, rec),
        "[0, 300]": get_wfs(wfs.waveforms, [ep], ch, nwfs, 0, 300, rec),
        "[600, 1000]": get_wfs(wfs.waveforms, [ep], ch, nwfs, 600, 1000, rec),
        "[2000, 5000]": get_wfs(wfs.waveforms, [ep], ch, nwfs, 2000, 5000, rec)
    }

    global fig
    if not has_option(op, 'same'):
        fig = go.Figure()

    np.seterr(divide='ignore')  # Ignore division warnings

    colors = ['blue', 'red', 'green', 'purple', 'orange']  # Different colors for plots

    for i, (label, selected_wfs) in enumerate(waveform_sets.items()):
        if not selected_wfs:
            print(f"No waveforms found for range {label}")
            continue

        fft_list_x = []
        fft_list_y = []

        # Compute FFT for each waveform in the selected set
        for wf in selected_wfs:
            tmpx, tmpy = fft(wf.adcs)  # Compute FFT
            fft_list_x.append(tmpx)
            fft_list_y.append(tmpy)

        # Compute mean FFT for the waveform set
        freq = np.mean(fft_list_x, axis=0)
        power = np.mean(fft_list_y, axis=0)

        # Plot FFT for this set
        fig.add_trace(go.Scatter(
            x=freq,
            y=power,
            mode='lines',
            name=f"FFT {label}",  # Label each line
            line=dict(color=colors[i % len(colors)], width=1)
        ))

    # Configure the layout
    fig.update_layout(
        title="Superimposed FFT of Selected Waveforms",
        xaxis_title="Frequency [MHz]",
        yaxis_title="Power [dB]",
        xaxis_type='log'
    )

    # Display the figure
    fig.show()

