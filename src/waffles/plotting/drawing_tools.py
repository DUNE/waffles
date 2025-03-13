from waffles.plotting.drawing_tools_utils import *
from typing import Union

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
        histogram_trace = get_histogram(times, nbins, tmin, tmax, color)
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

def plot_histogram_function(channel_ws, idx, figure, row, col, nbins, xmin, xmax, total_rows, total_cols):
    """
    Función para generar el histograma de un canal específico.
    """
    # Extraer los tiempos de las waveforms del canal específico
    times = [wf._Waveform__timestamp - wf._Waveform__daq_window_timestamp for wf in channel_ws.waveforms]

    # Si no hay datos, no graficar nada
    if not times:
        print(f"No waveforms for channel {channel_ws.channel} at (row {row}, col {col})")
        return

    # Generar el histograma
    histogram = get_histogram(times, nbins, xmin, xmax, line_width=0.5)

    # Añadir el histograma al subplot correspondiente
    figure.add_trace(histogram, row=row, col=col)


def plot_grid_to_interval(wfset: WaveformSet,                
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
        fig = go.Figure()
        
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

    # Obtener el tamaño de la cuadrícula
    total_rows = grid.ch_map.rows  
    total_cols = grid.ch_map.columns  

    # Pasar la función correcta para graficar histogramas, con filtrado por canal
    fig = plot_CustomChannelGrid(
        grid, 
        plot_function=lambda channel_ws, idx, figure_, row, col, *args, **kwargs: plot_histogram_function(
            channel_ws, idx, figure_, row, col, nbins, xmin, xmax, total_rows, total_cols
        ),
        x_axis_title='Time offset',  # Se configura después en función de la posición
        y_axis_title='Entries',  # Se configura después en función de la posición
        figure_title=f'Time offset histogram for APA {apa}',
        share_x_scale=True,
        share_y_scale=True
)
    write_image(fig, 800, 1200)




