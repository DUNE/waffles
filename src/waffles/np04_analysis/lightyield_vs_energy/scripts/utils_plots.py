from imports import *

def searching_for_beam_events(wfset, show : bool = False, save : bool = True, bin : int = 100000, x_min = None, x_max = None, beam_min = None, beam_max = None, output_folder : str = 'output'):
    timeoffset_list_DAQ = []
    for wf in wfset.waveforms: 
        timeoffset_list_DAQ.append(np.float32(np.int64(wf.timestamp)-np.int64(wf.daq_window_timestamp)))
    
    if x_min is None:
        x_min = min(timeoffset_list_DAQ)
    if x_max is None:
        x_max = max(timeoffset_list_DAQ)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=timeoffset_list_DAQ,
        nbinsx=bin,  
        marker=dict(color='blue', line=dict(color='black', width=1))))
    
    if beam_min is not None:
        fig.add_shape(
            type="line",
            x0=beam_min,
            x1=beam_min,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            name=f"Beam Min \n(x = {beam_min})")

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", width=2, dash="dash"),
            showlegend=True,
            name=f"Beam Min \n(x = {beam_min})"))

    if beam_max is not None:
        fig.add_shape(
            type="line",
            x0=beam_max,
            x1=beam_max,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            name=f"Beam Max \n(x = {beam_max})")

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", width=2, dash="dash"),
            showlegend=True,
            name=f"Beam Max \n(x = {beam_max})"))
    
    fig.update_layout(
        title="Interactive Histogram of Time Offsets",
        xaxis_title="Time Offset",
        yaxis_title="Count",
        xaxis=dict(range=[x_min, x_max]),  
        template="plotly_white",
        bargap=0.1)
    
    if show:
        fig.show()
    
    if save:
        if beam_min is not None and beam_max is not None:
            fig.update_layout(xaxis=dict(range=[beam_min - 200, beam_max + 200]))
        fig.write_image(f"{output_folder}/beam_timeoffset_histogram.png", scale=2)


# def plotting_overlap_wf(wfset, n_wf: int = 50, show : bool = True, save : bool = False, x_min=None, x_max=None, y_min=None, y_max=None, int_ll=None, int_ul=None, baseline=None, output_folder : str = 'output', deconvolution : bool = False, analysis_label : str = ''):
#     fig = go.Figure()

#     for i in range(n_wf):
        
#         y = wfset.waveforms[i].adcs
            
#         fig.add_trace(go.Scatter(
#             x=np.arange(len(y)) + wfset.waveforms[i].time_offset,
#             y=y,
#             mode='lines',
#             line=dict(width=0.5),
#             showlegend=False))

#     xaxis_range = dict(range=[x_min, x_max]) if x_min is not None and x_max is not None else {}
#     yaxis_range = dict(range=[y_min, y_max]) if y_min is not None and y_max is not None else {}

#     fig.update_layout(
#         xaxis_title="Time ticks",
#         yaxis_title="Adcs",
#         xaxis=xaxis_range,  
#         yaxis=yaxis_range,  
#         margin=dict(l=50, r=50, t=20, b=50),
#         template="plotly_white",
#         legend=dict(
#             x=1,  
#             y=1,  
#             xanchor="right",
#             yanchor="top",
#             orientation="v", 
#             bgcolor="rgba(255, 255, 255, 0.8)" ))

#     if int_ll is not None:
#         fig.add_shape(
#             type="line",
#             x0=int_ll,
#             x1=int_ll,
#             y0=0,
#             y1=1,
#             xref="x",
#             yref="paper",
#             line=dict(color="coral", width=2, dash="dash"),
#             name=f"Lower integral limit \n(x = {int_ll})"
#         )

#         fig.add_trace(go.Scatter(
#             x=[None], y=[None],
#             mode='lines',
#             line=dict(color="coral", width=2, dash="dash"),
#             showlegend=True,
#             name=f"Lower integral limit \n(x = {int_ll})"
#         ))

#     if int_ul is not None:
#         fig.add_shape(
#             type="line",
#             x0=int_ul,
#             x1=int_ul,
#             y0=0,
#             y1=1,
#             xref="x",
#             yref="paper",
#             line=dict(color="chocolate", width=2, dash="dash"),
#             name=f"Upper integral limit \n(x = {int_ul})"
#         )

#         fig.add_trace(go.Scatter(
#             x=[None], y=[None],
#             mode='lines',
#             line=dict(color="chocolate", width=2, dash="dash"),
#             showlegend=True,
#             name=f"Upper integral limit \n(x = {int_ul})"
#         ))

#     if baseline is not None:
#         fig.add_shape(
#             type="line",
#             x0=0,
#             x1=1,
#             y0=baseline,
#             y1=baseline,
#             xref="paper",
#             yref="y",
#             line=dict(color="red", width=1.5, dash="dash"),
#             name=f"Baseline \n(y = {baseline})"
#         )

#         fig.add_trace(go.Scatter(
#             x=[None], y=[None],
#             mode='lines',
#             line=dict(color="red", width=1.5, dash="dash"),
#             showlegend=True,
#             name=f"Baseline \n(y = {baseline})"
#         ))

#     if save:
#         fig.write_image(f"{output_folder}/waveform_plot.png", scale=2)
    
#     if show:
#         fig.show()




# Function to plot waveforms with/without peaks
def plotting_overlap_wf_PEAK_NEW(wfset, n_wf: int = 50, show : bool = True, save : bool = False, x_min=None, x_max=None, y_min=None, y_max=None, int_ll=None, int_ul=None, baseline=None, output_folder : str = 'output', analysis_label : str = 'test_peak_finding', peak_bool : bool = False, peak_beam_bool : bool = False):
    fig = go.Figure()

    if n_wf > len(wfset.waveforms):
        n_wf = len(wfset.waveforms)

    for i in range(n_wf):
        y = wfset.waveforms[i].adcs 
        x = np.arange(len(y))

            
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(width=0.5),
            showlegend=False))

        if peak_bool:
            fig.add_trace(go.Scatter(
                x=wfset.waveforms[i].analyses[analysis_label].result['peak_time'],
                y=wfset.waveforms[i].analyses[analysis_label].result['peak_amplitude'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle'),
                name='Peaks'
            ))

        
        if peak_beam_bool:
            fig.add_trace(go.Scatter(
                x=wfset.waveforms[i].analyses[analysis_label].result['beam_peak_time'],
                y=wfset.waveforms[i].analyses[analysis_label].result['beam_peak_amplitude'],
                mode='markers',
                marker=dict(color='blue', size=8, symbol='circle'),
                name='Beam peaks'
            ))

    xaxis_range = dict(range=[x_min, x_max]) if x_min is not None and x_max is not None else {}
    yaxis_range = dict(range=[y_min, y_max]) if y_min is not None and y_max is not None else {}

    fig.update_layout(
        xaxis_title="Time ticks",
        yaxis_title="Adcs",
        xaxis=xaxis_range,  
        yaxis=yaxis_range,  
        margin=dict(l=50, r=50, t=20, b=50),
        template="plotly_white",
        legend=dict(
            x=1,  
            y=1,  
            xanchor="right",
            yanchor="top",
            orientation="v", 
            bgcolor="rgba(255, 255, 255, 0.8)" ))

    if int_ll is not None:
        fig.add_shape(
            type="line",
            x0=int_ll,
            x1=int_ll,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="coral", width=2, dash="dash"),
            name=f"Lower integral limit \n(x = {int_ll})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="coral", width=2, dash="dash"),
            showlegend=True,
            name=f"Lower integral limit \n(x = {int_ll})"
        ))

    if int_ul is not None:
        fig.add_shape(
            type="line",
            x0=int_ul,
            x1=int_ul,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="chocolate", width=2, dash="dash"),
            name=f"Upper integral limit \n(x = {int_ul})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="chocolate", width=2, dash="dash"),
            showlegend=True,
            name=f"Upper integral limit \n(x = {int_ul})"
        ))

    if baseline is not None:
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=baseline,
            y1=baseline,
            xref="paper",
            yref="y",
            line=dict(color="red", width=1.5, dash="dash"),
            name=f"Baseline \n(y = {baseline})"
        )

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", width=1.5, dash="dash"),
            showlegend=True,
            name=f"Baseline \n(y = {baseline})"
        ))

    if save:
        fig.write_image(f"{output_folder}/waveform_plot.png", scale=2)
    
    if show:
        fig.show()