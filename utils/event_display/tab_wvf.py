import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc,html
import numpy as np
import pandas as pd # remove this dependency later
import yaml

from utils import custom_plotly_layout

class WaveformDisplay:
    '''
    This class is used to visualize the waveforms:
    - Event by event
    - All waveforms
    You can choose the number of waveforms and APAs to display.

    '''
    def __init__(self,):
        self.generate_layout()

    def construct_waveforms(self):
        '''
        This function creates the figures to acomodate the grids of accumulated waveforms if they do not exist.
        '''

        #check if any of the 7 figures has title to maintain the waveforms
        try: 
            for i in range(7): 
                title = self.waveforms[i].layout.title.text
                if title !=None: print(f"We already have some waveforms for ep{title}!")
                del title
        # if not, create the waveforms
        except AttributeError: 
            print("Creating waveforms from scratch!")
            self.waveforms = [go.Figure() for i in range(7)]

        phys_pos = [('50%','33%',),('50%','33%'),('50%','33%'),('50%','50%'),('50%','25%'),('50%','25%'),('50%','50%')]
        self.graphs = [dcc.Graph(figure=fig, style={'width': phys_pos[f][0],'height': phys_pos[f][1], 'display': 'inline-block'}) for f,fig in enumerate(self.waveforms)]
    
    def plot_waveform(self, raw_waveforms, ep, n_wvf):
        '''
        Temporary function to generate grid from loaded root file.
        In the future we will use the waffles methods and classes to produce this figures.
        '''

        print(f"Plotting waveforms for endpoint {ep}") #debugging, remove later
        with open('channel_map.yml', 'r') as file: physical_map = yaml.safe_load(file)
        
        all_eps = ["104","105","107","109","111","112","113"]
        ep_idx = np.where(np.array(all_eps)==ep)[0][0]
        cols = 4
        rows = int(np.ceil(len(physical_map[ep]) / cols))
        heights ={1:300,2:400,4:550,9:1000,10:1200} #look for a better way if waffles method do not solve this
        print(heights[rows])
        print(f"Rows: {rows}, Cols: {cols}")
        subtitles = [f"Ch: {ch[3:]}, N: {n_wvf}" for ch in physical_map[ep]]
        self.waveforms[ep_idx] = make_subplots(rows, cols,
                                       x_title='Time [ticks]',
                                       y_title='ADC counts',
                                       vertical_spacing=0.04,
                                       horizontal_spacing=0.02,
                                       shared_xaxes=True, shared_yaxes=True, 
                                       subplot_titles=subtitles)
        self.waveforms[ep_idx] = custom_plotly_layout(self.waveforms[ep_idx],legend_pos=["out","v"],figsize=(1200,heights[rows]))

        appeared_wf = set()
        for ch, channel in enumerate(physical_map[ep]):
            ch_idx = np.where(raw_waveforms['channel']==int(channel))[0]
            raw_wf = raw_waveforms['adcs'][ch_idx][:n_wvf]
            if len(raw_wf) > 0:
                for widx,w in enumerate(raw_wf):
                    w_p = np.multiply(w-np.mean(w[:40]),-1)
                    if np.max(w_p) < 16 and np.max(w_p) > 60: continue
                    row = ch // cols + 1 
                    col = ch % cols + 1
                    showlegend = widx not in appeared_wf
                    appeared_wf.add(widx)
                    # Add a trace to the subplot
                    self.waveforms[ep_idx].add_trace(go.Scatter(y=w_p[0:500], line=dict(width=0.4), name=f"Event {widx}", legendgroup=f'group{widx}', showlegend=showlegend), row=row, col=col)
                    self.waveforms[ep_idx].update_yaxes(range=[-30,250])
        self.waveforms[ep_idx].update_layout(title={'text':f"Waveforms for endpoint {ep}",'y':0.9})
        self.waveforms[ep_idx].show() ## TODO: remove this when we opimize the plots to appear quickly --> waffles classes

    def generate_layout(self):
        self.construct_waveforms()
        self.layout = html.Div(
            [
                dcc.Location(id="url"),
                html.H1(children="Plot all your Waveforms together!", style={"textAlign": "center"}),
                html.Div(id='waveform',children=self.graphs),
            ], style={"max-width": "100%", "margin": "auto"}
        )