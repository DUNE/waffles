import os, yaml, pickle, json
import plotly.subplots as psu
import plotly.graph_objects as go
from dash import (Dash, dcc, html, callback_context)
from dash.dependencies import Input, Output
# from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import waffles.input.raw_hdf5_reader as reader
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.BasicWfAna import BasicWfAna 
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.np04_data.ProtoDUNE_HD_APA_maps import APA_map
from waffles.np04_analysis.np04_ana import allow_certain_endpoints
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.plotting.display.set_server import SetServer

class Display:
    """
    A class to create a Dash app for displaying the results of the NP04 PDS analysis.
    """
    def __init__(
        self,
    ):
        """
        Initialize the Display class.

        Args (update)????:
            service_prefix (str, optional): _description_. Defaults to os.getenv("JUPYTERHUB_SERVICE_PREFIX", "/").
            server_url (_type_, optional): _description_. Defaults to "https://lxplus940.cern.ch"
            port (int, optional): _description_. Defaults to 8050.
        """

        server = SetServer()

        config = server.get_config()
        
        self.service_prefix = config['service_prefix']
        self.server_url = config['server_url']
        self.port = config['port']
        self.jupyter_mode = config['jupyter_mode']
        
        if self.jupyter_mode == "inline":
            display_path = os.getcwd()
            self.waffles_path = display_path
        else: 
            display_path = os.path.dirname(os.path.abspath(__file__))
            self.waffles_path = "/".join(display_path.split('/')[:-4])
        print(self.waffles_path)
        # self.waffles_path = "/afs/cern.ch/work/l/lperez/ProtoDUNE-HD/waffles/"
        self.basefolder = ''
        self.run_folder = []
        self.root_files = []
        self.my_file  = ''
        self.loaded   = False
        self.wfset    = None
        self.aux_plot = []
        
        self.layout = None
        self.rows = 10
        self.cols = 4
        self.figures  = [go.Figure() for _ in range(4)] # In pple we will need 4 figures (one per APA)
        self.phys_pos = [('50%', '130vh'), ('50%', '130vh'), ('50%', '130vh'), ('50%', '130vh')]
        self.graphs   = [dcc.Graph(figure=fig, style={'width': self.phys_pos[f][0], 'height': self.phys_pos[f][1], 'display': 'inline-block'}) for f, fig in enumerate(self.figures)]

        self.geometry_info = {
            'det': [],
        }
        
        self.apa2ep = {"1":[104,105,107], "2":[109], "3":[111], "4":[112,113]}

        self.construct_app()
        self.construct_widgets()
        self.run_app()


    def adjust_iframe_height(self, height=1000):
        """
        Generates a script to adjust the iframe height for the Dash app when running in Jupyter.
        Parameters:
            height (int): The desired height of the iframe in pixels.
        """

        from IPython.display import display, HTML
        script = f"""
        <script>
        // You might need to adjust the selector depending on your Jupyter environment
        const iframes = document.querySelectorAll('iframe');
        iframes.forEach(function(iframe) {{ iframe.style.height = '{height}px'; }});
        </script>
        """
        display(HTML(script))

    def construct_app(self):
        """
        Construct the Dash app.
        """

        if self.jupyter_mode == "inline":
            self.app = Dash( __name__,
                            requests_pathname_prefix=f"{self.service_prefix}proxy/{self.port}/",
                            # requests_pathname_prefix=f"{self.service_prefix}proxy/{self.port}/",
                            external_stylesheets=[dbc.themes.FLATLY] )
        else:
            self.app = Dash( __name__,
                            external_stylesheets=[dbc.themes.FLATLY] )

        """Get the custom style file"""
        with open(f'assets/styles.yaml', 'r') as file: styles = yaml.safe_load(file)

        """Define the navbar"""
        self.navbar = html.Div(
            children=[
                html.A(
                    href="https://github.com/DUNE/waffles",
                    target="_blank",  # Opens the link in a new tab
                    children=[html.Img(src=f'assets/neutrino.png', style={'height': '35px', 'marginRight': '15px', 'float':'right'}),],
                    style={'display': 'flex', 'alignItems': 'center', 'textDecoration': 'none'},
                ),
                html.H3("DUNE PDS TOOLS", style={'font-weight': 'bold'}),
            ],
            style=styles['NAVBAR_STYLE']
        )

        """Define the sidebar"""
        self.sidebar = html.Div(
            [
                # DUNE logo
                html.Div(
                    children=[
                            html.Img(src=(f'assets/WAFFLES.PNG'), style={'height': '100px', 'marginRight': '15px'}),
                            html.H3("WAFFLES DISPLAY", style={'font-weight': 'bold'}),
                            ],
                    style={'display': 'flex', 'alignItems': 'center'}
                ),

                # Standard folder selection dropdown
                html.Hr(style={'border': '3px solid #ffffff', 'height': '0px'}),

                html.P("Choose your format + location ⚙️"),
                html.Div([
                            dcc.RadioItems(
                                options=[
                                    {'label': 'LOCAL', 'value': 'LOCAL'},
                                    {'label': 'RUCIO', 'value': 'RUCIO'}
                                ],
                                value='LOCAL',  # Default selected value
                                id="location",
                                inline=True,
                                labelStyle={'margin-right': '20px'},  # Add margin to separate the items
                                style={'width': '30vw'}
                            ),
                            html.H1(),
                            dcc.RadioItems(
                                options=[
                                    {'label': '.pkl',  'value': '.pkl'},
                                    {'label': '.hdf5', 'value': '.hdf5'}
                                ],
                                value='.pkl',  # Default selected value
                                id="file_format",
                                inline=True,
                                labelStyle={'margin-right': '42px'},  # Add margin to separate the items
                                style={'width': '30vw'}
                            ),
                        ]),
                

                html.Hr(style={'border': '3px solid #ffffff', 'height': '0px'}),
                # Text box for writing load_folders
                dbc.Label("📂 Folder"),
                dbc.Input(
                    placeholder="Enter the base folder path",
                    type="text",
                    id='folder_input',
                    size='sm',
                    value=self.basefolder,
                ),
                html.H2(),

                # Dropdown which lists available run_folders
                html.Label('📝 Run'),
                dcc.Dropdown(
                    id='run_dropdown',
                    searchable=True,
                    placeholder='Select a run folder...',
                    style={'color': "#000000"}
                ),
                html.H2(),

                # Event display selector
                html.H2(),
                html.Hr(),
                html.P(
                    "Choose a Display",
                    className="lead"
                ),
                dcc.Dropdown(
                    id='display_dropdown',
                    value='Home',
                    options=['Home', 'Event Display', 'Individual Waveform Persistence', 'Heatmap Persistence', 'Fast Fourier Transform (Noise)'],
                    style={'color': "#000000"}
                ),
                html.Hr(style={'border': '1px solid #ffffff', 'height': '0px'}),
                html.Div([
                            html.Div([
                                        html.P("Choose your APA(s):"),
                                        dcc.Checklist(
                                                        ['1', '2', '3', '4'], # options
                                                        ['3','4'], # values
                                                        id="apas",
                                                        inline=True,
                                                        labelStyle={'margin-right': '20px'},  # Add margin to separate the items
                                                        style={'width': '30vw'}),
                                    ]),
                            html.Div(style={'width': '5vw'}),
                            html.Div([
                                        html.H2(),
                                        html.P("Number of waveforms"),
                                        dcc.Input(id="n_wvf", type="number", placeholder="", debounce=False, min=5, max=1500, step=20, value=500),
                                        html.H2(),
                                        html.P("Event number"),
                                        dcc.Input(id='number_event', type='number', style={'width': '10vw'}),
                                    ], style={'width': '45vw'}),
                        ], style={'width': '30vw'}),
                html.H2(),
                html.Div([
                            html.H2(),
                            dbc.Button("PLOT", id="plot-button", color="primary", n_clicks=0, style={'width': '19vw'}),
                        ])
            ],
            style=styles['SIDEBAR_STYLE'],
        )

        # Define content for the tabs
        self.content = html.Div(id="page-content", style=styles["CONTENT_STYLE"])
        # Define the layout
        self.app.layout = html.Div(style={'overflow': 'scroll'}, children=[
            dcc.Location(id="url"),
            self.navbar,
            self.sidebar,
            self.content,
        ])

    def construct_widgets(self):
        '''
        Construct the widgets for the Dash app.
        '''
        # Callbacks to update the content based on which tab is selected
        @self.app.callback(
            Output('page-content',     'children'),
            [Input('display_dropdown', 'value'),
             Input('folder_input',     'value'),
             Input('run_dropdown',     'value'),
             Input('n_wvf',            'value'),
             Input('plot-button',      'n_clicks'),
             Input('number_event',     'value'),
             Input('apas',             'value')
            ]
        )
        def render_tab_content(pathname,my_folder,my_run,n_wvf,plot_nclicks,event,apa_value):
            '''
            Callback to render the content of the tabs. Depending on the selected tab, it will render different content.
            Also it will check if the user has selected a file to plot and clicked the plot button.
            '''
            
            triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else ''
            if my_run is not None and my_folder is not None and triggered_id == 'plot-button' and plot_nclicks > 0:
                print("You clicked the button! (and we have the info needed)")
                if self.wfset is not None:
                    print("We have a wvset to plot!")
                else:
                    self.my_file = my_folder +"/"+ my_run
                    
                    if self.my_file.endswith('.pkl'):
                        print("Loading pickle file...")
                        with open(self.my_file, 'rb') as f:
                            self.wfset = pickle.load(f)
                    if self.my_file.endswith('.hdf5'):
                        print("Loading hdf5 file...")
                        self.wfset = reader.WaveformSet_from_hdf5_file( self.my_file, read_full_streaming_data = False )      
                    if self.my_file.endswith('.txt'):
                        print("Loading rucio file...")
                        filepaths = reader.get_filepaths_from_rucio(self.my_file)
                        self.wfset = reader.WaveformSet_from_hdf5_file( filepaths[:1], read_full_streaming_data = False )      

                    self.loaded = True
                print("\nWAVEFORM SET GENERATED CORRECTLY!\n") ## TODO:Print this below the run dropdown
            else:
                print("No file selected")
                
            #### HOME PAGE ####
            if pathname == "Home": 
                print("Home page")
                self.layout = html.Div([
                    html.H1(children="Welcome to our main page! 🏠 ", style={"textAlign": "center"}),
                    html.Hr(),
                    html.P(dcc.Markdown('''For visualizing a run you can either have it locally or search it with the rucio paths.''')),
                    html.P(dcc.Markdown('''You need to select the format of the files you are going to load and their location.''')),
                    html.P(dcc.Markdown('''Please notice that the optimal plotting strategy is to save you WaveformSet in a `.pkl` so we just need to plot it here.''')),
                    html.P(dcc.Markdown('''* There are some default folders available when you select `LOCAL` or `RUCIO`. You can also enter yours in the `📂 Folder`.''')),
                    html.P(dcc.Markdown('''* Then you need to choose between the runs that are inside `📝 Run`''')),
                    html.P(dcc.Markdown('''* Choose your visualizer with the dropdown menu :)''')),
                    html.P(dcc.Markdown('''* Finally, you need to select the enpoints to want to visualize and if it proceed the number of waveforms to accumulate.''')),
                    html.P(dcc.Markdown('''* Do not forget to push the `Plot` buttom to produce them.''')),
                ], style={"max-width": "100%", "margin": "auto"})
            
            ### EVENT DISPLAY ###
            elif pathname == "Event Display": 
                print(f"Event number: {event}")
                if self.loaded:
                    print("We have a wvset to plot!")
                    for adx, apa in enumerate(apa_value):
                        eps = self.apa2ep[apa]
                        filter_wfset = WaveformSet.from_filtered_WaveformSet( self.wfset, allow_certain_endpoints, eps )
                        self.figures[int(apa)-1] = psu.make_subplots( self.rows, self.cols,
                                                                      x_title='Time [ticks]',
                                                                      y_title='ADC counts',
                                                                      vertical_spacing=0.04,
                                                                      horizontal_spacing=0.02)
                        grid_apa = ChannelWsGrid( APA_map[int(apa)], self.wfset, compute_calib_histo = False )
                        plot_ChannelWsGrid( grid_apa,
                                            figure = self.figures[int(apa)-1],
                                            share_x_scale = False,
                                            share_y_scale = False,
                                            mode = 'overlay',
                                            analysis_label = None,
                                            plot_event = True,
                                            event_id = event,
                                            verbose = True)
                        # self.figures[int(apa)-1].show()
                        self.graphs = [dcc.Graph(figure=fig, style={'width': self.phys_pos[f][0],'height': self.phys_pos[f][1], 'display': 'inline-block'}) for f,fig in enumerate(self.figures)]

                self.layout = html.Div(
                                [
                                    dcc.Location(id="url"),
                                    html.H1(children="Event display!", style={"textAlign": "center"}),
                                    html.Div(id='event',children=self.graphs),
                                ], style={"max-width": "100%", "margin": "auto"}
                            )
                
            ### INDIVIDUAL WAVEFORM PERSISTENCE ###
            elif pathname == "Individual Waveform Persistence": 
                if self.loaded:
                    print("We have a wvset to plot!")
                    for adx, apa in enumerate(apa_value):
                        self.figures[int(apa)-1] = psu.make_subplots( self.rows,self.cols, 
                                                                      x_title='Time [ticks]',
                                                                      y_title='ADC counts',
                                                                      vertical_spacing=0.04,
                                                                      horizontal_spacing=0.02)
                        grid_apa = ChannelWsGrid( APA_map[int(apa)], self.wfset, compute_calib_histo = False )
                        plot_ChannelWsGrid( grid_apa,
                                            figure = self.figures[int(apa)-1],
                                            share_x_scale = False,
                                            share_y_scale = False,
                                            mode = 'overlay',
                                            wfs_per_axes = n_wvf,
                                            analysis_label = None,
                                            verbose = True)
                        # self.figures[int(apa)-1].show()
                        self.graphs = [dcc.Graph(figure=fig, style={'width': self.phys_pos[f][0],'height': self.phys_pos[f][1], 'display': 'inline-block'}) for f,fig in enumerate(self.figures)]

                self.layout = html.Div(
                                [
                                    dcc.Location(id="url"),
                                    html.H1(children="Plot your Waveforms together!", style={"textAlign": "center"}),
                                    html.Div(id='waveform',children=self.graphs),
                                ], style={"max-width": "100%", "margin": "auto"}
                            )
                
            ### HEATMAP PERSISTENCE ###
            elif pathname == "Heatmap Persistence":   
                if self.loaded:
                    print("We have a wvset to plot! Computing basic analysis...")
                    #Define the analysis parameters
                    json_file_path = f"{self.waffles_path}/conf/{self.my_file.split('/')[-1].split('_')[0]}.json"
                    with open(json_file_path, 'r') as json_file:
                        analysis_conf = json.load(json_file)
                    analysis_label = list(analysis_conf.keys())[0]
                    
                    baseline_limits  = analysis_conf[analysis_label]['base_lim'] #[0, 100, 900, 1000]
                    input_parameters = IPDict(baseline_limits = baseline_limits)
                    input_parameters['int_ll'] = analysis_conf[analysis_label]['int_ll']
                    input_parameters['int_ul'] = analysis_conf[analysis_label]['int_ul']
                    input_parameters['amp_ll'] = analysis_conf[analysis_label]['amp_ll']
                    input_parameters['amp_ul'] = analysis_conf[analysis_label]['amp_ul']
                    checks_kwargs = IPDict()
                    checks_kwargs['points_no'] = self.wfset.points_per_wf
                    
                    _ = self.wfset.analyse( 'standard',
                                       BasicWfAna,
                                       input_parameters,
                                       *[], # *args,
                                       analysis_kwargs = {},
                                       checks_kwargs = checks_kwargs,
                                       overwrite = True)
                    
                    for adx, apa in enumerate(apa_value):
                        self.figures[int(apa)-1] = psu.make_subplots( self.rows,self.cols, 
                                                                      x_title='Time [ticks]',
                                                                      y_title='ADC counts',
                                                                      vertical_spacing=0.04,
                                                                      horizontal_spacing=0.02)
                        grid_apa = ChannelWsGrid( APA_map[int(apa)], self.wfset, compute_calib_histo = False )
                        plot_ChannelWsGrid( grid_apa,
                                            figure = self.figures[int(apa)-1],
                                            share_x_scale = False,
                                            share_y_scale = False,
                                            mode = 'heatmap',
                                            wfs_per_axes = n_wvf,
                                            analysis_label = 'standard',
                                            verbose = True)
                        # self.figures[int(apa)-1].show()
                        self.graphs = [dcc.Graph(figure=fig, style={'width': self.phys_pos[f][0],'height': self.phys_pos[f][1], 'display': 'inline-block'}) for f,fig in enumerate(self.figures)]

                self.layout = html.Div(
                        [
                            dcc.Location(id="url"),
                            html.H1(children="Heatmaps", style={"textAlign": "center"}),
                            html.Div(id='heatmap',children=self.graphs),
                        ], style={"max-width": "100%", "margin": "auto"}
                        )    
                
            ### FAST FOURIER TRANSFORM (NOISE) ###      
            elif pathname == "Fast Fourier Transform (Noise)":  
                if self.loaded:
                    print("We have a wvset to plot!")
                    print("Working on the methods to plot the fft")

                self.layout = html.Div(
                        [
                            dcc.Location(id="url"),
                            html.H1(children="🚧 FFT plots", style={"textAlign": "center"}),
                            html.Div(id='fft',children=self.graphs),
                        ], style={"max-width": "100%", "margin": "auto"}
                        )  

            # If the user tries to reach a different page, return a 404 message
            else:
                self.layout = html.Div(
                    [
                        html.H1("404: Not found", className="text-danger"),
                        html.Hr(),
                        html.P(f"The pathname {pathname} was not recognised..."),
                    ],
                    className="p-3 bg-light rounded-3",
                )

            return self.layout

        # Callbacks to update dropdown options
        @self.app.callback(
            Output('folder_input', 'value'),
            Input('location', 'value'),
        )
        def update_basefolder(location):
            if location == 'LOCAL':
                basefolder = f'{self.waffles_path}/data/'
            else:
                basefolder = '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths'
            return basefolder

        @self.app.callback(
            Output('run_dropdown', 'options'),
            [Input('folder_input',  'value'),
             Input('file_format',  'value')],
        )
        def update_runfolder(basefolder,file_format):
            """ Update the runfolder dropdown with the available folders """
            if basefolder:
                if basefolder[-1] != '/': basefolder += '/'
            self.basefolder = basefolder

            options = []
            if basefolder and os.path.isdir(basefolder):
                self.run_folder = sorted(os.listdir(basefolder))
                if "1_rucio_paths" not in basefolder:
                    self.run_folder = [folder for folder in self.run_folder if folder.endswith(f'{file_format}')]
                options = [{'label': file, 'value': file} for file in self.run_folder]
                return options
            return []


    # Run the Dash app
    def run_app(self):
        self.app.run_server(
            jupyter_mode=self.jupyter_mode,
            jupyter_server_url=self.server_url,
            host="localhost",
            port=self.port,
        )
        if self.jupyter_mode == "inline":
            self.adjust_iframe_height(height=1500)