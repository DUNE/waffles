import os, yaml,glob, h5py
import numpy as np
from dash import (Dash, dcc, html, callback_context)
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from uproot import open as uproot_open

from set_server import SetServer
from tab_wvf import WaveformDisplay

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
        self.wvf_display = WaveformDisplay()

        config = server.get_config()
        self.service_prefix = config['service_prefix']
        self.server_url = config['server_url']
        self.port = config['port']
        self.jupyter_mode = config['jupyter_mode']

        self.folder = ''
        self.files = []
        self.my_file = ''

        self.standard_folders = [
            {'label': 'RAW_PDS'  , 'value': '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/2_daq_root'},
            {'label': 'ANA_PDS'  , 'value': '/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Comissioning/waffles/3_ana_root'},
        ]

        self.geometry_info = {
            'det': [],
        }

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
        iframes.forEach(function(iframe) {{
            iframe.style.height = '{height}px';
        }});
        </script>
        """
        display(HTML(script))

    def construct_app(self):
        """
        Construct the Dash app.
        """

        if self.jupyter_mode == "inline":
            self.app = Dash(
                __name__,
                requests_pathname_prefix=f"{self.service_prefix}proxy/{self.port}/",
                external_stylesheets=[dbc.themes.FLATLY]
            )
        else:
            self.app = Dash(
                __name__,
                external_stylesheets=[dbc.themes.FLATLY]
            )

        """Get the custom style file"""
        with open('assets/styles.yaml', 'r') as file: styles = yaml.safe_load(file)

        """Define the navbar"""
        self.navbar = html.Div(
            children=[
                html.A(
                    href="https://github.com/DUNE/waffles",
                    target="_blank",  # Opens the link in a new tab
                    children=[html.Img(src='/assets/neutrino.png', style={'height': '35px', 'marginRight': '15px', 'float':'right'}),],
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
                            html.Img(src=('/assets/WAFFLES.PNG'), style={'height': '100px', 'marginRight': '15px'}),
                            html.H3("WAFFLES DISPLAY", style={'font-weight': 'bold'}),
                            ],
                    style={'display': 'flex', 'alignItems': 'center'}
                ),

                # Standard folder selection dropdown
                html.Hr(style={'border': '3px solid #ffffff', 'height': '0px'}),
                html.P("Default Folders ⚙️"),
                dcc.Dropdown(
                    id='folder_dropdown',
                    options=self.standard_folders,
                    style={'color': "#000000"},
                ),
                html.Hr(style={'border': '3px solid #ffffff', 'height': '0px'}),

                # Text box for writing folders
                dbc.Label("📂 Folder"),
                dbc.Input(
                    placeholder="Enter the folder",
                    type="text",
                    id='folder_input',
                    size='sm',
                    value=''
                ),
                html.H2(),

                # Dropdown which lists available files
                html.Label('📝 Run'),
                dcc.Dropdown(
                    id='file_dropdown',
                    searchable=True,
                    placeholder='Select a file...',
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
                    options=['Home', 'Individual Waveform Persistence', 'Heatmap Persistence', 'Fast Fourier Transform (Noise)'],
                    style={'color': "#000000"}
                ),
                html.Hr(style={'border': '1px solid #ffffff', 'height': '0px'}),
                html.Div([
                            html.Div([
                                        html.P("Choose your Endpoints:"),
                                        dcc.Checklist(
                                                        ['104', '105', '107', '109', '111', '112', '113'], # options
                                                        ['111','112','113'], # values
                                                        id="endpoints",
                                                        style={'width': '30vw'}),
                                    ]),
                            html.Div(style={'width': '5vw'}),
                            html.Div([
                                        html.H2(),
                                        html.P("Number of waveforms"),
                                        dcc.Input(id="n_wvf", type="number", placeholder="", debounce=False, min=10, max=1500, step=20, value=500)
                                    ], style={'width': '45vw'}),
                        ], style={'width': '30vw'}),
                html.H2(),
                html.Div([
                            html.H2(),
                            dbc.Button("PLOT", id="plot-button", color="primary", n_clicks=0, style={'width': '19vw'}),
                        ])
                # html.Button("PLOT", id="plot-button", n_clicks=0),
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
        # Callbacks to update the content based on which tab is selected
        @self.app.callback(
            Output("page-content", "children"),
            [Input("display_dropdown", "value")]
        )
        def render_tab_content(pathname):
            if pathname == "Home": 
                return html.Div([
                    html.H1(children="Welcome to our main page! 🏠 ", style={"textAlign": "center"}),
                    html.Hr(),
                    html.P(dcc.Markdown('''For visualizing you need to make sure that the `.root` files for the run you are interested in are already processed.''')),
                    html.P(dcc.Markdown('''If not, go to your terminal and execute `python 00_HDF5toROOT.py`.''')),
                    html.P(dcc.Markdown('''Once the `.root` files are ready, you need to select the folder where they are located.''')),
                    html.P(dcc.Markdown('''* There are some default folders available for you to choose from that will pick the data that is in the common `/eos/` folder.''')),
                    html.P(dcc.Markdown('''* If you do not want any of those you can just enter your folder in the `📂 Folder`''')),
                    html.P(dcc.Markdown('''* Then you need to choose between the runs that are inside `📝 Run`''')),
                    html.P(dcc.Markdown('''* Choose your visualizer with the dropdown menu :)''')),
                    html.P(dcc.Markdown('''* Finally, you need to select the enpoints to want to visualize and if it proceed the number of waveforms to accumulate.''')),
                    html.P(dcc.Markdown('''* Do not forget to push the `Plot` buttom to produce them.''')),
                ], style={"max-width": "100%", "margin": "auto"})

            elif pathname == "Individual Waveform Persistence": return self.wvf_display.layout 
            elif pathname == "Heatmap Persistence":             return html.H1("🚧 Persistence plots")
            elif pathname == "Fast Fourier Transform (Noise)":  return html.H1("🚧 FFT plots")      

            # If the user tries to reach a different page, return a 404 message
            return html.Div(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ],
                className="p-3 bg-light rounded-3",
            )

        @self.app.callback(
            Output('folder_input', 'value'),
            Input('folder_dropdown', 'value')
        )
        def update_folder(folder):
            return folder

        # Callback to update dropdown options
        @self.app.callback(
            Output('file_dropdown', 'options'),
            Input('folder_input', 'value'),
        )
        def update_folder_files(folder):
            """Check that folder has a '/' at the end"""
            if folder:
                if folder[-1] != '/': folder += '/'
            self.folder = folder

            options = []
            if folder and os.path.isdir(folder):
                self.files = sorted(os.listdir(folder))
                options = [{'label': file, 'value': file} for file in self.files]
                return options
            return []


        @self.app.callback(
            Output('waveform', 'children'),
            [Input('folder_input', 'value'),
             Input('file_dropdown', 'value'),
             Input('n_wvf', 'value'),
             Input('plot-button', 'n_clicks'),
             Input('endpoints', 'value')]
        )
        def update_waveform(my_folder,my_file,n_wvf,plot_nclicks,endpoint_value):
            # print(f"Folder: {my_folder}, File: {my_file}, N waveforms: {n_wvf}, Plot clicks: {plot_nclicks}, Endpoints: {endpoint_value}")
            if my_file is not None and my_folder is not None:
                try:
                    my_file = my_folder +"/"+ my_file +"/"+my_file.split('/')[-1].replace("run_","")+'_0000.root'
                    root_file = uproot_open(my_file)
                    raw_waveforms = {}
                    raw_waveforms['adcs']    = root_file['raw_waveforms']['adcs'].array()
                    raw_waveforms['channel'] = root_file['raw_waveforms']['channel'].array()
                    print("\nRAW WAVEFORMS LOADED!\n") ## TODO:Print this below the run dropdown
                except Exception:
                    pass

                triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0] if callback_context.triggered else ''
                if triggered_id == 'plot-button' and plot_nclicks > 0:
                    try:
                        print(f"Plotting waveforms for {endpoint_value}")
                        for e,ep in enumerate(endpoint_value):
                            self.wvf_display.plot_waveform(raw_waveforms, ep, n_wvf)
                            print("\nPLOTTED\n")
                            self.wvf_display.generate_layout() 
                            # phys_pos = [('17%','25%',),('17%','25%'),('17%','25%'),('50%','25%'),('50%','25%'),('37%','25%'),('12%','50%')]
                            phys_pos = [('50%','33%',),('50%','33%'),('50%','33%'),('50%','50%'),('50%','25%'),('50%','25%'),('50%','50%')]
                            self.wvf_display.graphs = [dcc.Graph(figure=fig, style={'width': phys_pos[f][0],'height': phys_pos[f][1], 'display': 'inline-block'}) for f,fig in enumerate(self.wvf_display.waveforms)]
                        return self.wvf_display.graphs
                    except Exception as e:
                        print(f"Error plotting waveforms: {e}")
                        return []
                else:
                    print("no plot")
                    raise PreventUpdate

    def run_app(self):
        self.app.run_server(
            jupyter_mode=self.jupyter_mode,
            jupyter_server_url=self.server_url,
            host="localhost",
            port=self.port,
        )
        if self.jupyter_mode == "inline":
            self.adjust_iframe_height(height=1500)