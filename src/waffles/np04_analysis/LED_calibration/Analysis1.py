# import all necessary files and classes
from waffles.np04_analysis.LED_calibration.imports import *

# import all tunable parameters
from waffles.np04_analysis.LED_calibration.params import *


class Analysis1(WafflesAnalysis):

    def __init__(self):
        pass        

    ##################################################################
    def arguments(self, parse: argparse.ArgumentParser):                

        parse.add_argument('-a','--apa',      type=int,   required=True,  help="APA number")
        parse.add_argument('-p','--pde',      type=float, required=True,  help="photon detection efficiency")
        parse.add_argument('-b','--batch',    type=int,   required=True,  help="calibration batch")
        parse.add_argument('-ns','--noshow',  action="store_true",        help="do not show figures")

    ##################################################################
    def initialize(self, args):                

        # store the user arguments into data members
        self.show_figures = args['noshow']
        self.apa    = args['apa']
        self.pde    = args['pde']
        self.batch  = args['batch']
        self.data_folderpath = self.path_to_input_file

        # calibration configurations 
        #self.run_to_config_      =      run_to_config[self.batch][self.apa][self.pde]  # runs for a given configuration
        #self.config_to_channels_ = config_to_channels[self.batch][self.apa][self.pde]  # channels for each run
        self.excluded_channels_  =  excluded_channels[self.batch]

        # self.read_input_loop = [[2,2,0.4],
        #                         [2,2,0.45],
        #                         [2,2,0.5],
        #                         [2,3,0.4],
        #                         [2,3,0.45],
        #                         [2,3,0.5],
        #                         [2,4,0.4],
        #                         [2,4,0.45],
        #                         [2,2,0.5],
        #                         [3,2,0.4],
        #                         [3,2,0.45],
        #                         [3,2,0.5],
        #                         [3,3,0.4],
        #                         [3,3,0.45],
        #                         [3,3,0.5],
        #                         [3,4,0.4],
        #                         [3,4,0.45],
        #                         [3,2,0.5]                                                                                            
        #                         ]

    ##################################################################
    def read_input(self):

        # self.batch = self.read_input_itr[0]
        # self.apa   = self.read_input_itr[1]
        # self.pde   = self.read_input_itr[2]

        first = True
        self.wfset = None

        # loop over runs
        runs = run_to_config[self.batch][self.apa][self.pde]
        for run in runs.keys():
            # loop over endpoints using that run for calibration
            channels_and_endpoints = config_to_channels[self.batch][self.apa][self.pde][runs[run]]
            for endpoint in channels_and_endpoints.keys():
                
                # list of channels in that endpoint using that run for calibration
                channels = channels_and_endpoints[endpoint]

                print("\n Now loading waveforms from:")
                print(f" - run {run}")
                print(f" - endpoint {endpoint}")
                print(f" - channels {channels} \n")         
                
                # data folder for that run, and the batch, apa and pde specified in params.py
                input_filepath = led_utils.get_input_filepath(self.data_folderpath, self.batch, run, self.apa, self.pde)

                # read all files for the given run
                new_wfset = led_utils.read_data(input_filepath, self.batch, self.apa, 0.01, is_folder=False)

                # keep only waveforms in the necessary endpoint and channels
                new_wfset = led_utils.get_wfset_in_channels(new_wfset, endpoint, channels)                     
                
                if first:
                    self.wfset = new_wfset
                    first=False
                else:
                    self.wfset.merge(new_wfset)

        return True

    ##################################################################
    def analyze(self):

        """ ------------- analyse the waveform set --------------- """

        # get parameters input for the analysis of the waveforms
        input_parameters = led_utils.get_analysis_params()
        checks_kwargs = IPDict()
        checks_kwargs['points_no'] = self.wfset.points_per_wf
    
        # analise the waveform: compute baseline, integral and amplitud
        _ = self.wfset.analyse(analysis_label, BasicWfAna, input_parameters, *[],  # *args,
            analysis_kwargs={}, checks_kwargs=checks_kwargs, overwrite=True,)
                
        """ ------------- Compute charge histogram --------------- """

        # Create a grid of WaveformSets for each channel in one APA, and compute the charge histogram
        # Julio suggest decoupling creating the grid and creating the calib histo
        grid_apa = ChannelWsGrid(APA_map[self.apa], self.wfset, compute_calib_histo=True, 
                                bins_number=nbins, domain=np.array((-10000.0, 50000.0)),
                                variable="integral",analysis_label=None,)

        """ ------------- Fit peaks of charge histogram -------------"""

        # fit peaks in that grid
        fit_peaks_of_ChannelWsGrid(grid_apa, max_peaks, prominence, half_points_to_fit, 
                                    initial_percentage, percentage_step,)

        """ ------------- Plot charge histograms ------------- """

        figure = plot_ChannelWsGrid(grid_apa, figure=None, share_x_scale=False,
                            share_y_scale=False, mode="calibration", wfs_per_axes=None,
                            analysis_label=analysis_label, plot_peaks_fits=True,
                            detailed_label=False, verbose=True,)

        title = f"APA {self.apa} - Runs {list(self.wfset.runs)}"
        figure.update_layout(title={"text": title,"font": {"size": 24}}, width=1100, height=1200, showlegend=True,)

        if not self.show_figures:
            figure.show()

        # figure.write_image(f"{plots_saving_filepath}/apa_{self.apa}_calibration_histograms.png")

        """ ------------- Compute gain and S/N ------------- """

        # compute gain and S/N for the grid
        self.output_data = led_utils.get_gain_and_sn(grid_apa, self.excluded_channels_[self.apa][self.pde])

        return True

    ##################################################################
    def write_output(self):

        """ ------------- Save results to a dataframe ------------- """

        led_utils.save_data_to_dataframe(self.output_data, 
                                         self.path_to_output_file,
                                         self)
