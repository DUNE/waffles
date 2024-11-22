# import all necessary files and classes
from waffles.np04_analysis.LED_calibration.imports import *

# import all tunable parameters
from waffles.np04_analysis.LED_calibration.params import *


class analysis(WafflesAnalysis):

    def __init__(self):
        pass        

    ##################################################################
    def arguments(self, parse: argparse.ArgumentParser):                

        parse.add_argument('-a','--apa',    type=int,   required=True,  help="APA number")
        parse.add_argument('-p','--pde',    type=float, required=True,  help="photon detection efficiency")
        parse.add_argument('-b','--batch',  type=int,   required=True,  help="calibration batch")


    ##################################################################
    def initialize(self, args):                

        # store the user arguments into data members
        self.apa_no = args['apa']
        self.pde    = args['pde']
        self.batch  = args['batch']
        self.data_folderpath = self.path_to_input_file

        # calibration configurations 
        self.run_to_config_      =      run_to_config[self.batch][self.apa_no][self.pde]  # runs for a given configuration
        self.config_to_channels_ = config_to_channels[self.batch][self.apa_no][self.pde]  # channels for each run
        self.excluded_channels_  =  excluded_channels[self.batch]

    ##################################################################
    def read_input(self):

        first = True
        self.wfset = None
        # loop over runs
        for run in self.run_to_config_.keys():
            # loop over endpoints using that run for calibration
            for endpoint in self.config_to_channels_[self.run_to_config_[run]].keys():
                
                # list of channels in that endpoint using that run for calibration
                channels = self.config_to_channels_[self.run_to_config_[run]][endpoint]

                # data folder for that run, and the batch, apa and pde specified in params.py
                input_filepath = led_utils.get_input_filepath(self.data_folderpath, self.batch, run, self.apa_no, self.pde)

                # read all files for the given run
                new_wfset = led_utils.read_data(input_filepath, self.batch, self.apa_no, 0.01, is_folder=False)

                # keep only waveforms in the necessary endpoint and channels
                new_wfset = led_utils.get_wfset_in_channels(new_wfset, endpoint, channels)
                
                # get parameters input for the analysis of the waveforms
                input_parameters = led_utils.get_analysis_params()
                checks_kwargs = IPDict()
                checks_kwargs['points_no'] = new_wfset.points_per_wf

                print("\n Now analysing waveforms from:")
                print(f" - run {run}")
                print(f" - endpoint {endpoint}")
                print(f" - channels {channels} \n")      

                # analise the waveform: compute baseline, integral and amplitud
                _ = new_wfset.analyse(analysis_label, BasicWfAna, input_parameters, *[],  # *args,
                    analysis_kwargs={}, checks_kwargs=checks_kwargs, overwrite=True,)
                
                if first:
                    self.wfset = new_wfset
                    first=False
                else:
                    self.wfset.merge(new_wfset)

    ##################################################################
    def analize(self):

        """ ------------- Compute charge histogram --------------- """

        # Create a grid of WaveformSets for each channel in one APA, and compute the charge histogram
        # Julio suggest decoupling creating the grid and creating the calib histo
        grid_apa = ChannelWsGrid(APA_map[self.apa_no], self.wfset, compute_calib_histo=True, 
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

        title = f"APA {self.apa_no} - Runs {list(self.wfset.runs)}"
        figure.update_layout(title={"text": title,"font": {"size": 24}}, width=1100, height=1200, showlegend=True,)
        figure.show()

        # figure.write_image(f"{plots_saving_filepath}/apa_{self.apa_no}_calibration_histograms.png")

        """ ------------- Compute gain and S/N ------------- """

        # compute gain and S/N for the grid
        self.output_data = led_utils.get_gain_and_sn(grid_apa, self.excluded_channels_[self.apa_no][self.pde])

    ##################################################################
    def write_output(self):

        """ ------------- Save results to a dataframe ------------- """

        led_utils.save_data_to_dataframe(self.output_data, 
                                         self.path_to_output_file,
                                         self)
