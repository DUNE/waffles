from waffles.np04_analysis.led_calibration.imports import *

class Analysis1(WafflesAnalysis):

    def __init__(self):
        pass

    @classmethod
    def get_input_params_model(
        cls
    ) -> type:
        """Implements the WafflesAnalysis.get_input_params_model()
        abstract method. Returns the InputParams class, which is a
        Pydantic model class that defines the input parameters for
        this analysis.
        
        Returns
        -------
        type
            The InputParams class, which is a Pydantic model class"""
        
        class InputParams(BaseInputParams):
            """Validation model for the input parameters of the LED
            calibration analysis."""

            apas: list = Field(
                ...,
                description="APA number",
                example=[2]
            )

            pdes: list = Field(
                ...,
                description="Photon detection efficiency",
                example=[0.4]
            )

            batches: list = Field(
                ...,
                description="Calibration batch number",
                example=[2]
            )

            show_figures: bool = Field(
                default=False,
                description="Whether to show the produced "
                "figures",
            )

            max_peaks: int = Field(
                default=2,
                description="Maximum number of peaks to "
                "fit in each charge histogram",
            )

            prominence: float = Field(
                default=0.15,
                description="Minimal prominence, as a "
                "fraction of the y-range of the charge "
                "histogram, for a peak to be detected",
            )

            half_points_to_fit: int = Field(
                default=2,
                description="The number of points to "
                "fit on either side of the peak maximum. "
                "P.e. setting this to 2 will fit 5 points "
                "in total: the maximum and 2 points on "
                "either side."
            )

            initial_percentage: float = Field(
                default=0.15,
                description="It has to do with the peak "
                "finding algorithm. It is given to the "
                "'initial_percentage' parameter of the "
                "'fit_peaks_of_ChannelWsGrid()' function. "
                "Check its docstring for more information."
            )

            percentage_step: float = Field(
                default=0.05,
                description="It has to do with the peak "
                "finding algorithm. It is given to the "
                "'percentage_step' parameter of the "
                "'fit_peaks_of_ChannelWsGrid()' function. "
                "Check its docstring for more information."
            )

            plots_saving_folderpath: str = Field(
                default="./",
                description="Path to the folder where "
                "the plots will be saved."
            )

        return InputParams

    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
        """Implements the WafflesAnalysis.initialize() abstract
        method. It defines the attributes of the Analysis1 class.
        
        Parameters
        ----------
        input_parameters : BaseInputParams
            The input parameters for this analysis
            
        Returns
        -------
        None
        """

        self.analyze_loop = [None,]
        self.params = input_parameters
        self.wfset = None
        self.output_data = None

        self.read_input_loop_1 = self.params.batches
        self.read_input_loop_2 = self.params.apas
        self.read_input_loop_3 = self.params.pdes


    def read_input(self) -> bool:
        """Implements the WafflesAnalysis.read_input() abstract
        method. It loads a WaveformSet object into the self.wfset
        attribute which matches the input parameters, namely the
        APA number, the PDE and the batch number. The final
        WaveformSet is the result of merging different WaveformSet
        objects, each of which comes from a different run.
        The decision on which run contributes to which channel
        is done based on the configuration files, namely on the
        config_to_channels and run_to_config variables, which are
        imported from files in the configs/calibration_batches
        directory.
            
        Returns
        -------
        bool
            True if the method ends execution normally
        """

        self.batch = self.read_input_itr_1
        self.apa   = self.read_input_itr_2
        self.pde   = self.read_input_itr_3

        print(f"Processing runs for batch",self.batch,
                ", APA", self.apa,
                "and PDE", self.pde
                )

        first = True

        # Reset the WaveformSet
        self.wfset = None

        # get all runs for a given calibration batch, apa and PDE value
        runs = run_to_config[self.batch][self.apa][self.pde]
        
        # Loop over runs
        for run in runs.keys():
            channels_and_endpoints = config_to_channels[self.batch][self.apa][self.pde][runs[run]]
            
            # Loop over endpoints using the current run for calibration
            for endpoint in channels_and_endpoints.keys():
                
                # List of channels in that endpoint using that run for calibration
                channels = channels_and_endpoints[endpoint]

                print("  - Loading waveforms from "
                        f"run {run},"
                        f"endpoint {endpoint},"
                        f"channels {channels}"
                    )         
                
                # Get the filepath to the input data for this run
                input_filepath = led_utils.get_input_filepath(
                    self.params.input_path, 
                    self.batch,
                    self.apa,
                    self.pde,
                    run
                )

                # Read all files for the given run
                new_wfset = led_utils.read_data(
                    input_filepath,
                    self.batch,
                    self.apa,
                    is_folder=False,
                    stop_fraction=1.,
                )

                # Keep only the waveforms coming from 
                # the targeted channels for this run
                new_wfset = new_wfset.from_filtered_WaveformSet(
                    new_wfset,
                    led_utils.comes_from_channel,
                    endpoint,
                    channels
                )

                if first:
                    self.wfset = new_wfset
                    first=False
                else:
                    self.wfset.merge(new_wfset)

        return True

    def analyze(self) -> bool:
        """
        Implements the WafflesAnalysis.analyze() abstract method.
        It performs the analysis of the waveforms contained in the
        self.wfset attribute, which consists of the following steps:

        1. Analyze the waveforms in the WaveformSet by computing
        their baseline and integral.
        2. Create a grid of WaveformSets, so that their are ordered
        according to the APA ordering, and all of the waveforms in a
        WaveformSet come from the same channel.
        3. Compute the charge histogram for each channel in the grid
        4. Fit peaks of each charge histogram
        5. Plot charge histograms
        6. Compute gain and S/N for every channel.
        
        Returns
        -------
        bool
            True if the method ends execution normally
        """

        # ------------- Analyse the waveform set -------------

        print(f"  1. Analizyng WaveformSet with {len(self.wfset.waveforms)} waveforms")

        # get parameters input for the analysis of the waveforms
        analysis_params = led_utils.get_analysis_params(
            self.apa,
            # Will fail when APA 1 is analyzed
            run=None
        )

        checks_kwargs = IPDict()
        checks_kwargs['points_no'] = self.wfset.points_per_wf

        self.analysis_name = 'standard'
    
        # Analyze all of the waveforms in this WaveformSet:
        # compute baseline, integral and amplitud
        _ = self.wfset.analyse(
            self.analysis_name,
            BasicWfAna,
            analysis_params,
            *[],  # *args,
            analysis_kwargs={},
            checks_kwargs=checks_kwargs,
            overwrite=True
        )
                
        # ------------- Compute charge histogram -------------

        print(f"  2. Computing Charge histogram")

        # Create a grid of WaveformSets for each channel in one
        # APA, and compute the charge histogram for each channel
        self.grid_apa = ChannelWsGrid(
            APA_map[self.apa],
            self.wfset,
            compute_calib_histo=True, 
            bins_number=led_utils.get_nbins_for_charge_histo(
                self.pde,
                self.apa
            ),
            domain=np.array((-10000.0, 50000.0)),
            variable="integral",
            analysis_label=self.analysis_name
        )

        # ------------- Fit peaks of charge histogram -------------

        print(f"  3. Fit peaks")

        # Fit peaks of each charge histogram
        fit_peaks_of_ChannelWsGrid(
            self.grid_apa,
            self.params.max_peaks,
            self.params.prominence,
            self.params.half_points_to_fit, 
            self.params.initial_percentage,
            self.params.percentage_step
        )
    
        # ------------- Compute gain and S/N ------------- 

        print(f"  4. Computing S/N and gain")

        # Compute gain and S/N for every channel
        self.output_data = led_utils.get_gain_and_snr(
            self.grid_apa, 
            excluded_channels[self.batch][self.apa][self.pde]
        )

        return True

    def write_output(self) -> bool:
        """Implements the WafflesAnalysis.write_output() abstract
        method. It saves the results of the analysis to a dataframe,
        which is written to a pickle file.

        Returns
        -------
        bool
            True if the method ends execution
        """

        base_file_path = f"{self.params.output_path}"\
            f"/batch_{self.batch}_apa_{self.apa}_pde_{self.pde}"

        # ------------- Save the charge histogram plot ------------- 

        figure = plot_ChannelWsGrid(
            self.grid_apa,
            figure=None,
            share_x_scale=False,
            share_y_scale=False,
            mode="calibration",
            wfs_per_axes=None,
            analysis_label=self.analysis_name,
            plot_peaks_fits=True,
            detailed_label=False,
            verbose=True
        )

        title = f"APA {self.apa} - Runs {list(self.wfset.runs)}"

        figure.update_layout(
            title={
                "text": title,
                "font": {"size": 24}
            }, 
            width=1100,
            height=1200,
            showlegend=True
        )

        if self.params.show_figures:
            figure.show()

        fig_path = f"{base_file_path}_calib_histo.png"
        figure.write_image(f"{fig_path}")

        print(f"  charge histogram plots saved in {fig_path}")

        # ------------- Save calibration results to a dataframe -------------

        df_path = f"{base_file_path}_df.pkl"

        led_utils.save_data_to_dataframe(
            self,
            self.output_data, 
            df_path,
        )

        print(f"  dataframe with S/N and gain saved in {df_path}")

        return True