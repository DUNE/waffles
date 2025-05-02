from waffles.np02_analysis.led_calibration.imports import *


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
        this example analysis.
        
        Returns
        -------
        type
            The InputParams class, which is a Pydantic model class
        """
        
        class InputParams(BaseInputParams):
            """Validation model for the input parameters of the
            example calibration.
            """

            runs: list[int] = Field(
                ...,
                description="Run numbers of the runs to be read",
                example=[27906, 27907]
            )
            
            det: str = Field(
                ...,
                description= "Membrane, Cathode or PMT",
                example = "Membrane"
            )
            
            det_id: list = Field(
                ...,
                description="TCO [1] and no-tco [2] membrane, TCO [1] and no-tco [2] cathode, and PMTs",
                example=[2]
            )
            
            pdes: list = Field(
                ...,
                description="Photon detection effiency",
                example=[0.4]
            )

            ch: int = Field(
                ...,
                description="Channels to analyze",
                example=[-1] # Alls
            )
            
            tmin: int = Field(
                ...,
                description="Lower time limit considered for the analyzed waveforms",
                example=[-1000] # Alls
            )
            
            tmax: int = Field(
                ...,
                description="Up time limit considered for the analyzed waveforms",
                example=[1000] # Alls
            )

            rec: list = Field(
                ...,
                description="Records",
                example=[-1] #Alls
            )
            
            nwfs: int = Field(
                ...,
                description="Number of waveforms to analyze",
                example=[-1] #Alls
            )
            
            nwfs_plot: int = Field(
                ...,
                description="Number of waveforms to plot",
                example=[-1] #Alls
            )
            
            nbins: int = Field(
                ...,
                description="Number of bins",
                example=110
            )
            
            integration_intervals: list = Field(
                ...,
                description="Intervals for integration",
                example=[20,30,40] 
            )
            
            correct_by_baseline: bool = Field(
                default=True,
                description="Whether the baseline of each waveform "
                "is subtracted before computing the average waveform"
            )

            input_path: str = Field(
                default="/data",
                description="Imput path"
            )
            
            output_path: str = Field(
                default="/output",
                description="Output path"
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
            
            show_figures: bool = Field(
                default=True,
                description="Whether to show the produced "
                "figures",
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

        # Save the input parameters into an Analysis1 attribute
        # so that they can be accessed by the other methods
        self.params = input_parameters
        self.nbins=self.params.nbins
        self.integration_intervals=self.params.integration_intervals

        self.read_input_loop_1 = self.params.runs
        self.read_input_loop_2 = self.params.det_id
        self.read_input_loop_3 = self.params.pdes
        self.analyze_loop = [None,] 

        self.wfset = None


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

        self.run    = self.read_input_itr_1
        self.det_id = self.read_input_itr_2
        self.pde   = self.read_input_itr_3
        
        self.det_id_name=lc_utils.get_det_id_name(self.det_id)
        
        print(f"Processing runs for {self.det_id_name} {self.params.det} and PDE {self.pde}")

        first = True

        # Reset the WaveformSet
        self.wfset = None

        # get all runs for a given calibration batch, apa and PDE value
        runs = configs[self.det_id][self.pde]
        
        # Loop over runs
        for run in runs.keys():
            
            channels_and_endpoints = config_to_channels[self.det_id][self.pde][runs[run]]
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
                input_filepath = lc_utils.get_input_filepath(
                    run
                )
                print(f"  - Input file path: {input_filepath}")
                # Read all files for the given run
                new_wfset = lc_utils.read_data(
                    input_filepath,
                )

                # Keep only the waveforms coming from 
                # the targeted channels for this run
                new_wfset = new_wfset.from_filtered_WaveformSet(
                    new_wfset,
                    lc_utils.comes_from_channel,
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

        # ------------- Analyze the waveform set -------------
        
        print(" 1. Starting the analysis")
        eps = lc_utils.get_endpoints(self.params.det, self.det_id)

        # Select the waveforms and the corresponding waveformset in a specific time interval of the DAQ window
        selected_wfs, selected_wfset = lc_utils.get_wfs(self.wfset.waveforms, eps, self.params.ch, self.params.nwfs, self.params.tmin, self.params.tmax, self.params.rec, adc_max_threshold=15000)

        '''
        
        # Led calibration analysis once the inegtration and baseline limits are set
        
        print(f" 2. Analyzing WaveformSet with {len(selected_wfs)} waveforms between tmin={self.params.tmin} and tmax={self.params.tmax}")

        analysis_params = lc_utils.get_analysis_params()
        
        checks_kwargs = IPDict()
        checks_kwargs['points_no'] = self.wfset.points_per_wf

        self.analysis_name = 'standard'
        # Perform the analysis
        _ = self.wfset.analyse(
            self.analysis_name,
            BasicWfAna,
            analysis_params,
            *[],  # *args,
            analysis_kwargs={},
            checks_kwargs=checks_kwargs,
            overwrite=True
        )

        # Create the grid for charge histograms
        self.grid = lc_utils.get_grid(
            selected_wfs,
            self.params.det,
            self.det_id,
            self.nbins,
            self.analysis_name
        )
   
        # Fit the peaks in each channel's charge histogram
        fit_peaks_of_ChannelWsGrid(
            self.grid,
            self.params.max_peaks,
            self.params.prominence,
            self.params.half_points_to_fit,
            self.params.initial_percentage,
            self.params.percentage_step
        )

        checks_kwargs = IPDict()
        checks_kwargs['points_no'] = self.wfset.points_per_wf

        self.analysis_name = 'standard'
        
        '''
        # Computing of the S/N to establish the proper integration limits
        
        print(f" 2. Analyzing WaveformSet with {len(selected_wfs)} waveforms between tmin={self.params.tmin} and tmax={self.params.tmax}")

        analysis_params = lc_utils.get_analysis_params()
        
        checks_kwargs = IPDict()
        checks_kwargs['points_no'] = self.wfset.points_per_wf
        
        self.analysis_name = 'standard'
        
        snr_by_interval = {}
        snr_labels = []
        
        print('The intervals for intergration are', self.integration_intervals)
        
        for interval in self.integration_intervals:

            left = int(interval * 0.2)  # 20% of the interval to the left
            right = interval - left     # The rest of the interval to the right
            
            # The peak is around 80
            left = 80-int(interval * 0.2)
            right = 80+int(interval * 0.8)
            
            print(f"\n>>> Analyzing with interval: [{left}, {right}]")

            # Set parameters for this interval
            analysis_params = lc_utils.get_analysis_params()
            analysis_params['starting_tick'] = left
            analysis_params['integ_window'] = interval
            analysis_params['int_ll'] = left
            analysis_params['int_ul'] = right
            analysis_params['amp_ll'] = 60
            analysis_params['amp_ul'] = 180

            # Perform the analysis
            _ = self.wfset.analyse(
                self.analysis_name,
                BasicWfAna,
                analysis_params,
                *[],  # *args,
                analysis_kwargs={},
                checks_kwargs=checks_kwargs,
                overwrite=True
            )

            # Create the grid for charge histograms
            self.grid = lc_utils.get_grid(
                selected_wfs,
                self.params.det,
                self.det_id,
                self.nbins,
                self.analysis_name
            )
   
   
            # Fit the peaks in each channel's charge histogram
            fit_peaks_of_ChannelWsGrid(
                self.grid,
                self.params.max_peaks,
                self.params.prominence,
                self.params.half_points_to_fit,
                self.params.initial_percentage,
                self.params.percentage_step
            )
            
            # Plot the charge histogram for this interval
            figure = plot_ChannelWsGrid(
                self.grid,
                figure=None,
                share_x_scale=False,
                share_y_scale=False,
                mode="calibration",
                analysis_label=self.analysis_name,
                plot_peaks_fits=True,
                detailed_label=False,
                verbose=True
            )
            figure.update_layout(
                title=f"Charge Histogram for Interval [{left}, {right}]",
                width=1100,
                height=1200,
                showlegend=True
            )

            if self.params.show_figures:
                figure.show()

            # Calculate S/N for each channel
            snr_data = lc_utils.get_gain_and_snr(
                self.grid,
                excluded_channels[self.det_id][self.pde]
            )
            
            # Store S/N values by interval
            snr_by_interval[interval] = snr_data
            snr_labels.append(f"Interval [{left}, {right}]")
            

        # Print S/N results for all intervals
        print("S/N Results by Interval:")
        print(snr_by_interval.items())
    

        lc_utils.plot_snr_per_channel_grid(snr_by_interval, self.params.det, self.det_id, title=f"S/N vs Integral intervals - Detector {self.params.det} {self.det_id_name} - Runs {list(self.wfset.runs)}")

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
        base_file_path = self.params.output_path + "/run_" + str(self.run) + "_" + self.det_id_name + "_" + self.params.det
        
        figure0 = plot_ChannelWsGrid(
                self.grid,
                share_x_scale=False,
                share_y_scale=False,
                mode="overlay",
                wfs_per_axes=self.params.nwfs_plot,
                analysis_label=self.analysis_name,
                detailed_label=False,
                verbose=True
                )

        title0 = f"Waveforms for {self.params.det} {self.det_id_name} - Runs {list(self.wfset.runs)}"

        figure0.update_layout(
                title={
                    "text": title0,
                    "font": {"size": 24}
                }, 
                width=1100,
                height=1200,
                showlegend=True
        )

        if self.params.show_figures:
            figure0.show()
            
        fig_path0 = f"{base_file_path}_wfs.png"
        figure0.write_image(f"{fig_path0}")

        print(f"  waveforms plots saved in {fig_path0}")
        
        
        '''
        
        figure0 = plot_ChannelWsGrid(
            self.grid,
            share_x_scale=False,
            share_y_scale=False,
            mode="overlay",
            wfs_per_axes=self.params.nwfs_plot,
            analysis_label=self.analysis_name,
            detailed_label=False,
            verbose=True
        )

        title0 = f"Waveforms for {self.params.det} {self.det_id_name} - Runs {list(self.wfset.runs)}"

        figure0.update_layout(
            title={
                "text": title0,
                "font": {"size": 24}
            }, 
            width=1100,
            height=1200,
            showlegend=True
        )

        if self.params.show_figures:
            figure0.show()

        fig_path0 = f"{base_file_path}_wfs.png"
        figure0.write_image(f"{fig_path0}")

        print(f"  charge histogram plots saved in {fig_path0}")

        # ------------- Save the charge histogram plot ------------- 

        figure = plot_ChannelWsGrid(
            self.grid,
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

        title = f"{self.det_id_name} - Runs {list(self.wfset.runs)}"

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

        lc_utils.save_data_to_dataframe(
            self,
            self.output_data, 
            df_path,
        )

        print(f"  dataframe with S/N and gain saved in {df_path}")
        '''
        return True