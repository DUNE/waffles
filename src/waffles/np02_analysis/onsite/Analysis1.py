from waffles.np02_analysis.onsite.imports import *

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

            ch: list = Field(
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
                description="Intervals of intergration",
                example=[-1] #Alls
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

            validate_items = field_validator(
                "runs",
                mode="before"
            )(wcu.split_comma_separated_string)
            
            
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
        self.read_input_loop_3 = [None]
        self.analyze_loop = [None,] 

        self.wfset = None

    def read_input(self) -> bool:
        """Implements the WafflesAnalysis.read_input() abstract
        method. For the current iteration of the read_input loop,
        which fixes a run number, it reads the first
        self.params.waveforms_per_run waveforms from the first rucio
        path found for this run, and creates a WaveformSet out of them,
        which is assigned to the self.wfset attribute.
            
        Returns
        -------
        bool
            True if the method ends execution normally
        """
        self.run    = self.read_input_itr_1
        self.det_id = self.read_input_itr_2
        
        
        print(
            "In function Analysis1.read_input(): "
            f"Now reading waveforms for run {self.run} ..."
        )
        
        try:
            wfset_path = self.params.input_path
            #self.wfset=WaveformSet_from_hdf5_pickle(wfset_path)   
            self.wfset=load_structured_waveformset(wfset_path)   
        except FileNotFoundError:
            raise FileNotFoundError(f"File {wfset_path} was not found.")

        return True
    
    def analyze(self) -> bool:
        """Implements the WafflesAnalysis.analyze() abstract method.
        It performs the analysis of the waveforms contained in the
        self.wfset attribute, which consists of the following steps:

        1. If self.params.correct_by_baseline is True, the baseline
        for each waveform in self.wfset is computed and used in
        the computation of the mean waveform.
        2. A WaveformAdcs object is created which matches the mean
        of the waveforms in self.wfset.
        
        Returns
        -------
        bool
            True if the method ends execution normally
        """
        # ------------- Analyse the waveform set -------------
        
        print(" 1. Starting the analysis")
        
        # Obtain the endpoints from the detector
        eps = os_utils.get_endpoints(self.params.det, self.det_id)
        
        # Select the waveforms and the corresponding waveformset in a specific time interval of the DAQ window
        self.selected_wfs1, self.selected_wfset1= os_utils.get_wfs(self.wfset.waveforms, eps, self.params.ch, self.params.nwfs, self.params.tmin, self.params.tmax, self.params.rec, adc_max_threshold=15000)
        
        print(f" 2. Analyzing WaveformSet with {len(self.selected_wfs1)} waveforms between tmin={self.params.tmin} and tmax={self.params.tmax}")
        
        print(f" 3. Creating the grid")

        analysis_params = os_utils.get_analysis_params()
        
        checks_kwargs = IPDict()
        checks_kwargs['points_no'] = self.selected_wfset1.points_per_wf
        
        self.analysis_name = 'baseline_computation'
    
        _ = self.selected_wfset1.analyse(
            self.analysis_name,
            BasicWfAna2,
            analysis_params,
            *[],  # *args,
            analysis_kwargs={},
            checks_kwargs=checks_kwargs,
            overwrite=True
        )
        
        self.selected_wfs2, self.selected_wfset2 =os_utils.baseline_cut(self.selected_wfs1)
        #self.selected_wfs2=self.selected_wfs1
        # Create a grid of WaveformSets for each channel in one detector, and compute the corresponding function for each channel
        
        print(f" 3. After the filter we have {len(self.selected_wfs2)} waveforms")
        
        self.grid = os_utils.get_grid(self.selected_wfs2, self.params.det, self.det_id)
        
        # Computing of the S/N to establish the proper integration limits
        
        print(f" 4. Computing the charge histograms")


        checks_kwargs['points_no'] = self.selected_wfset2.points_per_wf
        
        self.analysis_name2 = 'charge_histogram'
        
        snr_by_interval = {}
        snr_labels = []
        
        print('The intervals for intergration are', self.integration_intervals)
        
        for interval in self.integration_intervals:

            left = int(interval * 0.2)  # 20% of the interval to the left
            right = interval - left     # The rest of the interval to the right
            
            # The peak is around 262
            left = 262-int(interval * 0.2)
            right = 262+int(interval * 0.8)
            
            print(f"\n>>> Analyzing with interval: [{left}, {right}]")

            # Set parameters for this interval
        
            analysis_params['starting_tick'] = left
            analysis_params['integ_window'] = interval
            analysis_params['int_ll'] = left
            analysis_params['int_ul'] = right
            
            print('analysis_params',analysis_params)
            
            # Perform the analysis
            _ = self.selected_wfset2.analyse(
                self.analysis_name2,
                BasicWfAna2,
                analysis_params,
                *[],  # *args,
                analysis_kwargs={},
                checks_kwargs=checks_kwargs,
                overwrite=True
            )
            print(self.selected_wfset2.waveforms[1000].analyses['charge_histogram'].result['baseline'])
            print(self.selected_wfset2.waveforms[1000].analyses['charge_histogram'].result['integral'])
            print(self.selected_wfset2.waveforms[1000].adcs[left:right])


            
            # Create the grid for charge histograms
            self.grid_charge = os_utils.get_grid_charge(
                self.selected_wfs2,
                self.params.det,
                self.det_id,
                self.nbins,
                self.analysis_name2
            )
   
   
            # Fit the peaks in each channel's charge histogram
            fit_peaks_of_ChannelWsGrid(
                self.grid_charge,
                self.params.max_peaks,
                self.params.prominence,
                self.params.half_points_to_fit,
                self.params.initial_percentage,
                self.params.percentage_step
            )
            
            # Plot the charge histogram for this interval
            figure = plot_ChannelWsGrid(
                self.grid_charge,
                figure=None,
                share_x_scale=False,
                share_y_scale=False,
                mode="calibration",
                analysis_label=self.analysis_name2,
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
            snr_data = os_utils.get_gain_and_snr(
                self.grid_charge
            )
            
            # Store S/N values by interval
            snr_by_interval[interval] = snr_data
            snr_labels.append(f"Interval [{left}, {right}]")
            

        # Print S/N results for all intervals
        print("S/N Results by Interval:")
        print(snr_by_interval.items())
        
        det_id_name=os_utils.get_det_id_name(self.det_id)

        os_utils.plot_snr_per_channel_grid(snr_by_interval, self.params.det, self.det_id, title=f"S/N vs Integral intervals - Detector {self.params.det} {det_id_name} - Runs {list(self.wfset.runs)}")
        
        return True

    def write_output(self) -> bool:
        """Implements the WafflesAnalysis.write_output() abstract
        method. It saves the mean waveform, which is a WaveformAdcs
        object, to a pickle file.

        Returns
        -------
        bool
            True if the method ends execution normally
        """

        det_id_name=os_utils.get_det_id_name(self.det_id)
        
        base_file_path = f"{self.params.output_path}"\
            f"run_{self.run}_{det_id_name}_{self.params.det}"
           
        # ------------- Save the waveforms plot ------------- 

        figure0 = plot_ChannelWsGrid(
            self.grid,
            figure=None,
            share_x_scale=False,
            share_y_scale=False,
            mode="heatmap",
            wfs_per_axes=self.params.nwfs_plot,
            analysis_label=self.analysis_name2,
            detailed_label=False,
            verbose=True
        )

        title0 = f"Heatmap for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

        figure0.update_layout(
            title={
                "text": title0,
                "font": {"size": 24}
            }, 
            width=1100,
            height=1200,
            showlegend=True
        )
        
        figure0.add_annotation(
            x=0.5,
            y=-0.05, 
            xref="paper",
            yref="paper",
            text="Timeticks",
            showarrow=False,
            font=dict(size=16)
        )
        figure0.add_annotation(
            x=-0.07,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Entries",
            showarrow=False,
            font=dict(size=16),
            textangle=-90
        )
        
        
        if self.params.show_figures:
            figure0.show()

        fig0_path = f"{base_file_path}_heatmap.png"
        figure0.write_image(f"{fig0_path}")

        print(f" \n Heatmap plots saved in {fig0_path}")
 
        # ------------- Save the waveforms plot ------------- 
        
        
        figure1 = plot_CustomChannelGrid(
            self.grid, 
            plot_function=lambda channel_ws, figure_, row, col: os_utils.plot_wfs(
                channel_ws, figure_, row, col, offset=False),
            share_x_scale=True,
            share_y_scale=True,
            show_ticks_only_on_edges=True 
        )

        title1 = f"Waveforms for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

        figure1.update_layout(
            title={
                "text": title1,
                "font": {"size": 24}
            },
            width=1100,
            height=1200,
            showlegend=True
        )
        
        figure1.add_annotation(
            x=0.5,
            y=-0.05, 
            xref="paper",
            yref="paper",
            text="Timeticks",
            showarrow=False,
            font=dict(size=16)
        )
        figure1.add_annotation(
            x=-0.07,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Entries",
            showarrow=False,
            font=dict(size=16),
            textangle=-90
        )

        if self.params.show_figures:
            figure1.show()
            
        fig1_path = f"{base_file_path}_wfs_2.png"
        figure1.write_image(f"{fig1_path}")
        print(f"\n Waveforms saved in {fig1_path}")
        
        
        '''
        figure1 = plot_CustomChannelGrid(
            self.grid_plot, 
            plot_function=lambda channel_ws, figure_, row, col: os_utils.plot_wfs(
                channel_ws, figure_, row, col, offset=False),
            share_x_scale=True,
            share_y_scale=True,
            show_ticks_only_on_edges=True 
        )

        title1 = f"Waveforms for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

        figure1.update_layout(
            title={
                "text": title1,
                "font": {"size": 24}
            },
            width=1100,
            height=1200,
            showlegend=True
        )
        
        figure1.add_annotation(
            x=0.5,
            y=-0.05, 
            xref="paper",
            yref="paper",
            text="Timeticks",
            showarrow=False,
            font=dict(size=16)
        )
        figure1.add_annotation(
            x=-0.07,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Entries",
            showarrow=False,
            font=dict(size=16),
            textangle=-90
        )

        if self.params.show_figures:
            figure1.show()
            
        fig1_path = f"{base_file_path}_wfs_2.png"
        figure1.write_image(f"{fig1_path}")
        print(f"\n Waveforms saved in {fig1_path}")
        
        #--------------- S/N ratio ----------------
           
        intervals = list(range(5, 400, 20))
        print('intervals',intervals)
        
        figure2 = plot_CustomChannelGrid(
            self.grid, 
            plot_function=lambda channel_ws, figure_, row, col: os_utils.plot_sigma_to_noise_vs_interval(
                channel_ws, figure_, row, col, intervals),
            share_x_scale=True,
            share_y_scale=True,
            show_ticks_only_on_edges=True 
        )

        title2 = f"S/N ratio for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

        figure2.update_layout(
            title={
                "text": title2,
                "font": {"size": 24}
            },
            width=1100,
            height=1200,
            showlegend=True
        )
        
        figure2.add_annotation(
            x=0.5,
            y=-0.05, 
            xref="paper",
            yref="paper",
            text="Timeticks",
            showarrow=False,
            font=dict(size=16)
        )
        figure2.add_annotation(
            x=-0.07,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Entries",
            showarrow=False,
            font=dict(size=16),
            textangle=-90
        )

        if self.params.show_figures:
            figure2.show()
            
        fig2_path = f"{base_file_path}_snr.png"
        figure2.write_image(f"{fig2_path}")
        print(f"\n Waveforms saved in {fig2_path}")
       
        # ------------- Save the average waveform plot wit-------------
       
        figure3 = plot_CustomChannelGrid(
            self.grid, 
            plot_function=lambda channel_ws, figure_, row, col: os_utils.plot_avg_waveform_with_peak_and_intervals(
                channel_ws, figure_, row, col, intervals),
            share_x_scale=True,
            share_y_scale=True,
            show_ticks_only_on_edges=True 
        )

        title3 = f"Average waveform for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

        figure3.update_layout(
            title={
                "text": title3,
                "font": {"size": 24}
            },
            width=1100,
            height=1200,
            showlegend=True
        )
        
        figure3.add_annotation(
            x=0.5,
            y=-0.05, 
            xref="paper",
            yref="paper",
            text="Timeticks",
            showarrow=False,
            font=dict(size=16)
        )
        figure3.add_annotation(
            x=-0.07,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Entries",
            showarrow=False,
            font=dict(size=16),
            textangle=-90
        )

        if self.params.show_figures:
            figure3.show()
            
        fig3_path = f"{base_file_path}_av_wf.png"
        figure3.write_image(f"{fig3_path}")
        print(f"\n Waveforms saved in {fig3_path}") 
        
        
        
        # ------------- Save the sigma histograms  ------------- 
        
        figure2 = plot_CustomChannelGrid(
            self.grid, 
            plot_function=lambda channel_ws, figure_, row, col: os_utils.plot_sigma_function(
                channel_ws, figure_, row, col, self.nbins),
            share_x_scale=True,
            share_y_scale=True,
            show_ticks_only_on_edges=True 
        )

        
        title2 = f"Sigma histograms for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

        figure2.update_layout(
            title={
                "text": title2,
                "font": {"size": 24}
            },
            width=1100,
            height=1200,
            showlegend=True
        )
        
        figure2.add_annotation(
            x=0.5,
            y=-0.05, 
            xref="paper",
            yref="paper",
            text="Sigma",
            showarrow=False,
            font=dict(size=16)
        )
        figure2.add_annotation(
            x=-0.07,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Entries",
            showarrow=False,
            font=dict(size=16),
            textangle=-90
        )

        if self.params.show_figures:
            figure2.show()
                        
        fig2_path = f"{base_file_path}_sigma_hist.png"
        figure2.write_image(f"{fig2_path}")

        print(f"\n Sigma histograms saved in {fig2_path}")
        
        
        # ------------- Save the FFT plots  ------------- 
        

        figure3= plot_CustomChannelGrid(
                self.grid, 
                plot_function=lambda channel_ws, figure_, row, col: os_utils.plot_meanfft_function(
                    channel_ws, figure_, row, col),
                share_x_scale=True,
                share_y_scale=True,
                show_ticks_only_on_edges=True,
                log_x_axis=True
            )
  
        
        title3 = f"Superimposed FFT for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

        figure3.update_layout(
            title={
                "text": title3,
                "font": {"size": 24}
            },
            width=1100,
            height=1200,
            showlegend=True
        )
        
        figure3.add_annotation(
            x=0.5,
            y=-0.05, 
            xref="paper",
            yref="paper",
            text="Frequency [MHz]",
            showarrow=False,
            font=dict(size=16)
        )
        figure3.add_annotation(
            x=-0.07,
            y=0.5,
            xref="paper",
            yref="paper",
            text="Power [dB]",
            showarrow=False,
            font=dict(size=16),
            textangle=-90
        )

        if self.params.show_figures:
            figure3.show()
                    
        fig3_path = f"{base_file_path}_meanfft.png"
        figure3.write_image(f"{fig3_path}")

        print(f" \n Mean FFT plots saved in {fig3_path}")

        '''
        return True