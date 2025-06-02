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
                example=[27906]
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
                example=[-1000] 
            )
            
            tmax: int = Field(
                ...,
                description="Up time limit considered for the analyzed waveforms",
                example=[1000] 
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
                description="Number of bins for the histograms",
                example=110
            )
            
            thr_adc: int = Field(
                ...,
                description="A thrshold for the ADC values",
                example=8000
            )
            
            wf_peak: int = Field(
                ...,
                description="Approximate position of the photoelectron peak along the timeticks axis",
                example=262
            )
            
            integration_intervals: list = Field(
                ...,
                description="Intervals of intergration",
                example=[-1] #Alls
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
            
            save_processed_wfset: bool = Field(
                default=True,
                description="Whether to save the processed wfset"
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
        self.thr_adc=self.params.thr_adc
        self.wf_peak=self.params.wf_peak    
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
            self.wfset=load_structured_waveformset(wfset_path)   
        except FileNotFoundError:
            raise FileNotFoundError(f"File {wfset_path} was not found.")

        return True
    
    def analyze(self) -> bool:
        """Implements the WafflesAnalysis.analyze() abstract method.
        It performs the analysis of the waveforms contained in the
        self.wfset attribute.
        Returns
        -------
        bool
            True if the method ends execution normally
        """
        # ------------- Analyse the waveform set -------------
        
        print("\n 1. Starting the analysis")
        
        # Obtain the endpoints from the detector
        eps = lc_utils.get_endpoints(self.params.det, self.det_id)
        
        # Select the waveforms and the corresponding waveformset in a specific time interval of the DAQ window
        self.selected_wfs1, self.selected_wfset1= lc_utils.get_wfs(self.wfset.waveforms, eps, self.params.ch, self.params.nwfs, self.params.tmin, self.params.tmax, self.params.rec, adc_max_threshold=15000)
   
        self.grid_raw=lc_utils.get_grid(self.selected_wfs1, self.params.det, self.det_id)
        
        if self.params.tmin == -1 and self.params.tmax == -1:
            print(f"\n 2. Analyzing WaveformSet with {len(self.selected_wfs1)} waveforms, in no specific time interval (tmin=-1 and tmax=-1).")
        else:
            print(f"\n 2. Analyzing WaveformSet with {len(self.selected_wfs1)} waveforms between tmin={self.params.tmin} and tmax={self.params.tmax}")

        analysis_params = lc_utils.get_analysis_params()
        
        checks_kwargs = IPDict()
        checks_kwargs['points_no'] = self.selected_wfset1.points_per_wf
        
        print(f"\n 3. Computing the baseline of the raw waveforms")
        
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
        
        self.selected_wfs2, self.selected_wfset2 =lc_utils.baseline_cut(self.selected_wfs1)
        
        print(f"\n 4. After aplying the baseline cut, we have {len(self.selected_wfs2)} waveforms")
        
        self.selected_wfs3, self.selected_wfset3 =lc_utils.adc_cut(self.selected_wfs2, thr_adc=self.thr_adc)
        
        self.grid_filt1= lc_utils.get_grid(self.selected_wfs2, self.params.det, self.det_id)
        
        if self.thr_adc != -1:
            print(f"\n 5. After applying a filter on the ADC values, we have {len(self.selected_wfs3)} waveforms.")
        else:
            print(f"\n 5. No more filters are applied.")
        
        self.grid_filt2= lc_utils.get_grid(self.selected_wfs3, self.params.det, self.det_id)
        
        print(f"\n 6. Computing the charge histograms, to establish the proper integration limits. Possibilities:{self.integration_intervals}")

        checks_kwargs['points_no'] = self.selected_wfset3.points_per_wf
        
        all_full_data_by_interval = {}
        snr_by_interval = {}
        snr_labels = []
        
        for interval in self.integration_intervals:

            left = int(interval * 0.2)  # 20% of the interval to the left
            right = interval - left     # The rest of the interval to the right
            
            left = self.wf_peak-int(interval * 0.2)
            right = self.wf_peak+int(interval * 0.8)
            
            print(f"\n >>> Analyzing with interval: ({left}, {right})")

            # Set parameters for this interval
            
            analysis_params['analysis_name'] = right
            analysis_params['starting_tick'] = left
            analysis_params['integ_window'] = interval
            analysis_params['int_ll'] = left
            analysis_params['int_ul'] = right
            analysis_params['amp_ll'] = left
            analysis_params['amp_ul'] = right
            
 
            self.analysis_name2= f"charge_histogram_({left}, {right})"
      
            # Perform the analysis
            
            _ = self.selected_wfset3.analyse(
                self.analysis_name2,
                BasicWfAna2,
                analysis_params,
                *[],  # *args,
                analysis_kwargs={},
                checks_kwargs=checks_kwargs,
                overwrite=True
            )
         
            # Create the grid for charge histograms
            self.grid_charge = lc_utils.get_grid_charge(
                self.selected_wfs3,
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
                figure.show(method='external', renderer='browser')
                
            full_data = lc_utils.get_gain_snr_and_amplitude(self.grid_charge)

            all_full_data_by_interval[left, right] = full_data
            
            snr_data = {
                ep: {
                    ch: {'snr': ch_data['snr']}
                    for ch, ch_data in ch_dict.items()
                }
                for ep, ch_dict in full_data.items()
            }

            snr_by_interval[interval] = snr_data
            snr_labels.append(f"Interval [{left}, {right}]")
           
        print("\n >>> Data for every interval:")
        for interval, full_data in all_full_data_by_interval.items():
            print(f"\n Interval {interval}:")
            for ep, ch_dict in full_data.items():
                for ch, values in ch_dict.items():
                    print(f"  EP {ep}, CH {ch} → Gain: {values['gain']:.2f}, "
                        f"SNR: {values['snr']:.2f}, Mean0: {values['mean0']:.2f},  Std0: {values['std0']:.2f},  "
                        f"Mean1: {values['mean1']:.2f},  Std1: {values['std1']:.2f}")

        print("\n >>> Plotting the S/N ratio per channel:")
        det_id_name = lc_utils.get_det_id_name(self.det_id)
        lc_utils.plot_snr_per_channel_grid(
            snr_by_interval,
            self.params.det,
            self.det_id,
            title=f"S/N vs Integral intervals - Detector {self.params.det} {det_id_name} - Runs {list(self.wfset.runs)}"
        ) 
        
        print(f"\n 7. Computing the maximum S/N ratio per channel")
        
        self.best_snr_info_per_channel = lc_utils.find_best_snr_per_channel(all_full_data_by_interval)
        
        # Ask the user if they want to create a WaveformSet with the SPE waveforms
        response = input("\n ▶ Do you want to create a waveformset with the spe waveforms for the integration interval that maximizes the S/N ratio? [y/n]: ").strip().lower()
        if response != 'y':
            print("\n ⚠  No WaveformSet creation.")
            self.should_save_waveforms = False  
        else:
            self.should_save_waveforms = True   
            self.waveformsets_by_channel = lc_utils.select_waveforms_around_spe(self.best_snr_info_per_channel, self.selected_wfset3)

        # Compute average amplitude
        
        if isinstance(self.integration_intervals, list) and isinstance(self.params.ch, list) \
        and len(self.integration_intervals) == 1 and len(self.params.ch) == 1:
            interval = self.integration_intervals[0]
            left = self.wf_peak - int(interval * 0.2)
            right = self.wf_peak + int(interval * 0.8)
            interval = (left,right)
            ch = self.params.ch[0]
            lc_utils.compute_average_amplitude(self.selected_wfs3, interval, self.params.ch, self.params.runs, self.params.output_path)
            
        else:
            print("\n >>> The average amplitude is not calculated because there are multiple channels or/and intervals.")
        return True

    def write_output(self) -> bool:
        
        """Implements the WafflesAnalysis.write_output() abstract
        method. It plots the waveforms and saves the processed
        WaveformSet in an HDF5 file. 

        Returns
        -------
        bool
            True if the method ends execution normally
        """

        det_id_name=lc_utils.get_det_id_name(self.det_id)
        
        base_file_path = f"{self.params.output_path}"\
            f"run_{self.run}_{det_id_name}_{self.params.det}"   
            
        if getattr(self, "should_save_waveforms", False):
            
            print(f"\n >>> Saving the Waveformset with the spe waveforms in a hdf5 file")
  
            for (ep, ch), wfset in self.waveformsets_by_channel.items():
                input_filename = f"run_{self.run}_ep{ep}_ch{ch}_spe_wfset"
                lc_utils.save_waveform_hdf5(wfset, input_filepath=input_filename, output_filepath=self.params.output_path)

        if len(self.params.ch) > 1 or self.params.ch == [-1]:
            lc_utils.save_dict_to_json(self.best_snr_info_per_channel, f"{base_file_path}_data.json")
        
        print(f"\n 8. Performing several plots for visualization")
        
        if self.params.save_processed_wfset:
            print("\n>>> Saving the processed WaveformSet in an HDF5 file")
    
            lc_utils.save_waveform_hdf5(
                self.selected_wfset3,
                input_filepath=f"{base_file_path}_process_wfset",
                output_filepath=self.params.output_path
            )

        
        # ------------- Save the average waveform plot ------------- 
        
        if isinstance(self.integration_intervals, list) and isinstance(self.params.ch, list) \
        and len(self.integration_intervals) == 1 and len(self.params.ch) == 1:  
            figure0 = plot_ChannelWsGrid(
                    self.grid_filt2,
                    figure=None,
                    share_x_scale=False,
                    share_y_scale=False,
                    mode="average",
                    wfs_per_axes=len(self.selected_wfs3),
                    detailed_label=False,
                    verbose=True
            )

            title0 = f"Average waveforms for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

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
                figure0.show(method='external', renderer='browser')

            fig0_path = f"{base_file_path}_ch_{self.params.ch}_average"
            #figure0.write_html(f"{fig0_path}.html")
            #figure0.write_image(f"{fig0_path}.png")

            print(f" \n Average plots saved in {fig0_path}")
    
        user_input = input("\n Do you want to plot the raw waveforms? (y/n): ").strip().lower()
        if user_input == "y":
            
            # ------------- Save the raw waveforms plot ------------- 
            
            figure1 = plot_CustomChannelGrid(
                self.grid_raw, 
                plot_function=lambda channel_ws, figure_, row, col: lc_utils.plot_wfs(
                    channel_ws, figure_, row, col,nwfs_plot=self.params.nwfs_plot, offset=False),
                share_x_scale=True,
                share_y_scale=True,
                show_ticks_only_on_edges=True 
            )

            title1 = f"No filtered waveforms for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

            figure1.update_layout(
                title={"text": title1, "font": {"size": 24}},
                width=1100,
                height=1200,
                showlegend=True
            )

            figure1.add_annotation(
                x=0.5, y=-0.05, xref="paper", yref="paper",
                text="Timeticks", showarrow=False, font=dict(size=16)
            )
            figure1.add_annotation(
                x=-0.07, y=0.5, xref="paper", yref="paper",
                text="Entries", showarrow=False, font=dict(size=16), textangle=-90
            )

            if self.params.show_figures:
                figure1.show(method='external', renderer='browser')

            fig1_path = f"{base_file_path}_wfs_raw"
            #figure1.write_html(f"{fig1_path}.html")
            #figure1.write_image(f"{fig1_path}.png")
            print(f"\nNo filtered waveforms saved in {fig1_path}")

        if self.thr_adc != -1:
            figure12 = plot_CustomChannelGrid(
                self.grid_filt1, 
                plot_function=lambda channel_ws, figure_, row, col: lc_utils.plot_wfs(
                    channel_ws, figure_, row, col, nwfs_plot=self.params.nwfs_plot,  offset=False, baseline = False),
                share_x_scale=True,
                share_y_scale=True,
                show_ticks_only_on_edges=True 
            )

            title12 = f"Waveforms with baseline filter for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

            figure12.update_layout(
                title={"text": title12, "font": {"size": 24}},
                width=1100,
                height=1200,
                showlegend=True
            )

            figure12.add_annotation(
                x=0.5, y=-0.05, xref="paper", yref="paper",
                text="Timeticks", showarrow=False, font=dict(size=16)
            )
            figure12.add_annotation(
                x=-0.07, y=0.5, xref="paper", yref="paper",
                text="Entries", showarrow=False, font=dict(size=16), textangle=-90
            )

            if self.params.show_figures:
                figure12.show(method='external', renderer='browser')
                
            fig12_path = f"{base_file_path}_wfs_filt1"
            #figure12.write_html(f"{fig12_path}.html")
            #figure12.write_image(f"{fig12_path}.png")
            print(f"\n Waveforms with only baseline filter saved in {fig12_path}")
        
        
        # ------------- Save the filtered waveforms plot ------------- 
            
        figure13 = plot_CustomChannelGrid(
            self.grid_filt2, 
            plot_function=lambda channel_ws, figure_, row, col: lc_utils.plot_wfs(
                channel_ws, figure_, row, col, nwfs_plot=self.params.nwfs_plot, offset=False),
            share_x_scale=True,
            share_y_scale=True,
            show_ticks_only_on_edges=True 
        )
        
        title13 = f"Filtered waveforms for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

        figure13.update_layout(
            title={
                "text": title13,
                "font": {"size": 24}
            },
            width=1100,
            height=1200,
            showlegend=True
        )
        
        figure13.add_annotation(
            x=0.5,
            y=-0.05, 
            xref="paper",
            yref="paper",
            text="Timeticks",
            showarrow=False,
            font=dict(size=16)
        )
        figure13.add_annotation(
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
            figure13.show(method='external', renderer='browser')
        
        fig13_path = f"{base_file_path}_wfs_filt"
        figure13.write_html(f"{fig13_path}.html")
        #figure13.write_image(f"{fig13_path}.png")
        
        print(f"\n Filtered waveforms saved in {fig13_path}")
        
        
        # ------------- Save the persistence plot -------------
        
        if isinstance(self.integration_intervals, list) and isinstance(self.params.ch, list) \
        and len(self.integration_intervals) == 1 and len(self.params.ch) == 1:  
        
            figure2 = plot_ChannelWsGrid(
                    self.grid_filt2,
                    figure=None,
                    share_x_scale=False,
                    share_y_scale=False,
                    mode="heatmap",
                    wfs_per_axes=len(self.selected_wfs3),
                    analysis_label=self.analysis_name,
                    detailed_label=False,
                    verbose=True
                )

            title2 = f"Persistence of filtered waveforms for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

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
                figure2.show(method='external', renderer='browser')
            
            fig2_path = f"{base_file_path}_ch_{self.params.ch}_pers"
            #figure2.write_html(f"{fig2_path}.html")
            #figure2.write_image(f"{fig2_path}.png")
            print(f"\n Persistance saved in {fig2_path}")
         
        return True