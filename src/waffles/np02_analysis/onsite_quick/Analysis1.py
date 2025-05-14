from waffles.np02_analysis.onsite_quick.imports import *

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
            
            thr_adc: int = Field(
                ...,
                description="A thrshold for the ADC values",
                example=8000
            )
            
            wf_peak: int = Field(
                ...,
                description="A guess on where the photoelectron peaks are located in the timeticks axis",
                example=262
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
        
        print("\n 1. Starting the analysis")
        
        # Obtain the endpoints from the detector
        eps = osq_utils.get_endpoints(self.params.det, self.det_id)
        
        # Select the waveforms and the corresponding waveformset in a specific time interval of the DAQ window
        self.selected_wfs1, self.selected_wfset1= osq_utils.get_wfs(self.wfset.waveforms, eps, self.params.ch, self.params.nwfs, self.params.tmin, self.params.tmax, self.params.rec, adc_max_threshold=15000)
   
        self.grid=osq_utils.get_grid(self.selected_wfs1, self.params.det, self.det_id)
        
        if self.params.tmin == -1 and self.params.tmax == -1:
            print(f"\n 2. Analyzing WaveformSet with {len(self.selected_wfs1)} waveforms, no specific time interval (tmin=-1 and tmax=-1).")
        else:
            print(f"\n 2. Analyzing WaveformSet with {len(self.selected_wfs1)} waveforms between tmin={self.params.tmin} and tmax={self.params.tmax}")
        
        print(f"\n 3. Creating the grid")

        analysis_params = osq_utils.get_analysis_params()
        
        checks_kwargs = IPDict()
        checks_kwargs['points_no'] = self.selected_wfset1.points_per_wf
        
        # Computing the baseline to apply the baseline cut
        
        self.analysis_name = 'standard'
    
        _ = self.selected_wfset1.analyse(
            self.analysis_name,
            BasicWfAna2,
            analysis_params,
            *[],  # *args,
            analysis_kwargs={},
            checks_kwargs=checks_kwargs,
            overwrite=True
        )
        
        '''
        self.selected_wfs2, self.selected_wfset2 =os_utils.baseline_cut(self.selected_wfs1)
        
        print(f"\n 4. After aplying the baseline cut, we have {len(self.selected_wfs2)} waveforms")
        
        self.selected_wfs3, self.selected_wfset3 =os_utils.adc_cut(self.selected_wfs2, thr_adc=self.thr_adc)
        
        self.grid_filt1= os_utils.get_grid(self.selected_wfs2, self.params.det, self.det_id)
        
        if self.thr_adc != -1:
            print(f"\n 5. After applying a filter on the ADC values, we have {len(self.selected_wfs3)} waveforms.")
        else:
            print(f"\n 5. No more filters were applied.")
        
        self.grid_filt2= os_utils.get_grid(self.selected_wfs3, self.params.det, self.det_id)
        
        print(f"\n 6. Computing the charge histograms, to establish the proper integration limits. Possibilities:{self.integration_intervals}")

        checks_kwargs['points_no'] = self.selected_wfset3.points_per_wf
        
        '''
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

        det_id_name=osq_utils.get_det_id_name(self.det_id)
        
        base_file_path = f"{self.params.output_path}"\
            f"run_{self.run}_{det_id_name}_{self.params.det}"   
            

        # ------------- Save the raw waveforms plot ------------- 
            
        figure1 = plot_CustomChannelGrid(
                self.grid, 
                plot_function=lambda channel_ws, figure_, row, col: osq_utils.plot_wfs(
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
            figure1.show()

        fig1_path = f"{base_file_path}_wfs_raw"
        figure1.write_html(f"{fig1_path}.html")
        figure1.write_image(f"{fig1_path}.png")
        print(f"\nNo filtered waveforms saved in {fig1_path}")

        
        # ------------- Save the FFT plot ------------- 
        
            
        figure2 = plot_CustomChannelGrid(
            self.grid, 
            plot_function=lambda channel_ws, figure_, row, col: osq_utils.plot_meanfft_function(
                channel_ws, figure_, row, col),
            share_x_scale=True,
            share_y_scale=True,
            show_ticks_only_on_edges=True 
        )
        
        title2 = f"FFT for {det_id_name} {self.params.det} - Runs {list(self.wfset.runs)}"

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
        
        fig2_path = f"{base_file_path}_wfs_filt"
        figure2.write_html(f"{fig2_path}.html")
        figure2.write_image(f"{fig2_path}.png")
        
        print(f"\n FFT plots saved in {fig2_path}")

         
        return True