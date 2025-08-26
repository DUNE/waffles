from waffles.np04_analysis.ground_shakes.imports import *

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
            
            tmin_prec: int = Field(
                ...,
                description="Lower time limit considered for the analyzed waveforms",
                example=[-1000] # Alls
            )
            
            tmax_prec: int = Field(
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

        self.read_input_loop_1 = self.params.runs
        self.read_input_loop_2 = self.params.apas
        self.read_input_loop_3 = self.params.pdes
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
        self.run = self.read_input_itr_1
        self.apa = self.read_input_itr_2
        self.pde = self.read_input_itr_3
        
        print(
            "In function Analysis1.read_input(): "
            f"Now reading waveforms for run {self.run} ..."
        )
        
        try:
            wfset_path = self.params.input_path
            self.wfset=WaveformSet_from_hdf5_file(wfset_path)   
        except FileNotFoundError:
            raise FileNotFoundError(f"File {wfset_path} was not found. Make sure that the wfset file is named as 'wfset_0<run_number>.hdf5'")
        
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
        
        # Obtain the endpoints from the APA
        eps = gs_utils.get_endpoints(self.apa)
        
        # Select the waveforms and the corresponding waveformset in a specific time interval of the DAQ window
        selected_wfs, selected_wfset= gs_utils.get_wfs(self.wfset.waveforms, eps, self.params.ch, self.params.nwfs, self.params.tmin, self.params.tmax, self.params.rec)
        
        selected_wfs_plot, selected_wfset_plot= gs_utils.get_wfs(self.wfset.waveforms, eps, self.params.ch, 2000,self.params.tmin, self.params.tmax, self.params.rec)
        
        selected_wfs_prec, selected_wfset_prec= gs_utils.get_wfs(self.wfset.waveforms, eps, self.params.ch, self.params.nwfs, self.params.tmin_prec, self.params.tmax_prec, self.params.rec)
        
        selected_wfs_prec_plot, selected_wfset_prec_plot= gs_utils.get_wfs(self.wfset.waveforms, eps, self.params.ch, 2000, self.params.tmin_prec, self.params.tmax_prec, self.params.rec)
        
        print(f" 2. Analyzing WaveformSet with {len(selected_wfs)} waveforms between tmin={self.params.tmin} and tmax={self.params.tmax}")
        
        print(f" 3. Computing the number of ground shakes events per nanosecond")
        
        
        record_numbers=selected_wfset.record_numbers[self.run]
        
        self.record_number=len(record_numbers)
        
        print('Number of records:', self.record_number)
        
        trigger_window=selected_wfs[0].daq_window_timestamp
        trigger_window1=selected_wfs[len(selected_wfs)-1].daq_window_timestamp
        time_difference=(trigger_window1-trigger_window)*16
        records_per_second=self.record_number/time_difference
        
        print('Trigger window', trigger_window)
        print('Trigger window1', trigger_window1)
        print('Time difference in ns',time_difference)
        print('Records per ns',records_per_second)
        
        print(f" 4. Computing the std histogram for the min_tick values of every record")
        
        min_ticks_by_record=gs_utils.get_min_ticks(selected_wfs,record_numbers)
        self.std_by_record=gs_utils.get_std_min_ticks(min_ticks_by_record)
        
        print('min_ticks_by_record',min_ticks_by_record)
        
        print(f" 5. Creating the grids")
        
        # Create a grid of WaveformSets for each channel in one APA, and compute the corresponding function for each channel
        self.grid = gs_utils.get_grid(selected_wfs, self.apa, self.run)
        self.grid_plot = gs_utils.get_grid(selected_wfs_plot, self.apa, self.run)
        self.grid_prec = gs_utils.get_grid(selected_wfs_prec, self.apa, self.run)
        self.grid_prec_plot = gs_utils.get_grid(selected_wfs_prec_plot, self.apa, self.run)
        
        self.bins_number=gs_utils.get_nbins_for_charge_histo(
                self.pde,
                self.apa
            )
        
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
        
        base_file_path = f"{self.params.output_path}"\
            f"run_{self.run}_apa_{self.apa}"
            
        print(f" 6. Creating all the plots")    
        
        print('std_by_record',self.std_by_record)
        
        figure0 = plot_Histogram(self.std_by_record.values(),nbins=150, x_range=(-2,4))

        # Add the x-axis, y-axis and figure titles
        
        title0 = f"Std of the min ticks for every ground shake event in APA {self.apa} - Runs {list(self.wfset.runs)}"

        figure0.update_layout(
            title={
                "text": title0,
                "font": {"size": 24}
            },
            width=1100,
            height=1200,
            xaxis_title={
                "text": "Std",
                "font": {"size": 24}
            },
            yaxis_title={"text": "Entries",
                "font": {"size": 24}
            },
            
            showlegend=True
        )
  
        if self.params.show_figures:
            figure0.show()
            
        fig0_path = f"{base_file_path}_std_per_record.png"
        figure0.write_image(f"{fig0_path}")
        
        print(f"\n Stds per record saved in {fig0_path}") 
          
            
        # ------------- Save the waveforms plot ------------- 

        
        
        figure1 = plot_CustomChannelGrid(
            self.grid_plot, 
            plot_function=lambda channel_ws, figure_, row, col: gs_utils.plot_wfs(
                channel_ws, figure_, row, col,  offset=True),
            share_x_scale=True,
            share_y_scale=True,
            show_ticks_only_on_edges=True 
        )

        # Add the x-axis, y-axis and figure titles
        
        title1 = f"Waveforms for APA {self.apa} - Runs {list(self.wfset.runs)}"

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
            
        fig1_path = f"{base_file_path}_wfs.png"
        figure1.write_image(f"{fig1_path}")
        
        print(f"\n Waveforms saved in {fig1_path}")
        

        # ------------- Save the sigma histograms for the precursor ------------- 
        
        figure2 = plot_CustomChannelGrid(
            self.grid_prec, 
            plot_function=lambda channel_ws, figure_, row, col: gs_utils.plot_sigma_function(
                channel_ws, figure_, row, col, nbins=150),
            share_x_scale=True,
            share_y_scale=True,
            show_ticks_only_on_edges=True 
        )

        # Add the x-axis, y-axis and figure titles
        
        title2 = f"Sigma histograms of the precursor for APA {self.apa} - Runs {list(self.wfset.runs)}"

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
                        
        fig2_path = f"{base_file_path}_sigma_hist_prec.png"
        figure2.write_image(f"{fig2_path}")

        print(f"\n Sigma histograms for the precursor saved in {fig2_path}")
        
        
        # ------------- Save the FFT plots  ------------- 
        
        # Plot the sigma histograms of each channel
        figure3= plot_CustomChannelGrid(
                self.grid, 
                plot_function=lambda channel_ws, figure_, row, col: gs_utils.plot_meanfft_function(
                    channel_ws, figure_, row, col),
                share_x_scale=True,
                share_y_scale=True,
                show_ticks_only_on_edges=True,
                log_x_axis=True
            )
        
        # Add the x-axis, y-axis and figure titles
        
        title3 = f"Superimposed FFT for APA {self.apa} - Runs {list(self.wfset.runs)}"

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
        
        # ------------- Save the sigma histograms   ------------- 
        
        # Plot the sigma histograms of each channel
        figure4= plot_CustomChannelGrid(
                self.grid, 
                plot_function=lambda channel_ws, figure_, row, col: gs_utils.plot_sigma_function(
                    channel_ws, figure_, row, col, nbins=125),
                share_x_scale=True,
                share_y_scale=True,
                show_ticks_only_on_edges=True
            )
        
        # Add the x-axis, y-axis and figure titles
        
        title4 = f"Sigma histograms for APA {self.apa} - Runs {list(self.wfset.runs)}"

        figure4.update_layout(
            title={
                "text": title4,
                "font": {"size": 24}
            },
            width=1100,
            height=1200,
            showlegend=True
        )
        
        figure4.add_annotation(
            x=0.5,
            y=-0.05, 
            xref="paper",
            yref="paper",
            text="Sigma",
            showarrow=False,
            font=dict(size=16)
        )
        figure4.add_annotation(
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
            figure4.show()
                    
        fig3_path = f"{base_file_path}_sigma_hist.png"
        figure4.write_image(f"{fig3_path}")

        print(f" \n Sigma histograms saved in {fig3_path}")
        
        # -------- Precursors plots ----------
        
        
        figure5 = plot_CustomChannelGrid(
            self.grid_prec_plot, 
            plot_function=lambda channel_ws, figure_, row, col: gs_utils.plot_wfs(
                channel_ws, figure_, row, col,  offset=True),
            share_x_scale=True,
            share_y_scale=True,
            show_ticks_only_on_edges=True 
        )

        # Add the x-axis, y-axis and figure titles
        
        title5 = f"Precursors for APA {self.apa} - Runs {list(self.wfset.runs)}"

        figure5.update_layout(
            title={
                "text": title5,
                "font": {"size": 24}
            },
            width=1100,
            height=1200,
            showlegend=True
        )
        
        figure5.add_annotation(
            x=0.5,
            y=-0.05, 
            xref="paper",
            yref="paper",
            text="Timeticks",
            showarrow=False,
            font=dict(size=16)
        )
        figure5.add_annotation(
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
            figure5.show()
            
        fig5_path = f"{base_file_path}_wfs_prec.png"
        figure5.write_image(f"{fig5_path}")
        
        print(f"\n Precursors saved in {fig5_path}")
        
        
        return True