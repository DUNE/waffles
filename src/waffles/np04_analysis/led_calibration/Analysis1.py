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

            crps: list = Field(
                ...,
                description="CRP number",
                example=[2]
            ) # type: ignore

            pdes: list = Field(
                ...,
                description="Photon detection efficiency",
                example=[0.4]
            ) # type: ignore
            
            vgains: list = Field(
                ...,
                description="Vgain of DAPHNE",
                example=[]
            ) # type: ignore
            
            run: list = Field(
                ...,
                description="Run number",
                example=[]
            ) # type: ignore

            show_figures: bool = Field(
                default=False,
                description="Whether to show the produced "
                "figures",
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

        self.read_input_loop_1 = self.params.crps
        self.read_input_loop_2 = self.params.pdes
        self.read_input_loop_3 = self.params.vgains
                 

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
        
        self.crp  = self.read_input_itr_1
        self.pde   = self.read_input_itr_2
        self.vgain= self.read_input_itr_3
    
        
        print(f"Processing run:", "CRP", self.crp, "PDE", self.pde,
                "and vgain", self.vgain
                )

        first = True
        
        input_filepath = led_utils.get_input_filepath(
                self.params.input_path, 
                self.params.run
            )
        print(input_filepath)
        
        new_wfset = led_utils.read_data(
                input_filepath,
                self.crp,
                self.pde,
                self.vgain,
                is_folder=False,
                stop_fraction=1.,
            )
        self.wfset=new_wfset
        
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
            self.crp,
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
            APA_map[self.crp],
            self.wfset,
            compute_calib_histo=True, 
            bins_number=led_utils.get_nbins_for_charge_histo(
                self.pde,
                self.crp
            ),
            domain=np.array((-10000.0, 50000.0)),
            variable="integral",
            analysis_label=self.analysis_name
        )

        # ------------- Fit peaks of charge histogram -------------
        '''
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
        '''
        # ------------- Compute gain and S/N ------------- 
        '''
        print(f"  4. Computing S/N and gain")

        # Compute gain and S/N for every channel
        self.output_data = led_utils.get_gain_and_snr(
            self.grid_apa, 
            excluded_channels[self.batch][self.apa][self.pde]
        )
        '''
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
            f"/crp_{self.crp}_pde_{self.pde}_vgain_{self.vgain}"
            
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

        title = f"CRP {self.crp} - Run {list(self.wfset.runs)}"

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

        fig_path = f"{base_file_path}_charge_hist.png"
        figure.write_image(f"{fig_path}")

        print(f"  charge histogram plots saved in {fig_path}")

        return True