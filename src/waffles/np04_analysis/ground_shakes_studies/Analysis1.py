from waffles.np04_analysis.ground_shakes_studies.imports import *
import matplotlib.pyplot as plt

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
        
        wfset_path = self.params.input_path+f"processed_merged_run_{self.run}.hdf5"
        self.wfset=WaveformSet_from_hdf5_file(wfset_path)        
        
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
        
        print(f"  1. Analizyng WaveformSet with {len(self.wfset.waveforms)} waveforms")
        
        analysis_params = gs_utils.get_analysis_params(
            self.apa,
            # Will fail when APA 1 is analyzed
            run=None
        )
        print('The parameters for the analysis are:',analysis_params)
        
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
        
        # ----- Creating the grid and computing charge histogram -----------

        print(f"  2. Creating the grid and computing charge histogram")

        # Create a grid of WaveformSets for each channel in one
        # APA, and compute the charge histogram for each channel
        
        self.grid_apa = ChannelWsGrid(
            APA_map[self.apa],
            self.wfset,
            compute_calib_histo=True, 
            bins_number=gs_utils.get_nbins_for_charge_histo(
                self.pde,
                self.apa
            ),
            domain=np.array((-10000.0, 50000.0)),
            variable="integral",
            analysis_label=self.analysis_name
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
            f"/run_{self.run}_apa_{self.apa}_pde_{self.pde}"
            
        # ------------- Save the persistance plot ------------- 

        figure1 = plot_ChannelWsGrid(
            self.grid_apa,
            figure=None,
            share_x_scale=False,
            share_y_scale=False,
            mode="heatmap",
            wfs_per_axes=len(self.wfset.waveforms),
            analysis_label=self.analysis_name,
            detailed_label=False,
            verbose=True
        )

        title1 = f"Persistence for APA {self.apa} - Runs {list(self.wfset.runs)}"

        figure1.update_layout(
            title={
                "text": title1,
                "font": {"size": 24}
            }, 
            width=1100,
            height=1200,
            showlegend=True
        )

        if self.params.show_figures:
            figure1.show()

        fig1_path = f"{base_file_path}_persistence.png"
        figure1.write_image(f"{fig1_path}")

        print(f" \n Persistence plots saved in {fig1_path}")

        # ------------- Save the charge histogram plot ------------- 

        figure2 = plot_ChannelWsGrid(
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

        title2 = f"Charge histograms for APA {self.apa} - Runs {list(self.wfset.runs)}"

        figure2.update_layout(
            title={
                "text": title2,
                "font": {"size": 24}
            }, 
            width=1100,
            height=1200,
            showlegend=True
        )

        if self.params.show_figures:
            figure2.show()

        fig2_path = f"{base_file_path}_calib_histo.png"
        figure2.write_image(f"{fig2_path}")

        print(f" \n Charge histogram plots saved in {fig2_path}")

        return True