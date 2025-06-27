from waffles.np02_analysis.LED_trigger.imports import *

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
            """Input parameters.
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

            input_path: str = Field(
                default="/data",
                description="Input path"
            )
            
            output_path: str = Field(
                default="/output",
                description="Output path"
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
        self.selected_wfs1, self.selected_wfset1= lc_utils.get_wfs(self.wfset.waveforms, eps, self.params.ch, self.params.nwfs)
   
        self.grid_raw=lc_utils.get_grid(self.selected_wfs1, self.params.det, self.det_id)
        
        print(f"\n 2. Analyzing WaveformSet with {len(self.selected_wfs1)} waveforms")

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
