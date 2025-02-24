from waffles.np04_analysis.example_analysis.imports import *

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

            waveforms_per_run: int = Field(
                ...,
                description="The number of waveforms to read from "
                "the first rucio filepath for each run",
                example=500
            )

            correct_by_baseline: bool = Field(
                default=True,
                description="Whether the baseline of each waveform "
                "is subtracted before computing the average waveform"
            )

            rucio_paths_directory: str = Field(
                default="/eos/experiment/neutplatform/protodune/"
                "experiments/ProtoDUNE-II/PDS_Commissioning/"
                "waffles/1_rucio_paths",
                description="The directory where the rucio paths are "
                "stored, following this format: the file "
                "'0<run_number>.txt' contains one rucio path per line."
            )

            validate_items = field_validator(
                "runs",
                mode="before"
            )(wcu.split_comma_separated_string)

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
        self.read_input_loop_2 = [None,]
        self.read_input_loop_3 = [None,]
        self.analyze_loop = [None,]

        self.wfset = None
        self.output_data = None

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

        print(
            "In function Analysis1.read_input(): "
            f"Now reading waveforms for run {self.read_input_itr_1} ..."
        )

        # Get the rucio filepaths for the current run
        filepaths = reader.get_filepaths_from_rucio(
            self.params.rucio_paths_directory+f"/0{self.read_input_itr_1}.txt"
        )

        # Try to read the first self.params.waveforms_per_run waveforms
        # from the first rucio filepath found for the current run
        self.wfset = reader.WaveformSet_from_hdf5_file(
            filepaths[0],
            read_full_streaming_data=False,
            truncate_wfs_to_minimum=True,
            nrecord_start_fraction=0.0,
            nrecord_stop_fraction=1.0,
            subsample=1,
            wvfm_count=self.params.waveforms_per_run,
            allowed_endpoints=[],
            det='HD_PDS',
            temporal_copy_directory='/tmp',
            erase_temporal_copy=True
        )

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

        print(
            "In function Analysis1.analyze(): "
            f"Computing the mean waveform for {len(self.wfset.waveforms)} "
            f"waveforms of run {self.read_input_itr_1} ..."
        )

        # 1. Compute the baseline for each waveform in the WaveformSet
        if self.params.correct_by_baseline:
            baselines = [
                wnu.get_baseline(
                    wf,
                    lower_time_tick_for_median=0,
                    upper_time_tick_for_median=100
                ) for wf in self.wfset.waveforms
            ]
        else:
            baselines = [
                0. for _ in self.wfset.waveforms
            ]

        # 2. Compute the mean waveform
        mean_adcs = np.mean(
            np.array([
                wf.adcs - baseline for wf, baseline in zip(
                    self.wfset.waveforms, baselines
                )
            ]),
            axis=0
        )

        self.output_data = WaveformAdcs(
            16.,
            mean_adcs,
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

        output_filepath = f"{self.params.output_path}"\
            f"/mean_waveform_run_{self.read_input_itr_1}.pkl"

        print(
            "In function Analysis1.write_output(): "
            f"Saving the mean waveform to {output_filepath} ..."
        )

        with open(output_filepath, "wb") as file:
            pickle.dump(
                self.output_data,
                file
            )

        return True