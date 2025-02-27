from waffles.np04_analysis.light_yield_vs_e.imports import *

class Analysis1(WafflesAnalysis):
    def __init__(self):
        pass

    @classmethod
    def get_input_params_model(cls) -> type:
        """
        Implements the WafflesAnalysis.get_input_params_model() abstract method.
        Returns the InputParams class, which is a Pydantic model class defining the input parameters.
        """
        
        class InputParams(BaseInputParams):
            """Validation model for the input parameters of the analysis."""

            runs: list[int] = Field(
                ..., description="Run numbers to be read", example=[27906, 27907]
            )
            # waveforms_per_run: int = Field(
            #     ..., description="Number of waveforms to read per run", example=500
            # )
            correct_by_baseline: bool = Field(
                default=True,
                description="Whether to subtract the baseline before computing the average waveform"
            )
            rucio_paths_directory: str = Field(
                default=(
                    "/eos/experiment/neutplatform/protodune/"
                    "experiments/ProtoDUNE-II/PDS_Commissioning/"
                    "waffles/1_rucio_paths"
                ),
                description="Directory where rucio paths are stored, formatted as '0<run_number>.txt'"
            )
            
            validate_items = field_validator("runs", mode="before")(wcu.split_comma_separated_string)

        return InputParams

    def initialize(self, input_parameters: BaseInputParams) -> None:
        """
        Implements WafflesAnalysis.initialize() to define class attributes.
        """
        self.params = input_parameters

        self.read_input_loop_1 = self.params.runs
        self.read_input_loop_2 = [None]
        self.read_input_loop_3 = [None]
        self.analyze_loop = [None]

        self.wfset = None
        self.output_data = None

    def read_input(self) -> bool:
        """
        Reads waveforms for the current run and creates a WaveformSet.
        """
        print(f"In Analysis1.read_input(): Reading waveforms for run {self.read_input_itr_1}...")
        
        filepaths = reader.get_filepaths_from_rucio(
            f"{self.params.rucio_paths_directory}/0{self.read_input_itr_1}.txt"
        )

        self.wfset = reader.WaveformSet_from_hdf5_files(
            filepaths[:2],
            read_full_streaming_data=False,
            truncate_wfs_to_minimum=True,
            nrecord_start_fraction=0.0,
            nrecord_stop_fraction=1.0,
            subsample=1,
            wvfm_count=1e9,
            allowed_endpoints=[],
            det='HD_PDS',
            temporal_copy_directory='/tmp',
            erase_temporal_copy=True
        )

        return True

    def analyze(self) -> bool:
        """
        Analyzes the waveforms of the current run.
        """
        print(f"In Analysis1.analyze(): Analyzing waveforms of run {self.read_input_itr_1}...")
        return True

    def write_output(self) -> bool:
        """
        Saves the waveform data to an output file.
        """
        output_filepath = f"{self.params.output_path}/mean_waveform_run_{self.read_input_itr_1}.hdf5"
        print(f"In Analysis1.write_output(): Saving waveform data to {output_filepath}...")

        WaveformSet_to_file(
            waveform_set=self.wfset,
            output_filepath=f"{self.params.output_path}waveformset.hdf5",
            overwrite=True,
            format="hdf5",
            compression="gzip",
            compression_opts=5
        )
        
        new_ws = WaveformSet_from_hdf5_pickle(
            'output/waveformset.hdf5'
        )
        print(new_ws)
        
        return True