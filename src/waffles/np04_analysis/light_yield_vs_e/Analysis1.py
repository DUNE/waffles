
from waffles.np04_analysis.tp.imports import *

class Analysis1(WafflesAnalysis):

    def __init__(self):
        pass

    @classmethod
    def get_input_params_model(cls) -> type:
        class InputParams(BaseInputParams):
            waveforms_per_run: int = Field(
                ...,
                description="Number of waveforms to read per run",
                example=500
            )

        return InputParams

    def initialize(self, input_parameters: BaseInputParams) -> None:
        """Initializes the analysis with input parameters."""
        self.params = input_parameters
        self.read_input_loop_1 = [self.params.waveforms_per_run]
        self.read_input_loop_2 = [None]
        self.read_input_loop_3 = [None]
        self.analyze_loop = [None]
        self.wfset = None
        self.output_data = None

    def read_input(self) -> bool:
        """Reads waveform data for the current run and creates a WaveformSet."""
        print(f"In Analysis1.read_input(): Reading waveforms for run {self.read_input_itr_1}...")

      
        self.wfset = reader.WaveformSet_from_hdf5_file(
            '/nfs/home/marroyav/np02vd_raw_run035398_0000_df-s04-d0_dw_0_20250210T110326.hdf5',
            read_full_streaming_data=False,
            truncate_wfs_to_minimum=False,
            nrecord_start_fraction=0.0,
            nrecord_stop_fraction=1.0,
            subsample=1,
            wvfm_count=self.params.waveforms_per_run,
            allowed_endpoints=[],
            det="VD_Membrane_PDS", 
            temporal_copy_directory='/tmp',
            erase_temporal_copy=True
        )

        return True

    def analyze(self) -> bool:
        print(f"In Analysis1.analyze(): Analyzing waveforms of run {self.read_input_itr_1}...")
        return True

    def write_output(self) -> bool:
        output_filepath = f"{self.params.output_path}/mean_waveform_run_{self.read_input_itr_1}.hdf5"
        print(f"In Analysis1.write_output(): Saving waveform data to {output_filepath}...")

        WaveformSet_to_file(waveform_set=self.wfset,
            output_filepath=f"{self.params.output_path}tp_waveformset.hdf5",
            overwrite=True,
            format="hdf5",
            compression = "gzip",
            compression_opts = 5,)
        new_ws=WaveformSet_from_hdf5_pickle(f'/nfs/home/marroyav/waffles/src/waffles/np04_analysis/tp/output/tp_waveformset.hdf5')
        print(new_ws)
        return True
