# import all necessary files and classes
from waffles.np04_analysis.beam_example.imports import *

class Analysis1(WafflesAnalysis):

    def __init__(self):
        pass        

    ##################################################################
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

            beam_input_path:          str = Field(...,    description="work in progress")
            wfs_input_path:           str = Field(...,    description="work in progress")
            events_output_path:       str = Field(...,    description="work in progress")
            wfset_light_output_path:  str = Field(...,    description="work in progress")
            wfset_heavy_output_path:  str = Field(...,    description="work in progress")
            delta_time:               float = Field(...,  description="work in progress")

        return InputParams

    ##################################################################
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    
        self.analyze_loop = [None,]
        self.params = input_parameters

        self.read_input_loop_1 = [None,]
        self.read_input_loop_2 = [None,]
        self.read_input_loop_3 = [None,]
        
    ##################################################################
    def read_input(self) -> bool:

        np04_file = f'{self.params.input_path}/{self.params.wfs_input_path}'
        beam_file = f'{self.params.input_path}/{self.params.beam_input_path}'
        
        print("Read np04 and beam information from: ")
        print("  np04 PDS hdf5 file: ", np04_file)
        print("  beam root file:       ", beam_file)

        # Read the two files and create BeamEvents combining their information
        self.events = events_from_hdf5_and_beam_files(np04_file, beam_file, self.params.delta_time)
        
        # sort events by timing
        self.events.sort(key=lambda x: x.ref_timestamp, reverse=False)

        print(f"\n {len(self.events)} events created from NP04 PDS and beam info")
        
        return True

    ##################################################################
    def analyze(self) -> bool:
        
        self.wfset_light = None
        self.wfset_heavy = None
        
        count_light = 0
        count_heavy = 0
        
        for event in self.events:
            if event.beam_info.tof > 0 and event.beam_info.tof < 110 and event.beam_info.p > 0:
                if event.wfset is not None:  # Asegurar que event.wfset no sea None
                    if self.wfset_light is None:
                        self.wfset_light = event.wfset
                    else:
                        self.wfset_light.merge(event.wfset)
                    count_light += 1  # Incrementar contador de eventos light
            
            elif event.beam_info.tof > 110 and event.beam_info.tof < 180 and event.beam_info.p > 0:
                if event.wfset is not None:  # Asegurar que event.wfset no sea None
                    if self.wfset_heavy is None:
                        self.wfset_heavy = event.wfset
                    else:
                        self.wfset_heavy.merge(event.wfset)
                    count_heavy += 1  # Incrementar contador de eventos heavy

        # Imprimir el número de eventos
        print(f'Events in wfset_light: {count_light}')
        print(f'Events in wfset_heavy: {count_heavy}')
        
        return True

    ##################################################################
    def write_output(self) -> bool:
        
        events_bytes=pickle.dumps(self.events)
        events_np=np.frombuffer(events_bytes,dtype=np.uint8)
        with h5py.File(self.params.events_output_path, "w") as hdf:
            hdf.create_dataset("wfset", data=events_np, compression="gzip") 
        print(f'\n Total events saved in file: {self.params.events_output_path}')
        
        # --------------------- Wfset_light -------------------
        
        wfset_light_bytes = pickle.dumps(self.wfset_light)
        wfset_light_np = np.frombuffer(wfset_light_bytes, dtype=np.uint8)  # Convert to NumPy array

        with h5py.File(self.params.wfset_light_output_path, "w") as hdf:
            hdf.create_dataset("wfset", data=wfset_light_np, compression="gzip") 
        print(f'\n Light waveforms saved in file: {self.params.wfset_light_output_path}')
        
        # --------------------- Wfset_heavy -------------------
        
        wfset_heavy_bytes = pickle.dumps(self.wfset_heavy)
        wfset_heavy_np = np.frombuffer(wfset_heavy_bytes, dtype=np.uint8)  # Convert to NumPy array

        with h5py.File(self.params.wfset_heavy_output_path, "w") as hdf:
            hdf.create_dataset("wfset", data=wfset_heavy_np, compression="gzip")
        print(f'\n Heavy waveforms saved in file: {self.params.wfset_heavy_output_path} \n')
        
        return True


       
