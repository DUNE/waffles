from waffles.np04_analysis.example_analysis.imports import *

class Analysis2(WafflesAnalysis):

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

            prominence: float = Field(
                ...,
                description="Minimal prominence for a peak to "
                "be detected",
                example=20
            )

        return InputParams

    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
        """Implements the WafflesAnalysis.initialize() abstract
        method. It defines the attributes of the Analysis2 class.
        
        Parameters
        ----------
        input_parameters : BaseInputParams
            The input parameters for this analysis
            
        Returns
        -------
        None
        """

        # Save the input parameters into an Analysis2 attribute
        # so that they can be accessed by the other methods
        self.params = input_parameters

        self.read_input_loop_1 = [None,]
        self.read_input_loop_2 = [None,]
        self.read_input_loop_3 = [None,]
        self.analyze_loop = [None,]

        self.wfs = {}
        self.peaks = {}
        self.output_data = None

    def read_input(self) -> bool:
        """Implements the WafflesAnalysis.read_input() abstract
        method. It reads every pickle file in the path given by
        self.params.input_path which contains a WaveformAdcs object,
        and stores them in the self.wfs attribute, which is a
        dictionary. The keys of the dictionary are the filenames
        of the pickle files, and the values are the WaveformAdcs
        objects read from the files.

        Returns
        -------
        bool
            True if the method ends execution normally
        """

        print(
            "In function Analysis2.read_input(): "
            f"Now reading waveforms from {self.params.input_path} ..."
        )

        for filename in os.listdir(self.params.input_path):
            if filename.endswith('.pkl'):
                filepath = os.path.join(
                    self.params.input_path,
                    filename
                )
                try:
                    with open(filepath, 'rb') as file:
                        self.wfs[filename] = pickle.load(file)

                        print(
                            "In function Analysis2.read_input(): "
                            f"Succesfully loaded {filename}"
                        )
                except Exception as error_message:
                    print(
                        "In function Analysis2.read_input(): "
                        f"Error loading {filename}: {error_message}"
                    )

        return True

    def analyze(self) -> bool:
        """Implements the WafflesAnalysis.analyze() abstract method.
        It performs the analysis of the waveforms contained in the
        self.wfs attribute, which, for each waveform wf in self.wfs,
        consists of running scipy.signal.find_peaks(wf.adcs) with
        the prominence parameter set to self.params.prominence. The
        results are stored in the self.peaks attribute, which is a
        dictionary. The keys of the dictionary are the filenames of
        the pickle files, and the values are the results of the
        scipy.signal.find_peaks() function.
        
        Returns
        -------
        bool
            True if the method ends execution normally
        """

        for filename in self.wfs.keys():

            print(
                "In function Analysis2.analyze(): "
                "Finding peaks for waveform coming "
                f"from {filename}"
            )

            wf = self.wfs[filename]
            peaks, _ = spsi.find_peaks(
                # Find peaks over the inverted waveform
                -1.*wf.adcs,
                prominence=self.params.prominence
            )
            self.peaks[filename] = peaks

        return True

    def write_output(self) -> bool:
        """Implements the WafflesAnalysis.write_output() abstract
        method. For each waveform in self.wfs, it saves a plot of
        it together with its spotted peaks.

        Returns
        -------
        bool
            True if the method ends execution normally
        """

        for filename in self.wfs.keys():

            fig, ax = plt.subplots()

            ax.plot(
                self.wfs[filename].adcs,
            )

            ax.scatter(
                self.peaks[filename],
                [self.wfs[filename].adcs[idx] for idx in self.peaks[filename]],
                color='red',
                marker='^',
                s=100,
                label='Spotted peaks'
            )

            ax.set_xlabel("Time tick")
            ax.set_ylabel("ADC value")
            fig.suptitle(f"Waveform coming from {filename}")
            ax.legend()

            output_filepath = f"{self.params.output_path}"\
                f"/{filename[:-4]}.png"

            print(
                "In function Analysis2.write_output(): "
                f"Saving the plotted mean waveform to {output_filepath} ..."
            )

            plt.savefig(
                output_filepath,
                format='png'
            )

            plt.close()

        return True