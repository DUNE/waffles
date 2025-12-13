from dataclasses import Field
from waffles.np04_analysis.vgain_analysis.imports import *
import gc
import weakref
import math

def list_defaultdict():
    return defaultdict(list)

class Analysis3(WafflesAnalysis):

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

            batches: list[int] = Field(
                ...,
                description="Number of the calibration batches "
                "to consider",
                example=[2]
            )

            apas: list[int] = Field(
                ...,
                description="Numbers of the APAs to consider",
                example=[2]
            )

            pdes: list[float] = Field(
                ...,
                description="Photon detection efficiencies to "
                "consider",
                example=[0.4]
            )

            channel_to_analyze: int = Field(
                ...,
                description="Single channel to analyze",
                example=10413
            )

            filter_type: list[str] = Field(
                ...,
                description="Type of high-pass filter to apply "
                "after the baseline subtraction. The available "
                "options are: 'Bessel', 'Butter', "
                "'Cheby_I', 'Cheby_II', 'Elliptic'.",
                example=['Bessel']
            )

            hpf_cutoff_frequency: list[str] = Field(
                ...,
                description="High-pass filter cutoff frequency",
                example=['160khz']
            )

            channels_per_run_filepath: str = Field(
                ...,
                description="Path to a CSV file which lists "
                "the channels which should be calibrated for "
                "each run. Apart from the run number and the "
                "targeted channels, each row contains a value "
                "for the batch number, the acquired APAs and "
                "the photon detection efficiency (PDE).",
                example='./configs/channels_per_run_database.csv'
            )

            excluded_channels_filepath: str = Field(
                ...,
                description="Path to a CSV file which lists "
                "the channels which should be excluded from "
                "the calibration for each combination of "
                "batch number, APA number and PDE",
                example='./configs/excluded_channels_database.csv'
            )

            hpf_filter_coefficients: str = Field(
                ...,
                description="Path to a JSON file which contains "
                "the high-pass filter coefficients to be applied "
                "after the baseline subtraction.",
                example='./configs/hpf_filter_coefficients.json'
            )

            boxcar_window_size: int = Field(
                ...,
                description="Boxcar filter window size to be applied after the HPF filter.",
                example=23
            )

            show_figures: bool = Field(
                default=False,
                description="Whether to show the produced "
                "figures",
            )

            verbose: bool = Field(
                default=False,
                description="Whether to print verbose messages "
                "or not"
            )

            baseline_analysis_label: str = Field(
                default='baseliner',
                description="Label for the baseline analysis",
            )

            hpf_analysis_label: str = Field(
                default='hpf_filter',
                description="Label for the high-pass filter analysis",
            )

            null_baseline_analysis_label: str = Field(
                default='null_baseliner',
                description="Label for the null baseline analysis",
            )

            baseline_limits: dict[int, list[int]] = Field(
                ...,
                description="Gives the region of the waveform "
                "which contains the ADC samples which will be "
                "used to compute the baseline.",
            )

            baseliner_std_cut: float = Field(
                default=3.0,
                description="Number of allowed standard deviations "
                "from a preliminary baseline estimate. The ADC "
                "samples that fall into the given range are "
                "considered in the definitive baseline computation.",
            )

            baseliner_type: str = Field(
                default="mean",
                description="How to compute the baseline out "
                "of the selected ADC samples",
            )

            lower_limit_wrt_baseline: float = Field(
                ...,
                description="It is used for the coarse selection "
                "cut. Its absolute value is the allowed margin for "
                "the waveform adcs below its baseline.",
                example=-200.
            )

            upper_limit_wrt_baseline: float = Field(
                ...,
                description="It is used for the coarse selection "
                "cut. Its absolute value is the allowed margin for "
                "the waveform adcs above its baseline.",
                example=40.
            )

            baseline_i_up: dict[int, int] = Field(
                ...,
                description="A dictionary whose keys refer to "
                "the APA number, and its values are the "
                "ADCs-array iterator value for the upper limit "
                "of the window which is considered to be the "
                "baseline region. If the waveform deviates from "
                "the baseline by more than a certain amount in "
                "this region, it will be excluded from the analysis.",
                example={1: 575, 2: 120, 3: 120, 4: 120}
            )

            signal_i_up: dict[int, int] = Field(
                ...,
                description="A dictionary whose keys refer to "
                "the APA number, and its values are the "
                "ADCs-array iterator value for the upper limit "
                "of the window where an upper-bound cut to the "
                "signal is applied",
                example={1: 650, 2: 165, 3: 165, 4: 165}
            )

            baseline_allowed_dev: float = Field(
                ...,
                description="Number of allowed baseline-STDs "
                "in the baseline region. I.e. the waveforms for "
                "which at least one ADC sample in the baseline "
                "region deviates from the baseline by more than "
                "this value times the baseline STD will be "
                "excluded.",
                example=4.0
            )

            signal_allowed_dev: float = Field(
                ...,
                description="Number of allowed baseline-STDs "
                "in the signal region. I.e. the waveforms for "
                "which at least one ADC sample in the signal "
                "region deviates from the signal by more than "
                "this value times the baseline STD will be "
                "excluded.",
                example=10.0
            )

            deviation_from_baseline: float = Field(
                ...,
                description="It is interpreted as a fraction of "
                "the signal amplitude, as measured from the "
                "baseline. The integration limits are adjusted "
                "so that only the part of the signal which "
                "exceeds this fraction is integrated.",
                example=0.2
            )

            lower_limit_correction: int = Field(
                default=0,
                description="Correction to be applied to the "
                "lower limit of the integration window",
            )

            upper_limit_correction: int = Field(
                default=0,
                description="Correction to be applied to the "
                "upper limit of the integration window",
            )

            integration_analysis_label: str = Field(
                default='integrator',
                description="Label for the integration analysis",
            )

            calib_histo_bins_number: dict[float, int] = Field(
                ...,
                description="Number of bins in the calibration "
                "histogram for each PDE. The keys are the "
                "PDEs, and the values are the number of bins "
                "in the calibration histogram."
            )

            calib_histo_lower_limit: float = Field(
                default=-10000.,
                description="Lower limit for the calibration "
                "histogram",
            )

            calib_histo_upper_limit: float = Field(
                default=50000.,
                description="Upper limit for the calibration "
                "histogram",
            )

            max_peaks: int = Field(
                ...,
                description="Maximum number of peaks to fit in "
                "each charge histogram",
                example=2
            )

            prominence: float = Field(
                ...,
                description="Minimal prominence, as a fraction "
                "of the y-range of the charge histogram, for a "
                "peak to be detected",
                example=0.15
            )

            initial_percentage: float = Field(
                default=0.15,
                description="Initial fraction of the calibration "
                "histogram to consider for the peak search"
            )

            percentage_step: float = Field(
                default=0.05,
                description="Step size for the percentage used "
                "in the peak search"
            )

            fit_type: str = Field(
                default='correlated_gaussians',
                description="Type of the fit to be used for the "
                "peaks in the charge histogram. It can be either "
                "'correlated_gaussians' or 'independent_gaussians'."
            )

            half_points_to_fit: int = Field(
                default=2,
                description="Only used if fit_type is set "
                "to 'independent_gaussians'. The number of "
                "points to fit on either side of the peak maximum."
            )

            std_increment_seed_fallback: float = Field(
                default=100.,
                description="Only used if fit_type is set "
                "to 'correlated_gaussians'. It is used when the "
                "peak finder predicts that the standard deviation "
                "of the second peak is less than the standard "
                "deviation of the first peak, which is incompatible "
                "with our fitting function. In this case, the "
                "seed for the standard deviation increment is set "
                "to this value."
            )

            ch_span_fraction_around_peaks: float = Field(
                default=0.03,
                description="Only used if fit_type is set to "
                "'correlated_gaussians'. Fraction of the charge "
                "histogram span around the extremal peaks to "
                "consider for the calibration histogram fit."
            )

            save_persistence_heatmaps: bool = Field(
                default=False,
                description="Whether to save the persistence "
                "heatmaps of the integrated waveforms or not"
            )

            output_dataframe_filename: str = Field(
                default='calibration_results.csv',
                description="Name of the output CSV file "
                "where the calibration results will be saved"
            )

            output_log_filename: str = Field(
                default='vGain_analysis_log.txt',
                description="Name of the output log file "
                "where the analysis log will be saved"
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
        self.grid_apa = None
        self.output_data = None

        self.filter_type = None
        self.filter_cutoff = None

        # Auxiliar waveformsets
        self.grid_apa_zero_baselined = None
        self.grid_apa_hpf_filtered = None

        self.grid_spe_idxs = defaultdict(list_defaultdict)
        self.grid_spe_parameters = defaultdict(list_defaultdict)

        self.io_data_dict = IODict()

        self._open_figures = []

        Path(f"{self.params.output_path}").mkdir(parents=True, exist_ok=True)

        self.output_log_file = open(
            f"{self.params.output_path}/{self.params.output_log_filename}", "a"
        )

        try:
            self.hpf_coefficients_dict = json.load(
                open(self.params.hpf_filter_coefficients, 'r')
            )
        except Exception as e:
            self.output_log_file.write(
                f"Error loading HPF coefficients with exception: {e}\n"
            )
            exit(1)

        self.read_input_loop_1 = self.params.batches
        self.read_input_loop_2 = self.params.apas
        self.read_input_loop_3 = self.params.pdes

        # columns: run, batch, acquired_apas, aimed_channels, pde
        try:
            self.channels_per_run = pd.read_csv(
                self.params.channels_per_run_filepath
            )
        except Exception as e:
            self.output_log_file.write(
                f"Error loading channels per run database with exception: {e}\n"
            )
            exit(1)

        # columns: batch, apa, pde, excluded_channels
        try:
            self.excluded_channels = pd.read_csv(
                self.params.excluded_channels_filepath
            )
        except Exception as e:
            self.output_log_file.write(
                f"Error loading excluded channels database with exception: {e}\n"
            )
            exit(1)

    def read_input(self) -> bool:
        """Implements the WafflesAnalysis.read_input() abstract
        method. It loads a WaveformSet object into the self.wfset
        attribute which matches the input parameters, namely the
        APA number, the PDE and the batch number. The final
        WaveformSet is the result of merging different WaveformSet
        objects, each of which comes from a different run. The
        decision on which run contributes to which channel is
        done based on the configuration files, namely on the
        self.channels_per_run attribute which is read from the
        configuration file whose path is given by the
        self.params.channels_per_run_filepath input parameter.

        Returns
        -------
        bool
            True if the method ends execution normally
        """

        self.filter_type = self.params.filter_type[0]
        self.filter_cutoff = self.params.hpf_cutoff_frequency[0]
        self.boxcar_window_size = self.params.boxcar_window_size
        
        self.batch = self.read_input_itr_1
        self.apa = self.read_input_itr_2
        self.pde = self.read_input_itr_3

        if self.params.verbose:
            print(
                "In function Analysis1.read_input(): "
                f"Processing runs for batch {self.batch}, "
                f"APA {self.apa}, "
                f"and PDE {self.pde}"
            )

        # apa_filter is a list of booleans, so that the
        # i-th entry is true if the current self.apa is
        # in the acquired_apas list for the i-th run in
        # the channels_per_run DataFrame
        apa_filter = pd.Series(
            [
                self.apa in led_utils.parse_numeric_list(aux)
                for aux in self.channels_per_run['acquired_apas']
            ]
        )

        # Filter the channels_per_run DataFrame to only
        # include runs from the current self.batch,
        # self.apa and self.pde
        filtered_channels_per_run = self.channels_per_run[
            (self.channels_per_run['batch'] == self.batch) &
            apa_filter &
            (self.channels_per_run['pde'] == self.pde)
        ]

        targeted_runs = list(
            filtered_channels_per_run['run'].values
        )

        if len(targeted_runs) == 0:
            if self.params.verbose:
                print(
                    "In function Analysis1.read_input(): "
                    f"Found no runs for batch {self.batch}, "
                    f"APA {self.apa} and PDE {self.pde}. "
                    "Skipping this configuration."
                )

            # WafflesAnalysis.execute() takes care of
            # skipping this iteration of the loop if
            # read_input() returns False, i.e. self.analyze()
            # and self.write_output() won't be executed
            # for this particular configuration
            return False

        fFirstRun = True

        # Reset the WaveformSet
        self.wfset = None
        


        # Loop over the list of runs for the current
        # batch, APA and PDE
        
        for i, run in enumerate(targeted_runs):

            channels = led_utils.parse_numeric_list(
                self.channels_per_run[
                    self.channels_per_run['run'] == run
                ]['aimed_channels'].values[0]
            )
            
            channel_to_analyze_ = self.params.channel_to_analyze
            if channel_to_analyze_ != 0:
                channels = [ch for ch in channels if ch == channel_to_analyze_]
                
            if self.params.verbose:
                print(
                    "In function Analysis1.read_input(): "
                    f"Reading the data for run {run}.",
                )

            if len(channels) == 0:
                if self.params.verbose:
                    print(
                        "The list of aimed channels is empty"
                        " for this run. Skipping this run."
                    )
                continue
            else:
                if self.params.verbose:
                    print(
                        f"The read channels are: {channels}."
                    )   
            
            # Get the filepaths to the input chunks for this run
            #input_filepaths = led_utils.get_input_filepaths_for_run(
            #    self.params.input_path,
            #    self.batch,
            #    self.pde,
            #    run
            #)

            #new_wfset = WaveformSet_from_pickle_files(
            #        filepath_list=input_filepaths,
            #        target_extension='.pkl',
            #        verbose=self.params.verbose
            #)
            try:
                input_filepaths = led_utils.get_input_filepaths_for_vgain_scan_run(
                        self.params.input_path,
                        self.batch,
                        self.pde,
                        run)

                new_wfset = led_utils.get_vgain_scan_waveformSet(
                        self.params.input_path,
                        self.batch,
                        input_filepaths)
            except Exception as e:
                print(f"Error in loading run {run} with exception: {e}")
                self.output_log_file.write(
                    f"Error in loading run {run} with exception: {e}\n"
                )
    
            # Keep only the waveforms coming from 
            # the targeted channels for this run.
            # This step is useless for cases when
            # the input pickles were already filtered
            # when copied from the original HDF5 files
            try:
                new_wfset = WaveformSet.from_filtered_WaveformSet(
                    new_wfset,
                    led_utils.comes_from_channel,
                    channels
                )

                if fFirstRun:
                    self.wfset = new_wfset
                    fFirstRun = False
                else:
                    self.wfset.merge(new_wfset)
            except Exception as e:
                print(f"Empty waveformSet in run {run} with exception: {e}")
                self.output_log_file.write(
                    f"Empty waveformSet in run {run} with exception: {e}\n"
                )
            
        return True

    def analyze(self) -> bool:
        """ Implements the WafflesAnalysis.analyze() abstract
        method. It performs the analysis of the waveforms contained
        in the self.wfset attribute, which consists of the following
        steps:

        1. Compute the baseline of each waveform
        2. Apply a coarse selection cut to the whole waveform
        set based on their maximum deviation from the baseline
        3. Compute the average baseline STD for each channel
        4. Apply a selection cut to each channel, based on
        its average baseline STD
        5. Subtract the baseline from each waveform and
        compute the average waveform of each channel
        6. Compute the integration window for each channel
        and integrate the waveforms
        7. Compute the calibration histogram for each channel
        and fit the first N peaks
        8. Out of the fit parameters, compute the gain and
        S/N for each channel
        
        Returns
        -------
        bool
            True if the method ends execution normally
        """
        # (re)create per-iteration containers that previous iterations may have cleaned up
        if self.io_data_dict is None:
            self.io_data_dict = IODict()
        # This is the only analysis stage which can be run on the merged WaveformSet,
        # since, for each waveform, it only depends on such waveform, and not on any
        # characteristics of the channel which it comes from

        baseliner_input_parameters = IPDict({
            'baseline_limits': self.params.baseline_limits[self.apa],
            'std_cut': self.params.baseliner_std_cut,
            'type': self.params.baseliner_type
        })

        checks_kwargs = IPDict({
            'points_no': self.wfset.points_per_wf
        })

        if self.params.verbose:
            print(
                "In function Analysis1.analyze(): "
                f"Running the baseliner on the merged "
                f"WaveformSet for batch {self.batch}, "
                f"APA {self.apa}, and PDE {self.pde} ... ",
                end=''
            )

        # Compute the baseline for the waveforms in the new WaveformSet
        _ = self.wfset.analyse(
            self.params.baseline_analysis_label,
            WindowBaseliner,
            baseliner_input_parameters,
            checks_kwargs=checks_kwargs,
            overwrite=True
        )

        if self.params.verbose:
            print("Finished.")

        # Add a dummy baseline analysis to the merged WaveformSet
        # We will use this for the integration stage after having
        # subtracted the actual baseline
        _ = self.wfset.analyse(
            self.params.null_baseline_analysis_label,
            StoreWfAna,
            {'baseline': 0.},
            overwrite=True
        )

        if self.params.verbose:
            print(
                "In function Analysis1.analyze(): "
                "Applying the coarse selection cut on  "
                "the merged WaveformSet for batch "
                f"{self.batch}, APA {self.apa}, and PDE "
                f"{self.pde} ... ",
                end=''
            )

        len_before_coarse_selection = len(self.wfset.waveforms)

        self.wfset = WaveformSet.from_filtered_WaveformSet(
            self.wfset,
            coarse_selection_for_led_calibration,
            self.params.baseline_analysis_label,
            self.params.lower_limit_wrt_baseline,
            self.params.upper_limit_wrt_baseline
        )

        len_after_coarse_selection = len(self.wfset.waveforms)

        if self.params.verbose:
            print(
                f"Kept {100.*(len_after_coarse_selection/len_before_coarse_selection):.2f}%"
                " of the waveforms"
            )
        
        self.grid_apa = None
        self.grid_apa_zero_baselined = None
        self.grid_apa_hpf_filtered = None

        # Separate the WaveformSet into a grid of WaveformSets,
        # so that each WaveformSet contains all of the waveforms
        # which come from the same channel
        self.grid_apa = ChannelWsGrid(   
            APA_map[self.apa],
            self.wfset,
            compute_calib_histo=False,
        )

        self.grid_apa_zero_baselined = ChannelWsGrid(   
            copy.deepcopy(APA_map[self.apa]),
            copy.deepcopy(self.wfset),
            compute_calib_histo=False,
        )

        self.grid_apa_hpf_filtered = ChannelWsGrid(   
            copy.deepcopy(APA_map[self.apa]),
            copy.deepcopy(self.wfset),
            compute_calib_histo=False,
        )
        mean_waveform_zero_baselined_io_dict = IODict()
        mean_waveform_hpf_filtered_io_dict = IODict()
        for endpoint in self.grid_apa.ch_wf_sets.keys():
            for channel in self.grid_apa.ch_wf_sets[endpoint].keys():

                if self.params.verbose:
                    print(
                        "In function Analysis1.analyze(): "
                        "Computing the average baseline STD "
                        f"of channel {endpoint}-{channel} "
                        f"(batch {self.batch}, APA {self.apa},"
                        f" PDE {self.pde}) ... ",
                        end=''
                    )

                average_baseline_std = led_utils.compute_average_baseline_std(
                    self.grid_apa.ch_wf_sets[endpoint][channel],
                    self.params.baseline_analysis_label
                )

                if self.params.verbose:
                    print(f"Found {average_baseline_std:.2f} ADCs.")
                    print(
                        "In function Analysis1.analyze(): "
                        "Applying the selection cut to channel "
                        f"{endpoint}-{channel} (batch {self.batch}"
                        f", APA {self.apa}, PDE {self.pde}) ... ",
                        end=''
                    )

                len_before_fine_selection = len(
                    self.grid_apa.ch_wf_sets[endpoint][channel].waveforms
                )

                # By applying the selection cut at this point, we avoid
                # integrating waveforms which will not make it through
                # the selection cut
                aux = WaveformSet.from_filtered_WaveformSet(
                    self.grid_apa.ch_wf_sets[endpoint][channel],
                    fine_selection_for_led_calibration,
                    self.params.baseline_analysis_label,
                    self.params.baseline_i_up[self.apa],
                    self.params.signal_i_up[self.apa],
                    average_baseline_std,
                    self.params.baseline_allowed_dev,
                    self.params.signal_allowed_dev
                )

                self.grid_apa.ch_wf_sets[endpoint][channel] = \
                    ChannelWs(*aux.waveforms)


                len_after_fine_selection = len(
                    self.grid_apa.ch_wf_sets[endpoint][channel].waveforms
                )

                if self.params.verbose:
                    print(
                        f"Kept {100.*(len_after_fine_selection/len_before_fine_selection):.2f}%"
                        " of the waveforms"
                    )
                    print(
                        "In function Analysis1.analyze(): "
                        "Subtracting the baseline from each waveform "
                        f"of channel {endpoint}-{channel} "
                        f"(batch {self.batch}, APA {self.apa},"
                        f" PDE {self.pde}) ... ",
                        end=''
                    )

                self.grid_apa.ch_wf_sets[endpoint][channel].apply(
                    subtract_baseline,
                    self.params.baseline_analysis_label,
                    show_progress=False
                )

                self.grid_apa_zero_baselined.ch_wf_sets[endpoint][channel] = copy.deepcopy(self.grid_apa.ch_wf_sets[endpoint][channel])
                mean_waveform_zero_baselined_io_dict[endpoint, channel] = self.grid_apa_zero_baselined.ch_wf_sets[endpoint][channel].compute_mean_waveform()

                #Here after the baseline subtraction I should put the HPF filter.
                coefficients = self.hpf_coefficients_dict[self.filter_type][self.filter_cutoff]
                self.grid_apa.ch_wf_sets[endpoint][channel].apply(
                    filter_waveform,
                    self.params.baseline_analysis_label,
                    filterType = 'IIR',
                    numerator = coefficients[0],
                    denominator = coefficients[1],
                    show_progress=False
                )

                #Here, Apply a 2MHz 3rd order LPF
                coefficients = self.hpf_coefficients_dict["LPF"]["2Mhz"]
                self.grid_apa.ch_wf_sets[endpoint][channel].apply(
                    filter_waveform,
                    self.params.baseline_analysis_label,
                    filterType = 'IIR',
                    numerator = coefficients[0],
                    denominator = coefficients[1],
                    show_progress=False
                )

                self.grid_apa_hpf_filtered.ch_wf_sets[endpoint][channel] = copy.deepcopy(self.grid_apa.ch_wf_sets[endpoint][channel])
                mean_waveform_hpf_filtered_io_dict[endpoint, channel] = self.grid_apa_hpf_filtered.ch_wf_sets[endpoint][channel].compute_mean_waveform()

                if self.params.verbose:
                    print("Finished.")
                    print(
                        "In function Analysis1.analyze(): "
                        "Computing the integration window for "
                        f"channel {endpoint}-{channel} "
                        f"(batch {self.batch}, APA {self.apa},"
                        f" PDE {self.pde}) ... ",
                        end=''
                    )

                # mean_wf = self.grid_apa.ch_wf_sets[endpoint][channel].\
                #     compute_mean_waveform()

                # limits = get_pulse_window_limits(
                #     mean_wf.adcs,
                #     0,
                #     self.params.deviation_from_baseline,
                #     self.params.lower_limit_correction,
                #     self.params.upper_limit_correction
                # )

                # if self.params.verbose:
                #     print(f"Found limits {limits[0]}-{limits[1]}.")
                #     print(
                #         "In function Analysis1.analyze(): "
                #         "Integrating the waveforms for channel "
                #         f"{endpoint}-{channel} (batch "
                #         f"{self.batch}, APA {self.apa}, PDE "
                #         f"{self.pde}) ... ",
                #         end=''
                #     )
                
                # integrator_input_parameters = IPDict({
                #     'baseline_analysis': self.params.null_baseline_analysis_label,
                #     'inversion': True,
                #     'int_ll': limits[0],
                #     'int_ul': limits[1],
                #     'amp_ll': limits[0],
                #     'amp_ul': limits[1]
                # })

                mean_filtered_wf = self.grid_apa.ch_wf_sets[endpoint][channel].compute_mean_waveform()
                boxcar_mean_wf = applyDiscreteFilter(
                    -1.0*mean_filtered_wf.adcs,
                    filterType = 'Boxcar',
                    boxcarFilterLength = self.boxcar_window_size
                )

                max_pos = np.argmax(boxcar_mean_wf)
                max_vale_avg = np.max(boxcar_mean_wf)

                integrator_input_parameters_boxcar = IPDict({
                    'inversion': True,
                    'window_length': self.boxcar_window_size,
                    'avg_window_length': 1,
                    'max_value_avg': max_vale_avg,
                    'max_pos': max_pos,
                    #'int_ll': limits[0],
                    #'int_ul': limits[1],
                    'amp_ll': 30,
                    'amp_ul': 200
                })

                checks_kwargs = IPDict({
                    'points_no': self.grid_apa.ch_wf_sets[endpoint][channel].\
                        points_per_wf
                })

                _ = self.grid_apa.ch_wf_sets[endpoint][channel].analyse(
                    self.params.integration_analysis_label,
                    BoxcarIntegrator,
                    integrator_input_parameters_boxcar,
                    checks_kwargs=checks_kwargs,
                    overwrite=True
                )

                if self.params.verbose:
                    print("Finished.")

        self.io_data_dict["mean_waveform_zero_baselined"] = mean_waveform_zero_baselined_io_dict
        self.io_data_dict["mean_waveform_hpf_filtered"] = mean_waveform_hpf_filtered_io_dict

        if self.params.verbose:
            print(
                "In function Analysis1.analyze(): "
                "Building the calibration histogram "
                "and fitting the peaks for batch "
                f"{self.batch}, APA {self.apa}, and "
                f"PDE {self.pde} ... ",
                end=''
            )
        
        calib_histo_lower_limit_, calib_histo_upper_limit_ = auto_domain_from_grid(
            self.grid_apa,
            analysis_label=self.params.integration_analysis_label,
            variable="integral",
            q_low=0.005,
            q_high=0.995,
            pad_frac=0.05,
        )

        self.grid_apa.compute_calib_histos(
            self.params.calib_histo_bins_number[self.pde],
            domain=np.array(
                (
                    self.params.calib_histo_lower_limit,
                    self.params.calib_histo_upper_limit
                )
            ),
            variable='integral',
            analysis_label=self.params.integration_analysis_label,
            verbose=self.params.verbose
        )

        fit_peaks_of_ChannelWsGrid( 
            self.grid_apa,
            self.params.max_peaks,
            self.params.prominence,
            self.params.initial_percentage,
            self.params.percentage_step,
            return_last_addition_if_fail=True,
            fit_type=self.params.fit_type,
            half_points_to_fit=self.params.half_points_to_fit,
            std_increment_seed_fallback=self.params.std_increment_seed_fallback,
            ch_span_fraction_around_peaks=self.params.ch_span_fraction_around_peaks,
            verbose=self.params.verbose
        )
        # Check which waveforms are within 2 std from the mean of the second peak
        #create a array of empty list with the same shape as the grid
        
        calib_hist_io_dict = IODict()
        for endpoint in self.grid_apa.ch_wf_sets.keys():
            for channel in self.grid_apa.ch_wf_sets[endpoint].keys():
                try:
                    u1 = self.grid_apa.ch_wf_sets[endpoint][channel].calib_histo.gaussian_fits_parameters['mean'][1][0]
                    std1 = self.grid_apa.ch_wf_sets[endpoint][channel].calib_histo.gaussian_fits_parameters['std'][1][0]
                except Exception as e:
                    u1 = 99999
                    std1 = 99
                    self.output_log_file.write(
                        f"Error getting μ₁ and σ₁ in APA:{self.apa} - endpoint: {endpoint} - channel:{channel}. Exception: {e}\n"
                    )
                calib_hist_io_dict[endpoint, channel] = self.grid_apa.ch_wf_sets[endpoint][channel].calib_histo
                print(f"Channel {endpoint}-{channel} has μ₁ = {u1} and σ₁ = {std1}")
                spe_idxs = []
                for idx, wf in enumerate(self.grid_apa.ch_wf_sets[endpoint][channel].waveforms):
                    integral = wf.get_analysis(self.params.integration_analysis_label).result['integral']
                    if integral < u1 + 2*std1 and integral > u1 - 2*std1:
                        spe_idxs.append(idx)
                if(len(spe_idxs) != 0):
                    mean_spe = self.grid_apa.ch_wf_sets[endpoint][channel].compute_mean_waveform(wf_idcs=spe_idxs)
                    self.grid_spe_parameters[endpoint][channel] = {
                        'min': np.min(mean_spe.adcs),
                        'max': np.max(mean_spe.adcs),
                        'amplitude': np.max(mean_spe.adcs) - np.min(mean_spe.adcs),
                        'dynamic_range': 2**14/(np.max(mean_spe.adcs) - np.min(mean_spe.adcs)),
                        'error_dr': False
                    }
                    self.grid_spe_idxs[endpoint][channel] = spe_idxs   
                else:
                    mean_spe = self.grid_apa.ch_wf_sets[endpoint][channel].compute_mean_waveform()
                    self.grid_spe_parameters[endpoint][channel] = {
                        'min': np.min(mean_spe.adcs),
                        'max': np.max(mean_spe.adcs),
                        'amplitude': np.max(mean_spe.adcs) - np.min(mean_spe.adcs),
                        'dynamic_range': 2**14/(np.max(mean_spe.adcs) - np.min(mean_spe.adcs)),
                        'error_dr': True
                    }
                    self.grid_spe_idxs[endpoint][channel] = list(range(len(self.grid_apa.ch_wf_sets[endpoint][channel].waveforms)))
            
        self.io_data_dict["calib_histograms"] = calib_hist_io_dict
        if self.params.verbose:
            print("Finished.")


        if self.params.verbose:
            print(
                "In function Analysis1.analyze(): "
                "Computing the gain and S/N for "
                f"batch {self.batch}, APA {self.apa}"
                f", and PDE {self.pde} ... "
            )

        # Filter the excluded_channels DataFrame to get only
        # the excluded channels for the current batch, APA and PDE
        filtered_excluded_channels = self.excluded_channels[
            (self.excluded_channels['batch'] == self.batch) &
            (self.excluded_channels['apa'] == self.apa) &
            (self.excluded_channels['pde'] == self.pde)
        ]

        self.output_data = led_utils.get_gain_and_snr(
            self.grid_apa,
            led_utils.parse_numeric_list(
                filtered_excluded_channels['excluded_channels'].values[0]
            ) if not filtered_excluded_channels.empty else [],
            reset_excluded_channels=True
        )

        return True

    def write_output(self) -> bool:
        """Implements the WafflesAnalysis.write_output() abstract
        method. It plots the calibration histograms for each channel
        (and optionally the persistence heatmaps) and saves the
        results of the analysis to a dataframe, which is written to
        a pickle file.

        Returns
        -------
        bool
            True if the method ends execution
        """
        output_path_ = f"{self.params.output_path}/vgain_run_{self.batch}/{self.pde}/apa_{self.apa}"
        Path(f"{output_path_}/data").mkdir(parents=True, exist_ok=True)
        Path(f"{output_path_}/plotcal1").mkdir(parents=True, exist_ok=True)
        Path(f"{output_path_}/plotcal2").mkdir(parents=True, exist_ok=True)
        Path(f"{output_path_}/plotmean1").mkdir(parents=True, exist_ok=True)
        Path(f"{output_path_}/plotmean2").mkdir(parents=True, exist_ok=True)

        base_file_path_data = f"{output_path_}/data/batch_{self.batch}_apa_{self.apa}_pde_{self.pde}_filter_{self.filter_type}_{self.filter_cutoff}_boxcar_{self.boxcar_window_size}"
        base_file_path_plotcal1 = f"{output_path_}/plotcal1/batch_{self.batch}_apa_{self.apa}_pde_{self.pde}_filter_{self.filter_type}_{self.filter_cutoff}_boxcar_{self.boxcar_window_size}"
        base_file_path_plotcal2 = f"{output_path_}"\
            f"/plotcal2/batch_{self.batch}_apa_{self.apa}_pde_{self.pde}_filter_{self.filter_type}_{self.filter_cutoff}_boxcar_{self.boxcar_window_size}"
        base_file_path_plotmean1 = f"{output_path_}"\
            f"/plotmean1/batch_{self.batch}_apa_{self.apa}_pde_{self.pde}_filter_{self.filter_type}_{self.filter_cutoff}_boxcar_{self.boxcar_window_size}"
        base_file_path_plotmean2 = f"{output_path_}"\
            f"/plotmean2/batch_{self.batch}_apa_{self.apa}_pde_{self.pde}_filter_{self.filter_type}_{self.filter_cutoff}_boxcar_{self.boxcar_window_size}"

        # Save the results to a pickle file
        self.io_data_dict["grid_spe_idxs"] = self.grid_spe_idxs
        self.io_data_dict["grid_spe_parameters"] = self.grid_spe_parameters
        self.io_data_dict["output_data"] = self.output_data
        # self.io_data_dict["params"] = self.params
        self.io_data_dict["batch"] = self.batch
        self.io_data_dict["apa"] = self.apa
        self.io_data_dict["pde"] = self.pde
        self.io_data_dict["filter_type"] = self.filter_type
        self.io_data_dict["filter_cutoff"] = self.filter_cutoff
        self.io_data_dict["boxcar_window_size"] = self.boxcar_window_size

        with open(f"{base_file_path_data}_analysis_output.pkl", "wb") as f:
            pickle.dump(self.io_data_dict, f)

        # Save the charge histogram plot
        figure = plot_ChannelWsGrid(
            self.grid_apa,
            figure=None,
            share_x_scale=False,
            share_y_scale=False,
            mode="calibration",
            wfs_per_axes=None,
            plot_peaks_fits=True,
            plot_sum_of_gaussians=True,
            detailed_label=False,
            verbose=self.params.verbose
        )

        title = f"Batch {self.batch}, APA {self.apa}, "
        title += f"PDE {self.pde} - Runs {list(self.wfset.runs)}"
        title_fontsize = 22
        figure_width = 1600
        figure_height = 1600
        subfigure_width = 1920
        subfigure_height = 1080

        figure.update_layout(
            title={
                "text": title,
                "font": {"size": title_fontsize}
            },
            width=figure_width,
            height=figure_height,
            showlegend=True
        )

        self._open_figures.append(figure)

        if self.params.show_figures:
            figure.show()

        fig_path = f"{base_file_path_plotcal1}_calibration_histograms.png"
        if self.params.verbose:
            print(
                "In function Analysis1.write_output(): "
                "Writing the fitted calibration histograms "
                f"for batch {self.batch}, APA {self.apa}, "
                f"and PDE {self.pde} to {fig_path} ... ",
                end=''
            )
    
        figure.write_image(f"{fig_path}", width=figure_width, height=figure_height, engine="kaleido")
        figure.write_html(f"{fig_path}.html")

        if self.params.verbose:
            print("Finished.")

         # Save each subplot individually
        rows = self.grid_apa.ch_map.rows
        cols = self.grid_apa.ch_map.columns

        for i in range(rows):
            for j in range(cols):
                channel_ws = self.grid_apa.get_channel_ws_by_ij_position_in_map(i,j)
                if channel_ws is None:
                    continue
                subplot_fig = pgo.Figure()
                # Add traces for this subplot (row/col are 1-based)
                for trace in figure.select_traces(row=i+1, col=j+1):
                    subplot_fig.add_trace(trace)
                # Optionally, set a simple title or axis labels
                subtitle = (
                    f"Endpoint: {self.grid_apa.ch_map.data[i][j].endpoint} — "
                    f"Channel: {self.grid_apa.ch_map.data[i][j].channel} — "
                    f"APA: {self.apa} — VGAIN: {self.batch} — PDE: {100*self.pde:.0f}%"
                )
                subplot_fig.update_layout(
                    title_text="Histogram of processed waveforms",
                    title_x=0.5,
                    title_font=dict(size=24),
                    margin=dict(t=120)  # room for the subtitle above the plot
                )
                subplot_fig.add_annotation(
                    text=subtitle,
                    xref="paper", yref="paper",
                    x=0.5, y=1.06,  # just above the main title line
                    showarrow=False,
                    xanchor="center",
                    font=dict(size=20)
                )
                subplot_fig.update_layout(
                    xaxis_title=r"$\mathrm{ADU}$",
                    yaxis_title=r"$\mathrm{Entries}$"
                )
                subplot_fig.update_xaxes(title_standoff=12)
                subplot_fig.update_yaxes(title_standoff=12)
                subplot_path = f"{base_file_path_plotcal2}_subplot_calibration_{i+1}_{j+1}.png"
                subplot_fig.write_image(subplot_path, width=subfigure_width, height=subfigure_height, engine="kaleido")
                subplot_fig.write_html(f"{subplot_path}.html")
                self._clear_plotly_figure(subplot_fig)
                if self.params.verbose:
                    print(f"Saved subplot ({i+1},{j+1}) to {subplot_path}")

        self._clear_plotly_figure(figure)
        
        figure = plot_ChannelWsGrid(
            self.grid_apa_hpf_filtered,
            figure=None,
            share_x_scale=False,
            share_y_scale=False,
            mode="average",
            wfs_per_axes=None,
            analysis_label=None,
            plot_analysis_markers=True,
            detailed_label=True,
            verbose=self.params.verbose
        )

        self._open_figures.append(figure)

        # Save each subplot individually
        rows = self.grid_apa_hpf_filtered.ch_map.rows
        cols = self.grid_apa_hpf_filtered.ch_map.columns

        for i in range(rows):
            for j in range(cols):
                channel_ws = self.grid_apa_hpf_filtered.get_channel_ws_by_ij_position_in_map(i,j)
                if channel_ws is None:
                    continue
                subplot_fig = pgo.Figure()
                # Add traces for this subplot (row/col are 1-based)
                for trace in figure.select_traces(row=i+1, col=j+1):
                    subplot_fig.add_trace(trace)
                # Optionally, set a simple title or axis labels
                subtitle = (
                    f"Endpoint: {self.grid_apa.ch_map.data[i][j].endpoint} — "
                    f"Channel: {self.grid_apa.ch_map.data[i][j].channel} — "
                    f"APA: {self.apa} — VGAIN: {self.batch} — PDE: {100*self.pde:.0f}%"
                )
                subplot_fig.update_layout(
                    title_text="Histogram of processed waveforms",
                    title_x=0.5,
                    title_font=dict(size=24),
                    margin=dict(t=120)  # room for the subtitle above the plot
                )
                subplot_fig.add_annotation(
                    text=subtitle,
                    xref="paper", yref="paper",
                    x=0.5, y=1.06,  # just above the main title line
                    showarrow=False,
                    xanchor="center",
                    font=dict(size=20)
                )
                subplot_fig.update_layout(
                    xaxis_title=r"$\mathrm{Sample}$",
                    yaxis_title=r"$\mathrm{ADU}$"
                )
                subplot_fig.update_xaxes(title_standoff=12)
                subplot_fig.update_yaxes(title_standoff=12)
                subplot_path = f"{base_file_path_plotmean1}_subplot_{i+1}_{j+1}.png"
                subplot_fig.write_image(subplot_path, width=subfigure_width, height=subfigure_height, engine="kaleido")
                subplot_fig.write_html(f"{subplot_path}.html")
                self._clear_plotly_figure(subplot_fig)
                if self.params.verbose:
                    print(f"Saved subplot ({i+1},{j+1}) to {subplot_path}")

        self._clear_plotly_figure(figure)
        # Single P.E. plots
        figure = plot_ChannelWsGrid(
            self.grid_apa_zero_baselined,
            figure=None,
            share_x_scale=False,
            share_y_scale=False,
            mode="average",
            wfs_per_axes=None,
            analysis_label=None,
            plot_analysis_markers=True,
            detailed_label=True,
            wfs_idcs=self.grid_spe_idxs,
            verbose=self.params.verbose
        )

        self._open_figures.append(figure)

        # Save each subplot individually
        rows = self.grid_apa_zero_baselined.ch_map.rows
        cols = self.grid_apa_zero_baselined.ch_map.columns

        for i in range(rows):
            for j in range(cols):
                channel_ws = self.grid_apa_zero_baselined.get_channel_ws_by_ij_position_in_map(i,j)
                if channel_ws is None:
                    continue
                subplot_fig = pgo.Figure()
                # Add traces for this subplot (row/col are 1-based)
                for trace in figure.select_traces(row=i+1, col=j+1):
                    subplot_fig.add_trace(trace)
                subtitle = (
                    f"Endpoint: {self.grid_apa.ch_map.data[i][j].endpoint} — "
                    f"Channel: {self.grid_apa.ch_map.data[i][j].channel} — "
                    f"APA: {self.apa} — VGAIN: {self.batch} — PDE: {100*self.pde:.0f}%"
                )
                subplot_fig.update_layout(
                    title_text="Histogram of processed waveforms",
                    title_x=0.5,
                    title_font=dict(size=24),
                    margin=dict(t=120)  # room for the subtitle above the plot
                )
                subplot_fig.add_annotation(
                    text=subtitle,
                    xref="paper", yref="paper",
                    x=0.5, y=1.06,  # just above the main title line
                    showarrow=False,
                    xanchor="center",
                    font=dict(size=20)
                )
                subplot_fig.update_layout(
                    xaxis_title=r"$\mathrm{Samples}$",
                    yaxis_title=r"$\mathrm{ADU}$"
                )
                subplot_fig.update_xaxes(title_standoff=12)
                subplot_fig.update_yaxes(title_standoff=12)
                subplot_path = f"{base_file_path_plotmean2}_subplot_SPE_{i+1}_{j+1}.png"
                subplot_fig.write_image(subplot_path, width=subfigure_width, height=subfigure_height, engine="kaleido")
                subplot_fig.write_html(f"{subplot_path}.html")
                self._clear_plotly_figure(subplot_fig)
                if self.params.verbose:
                    print(f"Saved subplot ({i+1},{j+1}) to {subplot_path}")

        # Save the persistence heatmaps
        self._clear_plotly_figure(figure)
        if self.params.save_persistence_heatmaps:

            aux_time_increment = {
                1: 80,
                2: 40,
                3: 40,
                4: 40
            }

            persistence_figure = plot_ChannelWsGrid(
                self.grid_apa,
                figure=None,
                share_x_scale=True,
                share_y_scale=True,
                mode='heatmap',
                wfs_per_axes=None,
                analysis_label=self.params.null_baseline_analysis_label,
                time_bins=aux_time_increment[self.apa],
                adc_bins=30,
                time_range_lower_limit=self.params.baseline_i_up[self.apa],
                time_range_upper_limit=self.params.baseline_i_up[self.apa] + aux_time_increment[self.apa],
                adc_range_above_baseline=10,
                adc_range_below_baseline=80,
                detailed_label=True,
                verbose=self.params.verbose
            )

            persistence_figure.update_layout(
                title = {
                    'text': title,
                    'font': {'size': title_fontsize}
                },
                width=figure_width,
                height=figure_height,
                showlegend=True
            )

            if self.params.show_figures:
                persistence_figure.show()

            fig_path = f"{base_file_path}_persistence_heatmaps.png"
            if self.params.verbose:
                print(
                    "In function Analysis1.write_output(): "
                    "Writing the persistence heatmaps "
                    f"for batch {self.batch}, APA {self.apa}, "
                    f"and PDE {self.pde} to {fig_path} ... ",
                    end=''
                )
        
            persistence_figure.write_image(f"{fig_path}")
            if self.params.verbose:
                print("Finished.")

        dataframe_output_path = os.path.join(
                output_path_,
                self.params.output_dataframe_filename
        )
        
        led_utils.save_data_to_dataframe_hpf(
            self.batch,
            self.apa,
            self.pde,
            self.filter_type,
            self.filter_cutoff,
            self.boxcar_window_size,
            self.grid_spe_parameters,
            self.output_data, 
            dataframe_output_path
        )

        if self.params.verbose:
            print(
                "In function Analysis1.write_output(): "
                "The calibration results have been saved "
                f"to {dataframe_output_path}"
            )

        # Aggressive cleanup to release memory before next iteration
        self.cleanup()
        return True

    def _clear_plotly_figure(self, fig):
        try:
            # Drop data/layout to break ref cycles
            fig.data = tuple()
            fig.layout = pgo.Layout()
        except Exception:
            pass
        try:
            del fig
        except Exception:
            pass

    def cleanup(self):
        """Drop large references and force a garbage collection."""
        # Clear any still-tracked figures
        for f in getattr(self, "_open_figures", []):
            self._clear_plotly_figure(f)
        self._open_figures = []

        # Drop heavy attributes
        self.wfset = None
        self.grid_apa = None
        self.grid_apa_zero_baselined = None
        self.grid_apa_hpf_filtered = None

        # Keep only minimal outputs
        self.io_data_dict = None

        # Reduce temp summaries as well (optional)
        # self._mean_waveform_zero_baselined_summary = None
        # self._mean_waveform_hpf_filtered_summary = None

        # Encourage Python to return arenas to the allocator
        gc.collect()
