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

            baseline_std_from_noise_results: bool = Field(
                default=False,
                description="If True, the baseline std is retrieved "
                "from external files whose paths are given by the "
                "daphne_configuration_database_filepath and the "
                "noise_results_dataframe_filepath input parameters. "
                "If False, the baseline std is computed from the data."
            )

            daphne_configuration_database_filepath: str | None = Field(
                default=None,
                description="Only used if the "
                "baseline_std_from_noise_results parameter is set to "
                "True. It is the path to the Daphne configuration JSON file.",
                example='../../np04_utils/DaphneConfigs.json'
            )

            noise_results_dataframe_filepath: str | None = Field(
                default=None,
                description="Only used if the "
                "baseline_std_from_noise_results parameter is set to "
                "True. It is the path to the noise results CSV file, "
                "which must contain the columns 'ConfigNumber', "
                "'OfflineCh' and 'RMS'.",
                example='../../np04_data/OfflineCh_RMS_Config_all.csv'
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

            calib_histo_bin_width: dict[float, float] = Field(
                ...,
                description="Bin width in the calibration "
                "histogram for each PDE. The keys are the "
                "PDEs, and the values are the bins width "
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

            gain_seeds_filepath: str | None = Field(
                default=None,
                description="If None, the number of bins and the "
                "domain of the charge calibration histogram are computed "
                "out of the calib_histo_bin_width, calib_histo_lower_limit "
                "and calib_histo_upper_limit input parameters. If it is "
                "defined, then it should give the path to a CSV file which "
                "contains, at least, the following columns: 'batch', 'APA', "
                "'PDE', 'endpoint', 'channel' and 'gain'. The channel-wise "
                "gain read from such file is used to compute a channel-wise "
                "domain for the charge calibration histograms.",
                example='./configs/gain_seeds.csv'
            )

            bins_per_charge_peak: int = Field(
                default=7,
                description="Used only if the gain_seeds_filepath parameter "
                "is defined. In that case, the calibration histogram (CH) "
                "domain is automatically set to [-1.*gain_seed, 5.*gain_seed], "
                "and the number of bins in the CH is computed as the domain "
                "width (i.e. 6.*gain_seed) multiplied by this parameter."
            )

            average_fallback: bool = Field(
                default=True,
                description="Used only if the gain_seeds_filepath parameter "
                "is defined. In that case, and if the given CSV lacks the "
                "gain-seed information for one or more of the required "
                "channels, then the domain (for the calibration histogram) "
                "for the lacking channels is set to the average domain of "
                "all of the other channels."
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

            sipm_vendor_filepath: str | None = Field(
                default=None,
                description="If None, the 'vendor' column of the " 
                "output CSV file will be filled with strings which "
                "match 'unavailable'. If it is defined, then it is the "
                "path to a CSV file which must contain the columns "
                "'endpoint', 'daphne_ch', and 'sipm', from which the "
                "endpoint, the channel and the vendor associated to "
                "each channel can be retrieved, respectively. In this "
                "case, the 'vendor' column of the output CSV file will "
                "be filled with the vendor information retrieved from "
                "this file.",
                example='../../np04_utils/PDHD_PDS_NewChannelMap.csv'
            ) 

            overwrite_output_dataframe: bool = Field(
                default=False,
                description="Whether to potentially overwrite existing "
                "rows in the output dataframe"
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
        self.integration_limits = None
        self.output_data = None

        self.read_input_loop_1 = self.params.batches
        self.read_input_loop_2 = self.params.apas
        self.read_input_loop_3 = self.params.pdes

        # columns: run, batch, acquired_apas, aimed_channels, pde
        self.channels_per_run = pd.read_csv(
            self.params.channels_per_run_filepath
        )

        # columns: batch, apa, pde, excluded_channels
        self.excluded_channels = pd.read_csv(
            self.params.excluded_channels_filepath
        )

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
            input_filepaths = led_utils.get_input_filepaths_for_run(
                self.params.input_path,
                self.batch,
                self.pde,
                run
            )

            new_wfset = WaveformSet_from_pickle_files(
                    filepath_list=input_filepaths,
                    target_extension='.pkl',
                    verbose=self.params.verbose
            )

            # Keep only the waveforms coming from 
            # the targeted channels for this run.
            # This step is useless for cases when
            # the input pickles were already filtered
            # when copied from the original HDF5 files
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

        # Separate the WaveformSet into a grid of WaveformSets,
        # so that each WaveformSet contains all of the waveforms
        # which come from the same channel
        self.grid_apa = ChannelWsGrid(   
            APA_map[self.apa],
            self.wfset,
            compute_calib_histo=False,
        )

        # Initialize the dictionary of integration limits to an
        # empty dictionary before looping over the channels
        self.integration_limits = {}

        for endpoint in self.grid_apa.ch_wf_sets.keys():
            for channel in self.grid_apa.ch_wf_sets[endpoint].keys():

                if self.params.baseline_std_from_noise_results:
                    if self.params.verbose:
                        print(
                            "In function Analysis1.analyze(): "
                            "Retrieving the average baseline STD "
                            f"of channel {endpoint}-{channel} "
                            f"(batch {self.batch}, APA {self.apa},"
                            f" PDE {self.pde}) ... "
                        )

                    average_baseline_std = get_average_baseline_std_from_file(
                        self.wfset.waveforms[0].run_number,
                        endpoint=endpoint,
                        channel=channel,
                        daphne_configuration_database_filepath=\
                            self.params.daphne_configuration_database_filepath
                            if self.params.daphne_configuration_database_filepath is not None
                            else "",
                        noise_results_dataframe_filepath=\
                            self.params.noise_results_dataframe_filepath
                            if self.params.noise_results_dataframe_filepath is not None
                            else ""
                    )
                else:
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

                mean_wf = self.grid_apa.ch_wf_sets[endpoint][channel].\
                    compute_mean_waveform()

                aux_limits = get_pulse_window_limits(
                    mean_wf.adcs,
                    0,
                    self.params.deviation_from_baseline,
                    self.params.lower_limit_correction,
                    self.params.upper_limit_correction
                )

                if self.params.verbose:
                    print(f"Found limits {aux_limits[0]}-{aux_limits[1]}.")
                    print(
                        "In function Analysis1.analyze(): "
                        "Integrating the waveforms for channel "
                        f"{endpoint}-{channel} (batch "
                        f"{self.batch}, APA {self.apa}, PDE "
                        f"{self.pde}) ... ",
                        end=''
                    )
                
                integrator_input_parameters = IPDict({
                    'baseline_analysis': self.params.null_baseline_analysis_label,
                    'inversion': True,
                    'int_ll': aux_limits[0],
                    'int_ul': aux_limits[1],
                    'amp_ll': aux_limits[0],
                    'amp_ul': aux_limits[1]
                })

                checks_kwargs = IPDict({
                    'points_no': self.grid_apa.ch_wf_sets[endpoint][channel].\
                        points_per_wf
                })

                _ = self.grid_apa.ch_wf_sets[endpoint][channel].analyse(
                    self.params.integration_analysis_label,
                    WindowIntegrator,
                    integrator_input_parameters,
                    checks_kwargs=checks_kwargs,
                    overwrite=True
                )

                if endpoint not in self.integration_limits.keys():
                    self.integration_limits[endpoint] = {}

                self.integration_limits[endpoint][channel] = aux_limits

                if self.params.verbose:
                    print("Finished.")

        if self.params.gain_seeds_filepath is None:
            bins_number = round(
                (
                    self.params.calib_histo_upper_limit - \
                    self.params.calib_histo_lower_limit
                ) / self.params.calib_histo_bin_width[self.pde]
            )

            domain = np.array(
                (
                    self.params.calib_histo_lower_limit,
                    self.params.calib_histo_upper_limit
                )
            )
        else:
            if self.params.verbose:
                print(
                    "In function Analysis1.analyze(): "
                    "Computing channel-wise number of bins "
                    "and domain (for the calibration histograms)"
                    " out of the gain seeds in file "
                    f"{self.params.gain_seeds_filepath}"
                )

            bins_number, domain = led_utils.get_nbins_and_channel_wise_domain(
                self.batch,
                self.apa,
                self.pde,
                self.params.gain_seeds_filepath,
                self.params.bins_per_charge_peak,
                -1.,
                5.,
                verbose=self.params.verbose
            )

        if self.params.verbose:
            print(
                "In function Analysis1.analyze(): "
                "Building the calibration histogram "
                "and fitting the peaks for batch "
                f"{self.batch}, APA {self.apa}, and "
                f"PDE {self.pde} ... ",
                end=''
            )

        self.grid_apa.compute_calib_histos(
            bins_number=bins_number,
            domain=domain,
            variable='integral',
            analysis_label=self.params.integration_analysis_label,
            average_fallback=self.params.average_fallback,
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

        base_file_path = f"{self.params.output_path}"\
            f"/batch_{self.batch}_apa_{self.apa}_pde_{self.pde}"

        # Save the charge histogram plot
        figure = plot_ChannelWsGrid(
            self.grid_apa,
            figure=None,
            share_x_scale=False,
            share_y_scale=False,
            mode="calibration",
            wfs_per_axes=None,
            plot_peaks_fits=True,
            plot_sum_of_gaussians=True if \
                self.params.fit_type == 'correlated_gaussians' \
                    else False,
            detailed_label=False,
            verbose=self.params.verbose
        )

        title = f"Batch {self.batch}, APA {self.apa}, "
        title += f"PDE {self.pde} - Runs {list(self.wfset.runs)}"
        title_fontsize = 22
        figure_width = 1100
        figure_height = 1200

        figure.update_layout(
            title={
                "text": title,
                "font": {"size": title_fontsize}
            },
            width=figure_width,
            height=figure_height,
            showlegend=True
        )

        if self.params.show_figures:
            figure.show()

        fig_path = f"{base_file_path}_calibration_histograms.png"
        if self.params.verbose:
            print(
                "In function Analysis1.write_output(): "
                "Writing the fitted calibration histograms "
                f"for batch {self.batch}, APA {self.apa}, "
                f"and PDE {self.pde} to {fig_path} ... ",
                end=''
            )
    
        figure.write_image(f"{fig_path}")
        if self.params.verbose:
            print("Finished.")

        # Save the persistence heatmaps
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
                self.params.output_path,
                self.params.output_dataframe_filename
        )
        
        led_utils.save_data_to_dataframe(
            self.batch,
            self.apa,
            self.pde,
            self.output_data,
            self.integration_limits,
            dataframe_output_path,
            sipm_vendor_filepath=\
                self.params.sipm_vendor_filepath,
            overwrite=self.params.overwrite_output_dataframe
        )

        if self.params.verbose:
            print(
                "In function Analysis1.write_output(): "
                "The calibration results have been saved "
                f"to {dataframe_output_path}"
            )

        return True