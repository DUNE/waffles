
from waffles.np04_analysis.vgain_analysis.imports import *


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

            top_level_vgain_configuration_file: str = Field(
                default = "configs/vgain_top_level.csv",
                description=""
            )

            vgain_channels_file: str = Field(
                default = "configs/vgain_channels.csv",
                description=""
            )

            rucio_paths_directory: str = Field(
                default = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths",
                description=""
            )
           
            runs: list = Field(
                default = [],
                description="Runs to analyze"
            )

            channels: list[list] = Field(
                    default = [[]],
                    description="Best channels in run"
                )

            vgain_filter: list = Field(
                default = [1596],
                description="Runs to analyze"
            )

            vgain_config_table: pd.core.frame.DataFrame = Field(
                    default = pd.read_csv("configs/vgain_top_level.csv"),
                    description="Best channels in run"
                )

            vgain_channels_table: pd.core.frame.DataFrame = Field(
                    default = pd.read_csv("configs/vgain_channels.csv"),
                    description="Best channels in run"
                )

            channels_dict: IPDict[int, str] = Field(
                    default = {},
                    description="Best channels in run"
                )
            
            model_config = ConfigDict(arbitrary_types_allowed=True)
            
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

        #vgain_config_table = pd.read_csv(self.params.top_level_vgain_configuration_file)
        #vgain_channels_table = pd.read_csv(self.params.vgain_channels_file)
        #print(type(vgain_config_table))
        vgain_filtered = self.params.vgain_config_table[self.params.vgain_config_table['vgain'].isin(self.params.vgain_filter)]
        channels_filtered = self.params.vgain_channels_table[self.params.vgain_channels_table['run'].isin(vgain_filtered['run'])]
        key = channels_filtered.keys()
        for run in channels_filtered[key[0]]:
            self.params.runs.append(run)
        for channels_ in channels_filtered[key[1]]:
            self.params.channels.append(channels_)
        for i in range(0,len(self.params.runs)):
            self.params.channels_dict[self.params.runs[i]] = self.params.channels[i+1]
        
        self.read_input_loop_1 = self.params.runs
        self.read_input_loop_2 = [None,]
        self.read_input_loop_3 = [None,]
        self.analyze_loop = [None,]

        self.wfset = None
        self.output_data = []

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
            #wvfm_count=self.params.waveforms_per_run,
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
        )

        #vgain_config_table = pd.read_csv(self.params.top_level_vgain_configuration_file)
        #vgain_channels_table = pd.read_csv(self.params.vgain_channels_file)
        #print('Here: ', self.analyze_itr, self.params.channels[self.analyze_itr])
        # skip = 0
        # print(self.read_input_itr_1)
        # print(self.params.channels_dict[self.read_input_itr_1])
        # if(self.params.channels_dict[self.read_input_itr_1] == '[]'):
        #     skip = 1
        # if(skip == 0):
        #     baseline_ = SBaseline(binsbase = None, threshold = 12, baselinestart = 0, baselinefinish = 1024, wait = 150, minimumfrac=0)
        #     filtering = 4
        #     endpoint_list = getEndpointList(self.params.channels_dict[self.read_input_itr_1])
        #     print(endpoint_list)
        #     run_map = createChannelMaps(endpoint_list)
        #     run = Run(self.read_input_itr_1, run_map, self.wfset, [])
        #     sync_run = {}
        #     for endpoint in endpoint_list:
        #         sync_run.update(run.synchronize_endpoint_waveform_set(endpoint))
        #     uniqueChannels = getUniqueChannelList(self.params.channels_dict[self.read_input_itr_1])
        #     print(uniqueChannels)

        #     for u_ch in uniqueChannels:
        #         ep = u_ch.endpoint
        #         ch_signal = u_ch.channel
        #         print(f'procesing run: {self.read_input_itr_1} - endpoint: {ep} - channel: {ch_signal}')
        #         signalWaveformList = sync_run[ep][ch_signal]
        #         signalWaveformSet = WaveformSet(*signalWaveformList)
        #         signalWaveformSet.analize(
                    
        #         )
                
                # filteredSignalWaveforms_tuple = percentileFilter(signalWaveformSet.waveforms,[0.25,0.75],'RMS')
                # filteredSignalWaveforms = filteredSignalWaveforms_tuple[0]
                # filteredSignalWaveforms_tuple = percentileFilter(filteredSignalWaveforms,[0.15,0.55],'SBaseline', binsbase = None, threshold = 12, baselinestart = 0, baselinefinish = 1024, wait = 150, minimumfrac=0)
                # filteredSignalWaveforms = filteredSignalWaveforms_tuple[0]
                # len_data = len(filteredSignalWaveforms)
                
                # avg_signal = np.zeros(1024)
                # avg_pedestal = np.zeros(len(filteredSignalWaveforms))
                # avg_trigger = np.zeros(len(avg_signal))
                # avg_filtered_signal = np.zeros(filteredSignalWaveforms[0].record_number)
                # pedestal_removed_signals = np.zeros((len(filteredSignalWaveforms),1024))
                # boxcar_filtered = np.zeros((len(filteredSignalWaveforms),1024))
                # pseudo_optimun = np.zeros((len(filteredSignalWaveforms),1024))
                
                # for i, waveform in enumerate(filteredSignalWaveforms):
                #     avg_pedestal[i] = baseline_.wfset_baseline(waveform, filtering = filtering)[0]
                #     wave_ = waveform.adcs - avg_pedestal[i]
                #     # wave_ = denoiser.apply_denoise(wave_, filtering)
                #     avg_signal = avg_signal + wave_
                #     pedestal_removed_signals[i,0:1024] = -wave_
                    
                # avg_signal = -avg_signal/len(filteredSignalWaveforms)
                
                # for i, waveform in enumerate(pedestal_removed_signals):
                #     boxcar_filtered[i,0:1024] = applyDiscreteFilter(waveform,filterType = 'Boxcar')
                #     pseudo_optimun[i,0:1024] = applyDiscreteFilter(waveform,filterType = 'FIR', numerator = np.flip(avg_signal[100:200]))
                # hist_boxcar = np.zeros(len_data)
                # hist_opt = np.zeros(len_data)
                # win = 2
                
                # max_boxcar_filtered_pos = np.argmax(np.mean(boxcar_filtered,axis=0))
                # max_pseudo_optimun_pos = np.argmax(np.mean(pseudo_optimun,axis=0))
                
                # for i, waveform in enumerate(boxcar_filtered):
                #     hist_boxcar[i] = np.mean(waveform[(max_boxcar_filtered_pos-win):(max_boxcar_filtered_pos+win)])
                # for i, waveform in enumerate(pseudo_optimun):
                #     hist_opt[i] = np.mean(waveform[max_pseudo_optimun_pos-win:max_pseudo_optimun_pos+win])

                
                # #self.output_data.append(hist_boxcar)
                # #self.output_data.append(hist_opt)
                # #self.output_data.append(boxcar_filtered)
                # #self.output_data.append(pseudo_optimun)

                # x = list(range(1024))
                # fig_signal_avg = pgo.Figure()
                # fig_signal_avg.add_trace(pgo.Scatter(x=x, y=avg_signal, mode='lines', name='Line Trace'))
                # fig_signal_avg.add_trace(pgo.Scatter(x=x, y=np.mean(boxcar_filtered,axis=0), mode='lines', name='Line Trace'))
                # fig_signal_avg.add_trace(pgo.Scatter(x=x, y=(np.max(avg_signal)/np.max(np.mean(pseudo_optimun,axis=0)))*np.mean(pseudo_optimun,axis=0), mode='lines', name='Line Trace'))
                # #self.output_data.append(fig_signal_avg)
                # vgain_value = self.params.vgain_config_table.loc[self.params.vgain_config_table['run'] == self.read_input_itr_1, 'vgain']
                # vgain_value = vgain_value.to_list()[0]
                # folder_path = 'output/vgain_' + str(vgain_value) + "/" + str(ep) + str(ch_signal)
                # os.makedirs(folder_path, exist_ok=True)
                # file_path = os.path.join(folder_path, "fig_signal_avg.png")
                # # Save the figure as a PNG
                # fig_signal_avg.write_image(file_path)
                # # Histogram 
                # file_path = os.path.join(folder_path, "histogram.png")
                # plt.figure()
                # plt.hist(hist_opt,bins=250, range=(-200, 1000));
                # plt.savefig(file_path, dpi=300, bbox_inches='tight')
                # plt.close()
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
        print(
            "In function Analysis1.write_output(): "
        )

        # Let's create a folder and save the data there:

        return True