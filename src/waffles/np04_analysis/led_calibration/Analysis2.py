from waffles.np04_analysis.led_calibration.imports import *

class Analysis2(WafflesAnalysis):

    def __init__(self):
        pass
    print('\n')
    print('Analysis 2 starting...')
    
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

            batches: list = Field(
                ...,
                description="Calibration batch number",
                example=[2]
            )

            show_figures: bool = Field(
                default=False,
                description="Whether to show the produced "
                "figures",
            )
            
            input_pkl: str = Field(
                default="./",
                description="Path to the folder where "
                "the pkl files will be saved."
            )
            
            output_pkl: str = Field(
                default="./",
                description="Path to the folder where "
                "the files from visualize.py (Analysis 2) will be saved."
            )
            
            # ---------- Parameters for the plots ----------
            
            variable: str = Field(
                description="Variable to represent in the plots: snr or gain."
            )
            colors: dict = Field(
                description="Colors for plots in Analysis 2."
            )
            symbols: dict = Field(
                description="Symbols for plots in Analysis 2."
            )
            translator: dict = Field(
                description="Translator for plots in Analysis 2."
            )
            y_label: dict = Field(
                description="ylabel for plots in Analysis 2."
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
            self.output_data = None
            
            self.batches = self.params.batches
            self.apas = self.params.apas
            self.pdes = self.params.pdes
            
            self.read_input_loop_1 = [0]
            self.read_input_loop_2 = [0]
            self.read_input_loop_3 = [0]
            
            '''
            
            self.read_input_loop_1 = self.params.batches
            self.read_input_loop_2 = self.params.apas
            self.read_input_loop_3 = self.params.pdes
            
            self.input_pkl = '/home/andrearf28/waffles/waffles/'
            self.path_to_output_folderpath = '/home/andrearf28/waffles/waffles/src/waffles/np04_analysis/led_calibration/output'
            '''
            
            
            print(f"The selected variable for these plots is {self.params.variable}")
            
            if self.params.variable not in ['gain', 'snr']:
                raise Exception('Either gain or snr must be selected')
            
            
    def read_input(self) -> bool:
    
        """Implements the WafflesAnalysis.read_input() abstract
        method. It loads a WaveformSet object into the self.wfset
        attribute which matches the input parameters, namely the
        APA number, the PDE and the batch number. The final
        WaveformSet is the result of merging different WaveformSet
        objects, each of which comes from a different run.
        The decision on which run contributes to which channel
        is done based on the configuration files, namely on the
        config_to_channels and run_to_config variables, which are
        imported from files in the configs/calibration_batches
        directory.
            
        Returns
        -------
        bool
            True if the method ends execution normally
        """

        # self.batch = self.read_input_itr_1
        # self.apa   = self.read_input_itr_2
        # self.pde   = self.read_input_itr_3
        
        self.read_input_itr_1=[0]
        self.read_input_itr_2=[0]
        self.read_input_itr_3=[0]
        
        '''
        # Lo que estaba antes:
        
        for batch in self.params.batches:

            aux_file_path = os.path.join(
                os.getcwd(), 
                f"{self.params.input_pkl}/batch_{batch}_apa_{self.apa}_pde_{self.pde}_df.pkl")
                
            with open(aux_file_path, "rb") as file:
                self.dataframes[batch] = pickle.load(file) 
        '''
                
        self.dataframes = {} 
     
        for batch in self.batches:
            
            combined_df = pd.DataFrame()

            for apa in self.apas:
                for pde in self.pdes:
                    # Construct the file path
                    aux_file_path = os.path.join(
                        os.getcwd(),
                        f"{self.params.input_pkl}/batch_{batch}_apa_{apa}_pde_{pde}_df.pkl"
                    )

                    # Try to load the file
                    try:
                        with open(aux_file_path, "rb") as file:
                            df = pickle.load(file)  # Load the dataframe
                            combined_df = pd.concat([combined_df, df], ignore_index=True)  # Concatenate to the combined dataframe
                            print(f"Loaded file: {aux_file_path}")
                    except FileNotFoundError:
                        print(f"File not found: {aux_file_path}")
                    except Exception as e:
                        print(f"Error loading file {aux_file_path}: {e}")

            # Store the combined dataframe for this batch in the dictionary
            self.dataframes[batch] = combined_df
                    
        #print('self_dataframes from read', self.dataframes) 
        
        return True 
        

    def analyze(self) -> bool:

        
        for batch in self.dataframes.keys():
            
            aux = [batch] * len(self.dataframes[batch])
            self.dataframes[batch]['batch_no'] = aux
            self.dataframes[batch]['batch_no'] = self.dataframes[batch]['batch_no'].astype(int)
        
                
        self.general_df = pd.concat(
            list(self.dataframes.values()), 
            ignore_index=True)

        # ------------- Prepare data for time plots -------------
        
        self.time = [led_utils.compute_timestamp(
                metadata[self.batch]['date_day'], 
                metadata[self.batch]['date_month'], 
                metadata[self.batch]['date_year']) for self.batch in self.params.batches ]

        self.time_labels = [
                f"{metadata[self.batch]['date_year']}/"
                f"{metadata[self.batch]['date_month']}/"
                f"{metadata[self.batch]['date_day']}" for self.batch in self.params.batches ]
        
        self.data_time=led_utils.prepare_data_time(
            self.params.apas,
            self.params.pdes,
            self.params.batches,
            self.params.variable,
            self.general_df
        )
        return True

    def write_output(self) -> bool:

        # ----------- Batch-wise plots -----------
        

        print(f"    1. Batch-wise plots, {self.params.variable} vs channels, are being created.")
        
        for k in range(len(self.params.apas)):
            
            apa_no=self.params.apas[k]
            
            for i in range(len(self.params.batches)):
                
                batch_no = self.params.batches[i]
                
                # Get the data for the given APA and batch
                current_df =self.general_df[
                    (self.general_df['APA'] == apa_no) & 
                    (self.general_df['batch_no'] == batch_no)]
        
                fig = pgo.Figure()

                for j in range(len(self.params.pdes)):
                
                    aux = current_df[current_df['PDE'] == self.params.pdes[j]]

                    fig.add_trace(pgo.Scatter(  
                        x=aux['channel_iterator'],
                        y=aux[self.params.variable],
                        mode='markers',
                        marker=dict(
                            size=5, 
                            color=self.params.colors[self.params.pdes[j]],
                            symbol=self.params.symbols[self.params.pdes[j]]),
                        name=f"PDE = {self.params.pdes[j]}",
                    ))
                title = f"{self.params.translator[self.params.variable]} per channel in APA {apa_no} - "\
                        f"Batch {batch_no} ({metadata[batch_no]['date_year']}/"\
                        f"{metadata[batch_no]['date_month']}/{metadata[batch_no]['date_day']}"\
                        f")"

                fig.update_layout(
                    title={
                            'text': title,
                            'font': {'size': 18},
                        },
                    xaxis_title='Channel',
                    yaxis_title=self.params.y_label[self.params.variable],
                    width=1000,
                    height=400,
                    showlegend=True,
                )

                labels = {}
                for j in range(current_df.shape[0]):
                    labels[current_df.iloc[j]['channel_iterator']] = f"{int(current_df.iloc[j]['endpoint'])}-{int(current_df.iloc[j]['channel'])}"

                fig.update_layout(
                    xaxis = dict(   
                        tickmode='array',
                        tickvals=list(labels.keys()),
                        ticktext=list(labels.values()),
                        tickangle=45,
                    )
                )
                
                if self.params.show_figures:
                    fig.show()
                
                fig.write_image(f"{self.params.output_pkl}/batchwise_batch_{batch_no}_apa_{apa_no}_{self.params.variable}.png")
        
        print(f"  Batch-wise plot saved in {self.params.output_pkl}")
                
        # ------------ APA-wise plot  -----------
        
        print(f"    2. APA-wise plots, {self.params.variable} vs time, are being created.")
        
        for self.apa in self.data_time.keys():

            fig = pgo.Figure()

            for self.pde in self.data_time[self.apa]:

                for channel_iterator in self.data_time[self.apa][self.pde].keys():
                    unique_channel = get_endpoint_and_channel(
                        self.apa,
                        channel_iterator
                    )

                    fig.add_trace(
                        pgo.Scatter(
                            x=self.time,
                            y=self.data_time[self.apa][self.pde][channel_iterator],
                            mode='lines+markers',
                            name=f"PDE = {self.pde}, channel {unique_channel}",
                            line=dict(
                                color=self.params.colors[self.pde],
                                width=0.5),
                            marker=dict(
                                size=5,
                                color=self.params.colors[self.pde],
                                symbol=self.params.symbols[self.pde])
                        )
                    )

            # Configurar y guardar la figura
            title = f"{self.params.translator[self.params.variable]} per channel in APA {self.apa}"
            fig.update_layout(
                title={'text': title, 'font': {'size': 18}},
                yaxis_title=self.params.y_label[self.params.variable],
                width=800,
                height=400,
                xaxis=dict(
                    tickmode='array',
                    tickvals=self.time,
                    ticktext=self.time_labels,
                    tickangle=15,
                    tickfont=dict(size=16)
                ),
                showlegend=True
            )

            if self.params.show_figures:
                fig.show()

            fig.write_image(f"{self.params.output_pkl}/apawise_apa_{self.apa}_{self.params.variable}_with_time.png")
        
        print(f"  APA-wise plot saved in {self.params.output_pkl}")
        
        return True
                
