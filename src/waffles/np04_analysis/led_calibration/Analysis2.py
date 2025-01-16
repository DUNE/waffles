from waffles.np04_analysis.led_calibration.imports import *

class Analysis2(WafflesAnalysis):

    def __init__(self):
        pass
    
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

            plots_saving_folderpath: str = Field(
                default="./",
                description="Path to the folder where "
                "the plots will be saved."
            )
            
            # Parameters for the plots

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

            self.read_input_loop_1 = self.params.batches
            self.read_input_loop_2 = self.params.apas
            self.read_input_loop_3 = self.params.pdes
    
            self.variable = 'snr'
            self.input_base_folderpath = '/home/andrearf28/waffles/waffles/src/waffles/np04_analysis/led_calibration/output'
            self.path_to_output_folderpath = '/home/andrearf28/waffles/waffles/src/waffles/np04_analysis/led_calibration/output'

            # Validación del atributo `self.variable`
            
            if self.variable not in ['gain', 'snr']:
                raise Exception('Either gain or snr must be selected')
            
            print('Llego aquí 1')
            
            
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

        self.batch = self.read_input_itr_1
        self.apa   = self.read_input_itr_2
        self.pde   = self.read_input_itr_3
        
        '''
        self.apas = [self.apa]
        self.batches = [self.batch]
        self.pdes = [self.pde]
        '''
        
        self.dataframes = {}

        for batch in self.params.batches:

            aux_file_path = os.path.join(
                os.getcwd(), 
                f"{self.input_base_folderpath}/batch_{self.batch}_apa_{self.apa}_pde_{self.pde}_df.pkl")
                
            with open(aux_file_path, "rb") as file:
                self.dataframes[batch] = pickle.load(file)
        
        print('Llego aquí 2')
        
        return True 
        

    def analyze(self) -> bool:
        
        # Prepare the data for the plot agains time
        
        self.time = [led_utils.compute_timestamp(
                metadata[batch_no]['date_day'], 
                metadata[batch_no]['date_month'], 
                metadata[batch_no]['date_year']) for batch_no in self.params.batches ]

        self.time_labels = [
                f"{metadata[batch_no]['date_year']}/"
                f"{metadata[batch_no]['date_month']}/"
                f"{metadata[batch_no]['date_day']}" for batch_no in self.params.batches ]

        for batch in self.dataframes.keys():

            aux = [batch] * len(self.dataframes[batch])
            self.dataframes[batch]['batch_no'] = aux
            self.dataframes[batch]['batch_no'] = self.dataframes[batch]['batch_no'].astype(int)
        
        # Aquí basicamente están los datos filtrados como un dataframe
        
        self.general_df = pd.concat(
            list(self.dataframes.values()), 
            ignore_index=True)
        
        print('self.general_df',self.general_df)
        
        print('Llego aquí 3')

        return True

    def write_output(self) -> bool:

        # Batch-wise plots
        
        for k in range(len(self.params.apas)):
            
            apa_no=self.params.apas[k]
            
            for i in range(len(self.params.batches)):
                batch_no = self.params.batches[i]
                
                # Get the data for the given APA and batch
                current_df =self.general_df[
                    (self.general_df['APA'] == apa_no) & 
                    (self.general_df['batch_no'] == batch_no)]
                
                if current_df.empty:
                    print(f"No data for APA {apa_no}, Batch {batch_no}")
                    continue  # Skip this iteration if there's no data for this APA and batch

                fig = pgo.Figure()

                for j in range(len(self.params.pdes)):
                    
                    print('Llego aquí 3.3')
                
                    aux = current_df[current_df['PDE'] == self.params.pdes[j]]

                    fig.add_trace(pgo.Scatter(  
                        x=aux['channel_iterator'],
                        y=aux[self.variable],
                        mode='markers',
                        marker=dict(
                            size=5, 
                            color=self.params.colors[self.params.pdes[j]],
                            symbol=self.params.symbols[self.params.pdes[j]]),
                        name=f"PDE = {self.params.pdes[j]}",
                    ))
                title = f"{self.params.translator[self.variable]} per channel in APA {apa_no} - "\
                        f"Batch {batch_no} ({metadata[batch_no]['date_year']}/"\
                        f"{metadata[batch_no]['date_month']}/{metadata[batch_no]['date_day']}"\
                        f")"

                fig.update_layout(
                    title={
                            'text': title,
                            'font': {'size': 18},
                        },
                    xaxis_title='Channel',
                    yaxis_title=self.params.y_label[self.variable],
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
                
                fig.write_image(f"{self.path_to_output_folderpath}/apa_{apa_no}_clustered_{self.variable}s.png")


                
        # Plot against time, apa-wise
        print('Llego aquí 4')
        
        
        for k in range(len(self.params.apas)):
    
            apa_no=self.params.apas[k]

            for i in range(len(self.params.pdes)):
                
                current_df = self.general_df[
                    (self.general_df['APA'] == apa_no) &
                    (self.general_df['PDE'] == self.params.pdes[i])]
                
                data[apa_no][self.params.pdes[i]] = {}

                possible_channel_iterators = current_df['channel_iterator'].unique()
                
                for channel_iterator in possible_channel_iterators:
                    
                    aux = current_df[current_df['channel_iterator'] == channel_iterator]
                    time_ordered_values_of_variable = []

                    # Here's why the data is ordered by batch number, i.e. ordered by time
                    for batch_no in self.params.batches:

                        aux2 = aux[aux['batch_no'] == batch_no]
                        if len(aux2) == 0:
                            print(f"Warning: Found no entry for APA {apa_no}, PDE {self.params.pdes[i]}, batch {batch_no} and channel iterator {channel_iterator}.")
                        elif len(aux2) == 1:
                            time_ordered_values_of_variable.append(
                                aux2[self.variable].values[0])
                        else:
                            raise Exception(f"Found more than one entry for APA {apa_no}, PDE {pdes[i]}, batch {batch_no} and channel iterator {channel_iterator}.")
                            
                    data[apa_no][self.params.pdes[i]][channel_iterator] = time_ordered_values_of_variable
        
        for apa_no in data.keys():
    
            fig= pgo.Figure()

            for pde in self.general_df[apa_no]():

                for channel_iterator in self.data[apa_no][pde].keys():

                    unique_channel = get_endpoint_and_channel(
                        apa_no, 
                        channel_iterator)

                    fig.add_trace(
                        pgo.Scatter(
                            x=self.time,
                            y=self.data[apa_no][pde][channel_iterator],
                            mode='lines+markers',
                            name=f"PDE = {pde}, channel {unique_channel}",
                            line=dict(
                                color=self.params.colors[pde[j]],
                                width=0.5),
                            marker=dict(
                                size=5,
                                color=self.params.colors[pde[j]],
                                symbol=self.params.symbols[pde[j]])
                        )
                    )
                    
            title = f"{self.params.translator[self.variable]} per channel in APA {apa_no}"

            fig.update_layout(
                title = {
                            'text': title,
                            'font': {'size': 18},
                        },
                #xaxis_title='Time',
                yaxis_title=self.params.y_label[self.variable],
                width=800,
                height=400,
            )

            fig.update_layout(
                xaxis=dict( 
                    tickmode='array',
                    tickvals=self.time,
                    ticktext=self.time_labels,
                    tickangle=15,
                    tickfont=dict(size=16)
                ),
                showlegend=self.showlegend
            )
                    
            if self.params.show_figures:
                fig.show()
            fig.write_image(f"{self.path_to_output_folderpath}/apa_{apa_no}_{self.variable}s_with_time.png")
            
        return True
                
