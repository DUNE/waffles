# import all necessary files and classes
from waffles.np04_analysis.led_calibration.imports import *

class visualizer(WafflesAnalysis):

    def __init__(self):
        pass        

    ##################################################################
    def arguments(self, parse: argparse.ArgumentParser):                

        parse.add_argument('-a','--apa',    type=int,   required=True,  help="APA number")
        parse.add_argument('-p','--pde',    type=float, required=True,  help="photon detection efficiency")
        parse.add_argument('-b','--batch',  type=int,   required=True,  help="calibration batch")


    ##################################################################
    def initialize(self, args):                

        # store the user arguments into data members
        self.apa_no = args['apa']
        self.pde    = args['pde']
        self.batch  = args['batch']
        self.data_folderpath = self.path_to_input_file

        self.apa_nos = [self.apa_no ]
        self.batches = [self.batch]
        self.variable = 'snr'
        showlegend = False

        if self.variable not in ['gain', 'snr']:
            raise Exception('Either gain or snr must be selected')


    ##################################################################
    def read_input(self):

        self.dataframes = {}

        for batch in self.batches:

            aux_file_path = os.path.join(
                os.getcwd(), self.path_to_input_file)
                #f"{self.input_base_folderpath}/batch_{batch}/output_pickles/led_calibration_data.pkl")

            with open(aux_file_path, "rb") as file:
                self.dataframes[batch] = pickle.load(file)
        
        return True

    ##################################################################
    def analyze(self):

        
        for batch in self.dataframes.keys():

            aux = [batch] * len(self.dataframes[batch])
            self.dataframes[batch]['batch_no'] = aux
            self.dataframes[batch]['batch_no'] = self.dataframes[batch]['batch_no'].astype(int)

        self.general_df = pd.concat(
            list(self.dataframes.values()), 
            ignore_index=True)

        return True

    ##################################################################
    def write_output(self):


        pdes = [0.40, 0.45, 0.50]
        colors = {
            0.4: 'black', 
            0.45: 'green', 
            0.5: 'red'}
        symbols = {
            0.4: 'circle', 
            0.45: 'square',
            0.5: 'diamond'}

        translator = {'gain': 'Gain', 'snr': 'SNR'}
        y_label = {'gain': 'center[1] - center[0]',
                'snr': '(center[1]-center[0])/sqrt( std[0]**2  + std[1]**2 )'}
        

        for apa_no in self.apa_nos:

            for i in range(len(self.batches)):
                
                batch_no = self.batches[i]

                # Get the data for the given APA and batch
                current_df = self.general_df[
                    (self.general_df['APA'] == apa_no) & 
                    (self.general_df['batch_no'] == batch_no)]

                fig = pgo.Figure()

                for j in range(len(pdes)):

                    aux = current_df[current_df['PDE'] == pdes[j]]

                    fig.add_trace(pgo.Scatter(  
                        x=aux['channel_iterator'],
                        y=aux[self.variable],
                        mode='markers',
                        marker=dict(
                            size=5, 
                            color=colors[pdes[j]],
                            symbol=symbols[pdes[j]]),
                        name=f"PDE = {pdes[j]}",
                    ))

                title = f"{translator[self.variable]} per channel in APA {apa_no} - "\
                        f"Batch {batch_no} ({metadata[batch_no]['date_year']}/"\
                        f"{metadata[batch_no]['date_month']}/{metadata[batch_no]['date_day']}"\
                        f")"

                fig.update_layout(
                    title={
                            'text': title,
                            'font': {'size': 18},
                        },
                    xaxis_title='Channel',
                    yaxis_title=y_label[self.variable],
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

                fig.show()
                fig.write_image(f"{self.path_to_output_folderpath}/batch_{batch_no}/{led_utils.get_apa_foldername(batch_no, apa_no)}/general_plots/apa_{apa_no}_clustered_{self.variable}s.png")
