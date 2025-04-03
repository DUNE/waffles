from waffles.np04_analysis.light_vs_hv.imports import *

class Analysis4(WafflesAnalysis):

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

            input_path:     str =  Field(default="./output/data_avg_wf.root",          
                                description= "File with the list of files to search for the data. In each each line must be only a file name, and in that file must be a collection of .fcls from the same run")
            output:         str =  Field(default="./output",          
                                description= "Output folder to save the filtered data")
            hv:         list =  Field(... ,          
                                description= "HV vector")
          

        return InputParams
    
    def initialize(
        self,
        input_parameters: BaseInputParams
    ) -> None:
    
        self.params = input_parameters

        self.file_name=self.params.input_path
        print(f"File name: {self.file_name}")

        self.hv=self.params.hv

        self.output = self.params.output

        self.read_input_loop=[None,]

    

    ##################################################################
    def read_input(self) -> bool:
        # open root file
        self.file = root.TFile(self.file_name, "READ")
        self.tree = self.file.Get("my_tree")
        
        return True
    #############################################################

    def analyze(self) -> bool:

        # number of events in ttree
        self.n_entries = self.tree.GetEntries()

        self.waveform=[]
        # get all waveforms
        for i in range (self.n_entries):
            self.tree.GetEntry(i)
            self.waveform.append(np.array(self.tree.avg_wf_dec_filt))  

        self.s3=[None for _ in range(self.n_entries)]
        self.max=[None for _ in range(self.n_entries)]

        for i in range(self.n_entries):
            
            aux=self.waveform[i][0:600]
            max_aux=np.argmax(aux)
            mean = np.mean(self.waveform[i][1024:1400])
            self.max[i]=max_aux
            self.s3[i]=np.sum((self.waveform[i]-mean)[max_aux:max_aux+150])
            
        self.s3=np.array(self.s3)/self.s3[0]
        self.s=self.s3#(self.s1+self.s2)/2
        self.hv=np.array(self.hv)/360
         
        self.params_s, self.params_covariance_s = curve_fit(birks_law,self.hv,self.s,maxfev=20000)
        
        print(self.params_s)
        print(self.params_covariance_s)
        print(self.max)                                         
        return True
    
    def write_output(self) -> bool:
        output_file_1=self.output + "/fit/"   
        

        # Data for ARIS coll
        x1 = [0.0, 0.05, 0.1, 0.2, 0.5]
        y1 = [1.0, 0.87542, 0.877808, 0.758476, 0.587763]
        ey1 = [0.0, 0.0287963, 0.022386, 0.016722, 0.0135073]

        # Data for Kubota et al
        x2 = [0, 0.132479, 0.184815, 0.576578, 1.13743]
        y2 = [1, 0.801131, 0.760566, 0.597039, 0.488523]

        # Data for 3x1
        x3 = [0.485]
        ex3 = [0.017]
        y3 = [0.577]
        ey3 = [0.022]

        # Data for ProtoDUNE-DP Run I
        x4 = [0.09]
        exm = [0.02]
        exp = [0.10]
        y4 = [0.833]
        ey4 = [0.007]

        # Data for ProtoDUNE-DP Run II
        x5 = [0, 0.497]
        y5 = [0, 0.62]
        ex5 = [0, 0.01]
        ey5 = [0, 0.014]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # ARIS coll
        ax.errorbar(x1, y1, yerr=ey1, fmt='o', markersize=6, label="ARIS", color='black')

        # Kubota et al
        ax.plot(x2, y2, 's', markersize=8, label="Kubota et al.", color='red')

        # 3x1
        ax.errorbar(x3, y3, xerr=ex3, yerr=ey3, fmt='^', markersize=8, label="WA105 demonstrator", color='blue')

        # ProtoDUNE-DP Run I
        ax.errorbar(x4, y4, xerr=[exm, exp], yerr=[ey4, ey4], fmt='o', markersize=8, label="ProtoDUNE-DP Run I", color='purple')

        # ProtoDUNE-DP Run II
        ax.errorbar(x5, y5, xerr=ex5, yerr=ey5, fmt='o', markersize=8, label="ProtoDUNE-DP Run II", color='orange')


        ax.errorbar(self.hv, self.s , label="ProtoDUNE-HD Preliminary average",fmt='o',color="green",marker="x" )

        x_axis=np.linspace(0,1,40)

        ax.errorbar(x_axis,birks_law(x_axis,*self.params_s),color="gray",label=f" fitted data: ProtoDUNE HD ", linestyle="--")
        
        # Styling
        ax.set_xlim(-0.1, 0.6)
        ax.set_ylim(0.5, 1.1)
        ax.set_xlabel("Drift field (kV/cm)", fontsize=14)
        ax.set_ylabel("S1_drift / S1_0", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

        # Save and show
        plt.tight_layout()
        plt.savefig(output_file_1+f"fit_light.png")
        plt.close()


        return True
