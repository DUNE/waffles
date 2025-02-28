# import all necessary files and classes
from waffles.np04_analysis.beam_example.imports import *

ROOT_IMPORTED = False
try: 
    from ROOT import TFile
    from ROOT import TTree
    ROOT_IMPORTED = True
except ImportError: 
    print(
        "[raw_ROOT_reader.py]: Could not import ROOT module. "
        "'pyroot' library options will not be available."
    )
    ROOT_IMPORTED = False
    pass


class Analysis2(WafflesAnalysis):

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

            events_output_path:         str = Field(...,          description="work in progress")
            events_summary_output_path: str = Field(...,          description="work in progress")            
            
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

        print(f"Reading events from pickle file: ", self.params.events_output_path)

        self.events = events_from_pickle_file(self.params.events_output_path)

        print(f"  {len(self.events)} events read")
        
        return True

    ##################################################################
    def analyze(self) -> bool:

        print(f"Analize the waveforms (compute baseline, amplitud and integral)")

        t0 = self.events[0].ref_timestamp
        
        # loop over events
        for e in self.events:

            # get the number of waveforms
            nwfs = len(e.wfset.waveforms) if e.wfset else 0            

            if not nwfs: continue

            # ------------- Analyse the waveform set -------------
            b_ll = 0
            b_ul = 100
            int_0 = 135
            int_1 = 145
            int_2 = 165
            
            # baseline limits
            bl = [b_ll, b_ul, 900, 1000]
            
            peak_finding_kwargs = dict( prominence = 20,rel_height=0.5,width=[0,75])
            ip_portion = IPDict(baseline_limits=bl,
                        int_ll=int_0,int_ul=int_1,amp_ll=int_0,amp_ul=int_1,
                        points_no=10,
                        peak_finding_kwargs=peak_finding_kwargs)
            ip_total = IPDict(baseline_limits=bl,
                        int_ll=int_0,int_ul=int_2,amp_ll=int_0,amp_ul=int_2,
                        points_no=10,
                        peak_finding_kwargs=peak_finding_kwargs)
            analysis_kwargs = dict(  return_peaks_properties = False)
            checks_kwargs   = dict( points_no = e.wfset.points_per_wf)
            #if wset.waveforms[0].has_analysis('standard') == False:
            
            # analyse the waveforms (copute baseline, amplitude and integral)
            
            a_portion=e.wfset.analyse('Portion_integral',BasicWfAna,ip_portion,checks_kwargs = checks_kwargs,overwrite=True)
            a_total=e.wfset.analyse('Total_integral',BasicWfAna,ip_total,checks_kwargs = checks_kwargs,overwrite=True)
            
            '''
            # dump event information when ROOT is not available
            if not ROOT_IMPORTED:

                print(f"Dump information about events:")
                
                # print information about the event
                print (e.record_number,
                       e.event_number,
                       e.first_timestamp-t0,
                       (e.last_timestamp-e.first_timestamp)*0.016,
                       ', p =', e.beam_info.p,
                       ', nwfs =', nwfs,
                       ', c0 =', e.beam_info.c0,
                       ', c1 =', e.beam_info.c1,
                       ', tof =', e.beam_info.tof)
            '''
        
        return True

    ##################################################################
    def write_output(self) -> bool:

        if not ROOT_IMPORTED:
            
            print(f'Saving events summary in root file with uproot: {self.params.events_summary_output_path}')
        
            # Inicializar listas vacías para almacenar datos
            data = {
                "evt": [],
                "p": [],
                "tof": [],
                "c0": [],
                "c1": [],
                "t": [],
                "nwfs": [],
                "a": [],
                "qport": [],
                "qtotal":[]
            }

            # Loop sobre eventos
            for e in self.events:
                qtotal = 0
                qport=0
                a = 0            
                if e.wfset:
                    for wf in e.wfset.waveforms:
                        qport += wf.get_analysis('Portion_integral').result['integral']
                        qtotal += wf.get_analysis('Total_integral').result['integral']
                        a += wf.get_analysis('Total_integral').result['amplitude']

                nwfs = len(e.wfset.waveforms) if e.wfset else 0
                if nwfs > 0:
                    qport /= nwfs
                    qtotal /= nwfs
                    a /= nwfs            

                # Agregar valores a las listas
                data["evt"].append(e.event_number)
                data["p"].append(e.beam_info.p)
                data["tof"].append(e.beam_info.tof)
                data["c0"].append(e.beam_info.c0)
                data["c1"].append(e.beam_info.c1)
                data["t"].append(e.beam_info.t)
                data["nwfs"].append(nwfs)
                data["qport"].append(qport)
                data["qtotal"].append(qtotal)
                data["a"].append(a)

            # Convertir listas a arrays de numpy
            for key in data:
                data[key] = np.array(data[key])

            # Escribir a un archivo ROOT usando uproot
            with uproot.recreate(self.params.events_summary_output_path) as file:
                file["tree"] = data  # Crear el árbol directamente desde el diccionario

            print(f"  {len(self.events)} events saved")
        
        else:

            print(f'Saving events summary in root file: {self.params.events_summary_output_path}')
            
            file = TFile(f'{self.params.events_summary_output_path}', 'recreate')
            tree = TTree("tree", "tree title")

            evt  = np.array([0], dtype=np.int32)        
            p    = np.array([0], dtype=np.float64)
            tof  = np.array([0], dtype=np.float64)
            c0   = np.array([0], dtype=np.int32)
            c1   = np.array([0], dtype=np.int32)
            t    = np.array([0], dtype=np.int64)
            nwfs = np.array([0], dtype=np.int32)
            qport    = np.array([0], dtype=np.float64)
            qtotal    = np.array([0], dtype=np.float64)
            a    = np.array([0], dtype=np.float64)                

            tree.Branch("evt", evt, 'normal/I')
            tree.Branch("p",   p,   'normal/D')
            tree.Branch("tof", tof, 'normal/D')
            tree.Branch("c0",  c0,  'normal/I')
            tree.Branch("c1",  c1,  'normal/I')
            tree.Branch("t",   t,   'normal/I')
            tree.Branch("nwfs",nwfs,'normal/I')
            tree.Branch("qport",   qport,   'normal/D')
            tree.Branch("qtotal",   qtotal,   'normal/D')
            tree.Branch("a",   a,   'normal/D')        
            
            # loop over events
            for e in self.events:

                qport[0]=0
                qtotal[0]=0
                a[0]=0            
                if e.wfset:
                    for wf in e.wfset.waveforms:
                        qport[0] += wf.get_analysis('Portion_integral').result['integral']
                        qtotal[0] += wf.get_analysis('Total_integral').result['integral']
                        a[0] += wf.get_analysis('Total_integral').result['amplitude']

                if nwfs>0:
                    qport[0] = qport[0]/(1.*nwfs)
                    qtotal[0] = qtotal[0]/(1.*nwfs)
                    a[0] = a[0]/(1.*nwfs)                

                evt[0] = e.event_number
                p[0]   = e.beam_info.p
                tof[0] = e.beam_info.tof
                c0[0]  = e.beam_info.c0
                c1[0]  = e.beam_info.c1
                t[0]   = e.beam_info.t
                nwfs[0]= len(e.wfset.waveforms) if e.wfset else 0                        
                
                
                tree.Fill()

            file.Write()
            file.Close()

            print(f"  {len(self.events)} events saved")
        
        return True
