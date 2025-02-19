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
            int_ll = 135
            int_ul = 165
            
            # baseline limits
            bl = [b_ll, b_ul, 900, 1000]
            
            peak_finding_kwargs = dict( prominence = 20,rel_height=0.5,width=[0,75])
            ip = IPDict(baseline_limits=bl,
                        int_ll=int_ll,int_ul=int_ul,amp_ll=int_ll,amp_ul=int_ul,
                        points_no=10,
                        peak_finding_kwargs=peak_finding_kwargs)
            analysis_kwargs = dict(  return_peaks_properties = False)
            checks_kwargs   = dict( points_no = e.wfset.points_per_wf)
            #if wset.waveforms[0].has_analysis('standard') == False:
            
            # analyse the waveforms (copute baseline, amplitude and integral)
            a=e.wfset.analyse('standard',BasicWfAna,ip,checks_kwargs = checks_kwargs,overwrite=True)

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
        
        return True

    ##################################################################
    def write_output(self) -> bool:

        if not ROOT_IMPORTED:
            return False


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
        q    = np.array([0], dtype=np.float64)
        a    = np.array([0], dtype=np.float64)                

        tree.Branch("evt", evt, 'normal/I')
        tree.Branch("p",   p,   'normal/D')
        tree.Branch("tof", tof, 'normal/D')
        tree.Branch("c0",  c0,  'normal/I')
        tree.Branch("c1",  c1,  'normal/I')
        tree.Branch("t",   t,   'normal/I')
        tree.Branch("nwfs",nwfs,'normal/I')
        tree.Branch("q",   q,   'normal/D')
        tree.Branch("a",   a,   'normal/D')        
        
        # loop over events
        for e in self.events:

            q[0]=0
            a[0]=0            
            if e.wfset:
                for wf in e.wfset.waveforms:
                    q[0] += wf.get_analysis('standard').result['integral']
                    a[0] += wf.get_analysis('standard').result['amplitude']

            if nwfs>0:
                q[0] = q[0]/(1.*nwfs)
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
