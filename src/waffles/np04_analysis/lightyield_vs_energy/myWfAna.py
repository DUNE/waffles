import numpy as np

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.utils.check_utils as wuc
from waffles.utils.baseline.baseline import SBaseline
import waffles.Exceptions as we

from pathlib import Path

def which_APA_for_the_ENDPOINT(endpoint: int):
    apa_endpoints = {1: {104, 105, 107}, 2: {109}, 3: {111}, 4: {112, 113}}
    for apa, endpoints in apa_endpoints.items():
        if endpoint in endpoints:
            return apa
    return None


def from_cutoff_to_sigma(cutoff_frequency):
    npoints = 1024//2 # The fft has half the length of the waveforms
    FFTFreq = 32.5
    binwidth = FFTFreq/npoints
    fc_in_ticks = cutoff_frequency/binwidth
    return fc_in_ticks/np.sqrt(np.log(2))


def gaus(x, sigma=20):
    return np.exp(-(x)**2/2/sigma**2)

def searching_maritza_template(apa, endpoint, daq_channel, map_path = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft/PDHD_PDS_ChannelMap.csv', maritza_template_folder = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft'):
    daphne_to_offline = {11207: 0, 11201: 1, 11217: 2, 11211: 3, 11307: 4, 11220: 5, 11226: 6, 11230: 7, 11236: 8, 11240: 9, 11205: 10, 11203: 11, 11215: 12, 11213: 13, 11305: 14, 11222: 15, 11224: 16, 11232: 17, 11234: 18, 11242: 19, 11202: 20, 11204: 21, 11212: 22, 11214: 23, 11302: 24, 11225: 25, 11223: 26, 11235: 27, 11233: 28, 11245: 29, 11200: 30, 11206: 31, 11210: 32, 11216: 33, 11300: 34, 11227: 35, 11221: 36, 11237: 37, 11231: 38, 11247: 39, 11106: 40, 11131: 41, 11107: 42, 11130: 43, 11146: 44, 11111: 45, 11117: 46, 11121: 47, 11147: 48, 11120: 49, 11104: 50, 11133: 51, 11105: 52, 11132: 53, 11144: 54, 11113: 55, 11115: 56, 11123: 57, 11145: 58, 11122: 59, 11103: 60, 11134: 61, 11102: 62, 11135: 63, 11143: 64, 11114: 65, 11112: 66, 11124: 67, 11142: 68, 11125: 69, 11101: 70, 11136: 71, 11100: 72, 11137: 73, 11141: 74, 11116: 75, 11110: 76, 11126: 77, 11140: 78, 11127: 79, 10920: 80, 10926: 81, 10930: 82, 10936: 83, 10900: 84, 10906: 85, 10910: 86, 10916: 87, 10940: 88, 10946: 89, 10922: 90, 10924: 91, 10932: 92, 10934: 93, 10902: 94, 10904: 95, 10912: 96, 10914: 97, 10942: 98, 10944: 99, 10925: 100, 10923: 101, 10935: 102, 10933: 103, 10905: 104, 10903: 105, 10915: 106, 10913: 107, 10945: 108, 10943: 109, 10927: 110, 10921: 111, 10937: 112, 10931: 113, 10907: 114, 10901: 115, 10917: 116, 10911: 117, 10947: 118, 10941: 119, 10400: 120, 10406: 121, 10410: 122, 10416: 123, 10500: 124, 10506: 125, 10521: 126, 10517: 127, 10710: 128, 10707: 129, 10402: 130, 10404: 131, 10412: 132, 10414: 133, 10502: 134, 10504: 135, 10523: 136, 10515: 137, 10712: 138, 10705: 139, 10405: 140, 10403: 141, 10415: 142, 10413: 143, 10505: 144, 10503: 145, 10524: 146, 10512: 147, 10715: 148, 10702: 149, 10407: 150, 10401: 151, 10417: 152, 10411: 153, 10507: 154, 10501: 155, 10526: 156, 10510: 157, 10717: 158, 10700: 159}
    daphne_channel = daq_channel + 100*endpoint
    apa_template_folder  = next((f for f in Path(maritza_template_folder).glob(f"*APA{apa}*") if f.is_dir()), None)
    martiza_template_file = next(apa_template_folder.glob(f"*APA{apa}_CH{daphne_to_offline[daphne_channel]}*.txt"), None)
    if martiza_template_file is None:
        # Maritza template not available (only 34 channels)
        return None
    else: 
        with open(martiza_template_file, "r") as file:
            maritza_values = [float(line.strip()) for line in file]
        return np.array(maritza_values)


class myWfAna(WfAna):
    """Stands for Basic Waveform Analysis. This 
    class inherits from WfAna. It implements a 
    basic analysis which is performed over a 
    certain WaveformAdcs object.

    Attributes
    ----------
    input_parameters: IPDict (inherited from WfAna)
    baseline_limits: list of int
        In case `baseline_method` is set to EasyMedian:
            It must have an even number of integers which
            must meet baseline_limits[i] < baseline_limits[i + 1].
            Given a WaveformAdcs object, wf, the points which
            are used for baseline calculation are
            wf.adcs[baseline_limits[2*i] - wf.time_offset :
            baseline_limits[(2*i) + 1] - wf.time_offset],
            with i = 0,1,...,(len(baseline_limits)/2) - 1. The
            upper limits are exclusive.
        In case `baseline_method` is set to SBaseline:
            This parameter is not used. 
    baseline_method: str
        The method used to compute the baseline.
        Values accepted: SBaseline, EasyMedian
    int_ll (resp. int_ul): int
        Stands for integration lower (resp. upper) limit.
        Iterator value for the first (resp. last) point
        of the Waveform that falls into the integration
        window. int_ll must be smaller than int_ul. These
        limits are inclusive. I.e. the points which are
        used for the integral calculation are
        wf.adcs[int_ll - wf.time_offset : int_ul + 1 - wf.time_offset].
    amp_ll (resp. amp_ul): int
        Stands for amplitude lower (resp. upper) limit.
        Iterator value for the first (resp. last) point
        of the Waveform that is considered to compute
        the amplitude of the Waveform. amp_ll must be smaller
        than amp_ul. These limits are inclusive. I.e., the
        points which are used for the amplitude calculation
        are wf.adcs[amp_ll - wf.time_offset : amp_ul + 1 - wf.time_offset].

    invert: bool
        Rather to invert the waveform or not 
    
    result: WfAnaResult (inherited from WfAna)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    @we.handle_missing_data
    def __init__(self, input_parameters: IPDict):
        """BasicWfAna class initializer. It is assumed that it is
        the caller responsibility to check the well-formedness
        of the input parameters, according to the attributes
        documentation in the class documentation. No checks
        are perfomed here.

        Parameters
        ----------
        input_parameters: IPDict
            This IPDict must contain the following keys:
                - 'baseline_limits' (list of int)
                - 'amp_ll' (int)
                - 'amp_ul' (int)
                - 'invert' (bool)
                - 'baseline_method' (str)
                - 'baseliner' (if baseline_method = SBaseline)
                
                - 'integration' (bool)
                - 'int_ll' (int)
                - 'int_ul' (int)
                
                - 'deconvolution' (bool) 
                - 'save_devonvolved_wf' (bool)
                - 'maritza_template' (bool)
                - 'template' (str) #if maritza_template = False + it must have a namae like xxxxx_endYYY_chKK #estensione?
                - 'gauss_filtering' (bool)
                - 'gauss_cutoff' (float) #MHz
        """

        self.__baseline_limits = input_parameters['baseline_limits']
        self.__amp_ll = input_parameters['amp_ll']
        self.__amp_ul = input_parameters['amp_ul']
        self.__baseline_method = input_parameters.get('baseline_method', "EasyMedian")
        if self.__baseline_method == "SBaseline":
            self.__baseliner: SBaseline = input_parameters['baseliner']
            
               
        # Deconvolution parameters
        self.__devonvolution = input_parameters['deconvolution']
        if self.__devonvolution:
            self.__save_devonvolved_wf = input_parameters['save_devonvolved_wf']
            self.__maritza_template = input_parameters['maritza_template']
            if not self.__maritza_template:
                self.__template = input_parameters['template'] #it must have a namae like xxxxx_endYYY_chKK.pkl
            
            self.__gauss_filtering = input_parameters['gauss_filtering']
            if self.__gauss_filtering:
                self.__gauss_cutoff = input_parameters['gauss_cutoff'] #MHz
            
        self.__onlyoptimal: bool = input_parameters.get('onlyoptimal', False)

        self.__invfactor = 1
        self.__invert = input_parameters.get('invert', False)
        if self.__invert:
            self.__invfactor = -1;
            
        # Integration parameters
        self.__integration = input_parameters['integration']
        if self.__integration:
            self.__int_ll = input_parameters['int_ll']
            self.__int_ul = input_parameters['int_ul']
            if self.__devonvolution:
                self.__int_ll_deconv_filter = input_parameters['int_ll_deconv_filter']
                self.__int_ul_deconv_filter = input_parameters['int_ul_deconv_filter']

        super().__init__(input_parameters)

    # Getters
    @property
    def baseline_limits(self):
        return self.__baseline_limits
    
    @property
    def integration(self):
        return self.__integration

    # @property
    # def int_ll(self):
    #     return self.__int_ll

    # @property
    # def int_ul(self):
    #     return self.__int_ul

    @property
    def amp_ll(self):
        return self.__amp_ll

    @property
    def amp_ul(self):
        return self.__amp_ul
    
    @property
    def deconvolution(self):
        return self.__deconvolution
    


    def analyse(self, waveform: WaveformAdcs) -> None:
        """With respect to the given WaveformAdcs object, this 
        analyser method does the following:

            - It computes the baseline as the median of the points
            that are considered, according to the documentation of
            the self.__baseline_limits attribute.
            - It calculates the integral of
            waveform.adcs[int_ll - waveform.time_offset:
            int_ul + 1 - waveform.time_offset].
            To do so, it assumes that the temporal resolution of
            the waveform is constant and approximates its integral
            to waveform.time_step_ns*np.sum(-b + waveform.adcs[int_ll -
            waveform.time_offset: int_ul + 1 - waveform.time_offset]),
            where b is the computed baseline.
            - It calculates the amplitude of
            waveform.adcs[amp_ll - waveform.time_offset: amp_ul + 1 -
            waveform.time_offset].

        Note that for these computations to be well-defined, it is
        assumed that

            - baseline_limits[0] - wf.time_offset >= 0
            - baseline_limits[-1] - wf.time_offset <= len(wf.adcs)
            - int_ll - wf.time_offset >= 0
            - int_ul - wf.time_offset < len(wf.adcs)
            - amp_ll - wf.time_offset >= 0
            - amp_ul - wf.time_offset < len(wf.adcs)

        For the sake of efficiency, these checks are not done.
        It is the caller's responsibility to ensure that these
        requirements are met.

        Parameters
        ----------
        waveform: WaveformAdcs
            The WaveformAdcs object which will be analysed

        Returns
        ----------
        None
        """
        
        optimized = True

        if self.__baseline_method == "EasyMedian":
            split_baseline_samples = [
                waveform.adcs[
                    self.__baseline_limits[2 * i] - waveform.time_offset:
                    self.__baseline_limits[(2 * i) + 1] - waveform.time_offset
                ]
                for i in range(len(self.__baseline_limits) // 2)
            ]

            baseline_samples = np.concatenate(split_baseline_samples)
            baseline = np.median(baseline_samples)
        else:
            baseline, optimized = self.__baseliner.compute_baseline(waveform.adcs, self.__baseliner.filtering)

        if self.__integration:
            integral_before = waveform.time_step_ns * self.__invfactor * (
                np.sum(
                    waveform.adcs[ self.__int_ll - waveform.time_offset : 
                                self.__int_ul + 1 - waveform.time_offset
                                ]
                ) - (( self.__int_ul - self.__int_ll + 1) * baseline))
        else:
            integral_before = None
        
        amplitude=(
            np.max(
                waveform.adcs[
                    self.__amp_ll - waveform.time_offset:
                    self.__amp_ul + 1 - waveform.time_offset
                ]
            ) - np.min(
                waveform.adcs[
                    self.__amp_ll - waveform.time_offset:
                    self.__amp_ul + 1 - waveform.time_offset
                ]
            )
        )

        if not optimized and self.__onlyoptimal:
            integral = np.nan
            amplitude = np.nan
        
        # Deconvolution    
        if self.__devonvolution:
            new_wf_adcs = waveform.adcs.astype(float) - baseline
            new_wf_adcs = -new_wf_adcs
            
            if self.__maritza_template:
                adc_template = searching_maritza_template(which_APA_for_the_ENDPOINT(waveform.endpoint),waveform.endpoint,waveform.channel)
            elif self.__template.split('ch')[-1].split('.')[0] == str(waveform.channel) and self.template.split('end')[-1].split('_')[0] == str(waveform.endpoint):
                import pickle
                with open(self.__template, 'rb') as f:
                    adc_template = np.roll(np.array(pickle.load(f)[0]), 0)
            else: 
                adc_template = None
                    
            if adc_template is None:
                integral_deconv = None
                integral_deconv_filtered = None
                deconvolved_wf = None
                filtered_deconvolved_wf = None
            else:
                new_wf_adcs = waveform.adcs.astype(float) - baseline
                new_wf_adcs = -new_wf_adcs
                    
                signal_fft = np.fft.fft(new_wf_adcs)
                template_fft = np.fft.fft(adc_template, n=len(new_wf_adcs))
                deconvolved_fft = signal_fft / template_fft    
                deconvolved_wf = np.fft.ifft(deconvolved_fft).real  
                
                if self.__integration:
                    integral_deconv = waveform.time_step_ns * self.__invfactor * (
                        np.sum(
                            deconvolved_wf[ self.__int_ll_deconv_filter - waveform.time_offset : 
                                        self.__int_ul_deconv_filter + 1 - waveform.time_offset
                                        ]
                        ) - (( self.__int_ul_deconv_filter - self.__int_ll_deconv_filter + 1) * baseline))
                else:
                    integral_deconv = None
                
                
                if self.__gauss_filtering:
                    _x = np.linspace(0, 1024, 1024, endpoint=False)
                    sigma = from_cutoff_to_sigma(self.__gauss_cutoff)
                    filter_gaus = np.array([gaus(x, sigma=sigma) for x in _x])
                    
                    filtered_deconvolved_fft = deconvolved_fft * filter_gaus
                    filtered_deconvolved_wf = np.fft.ifft(filtered_deconvolved_fft).real  
                
                    if self.__integration:
                        integral_deconv_filtered = waveform.time_step_ns * self.__invfactor * (
                            np.sum(
                                filtered_deconvolved_wf[ self.__int_ll_deconv_filter - waveform.time_offset : 
                                            self.__int_ul_deconv_filter + 1 - waveform.time_offset
                                            ]
                            ) - (( self.__int_ul_deconv_filter - self.__int_ll_deconv_filter + 1) * baseline))
                    else:
                        integral_deconv_filtered = None
                    
                else:
                   integral_deconv_filtered = None 
                   filtered_deconvolved_wf = None 
                   
                if not self.__save_devonvolved_wf:
                    deconvolved_wf = None 
                    filtered_deconvolved_wf = None
        else:
            integral_deconv=None,
            integral_deconv_filtered=None
            deconvolved_wf = None 
            filtered_deconvolved_wf = None

        self._WfAna__result = WfAnaResult(
            baseline=baseline,
            # For a deeper analysis for which we need (and
            # can afford the computation time) for this data,
            # this one might be set to np.min(baseline_samples),
            baseline_min=None,
            # np.max(baseline_samples),
            baseline_max=None,
            # and ~np.std(baseline_samples))
            baseline_rms=None,
            # Assuming that the waveform is inverted and
            # using linearity to avoid some multiplications
            integral_before=integral_before,
            amplitude=amplitude,
            integral_deconv=integral_deconv,
            integral_deconv_filtered=integral_deconv_filtered,
            deconvolved_wf = deconvolved_wf,     
            filtered_deconvolved_wf = filtered_deconvolved_wf,     
            )
        return

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(
            input_parameters: IPDict,
            points_no: int
    ) -> None:
        """This method performs three checks:

            - It checks whether the baseline limits, say bl, are
            well-formed, i.e. whether they meet

                0 <= bl[0] < bl[1] < ... < bl[-1] <= points_no - 1

            - It checks whether the integration window, say
            (int_ll, int_ul) is well-formed, i.e. whether it meets

                0 <= int_ll < int_ul <= points_no - 1

            - It checks whether the amplitude window, say
            (amp_ll, amp_ul) is well-formed, i.e. whether it meets

                0 <= amp_ll < amp_ul <= points_no - 1

        If any of these checks fail, an exception is raised.

        Parameters
        ----------
        input_parameters: IPDict
            The input parameters to be checked. It is the IPDict
            that can be potentially given to BasicWfAna.__init__
            to instantiate a BasicWfAna object.
        points_no: int
            The number of points in any waveform that could be
            analysed. It is assumed to be the same for all the
            waveforms.

        Returns - ((
        ----------
        None
        """
        
        if input_parameters["baseline_method"] == "EasyMedian":
            if not wuc.baseline_limits_are_well_formed(
                    input_parameters['baseline_limits'],
                    points_no):

                raise Exception(we.GenerateExceptionMessage(
                    1,
                    'BasicWfAna.check_input_parameters()',
                    f"The baseline limits ({input_parameters['baseline_limits']})"
                    " are not well formed."))
        
        # int_ul_ = input_parameters['int_ul']
        # if int_ul_ is None:
        #     int_ul_ = points_no - 1

        # if not wuc.subinterval_is_well_formed(
        #         input_parameters['int_ll'],
        #         int_ul_,
        #         points_no):

        #     raise Exception(we.GenerateExceptionMessage(
        #         2,
        #         'BasicWfAna.check_input_parameters()',
        #         f"The integration window ({input_parameters['int_ll']},"
        #         f" {int_ul_}) is not well formed. It must be a subset of"
        #         f" [0, {points_no})."))
                    
        # amp_ul_ = input_parameters['amp_ul']
        # if amp_ul_ is None:
        #     amp_ul_ = points_no - 1

        # if not wuc.subinterval_is_well_formed(
        #         input_parameters['amp_ll'],
        #         amp_ul_,
        #         points_no):

        #     raise Exception(we.GenerateExceptionMessage(
        #         3,
        #         'BasicWfAna.check_input_parameters()',
        #         f"The amplitude window ({input_parameters['amp_ll']},"
        #         f" {amp_ul_}) is not well formed. It must be a subset of"
        #         f" [0, {points_no})."))

