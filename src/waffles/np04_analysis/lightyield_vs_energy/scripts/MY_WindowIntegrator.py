import numpy as np
import pandas as pd

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.utils.check_utils as wuc
import waffles.Exceptions as we


def fbk_or_hpk(endpoint: int, channel: int):
    channel_vendor_map = {
    104: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK"},
    105: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 12: "FBK", 15: "FBK", 17: "FBK", 21: "HPK", 23: "HPK", 24: "HPK", 26: "HPK"},
    107: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK",
          10: "HPK", 12: "HPK", 15: "HPK", 17: "HPK"},
    109: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
          20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
    111: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
          20: "FBK", 21: "FBK", 22: "FBK", 23: "FBK", 24: "FBK", 25: "FBK", 26: "FBK", 27: "FBK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
    112: {0: "HPK", 1: "HPK", 2: "HPK", 3: "HPK", 4: "HPK", 5: "HPK", 6: "HPK", 7: "HPK",
          10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK",
          20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 42: "HPK", 45: "HPK", 47: "HPK"},
    113: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK"}}

    return channel_vendor_map[endpoint][channel]

bias_info = {
    'june': { 'FBK' : {'OV_V': 4.5, 'PDE': 0.45}, 'HPK' : {'OV_V': 3.0, 'PDE': 0.5}, 'batch' : 1 }, # FBK OV, HPK  OV, batch number to use for p.e. calibration
    'july': { 'FBK' : {'OV_V': 4.5, 'PDE': 0.45}, 'HPK' : {'OV_V': 2.5, 'PDE': 0.45}, 'batch' : 1 },
    'august': { 'FBK' : {'OV_V': 4.5, 'PDE': 0.45}, 'HPK' : {'OV_V': 2.5, 'PDE': 0.45}, 'batch' : 4 }, 
    'september': { 'FBK' : {'OV_V': 4.5, 'PDE': 0.45}, 'HPK' : {'OV_V': 2.5, 'PDE': 0.45}, 'batch' : 6 }}
df_calibration_45 = pd.read_csv("/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/calibration/calibration_results_45_pde.csv")
df_calibration_50 = pd.read_csv("/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/calibration/calibration_results_50_pde.csv")


class MY_WindowIntegrator(WfAna):
    """This WfAna subclass uses the trapezoidal
    rule to compute the integral of the points
    which belong to a certain time window of the
    WaveformAdcs object.

    Attributes
    ----------
    input_parameters: IPDict (inherited from WfAna)
    baseline_analysis: string
        The name of the analysis from which to
        retrieve the baseline, which is used
        by the integral computation. I.e. the
        analyses attribute of the integrated
        WaveformAdcs object must contain a
        key matching baseline_analysis. The
        value for such a key is a WfAna object
        whose result attribute must contain the
        baseline value to use under the
        'baseline' key.
    inversion: bool
        Whether to invert the waveform or not
        before computing the integral.
    int_ll (resp. int_ul): int 
        Stands for integration lower (resp. upper)
        limit. Iterator value for the first (resp.
        last) point of the Waveform that falls into
        the integration window. int_ll must be smaller
        than int_ul. These limits are inclusive. I.e.
        the points which are used for the integral
        calculation are
        wf.adcs[int_ll - wf.time_offset : int_ul + 1 - wf.time_offset].
    amp_ll (resp. amp_ul): int
        Stands for amplitude lower (resp. upper)
        limit. Iterator value for the first (resp. last)
        point of the Waveform that is considered to
        compute the amplitude of the Waveform. amp_ll
        must be smaller than amp_ul. These limits are
        inclusive. I.e., the points which are used for
        the amplitude calculation are
        wf.adcs[amp_ll - wf.time_offset : amp_ul + 1 - wf.time_offset].   
    period: str (June, July, August, September)
        Stands for the period in which the run was acquired and 
        is required for the computation of the number of photoelectrons. 
        If it is None, the information about spe are not used 
    result: WfAnaResult (inherited from WfAna)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    @we.handle_missing_data
    def __init__(self, input_parameters: IPDict):
        """WindowIntegrator class initializer. It is assumed
        that it is the caller responsibility to check the
        well-formedness of the input parameters, according
        to the attributes documentation in the class
        documentation. No checks are performed here.

        Parameters
        ----------
        input_parameters: IPDict
            This IPDict must contain the following keys:
                - 'baseline_analysis' (str)
                - 'inversion' (bool)
                - 'int_ll' (int)
                - 'int_ul' (int)
                - 'amp_ll' (int)
                - 'amp_ul' (int)
                - 'period' (str)
        """

        self.__baseline_analysis = input_parameters['baseline_analysis']
        self.__inversion = input_parameters['inversion']
        self.__int_ll = input_parameters['int_ll']
        self.__int_ul = input_parameters['int_ul']
        self.__amp_ll = input_parameters['amp_ll']
        self.__amp_ul = input_parameters['amp_ul']
        self.__period = input_parameters['period'].strip().lower()

        super().__init__(input_parameters)

    # Getters
    @property
    def baseline_analysis(self):
        return self.__baseline_analysis
    
    @property
    def inversion(self):
        return self.__inversion

    @property
    def int_ll(self):
        return self.__int_ll

    @property
    def int_ul(self):
        return self.__int_ul

    @property
    def amp_ll(self):
        return self.__amp_ll

    @property
    def amp_ul(self):
        return self.__amp_ul
    
    @property
    def period(self):
        return self.__period

    def analyse(self, waveform: WaveformAdcs) -> None:
        """With respect to the given WaveformAdcs object, this 
        analyser method does the following:

            - It calculates the integral of
            waveform.adcs[int_ll - waveform.time_offset:
            int_ul + 1 - waveform.time_offset].
            To do so, it assumes that the temporal resolution
            of the waveform is constant and approximates its
            integral to

                numpy.trapezoid(
                    y=(1. if not self.__inversion else -1.) * \
                    (-b + waveform.adcs[
                        int_ll - waveform.time_offset:
                        int_ul + 1 - waveform.time_offset
                    ]),
                    dx=waveform.time_step_ns
                )

            where b is the baseline which has been retrieved
            from the indicated previous analysis, up to the
            baseline_analysis attribute. 

            - It calculates the amplitude of
                
                waveform.adcs[
                    amp_ll - waveform.time_offset:
                    amp_ul + 1 - waveform.time_offset
                ]
            
            - It extrapolates the value of the spe from calibration runs,
            and returns:
                spe_amplitude_adcs = Amplitude of the single pe (in adcs)
                gain = Integral of the spe (using x=0.3) ,
                integral_pe = integral expressed in number of pe


        Note that for these computations to be well-defined, it is
        assumed that

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
        
        try:
            baseline_ana = waveform.analyses[self.__baseline_analysis]
        except KeyError:
            raise KeyError(
                we.GenerateExceptionMessage(
                    1,
                    'WindowIntegrator.analyse()',
                    f"The WaveformAdcs object does not contain an analysis"
                    f" with the name {self.__baseline_analysis}."
                )
            )
        else:
            try:
                baseline = baseline_ana.result['baseline']
            except KeyError:
                raise KeyError(
                    we.GenerateExceptionMessage(
                        2,
                        'WindowIntegrator.analyse()',
                        f"Found analysis {self.__baseline_analysis} in the"
                        f" WaveformAdcs object, but it does not contain"
                        f" a result with the key 'baseline'."
                    )
                )
        try:
            integral = np.trapezoid(
                y=(1. if not self.__inversion else -1.) * \
                (-1.*baseline + waveform.adcs[
                    self.__int_ll - waveform.time_offset:
                    self.__int_ul + 1 - waveform.time_offset
                ]),
                dx=waveform.time_step_ns
            )
        except AttributeError:
            integral = np.trapz(
                y=(1. if not self.__inversion else -1.) * \
                (-1.*baseline + waveform.adcs[
                    self.__int_ll - waveform.time_offset:
                    self.__int_ul + 1 - waveform.time_offset
                ]),
                dx=waveform.time_step_ns
            )

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

        vendor = fbk_or_hpk(waveform.endpoint, waveform.channel)

        # Computing number of photoelectrons
        if bias_info[self.__period][vendor]['PDE'] == 0.5:
            df_calibration = df_calibration_50
        elif bias_info[self.__period][vendor]['PDE'] == 0.45:
            df_calibration = df_calibration_45
        else:
            df_calibration = None
            print('PDE problem march')

        subset = df_calibration.loc[
            (df_calibration['batch'] == bias_info[self.__period]['batch']) &
            (df_calibration['endpoint'] == waveform.endpoint) &
            (df_calibration['channel'] == waveform.channel) &
            (df_calibration['OV_V'] == bias_info[self.__period][vendor]['OV_V'])
            ]
        
        spe_amplitude_adcs = subset['SPE_mean_adcs'].iloc[0]
        gain = subset['gain'].iloc[0] #spe integral
        integral_pe = integral/gain

        self._WfAna__result = WfAnaResult(
            integral=integral,
            amplitude=amplitude,
            spe_amplitude_adcs = spe_amplitude_adcs,
            gain = gain,
            integral_pe = integral_pe,
            integration_limits = (self.__int_ll, self.__int_ul)
        )
        return

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(
            input_parameters: IPDict,
            points_no: int
    ) -> None:
        """This method performs two checks:

            - It checks whether the integration window,
            (int_ll, int_ul) is well-formed, i.e. whether
            it meets

                0 <= int_ll < int_ul <= points_no - 1

            - It checks whether the amplitude window,
            (amp_ll, amp_ul) is well-formed, i.e. whether
            it meets

                0 <= amp_ll < amp_ul <= points_no - 1
            
            - It checks if the period is June, July, August or September 

        If any of these checks fail, an exception is raised.

        Parameters
        ----------
        input_parameters: IPDict
            The input parameters to be checked. It is the IPDict
            that can be potentially given to
            WindowIntegrator.__init__().
        points_no: int
            The number of points in any waveform that could be
            analysed. It is assumed to be the same for all the
            waveforms.

        Returns
        ----------
        None
        """

        int_ul_ = input_parameters['int_ul']
        if int_ul_ is None:
            int_ul_ = points_no - 1

        if not wuc.subinterval_is_well_formed(
            input_parameters['int_ll'],
            int_ul_,
            points_no
        ):
            raise Exception(we.GenerateExceptionMessage(
                1,
                'WindowIntegrator.check_input_parameters()',
                f"The integration window ({input_parameters['int_ll']},"
                f" {int_ul_}) is not well formed. It must be a subset of"
                f" [0, {points_no})."))
                    
        amp_ul_ = input_parameters['amp_ul']
        if amp_ul_ is None:
            amp_ul_ = points_no - 1

        if not wuc.subinterval_is_well_formed(
            input_parameters['amp_ll'],
            amp_ul_,
            points_no
        ):
            raise Exception(we.GenerateExceptionMessage(
                2,
                'WindowIntegrator.check_input_parameters()',
                f"The amplitude window ({input_parameters['amp_ll']},"
                f" {amp_ul_}) is not well formed. It must be a subset of"
                f" [0, {points_no})."))

        period = input_parameters['period'].strip().lower()
        if period not in {"june", "july", "august", "september"}:
            raise Exception(we.GenerateExceptionMessage(
                2,
                'WindowIntegrator.check_input_parameters()',
                f"The period ({input_parameters['period']},"
                f" is not june, july, august or september."))