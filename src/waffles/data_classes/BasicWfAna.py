import numpy as np

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IpDict
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.utils.check_utils as wuc
import waffles.Exceptions as we


class BasicWfAna(WfAna):

    """
    Stands for Basic Waveform Analysis. This class
    inherits from WfAna. It implements a basic
    analysis which is performed over a certain
    WaveformAdcs object.

    Attributes
    ----------
    InputParameters : IPDict (inherited from WfAna)
    BaselineLimits : list of int
        It must have an even number of integers which
        must meet BaselineLimits[i] < BaselineLimits[i + 1].
        Given a WaveformAdcs object, wf, the points which
        are used for baseline calculation are
        wf.adcs[BaselineLimits[2*i] - wf.time_offset :
        BaselineLimits[(2*i) + 1] - wf.time_offset],
        with i = 0,1,...,(len(BaselineLimits)/2) - 1. The
        upper limits are exclusive.
    IntLl (resp. IntUl) : int
        Stands for integration lower (resp. upper) limit.
        Iterator value for the first (resp. last) point
        of the Waveform that falls into the integration
        window. IntLl must be smaller than IntUl. These
        limits are inclusive. I.e. the points which are
        used for the integral calculation are
        wf.adcs[IntLl - wf.time_offset : IntUl + 1 - wf.time_offset].
    AmpLl (resp. AmpUl) : int
        Stands for amplitude lower (resp. upper) limit.
        Iterator value for the first (resp. last) point
        of the Waveform that is considered to compute
        the amplitude of the Waveform. AmpLl must be smaller
        than AmpUl. These limits are inclusive. I.e., the
        points which are used for the amplitude calculation
        are wf.adcs[AmpLl - wf.time_offset : AmpUl + 1 - wf.time_offset].
    Result : WfAnaResult (inherited from WfAna)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    @we.handle_missing_data
    def __init__(self, input_parameters: IpDict):
        """
        BasicWfAna class initializer. It is assumed that it is
        the caller responsibility to check the well-formedness
        of the input parameters, according to the attributes
        documentation in the class documentation. No checks
        are perfomed here.

        Parameters
        ----------
        input_parameters : IPDict
            This IPDict must contain the following keys:
                - 'baseline_limits' (list of int)
                - 'int_ll' (int)
                - 'int_ul' (int)
                - 'amp_ll' (int)
                - 'amp_ul' (int)
        """

        self.__baseline_limits = input_parameters['baseline_limits']
        self.__int_ll = input_parameters['int_ll']
        self.__int_ul = input_parameters['int_ul']
        self.__amp_ll = input_parameters['amp_ll']
        self.__amp_ul = input_parameters['amp_ul']

        super().__init__(input_parameters)

    # Getters
    @property
    def BaselineLimits(self):
        return self.__baseline_limits

    @property
    def IntLl(self):
        return self.__int_ll

    @property
    def IntUl(self):
        return self.__int_ul

    @property
    def AmpLl(self):
        return self.__amp_ll

    @property
    def AmpUl(self):
        return self.__amp_ul

    def analyse(self, Waveform: WaveformAdcs) -> None:
        """
        With respect to the given WaveformAdcs object, this analyser
        method does the following:

            - It computes the baseline as the median of the points
            that are considered, according to the documentation of
            the self.__baseline_limits attribute.
            - It calculates the integral of
            Waveform.adcs[IntLl - Waveform.time_offset :
            IntUl + 1 - Waveform.time_offset].
            To do so, it assumes that the temporal resolution of
            the Waveform is constant and approximates its integral
            to Waveform.TimeStep_ns*np.sum( -b + Waveform.adcs[IntLl -
            Waveform.time_offset : IntUl + 1 - Waveform.time_offset]),
            where b is the computed baseline.
            - It calculates the amplitude of
            Waveform.adcs[AmpLl - Waveform.time_offset : AmpUl + 1 -
            Waveform.time_offset].

        Note that for these computations to be well-defined, it is
        assumed that

            - BaselineLimits[0] - wf.time_offset >= 0
            - BaselineLimits[-1] - wf.time_offset <= len(wf.adcs)
            - IntLl - wf.time_offset >= 0
            - IntUl - wf.time_offset < len(wf.adcs)
            - AmpLl - wf.time_offset >= 0
            - AmpUl - wf.time_offset < len(wf.adcs)

        For the sake of efficiency, these checks are not done.
        It is the caller's responsibility to ensure that these
        requirements are met.

        Parameters
        ----------
        Waveform : WaveformAdcs
            The WaveformAdcs object which will be analysed

        Returns
        ----------
        None
        """

        split_baseline_samples = [
            Waveform.adcs[
                self.__baseline_limits[2 * i] - Waveform.time_offset:
                self.__baseline_limits[(2 * i) + 1] - Waveform.time_offset
            ]
            for i in range(len(self.__baseline_limits) // 2)
        ]

        baseline_samples = np.concatenate(split_baseline_samples)
        baseline = np.median(baseline_samples)

        self._WfAna__result = WfAnaResult(
            baseline=baseline,
            # Might be set to np.min(baseline_samples) (resp.
            baseline_min=None,
            # np.max(baseline_samples), ~np.std(baseline_samples))
            baseline_max=None,
            # for a deeper analysis for which we need (and
            baseline_rms=None,
            # can afford the computation time) for this data

            integral=Waveform.TimeStep_ns*(((
                self.__int_ul - self.__int_ll + 1)*baseline) - np.sum(
                # Assuming that the Waveform is
                Waveform.adcs[
                    self.__int_ll - Waveform.time_offset:
                        self.__int_ul + 1 - Waveform.time_offset])),
            # inverted and using linearity
            # to avoid some multiplications
            amplitude=(
                np.max(
                    Waveform.adcs[
                        self.__amp_ll - Waveform.time_offset:
                        self.__amp_ul + 1 - Waveform.time_offset
                    ]
                ) - np.min(
                    Waveform.adcs[
                        self.__amp_ll - Waveform.time_offset:
                        self.__amp_ul + 1 - Waveform.time_offset
                    ]
                )
            )
        )
        return

    @ staticmethod
    @ we.handle_missing_data
    def check_input_parameters(
            input_parameters: IpDict,
            points_no: int) -> None:
        """
        This method performs three checks:

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
        input_parameters : IPDict
            The input parameters to be checked. It is the IPDict
            that can be potentially given to BasciWfAna.__init__
            to instantiate a BasicWfAna object.
        points_no : int
            The number of points in any Waveform that could be
            analysed. It is assumed to be the same for all the
            waveforms.

        Returns
        ----------
        None
        """

        if not wuc.baseline_limits_are_well_formed(
                input_parameters['baseline_limits'],
                points_no):

            raise Exception(we.GenerateExceptionMessage(
                1,
                'BasicWfAna.check_input_parameters()',
                f"The baseline limits ({input_parameters['baseline_limits']})"
                " are not well formed."))
        int_ul_ = input_parameters['int_ul']
        if int_ul_ is None:
            int_ul_ = points_no - 1

        if not wuc.subinterval_is_well_formed(
                input_parameters['int_ll'],
                int_ul_,
                points_no):

            raise Exception(we.GenerateExceptionMessage(
                2,
                'BasicWfAna.check_input_parameters()',
                f"The integration window ({input_parameters['int_ll']},"
                f" {int_ul_}) is not well formed. It must be a subset of"
                f" [0, {points_no})."))
        amp_ul_ = input_parameters['amp_ul']
        if amp_ul_ is None:
            amp_ul_ = points_no - 1

        if not wuc.subinterval_is_well_formed(
                input_parameters['amp_ll'],
                amp_ul_,
                points_no):

            raise Exception(we.GenerateExceptionMessage(
                3,
                'BasicWfAna.check_input_parameters()',
                f"The amplitude window ({input_parameters['amp_ll']},"
                f" {amp_ul_}) is not well formed. It must be a subset of"
                f" [0, {points_no})."))
