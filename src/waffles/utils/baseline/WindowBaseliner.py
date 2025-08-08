import numpy as np

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.utils.check_utils as wuc
import waffles.Exceptions as we


class WindowBaseliner(WfAna):
    """This WfAna subclass implements a baseline
    computation based on the mean/median of the
    points which belong to a certain time window
    of the WaveformAdcs object.

    Attributes
    ----------
    input_parameters: IPDict (inherited from WfAna)
    baseline_limits: list of int
        It must have an even number of integers which
        must meet baseline_limits[i] < baseline_limits[i + 1].
        Given a WaveformAdcs object, wf, the points which
        are used for baseline calculation are
        wf.adcs[baseline_limits[2*i] - wf.time_offset :
        baseline_limits[(2*i) + 1] - wf.time_offset],
        with i = 0,1,...,(len(baseline_limits)/2) - 1. The
        upper limits are exclusive.
    std_cut: float
        To do an initial estimation of the baseline, say
        preliminary_baseline, the median of the points
        specified in the baseline_limits attribute
        documentation is computed. Also, the standard
        deviation of these points, say x, is computed. To
        compute the definitive baseline, the points which
        are used are those which are within the
        preliminary_baseline +/- (std_cut * x) range.
    type: str
        Either 'mean' or 'median'. It specifies whether
        the baseline is computed as the mean or the median
        of the points which have been selected up to the 
        baseline_limits and the std_cut attributes.
    result: WfAnaResult (inherited from WfAna)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    @we.handle_missing_data
    def __init__(self, input_parameters: IPDict):
        """WindowBaseliner class initializer. It is assumed that
        it is the caller responsibility to check the well-formedness
        of the input parameters, according to the attributes
        documentation in the class documentation. No checks
        are perfomed here.

        Parameters
        ----------
        input_parameters: IPDict
            This IPDict must contain the following keys:
                - 'baseline_limits' (list of int)
                - 'std_cut' (float)
                - 'type' (str)
        """

        self.__baseline_limits = input_parameters['baseline_limits']
        self.__std_cut = input_parameters['std_cut']
        self.__type = input_parameters['type']

        super().__init__(input_parameters)

    # Getters
    @property
    def baseline_limits(self):
        return self.__baseline_limits

    @property
    def std_cut(self):
        return self.__std_cut

    @property
    def type(self):
        return self.__type

    def analyse(self, waveform: WaveformAdcs) -> None:
        """This analyser method computes the baseline of the
        given WaveformAdcs object, say wf, according to the
        documentation of this class attributes. For this
        computation to be well-defined, it is assumed that

            - baseline_limits[0] - wf.time_offset >= 0
            - baseline_limits[-1] - wf.time_offset <= len(wf.adcs)

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

        split_baseline_samples = [
            waveform.adcs[
                self.__baseline_limits[2 * i] - waveform.time_offset:
                self.__baseline_limits[(2 * i) + 1] - waveform.time_offset
            ]
            for i in range(len(self.__baseline_limits) // 2)
        ]

        baseline_samples = np.concatenate(split_baseline_samples)

        preliminary_baseline = np.median(baseline_samples)

        allowed_samples_mask = \
            np.abs(baseline_samples - preliminary_baseline) <= \
                abs(self.__std_cut) * np.std(baseline_samples)

        baseline_samples = baseline_samples[allowed_samples_mask]

        # WindowBaseliner.check_input_parameters() takes care of
        # checking that self.__type is either 'mean' or 'median'
        baseline = np.mean(baseline_samples) \
            if self.__type == 'mean' else \
            np.median(baseline_samples)

        self._WfAna__result = WfAnaResult(
            baseline=baseline,
            baseline_std=np.std(baseline_samples)
        )
        return

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(
            input_parameters: IPDict,
            points_no: int
    ) -> None:
        """This method performs two checks:

            - It checks whether the baseline limits, say bl, are
            well-formed, i.e. whether they meet

                0 <= bl[0] < bl[1] < ... < bl[-1] <= points_no - 1

            - It checks whether the baseline type is valid, i.e.
            whether it meets

                self.__type in ['mean', 'median']

        If any of these checks fail, an exception is raised.

        Parameters
        ----------
        input_parameters: IPDict
            The input parameters to be checked. It is the IPDict
            that can be potentially given to
            WindowBaseliner.__init__().
        points_no: int
            The number of points in any waveform that could be
            analysed. It is assumed to be the same for all the
            waveforms.

        Returns
        ----------
        None
        """

        if not wuc.baseline_limits_are_well_formed(
            input_parameters['baseline_limits'],
            points_no
        ):
            raise Exception(we.GenerateExceptionMessage(
                1,
                'WindowBaseliner.check_input_parameters()',
                f"The baseline limits ({input_parameters['baseline_limits']})"
                " are not well formed."))

        if input_parameters['type'] not in ['mean', 'median']:
            raise Exception(
                we.GenerateExceptionMessage(
                    2,
                    'WindowBaseliner.check_input_parameters()',
                    f"The specified type ('{input_parameters['type']}')"
                    " must be either 'mean' or 'median'."
                )
            )