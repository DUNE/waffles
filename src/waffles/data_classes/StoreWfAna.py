from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.WfAnaResult import WfAnaResult

import waffles.Exceptions as we


class StoreWfAna(WfAna):
    """Stands for Store Waveform Analysis. This 
    class inherits from WfAna. It implements a 
    dummy analysis which is performed over a 
    certain WaveformAdcs object, which simply
    stores the given input parameters as if they
    were the result of a real analysis. This
    analysis does not depend on the actual
    waveform data.

    Attributes
    ----------
    input_parameters: IPDict (inherited from WfAna)
    result: WfAnaResult (inherited from WfAna)

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    @we.handle_missing_data
    def __init__(self, input_parameters: IPDict):
        """StoreWfAna class initializer.

        Parameters
        ----------
        input_parameters: IPDict
            The contents of this IPDict can be arbitrary
        """

        super().__init__(input_parameters)

    def analyse(self, waveform: WaveformAdcs) -> None:
        """This 'analysis' consists of setting to the result
        attribute the input parameters that were given to the
        class initializer. Note that this analysis is not
        dependent on the actual waveform data.

        Parameters
        ----------
        waveform: WaveformAdcs
            The WaveformAdcs object of interest

        Returns
        ----------
        None
        """

        self._WfAna__result = WfAnaResult(
            **self._WfAna__input_parameters
        )
        return

    @staticmethod
    @we.handle_missing_data
    def check_input_parameters(
            input_parameters: IPDict
    ) -> None:
        """This method performs no checks.

        Parameters
        ----------
        input_parameters: IPDict
            It is the IPDict that can be potentially given to
            StoreWfAna.__init__ to instantiate a StoreWfAna object.

        Returns
        ----------
        None
        """

        return
