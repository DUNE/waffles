import numpy as np
from collections import OrderedDict
from typing import Optional, TYPE_CHECKING

# Import only for type-checking, so as
# to avoid a runtime circular import
if TYPE_CHECKING:
    from waffles.data_classes.WfAna import WfAna

from waffles.data_classes.IPDict import IPDict
from waffles.Exceptions import GenerateExceptionMessage


class WaveformAdcs:

    # It is useful to have such a class so that tools which
    # only need the adcs information can be run even in situations 
    # where a Waveform does not have a defined timestamp,
    # endpoint or any other attribute which could be used to
    # identify a Waveform at a higher level. For example, the 
    # Waveform which is the result of a averaging over every 
    # Waveform for a certain channel could be analyzed so as 
    # to compute its baseline, but its timestamp is not defined, 
    # i.e. it makes no sense.

    """This class implements the adcs array of a Waveform.

    Attributes
    ----------
    time_step_ns: float
        The time step (in nanoseconds) for this Waveform
    adcs: unidimensional numpy array of integers
        The readout for this Waveform, in # of ADCs
    time_offset: int
        A time offset, in units of time_step_ns (i.e.
        time ticks) which will be used as a relative
        alignment among different WaveformAdcs
        objects for plotting and analysis purposes.
        It must be semipositive and smaller than
        len(self.__adcs)-1. It is set to 0 by default.
    analyses: OrderedDict of WfAna objects

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    # The restrictions over the time_offset attribute
    # ensure that there are always at least two points
    # left in the [0, 1, ..., len(self.__adcs) - 1] range,
    # so that baselines, integrals and amplitudes can be
    # computed using points in that range.

    def __init__(
        self, 
        time_step_ns: float,
        adcs: np.ndarray,
        time_offset: int = 0
    ):
        """WaveformAdcs class initializer

        Parameters
        ----------
        time_step_ns: float
        adcs: unidimensional numpy array of integers
        time_offset: int
            It must be semipositive and smaller than
            len(self.__adcs)-1. It is set to 0 by
            default.
        """

        # Shall we add add type checks here?

        self.__time_step_ns = time_step_ns
        self.__adcs = adcs

        # WaveformSet._set_time_offset()
        # takes care of the proper checks
        self.__set_time_offset(time_offset)

        # Initialize the analyses 
        # attribute as an empty OrderedDict.
        self.__analyses = OrderedDict()  

        # Do we need to add trigger primitives as attributes?

    # Getters
    @property
    def time_step_ns(self):
        return self.__time_step_ns

    @property
    def adcs(self):
        return self.__adcs

    @property
    def time_offset(self):
        return self.__time_offset

    @property
    def analyses(self):
        return self.__analyses

# For the moment there are no setters for
# the attributes of WaveformAdcs. I.e. you
# can only set the value of its attributes
# through WaveformAdcs.__init__. Here's an
# example of what a setter would look like.
#
#   @time_step_ns.setter
#   def time_step_ns(self, input):
#       self.__time_step_ns = input
#       return

    def __set_time_offset(self, input: float) -> None:
        """This method is not intended for user usage. It 
        is a setter for the time_offset attribute.

        Parameters
        ----------
        input: float
            The value which will be assigned to the time_offset
            attribute. It must be semipositive and smaller than
            len(self.__adcs)-1.

        Returns
        ----------
        None
        """

        if input < 0 or input >= len(self.__adcs)-1:

            raise Exception(GenerateExceptionMessage(
                1,
                'WaveformAdcs.__set_time_offset()',
                f"The given time offset ({input}) must belong to the"
                f" [0, {len(self.__adcs)-2}] interval."))
        else:
            self.__time_offset = input

        return

    def __slice_adcs(
        self,
        start: int,
        end: int
    ) -> None:
        """This method is not intended for user usage.
        No well-formedness checks are performed here.
        This method slices the self.__adcs attribute
        array to self.__adcs[start, end]. This method
        applies the change in place.

        Parameters
        ----------
        start: int
            Iterator value for the (inclusive) start of
            the slice
        end: int
            Iterator value for the (exclusive) end of
            the slice

        Returns
        ----------
        None
        """

        self.__adcs = self.__adcs[start:end]
        return
        
    def confine_iterator_value(self, input: int) -> int:
        """Confines the input integer to the range [0, len(self.__adcs) - 1].
        I.e returns 0 if input is negative, returns input if input belongs
        to the range [0, len(self.__adcs) - 1], 
        and returns len(self.__adcs) - 1 in any other case.

        Parameters
        ----------
        input: int

        Returns
        ----------
        int
        """

        if input < 0:
            return 0
        elif input < len(self.__adcs):
            return input
        else:
            return len(self.__adcs) - 1

    def analyse(
        self, 
        label: str,
        analysis_class: type,
        input_parameters: IPDict,
        *args,
        overwrite: bool = False,
        **kwargs
    ) -> dict:
        """This method creates an object of type analysis_class,
        which must be a class (type) which inherits from the
        WfAna class. Then, runs its analyse() method on the
        self WaveformAdcs object. The created analysis object
        is stored in the self.__analyses attribute, using the
        label parameter as its key. This method returns the
        output of the called method, even if it is None.

        Parameters
        ----------
        label: str
            Key for the new WfAna (or derived) object within
            the self.__analyses OrderedDict
        analysis_class: type
            Class (type) which must inherit from WfAna. For
            efficiency reasons, this check is not performed
            at this level (WaveformAdcs). The given class
            must have an analyse() method which takes a
            WaveformAdcs object as its first argument. The
            implementation of such method is enforced by the
            WfAna abstract class, but it is the user's
            responsibility to ensure that such method takes,
            indeed, a WaveformAdcs object as its first argument.
        input_parameters: IPDict
            The input parameters which will be passed to the
            analysis_class initializer. It is the user's
            responsibility to ensure that input_parameters
            contain the required information to initialize
            the analysis_class object, and that it is
            well-defined.
        *args
            Additional positional arguments which are given
            to the analyse() method of analysis_class
        overwrite: bool
            If True, the method will overwrite any existing
            WfAna (or derived) object with the same label
            (key) within self.__analyses.
        **kwargs
            Additional keyword arguments which are given
            to the analyse() method of analysis_class

        N.B.: It is preferred to keep track of all of the I/O
        information of the analysis in the WfAna attributes which
        are designed for that purpose, i.e. WfAna.input_parameters
        and WfAna.Result. However, *args and **kwargs are enabled
        to give more room for the user to potentially configure the
        analyse() method of WfAna (or derived) with parameters that
        have no interest from the point of view of the analysis.
        In the same way, the output of WfAna.analyse() is captured
        and returned by this method, even if it is None.

        Returns
        ----------
        output: object
            It is the output of the analyse() method of the
            analysis_class object which was created.
        """

        if not overwrite:
            if label in self.__analyses.keys():
                raise Exception(GenerateExceptionMessage(
                    1,
                    'WaveformAdcs.analyse()',
                    f"There is already an analysis with label '{label}'. "
                    f"If you want to overwrite it, set the 'overwrite' "
                    "parameter to True."))

        # Create the analysis object
        aux = analysis_class(input_parameters)

        # Run the analysis and grab
        # the output, even if it is None
        output = aux.analyse(self,
                             *args,
                             **kwargs)

        # Add the created analysis object
        # to the self.__analyses attribute
        self.__analyses[label] = aux
        
        return output

    # The WfAna class is not defined at runtime, only
    # during type-checking (see TYPE_CHECKING). Not
    # enclosing the type in quotes would raise a
    # `NameError: name 'WfAna' is not defined.`
    def get_analysis(
        self, 
        label: Optional[str] = None
    ) -> 'WfAna':
        """If the 'label' parameter is defined, then this
        method returns the WfAna object which has such
        label within the self.__analyses OrderedDict.
        If there is no analysis with such label, then
        this method raises a KeyError. If the 'label'
        parameter is not defined, then this method returns
        the last WfAna object added to self.__analyses. If
        there are no analyses, then this method raises an
        exception.

        Parameters
        ----------
        label: str
            The key for the WfAna object within the
            self.__analyses OrderedDict.

        Returns
        ----------
        output: WfAna
            The WfAna object which has the given label
        """

        if label is None:
            try:
                # Grabbing the last analysis
                output = next(reversed(self.__analyses.values()))
            except StopIteration:
                raise Exception(GenerateExceptionMessage(
                    1,
                    'WaveformAdcs.get_analysis()',
                    'The Waveform has not been analysed yet.'))
        else:
            try:
                output = self.__analyses[label]
            except KeyError:
                raise Exception(GenerateExceptionMessage(
                    2,
                    'WaveformAdcs.get_analysis()',
                    f"There is no analysis with label '{label}'."))
        return output