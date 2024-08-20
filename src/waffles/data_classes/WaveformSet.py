import inspect

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
from plotly import graph_objects as pgo
from plotly import subplots as psu

from waffles.data_classes.WaveformAdcs import waveform_adcs
from waffles.data_classes.WfAna import wf_ana
from waffles.data_classes.Map import map_
from waffles.data_classes.IPDict import ip_dict

import waffles.utils.numerical_utils as wun
import waffles.utils.filtering_utils as wuf

from waffles.Exceptions import generate_exception_message


class waveform_set:

    """
    This class implements a set of waveforms.

    Attributes
    ----------
    waveforms : list of Waveform objects
        waveforms[i] gives the i-th waveform in the set.
    points_per_wf : int
        Number of entries for the Adcs attribute of
        each Waveform object in this waveform_set object.
    runs : set of int
        It contains the run number of any run for which
        there is at least one waveform in the set.
    record_numbers : dictionary of sets
        It is a dictionary whose keys are runs (int) and
        its values are sets of record numbers (set of int).
        If there is at least one Waveform object within
        this waveform_set which was acquired during run n,
        then n belongs to record_numbers.keys(). record_numbers[n]
        is a set of record numbers for run n. If there is at
        least one waveform acquired during run n whose
        RecordNumber is m, then m belongs to record_numbers[n].
    available_channels : dictionary of dictionaries of sets
        It is a dictionary whose keys are run numbers (int),
        so that if there is at least one waveform in the set
        which was acquired during run n, then n belongs to
        available_channels.keys(). available_channels[n] is a
        dictionary whose keys are endpoints (int) and its
        values are sets of channels (set of int). If there
        is at least one Waveform object within this waveform_set
        which was acquired during run n and which comes from
        endpoint m, then m belongs to available_channels[n].keys().
        available_channels[n][m] is a set of channels for
        endpoint m during run n. If there is at least one
        waveform for run n, endpoint m and channel p, then p
        belongs to available_channels[n][m].
    MeanAdcs : waveform_adcs
        The mean of the adcs arrays for every waveform
        or a subset of waveforms in this waveform_set. It
        is a waveform_adcs object whose TimeStep_ns
        attribute is assumed to match that of the first
        waveform which was used in the average sum.
        Its Adcs attribute contains points_per_wf entries,
        so that MeanAdcs.Adcs[i] is the mean of
        self.waveforms[j].Adcs[i] for every value
        of j or a subset of values of j, within
        [0, len(self.__waveforms) - 1]. It is not
        computed by default. I.e. if self.MeanAdcs
        equals to None, it should be interpreted as
        unavailable data. Call the 'compute_mean_waveform'
        method of this waveform_set to compute it.
    MeanAdcsIdcs : tuple of int
        It is a tuple of integers which contains the indices
        of the waveforms, with respect to this waveform_set,
        which were used to compute the MeanAdcs.Adcs
        attribute. By default, it is None. I.e. if
        self.MeanAdcsIdcs equals to None, it should be
        interpreted as unavailable data. Call the
        'compute_mean_waveform' method of this waveform_set
        to compute it.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self, *waveforms):
        """
        waveform_set class initializer

        Parameters
        ----------
        waveforms : unpacked list of Waveform objects
            The waveforms that will be added to the set
        """

        # Shall we add type checks here?

        if len(waveforms) == 0:
            raise Exception(generate_exception_message(
                1,
                'waveform_set.__init__()',
                'There must be at least one waveform in the set.'))
        self.__waveforms = list(waveforms)

        if not self.check_length_homogeneity():
            raise Exception(generate_exception_message(
                2,
                'waveform_set.__init__()',
                'The length of the given waveforms is not homogeneous.'))

        self.__points_per_wf = len(self.__waveforms[0].Adcs)

        self.__runs = set()
        self.__update_runs(other_runs=None)

        self.__record_numbers = {}
        self.__update_record_numbers(other_record_numbers=None)

        self.__available_channels = {}
        # Running on an Apple M2, it took
        self.__update_available_channels(other_available_channels=None)
        # ~ 52 ms to run this line for a
        # waveform_set with 1046223 waveforms
        self.__mean_adcs = None
        self.__mean_adcs_idcs = None

    # Getters
    @property
    def waveforms(self):
        return self.__waveforms

    @property
    def points_per_wf(self):
        return self.__points_per_wf

    @property
    def runs(self):
        return self.__runs

    @property
    def record_numbers(self):
        return self.__record_numbers

    @property
    def available_channels(self):
        return self.__available_channels

    @property
    def mean_adcs(self):
        return self.__mean_adcs

    @property
    def mean_adcs_idcs(self):
        return self.__mean_adcs_idcs

    def get_set_of_endpoints(self) -> set:
        """
        This method returns a set which contains every endpoint
        for which there is at least one waveform in this
        waveform_set object.

        Returns
        ----------
        output : set of int
        """

        output = set()

        for run in self.__available_channels.keys():
            for endpoint in self.__available_channels[run].keys():
                output.add(endpoint)

        return output

    def get_run_collapsed_available_channels(self) -> dict:
        """
        This method returns a dictionary of sets of integers,
        say output, whose keys are endpoints. If there is
        at least one waveform within this set that comes from
        endpoint n, then n belongs to output.keys(). output[n]
        is a set of integers, so that if there is at least a
        waveform coming from endpoint n and channel m, then m
        belongs to output[n].

        Returns
        ----------
        output : dictionary of sets
        """

        output = {}

        for run in self.__runs:
            for endpoint in self.__available_channels[run].keys():
                try:
                    aux = output[endpoint]
                except KeyError:
                    output[endpoint] = set()
                    aux = output[endpoint]

                for channel in self.__available_channels[run][endpoint]:
                    aux.add(channel)

        return output

    def check_length_homogeneity(self) -> bool:
        """
        This method returns True if the Adcs attribute
        of every Waveform object in this waveform_set
        has the same length. It returns False if else.
        In order to call this method, there must be at
        least one waveform in the set.

        Returns
        ----------
        bool
        """

        if len(self.__waveforms) == 0:
            raise Exception(generate_exception_message(
                1,
                'waveform_set.check_length_homogeneity()',
                'There must be at least one waveform in the set.'))
        length = len(self.__waveforms[0].Adcs)
        for i in range(1, len(self.__waveforms)):
            if len(self.__waveforms[i].Adcs) != length:
                return False
        return True

    def __update_runs(self, other_runs: Optional[set] = None) -> None:
        """
        This method is not intended to be called by the user.
        This method updates the self.__runs attribute of this
        object. Its behaviour is different depending on whether
        the 'other_runs' parameter is None or not. Check its
        documentation for more information.

        Parameters
        ----------
        other_runs : set of int
            If it is None, then this method clears the self.__runs
            attribute of this object and then iterates through the
            whole waveform_set to fill such attribute according to
            the waveforms which are currently present in this
            waveform_set object. If the 'other_runs' parameter is
            defined, then it must be a set of integers, as expected
            for the runs attribute of a waveform_set object. In this
            case, the entries within other_runs which are not
            already present in self.__runs, are added to self.__runs.
            The well-formedness of this parameter is not checked by
            this method. It is the caller's responsibility to ensure
            it.

        Returns
        ----------
        None
        """

        if other_runs is None:
            self.__reset_runs()
        else:
            self.__runs = self.__runs.union(other_runs)

        return

    def __reset_runs(self) -> None:
        """
        This method is not intended for user usage.
        This method must only be called by the
        waveform_set.__update_runs() method. It clears
        the self.__runs attribute of this object and
        then iterates through the whole waveform_set to
        fill such attribute according to the waveforms
        which are currently present in this waveform_set
        object.
        """

        self.__runs.clear()

        for wf in self.__waveforms:
            self.__runs.add(wf.RunNumber)

        return

    def __update_record_numbers(
            self,
            other_record_numbers: Optional[Dict[int, set]] = None
    ) -> None:
        """
        This method is not intended to be called by the user.
        This method updates the self.__record_numbers attribute
        of this object. Its behaviour is different depending on
        whether the 'other_record_numbers' parameter is None or
        not. Check its documentation for more information.

        Parameters
        ----------
        other_record_numbers : dictionary of sets of int
            If it is None, then this method clears the
            self.__record_numbers attribute of this object
            and then iterates through the whole waveform_set
            to fill such attribute according to the waveforms
            which are currently present in this waveform_set.
            If the 'other_record_numbers' parameter is defined,
            then it must be a dictionary of sets of integers,
            as expected for the record_numbers attribute of a
            waveform_set object. In this case, the information
            in other_record_numbers is merged into the
            self.__record_numbers attribute of this object,
            according to the meaning of the self.__record_numbers
            attribute. The well-formedness of this parameter
            is not checked by this method. It is the caller's
            responsibility to ensure it.

        Returns
        ----------
        None
        """

        if other_record_numbers is None:
            self.__reset_record_numbers()

        else:
            for run in other_record_numbers.keys():
                if run in self.__record_numbers.keys():
                    # If this run is present in both, this waveform_set and
                    # the incoming one, then carefully merge the information

                    self.__record_numbers[run] = self.__record_numbers[
                        run].union(other_record_numbers[run])

                else:
                    # If this run is present in the incoming
                    # waveform_set but not in self, then
                    # simply get the information from the
                    # incoming waveform_set as a block

                    self.__record_numbers[run] = other_record_numbers[run]
        return

    def __reset_record_numbers(self) -> None:
        """
        This method is not intended for user usage.
        This method must only be called by the
        waveform_set.__update_record_numbers() method. It clears
        the self.__record_numbers attribute of this object and
        then iterates through the whole waveform_set to fill such
        attribute according to the waveforms which are currently
        present in this waveform_set object.
        """

        self.__record_numbers.clear()

        for wf in self.__waveforms:
            try:
                self.__record_numbers[wf.RunNumber].add(wf.RecordNumber)
            except KeyError:
                self.__record_numbers[wf.RunNumber] = set()
                self.__record_numbers[wf.RunNumber].add(wf.RecordNumber)
        return

    def __update_available_channels(
        self,
        other_available_channels: Optional[Dict[int, Dict[int, set]]] = None
    ) -> None:
        """
        This method is not intended to be called by the user.
        This method updates the self.__available_channels
        attribute of this object. Its behaviour is different
        depending on whether the 'other_available_channels'
        parameter is None or not. Check its documentation for
        more information.

        Parameters
        ----------
        other_available_channels : dictionary of dictionaries of sets
            If it is None, then this method clears the
            self.__available_channels attribute of this object
            and then iterates through the whole waveform_set
            to fill such attribute according to the waveforms
            which are currently present in this waveform_set.
            If the 'other_available_channels' parameter is
            defined, then it must be a dictionary of dictionaries
            of sets of integers, as expected for the
            available_channels attribute of a waveform_set object.
            In this case, the information in other_available_channels
            is merged into the self.__available_channels attribute
            of this object, according to the meaning of the
            self.__available_channels attribute. The well-
            formedness of this parameter is not checked by this
            method. It is the caller's responsibility to ensure it.

        Returns
        ----------
        None
        """

        if other_available_channels is None:
            self.__reset_available_channels()

        else:
            for run in other_available_channels.keys():
                if run in self.__available_channels.keys():
                    # If this run is present in both, this waveform_set and
                    # the incoming one, then carefully merge the information

                    for endpoint in other_available_channels[run].keys():
                        # If this endpoint for this run is present
                        if endpoint in self.__available_channels[run].keys():
                            # in both waveform sets, then carefully
                            # merge the information.

                            self.__available_channels[run][
                                endpoint] = self.__available_channels[
                                    run][endpoint].union(
                                        other_available_channels[run][endpoint]
                            )

                        else:
                            # If this endpoint for this run is present in the
                            # incoming waveform_set but not in
                            # self, then simply get the information from the
                            # incoming waveform_set as a block

                            self.__available_channels[run][
                                endpoint] = other_available_channels[run][
                                    endpoint]

                else:
                    # If this run is present in the incoming
                    # waveform_set but not in self, then
                    # simply get the information from the
                    # incoming waveform_set as a block

                    self.__available_channels[
                        run] = other_available_channels[run]
        return

    def __reset_available_channels(self) -> None:
        """
        This method is not intended for user usage.
        This method must only be called by the
        waveform_set.__update_available_channels() method. It clears
        the self.__available_channels attribute of this object and
        then iterates through the whole waveform_set to fill such
        attribute according to the waveforms which are currently
        present in this waveform_set object.
        """

        self.__available_channels.clear()

        for wf in self.__waveforms:
            try:
                aux = self.__available_channels[wf.RunNumber]

                try:
                    aux[wf.Endpoint].add(wf.Channel)

                except KeyError:
                    aux[wf.Endpoint] = set()
                    aux[wf.Endpoint].add(wf.Channel)

            except KeyError:
                self.__available_channels[wf.RunNumber] = {}
                self.__available_channels[wf.RunNumber][wf.Endpoint] = set()
                self.__available_channels[wf.RunNumber][wf.Endpoint].add(
                    wf.Channel)
        return

    def analyse(
            self, label: str,
            analysis_class: type,
            input_parameters: ip_dict,
            *args,
            analysis_kwargs: dict = {},
            checks_kwargs: dict = {},
            overwrite: bool = False
    ) -> dict:
        """
        For each Waveform in this waveform_set, this method
        calls its 'analyse' method passing to it the parameters
        given to this method. In turn, Waveform.analyse()
        (actually waveform_adcs.analyse()) creates an object
        of type analysis_class (which must be a class which
        inherits from the WfAna class) and runs its analyse()
        method on the current Waveform object. The created
        analysis object is stored in the Analyses attribute
        of the Waveform object, using the given label parameter
        as its key. This method returns a dictionary, say x,
        where the keys are indices of the waveforms in this
        waveform_set, so that x[i] is the output of the
        self.__waveforms[i].analyse() method.

        Parameters
        ----------
        label : str
            For every analysed waveform, this is the key
            for the new WfAna (or derived) object within its
            Analyses attribute.
        analysis_class : type
            Class (type) which must inherit from WfAna. The
            given class must have an analyse() method which
            takes a waveform_adcs object as its first argument
            (after self).
        input_parameters : IPDict
            The input parameters which will be passed to the
            analysis_class initializer by the waveform_adcs.analyse()
            method, for each analysed waveform. It is the
            user's responsibility to ensure that
            input_parameters contain the required information
            to initialize the analysis_class object, and that
            it is well-defined.
        *args
            Additional positional arguments which are given
            to the Waveform.analyse() (actually waveform_adcs.analyse())
            for each analysed waveform, which in turn,
            are given to the analyse() method of analysis_class.
        analysis_kwargs : dict
            Additional keyword arguments which are given
            to the Waveform.analyse() (actually waveform_adcs.analyse())
            for each analysed waveform, which in turn,
            are given to the analyse() method of analysis_class.
        checks_kwargs : dict
            Additional keyword arguments which are given
            to the check_input_parameters() method of
            the analysis_class class.
        overwrite : bool
            If True, for every analysed Waveform, its
            'analyse' method will overwrite any existing
            WfAna (or derived) object with the same label
            (key) within its Analyses attribute.

        Returns
        ----------
        output : dict
            output[i] gives the output of
            self.__waveforms[i].analyse(...), which is a
            dictionary containing any additional information
            of the analysis which was performed over the
            i-th waveform of this waveform_set. Such
            dictionary is empty if the analyser method gives
            no additional information.
        """

        if not issubclass(analysis_class, wf_ana):
            raise Exception(generate_exception_message(
                1,
                'waveform_set.analyse()',
                'The analysis class must be derived from the WfAna class.'))

        analysis_class.check_input_parameters(
            input_parameters,
            **checks_kwargs)

        # analysis_class may have not implemented an abstract method
        signature = inspect.signature(analysis_class.analyse)
        # of WfAna, p.e. analyse(), and still produce no errors until
        # an object of such class is instantiated. If that's the case,
        # 'signature' is actually the signature of WfAna.analyse() and
        # inspecting it is dumb, since we would not be checking the
        # signature of analysis_class.analyse(). This is not a big deal,
        # though, because the user will, anyway, encounter a descriptive
        # error when trying to run the analysis for the first waveform,
        # where an object of analysis_class is instantiated.
        try:

            aux = list(signature.parameters.keys())[1]
            if aux != 'waveform':  # The first parameter is 'self'

                raise Exception(generate_exception_message(
                    2,
                    "waveform_set.analyse()",
                    "The name of the first parameter of the 'analyse()'"
                    f" method ('{aux}') of the given analysis class"
                    f" ({analysis_class.__name__}) must be 'waveform'."))

            if signature.parameters['waveform'].annotation != waveform_adcs:
                raise Exception(generate_exception_message(
                    3,
                    "waveform_set.analyse()",
                    "The 'waveform' parameter of the 'analyse()' "
                    "method of the given analysis class"
                    f" ({analysis_class.__name__}) must be hinted as a "
                    "waveform_adcs object."))
        except IndexError:
            raise Exception(generate_exception_message(
                4,
                "waveform_set.analyse()",
                "The 'analyse()' method of the given analysis class "
                f"({analysis_class.__name__}) must take at least"
                " one parameter."))
        output = {}

        for i in range(len(self.__waveforms)):
            output[i] = self.__waveforms[i].analyse(label,
                                                    analysis_class,
                                                    input_parameters,
                                                    *args,
                                                    overwrite=overwrite,
                                                    **analysis_kwargs)
        return output

    def compute_mean_waveform(
            self, *args,
            wf_idcs: Optional[List[int]] = None,
            wf_selector: Optional[Callable[
                ...,
                bool]] = None,
            **kwargs) -> waveform_adcs:
        """
        If wf_idcs is None and wf_selector is None,
        then this method creates a waveform_adcs
        object whose Adcs attribute is the mean
        of the adcs arrays for every waveform in
        this waveform_set. If wf_idcs is not None,
        then such mean is computed using the adcs
        arrays of the waveforms whose iterator
        values, with respect to this waveform_set,
        are given in wf_idcs. If wf_idcs is None
        but wf_selector is not None, then such
        mean is computed using the adcs arrays
        of the waveforms, wf, within this
        waveform_set for which
        wf_selector(wf, *args, **kwargs) evaluates
        to True. In any case, the TimeStep_ns
        attribute of the newly created waveform_adcs
        object assumed to match that of the first
        waveform which was used in the average sum.

        In any case, the resulting waveform_adcs
        object is assigned to the
        self.__mean_adcs attribute. The
        self.__mean_adcs_idcs attribute is also
        updated with a tuple of the indices of the
        waveforms which were used to compute the
        mean waveform_adcs. Finally, this method
        returns the averaged waveform_adcs object.

        Parameters
        ----------
        *args
            These arguments only make a difference if
            the 'wf_idcs' parameter is None and the
            'wf_selector' parameter is suitable defined.
            For each waveform, wf, these are the
            positional arguments which are given to
            wf_selector(wf, *args, **kwargs) as *args.
        wf_idcs : list of int
            If it is not None, then it must be a list
            of integers which must be a valid iterator
            value for the __waveforms attribute of this
            waveform_set. I.e. any integer i within such
            list must satisfy
            0 <= i <= len(self.__waveforms) - 1. Any
            integer which does not satisfy this condition
            is ignored. These integers give the waveforms
            which are averaged.
        wf_selector : callable
            This parameter only makes a difference if
            the 'wf_idcs' parameter is None. If that's
            the case, and 'wf_selector' is not None, then
            it must be a callable whose first parameter
            must be called 'waveform' and its type
            annotation must match the Waveform class.
            Its return value must be annotated as a
            boolean. In this case, the mean waveform
            is averaged over those waveforms, wf, for
            which wf_selector(wf, *args, **kwargs)
            evaluates to True.
        *kwargs
            These keyword arguments only make a
            difference if the 'wf_idcs' parameter is
            None and the 'wf_selector' parameter is
            suitable defined. For each waveform, wf,
            these are the keyword arguments which are
            given to wf_selector(wf, *args, **kwargs)
            as **kwargs.

        Returns
        ----------
        output : np.ndarray
            The averaged adcs array
        """

        if len(self.__waveforms) == 0:
            raise Exception(generate_exception_message(
                1,
                'waveform_set.compute_mean_waveform()',
                'There are no waveforms in this waveform_set object.'))
        if wf_idcs is None and wf_selector is None:

            output = self.__compute_mean_waveform_of_every_waveform()
            # Average over every
            # waveform in this waveform_set
        elif wf_idcs is None and wf_selector is not None:

            signature = inspect.signature(wf_selector)

            wuf.check_well_formedness_of_generic_waveform_function(signature)

            output = self.__compute_mean_waveform_with_selector(
                wf_selector,
                *args,
                **kwargs)
        else:

            fWfIdcsIsWellFormed = False
            for idx in wf_idcs:
                if self.is_valid_iterator_value(idx):

                    fWfIdcsIsWellFormed = True
                    break                       # Just make sure that there
                    # is at least one valid
                    # iterator value in the given list

            if not fWfIdcsIsWellFormed:
                raise Exception(generate_exception_message(
                    2,
                    'waveform_set.compute_mean_waveform()',
                    'The given list of waveform indices is empty or it does '
                    'not contain even one valid iterator value in the given '
                    'list. I.e. there are no waveforms to average.'))

            output = self.__compute_mean_waveform_of_given_waveforms(
                wf_idcs)  # In this case we also need to remove indices
            # redundancy (if any) before giving wf_idcs to
            # waveform_set.__compute_mean_waveform_of_given_waveforms.
            # This is a open issue for now.
        return output

    def __compute_mean_waveform_of_every_waveform(self) -> waveform_adcs:
        """
        This method should only be called by the
        waveform_set.compute_mean_waveform() method,
        where any necessary well-formedness checks
        have already been performed. It is called by
        such method in the case where both the 'wf_idcs'
        and the 'wf_selector' input parameters are
        None. This method sets the self.__mean_adcs
        and self.__mean_adcs_idcs attributes according
        to the waveform_set.compute_mean_waveform()
        method documentation. It also returns the
        averaged waveform_adcs object. Refer to the
        waveform_set.compute_mean_waveform() method
        documentation for more information.

        Returns
        ----------
        output : np.ndarray
            The averaged adcs array
        """

        # waveform_set.compute_mean_waveform()
        aux = self.waveforms[0].Adcs
        # has already checked that there is at
        # least one waveform in this waveform_set
        for i in range(1, len(self.__waveforms)):
            aux += self.waveforms[i].Adcs

        output = waveform_adcs(
            self.__waveforms[0].TimeStep_ns,
            aux/len(self.__waveforms),
            time_offset=0)

        self.__mean_adcs = output
        self.__mean_adcs_idcs = tuple(range(len(self.__waveforms)))

        return output

    def __compute_mean_waveform_with_selector(
            self, wf_selector: Callable[..., bool],
            *args,
            **kwargs) -> waveform_adcs:
        """
        This method should only be called by the
        waveform_set.compute_mean_waveform() method,
        where any necessary well-formedness checks
        have already been performed. It is called by
        such method in the case where the 'wf_idcs'
        parameter is None and the 'wf_selector'
        parameter is suitably defined. This method
        sets the self.__mean_adcs and
        self.__mean_adcs_idcs attributes according
        to the waveform_set.compute_mean_waveform()
        method documentation. It also returns the
        averaged waveform_adcs object. Refer to the
        waveform_set.compute_mean_waveform() method
        documentation for more information.

        Parameters
        ----------
        wf_selector : callable
        *args
        **kwargs

        Returns
        ----------
        output : np.ndarray
            The averaged adcs array
        """

        added_wvfs = []

        aux = np.zeros((self.__points_per_wf,))

        for i in range(len(self.__waveforms)):
            if wf_selector(self.__waveforms[i], *args, **kwargs):
                aux += self.__waveforms[i].Adcs
                added_wvfs.append(i)

        if len(added_wvfs) == 0:
            raise Exception(generate_exception_message(
                1,
                'waveform_set.__compute_mean_waveform_with_selector()',
                'No waveform in this waveform_set object '
                'passed the given selector.'))

        output = waveform_adcs(
            self.__waveforms[added_wvfs[0]].TimeStep_ns,
            aux/len(added_wvfs),
            time_offset=0)

        self.__mean_adcs = output
        self.__mean_adcs_idcs = tuple(added_wvfs)

        return output

    def __compute_mean_waveform_of_given_waveforms(
            self, wf_idcs: List[int]) -> waveform_adcs:
        """
        This method should only be called by the
        waveform_set.compute_mean_waveform() method,
        where any necessary well-formedness checks
        have already been performed. It is called by
        such method in the case where the 'wf_idcs'
        parameter is not None, regardless the input
        given to the 'wf_selector' parameter. This
        method sets the self.__mean_adcs and
        self.__mean_adcs_idcs attributes according
        to the waveform_set.compute_mean_waveform()
        method documentation. It also returns the
        averaged waveform_adcs object. Refer to the
        waveform_set.compute_mean_waveform() method
        documentation for more information.

        Parameters
        ----------
        wf_idcs : list of int

        Returns
        ----------
        output : np.ndarray
            The averaged adcs array
        """

        added_wvfs = []

        aux = np.zeros((self.__points_per_wf,))

        for idx in wf_idcs:
            try:
                # waveform_set.compute_mean_waveform() only checked that there
                # is at least one valid iterator value, but we need to handle
                # the case where there are invalid iterator values

                aux += self.__waveforms[idx].Adcs
            except IndexError:
                continue
                # Ignore the invalid iterator values as specified in the
                # waveform_set.compute_mean_waveform() method documentation
            else:
                added_wvfs.append(idx)

        output = waveform_adcs(
            self.__waveforms[added_wvfs[0]].TimeStep_ns,
            # len(added_wvfs) must be at least 1.
            aux/len(added_wvfs),
            # This was already checked by
            # waveform_set.compute_mean_waveform()
            time_offset=0)
        self.__mean_adcs = output
        self.__mean_adcs_idcs = tuple(added_wvfs)

        return output

    def is_valid_iterator_value(self, iterator_value: int) -> bool:
        """
        This method returns True if
        0 <= iterator_value <= len(self.__waveforms) - 1,
        and False if else.
        """

        if iterator_value < 0:
            return False
        elif iterator_value <= len(self.__waveforms) - 1:
            return True
        else:
            return False

    def filter(
            self, wf_filter: Callable[..., bool],
            *args,
            actually_filter: bool = False,
            return_the_staying_ones: bool = True,
            **kwargs) -> List[int]:
        """
        This method filters the waveforms in this waveform_set
        using the given wf_filter callable. I.e. for each
        Waveform object, wf, in this waveform_set, it runs
        wf_filter(wf, *args, **kwargs). This method returns
        a list of indices for the waveforms which got the
        same result from the filter.

        Parameters
        ----------
        wf_filter : callable
            It must be a callable whose first parameter
            must be called 'waveform' and its type
            annotation must match the Waveform class.
            Its return value must be annotated as a
            boolean. The waveforms that are filtered
            out are those for which
            wf_filter(waveform, *args, **kwargs)
            evaluates to False.
        *args
            For each waveform, wf, these are the
            positional arguments which are given to
            wf_filter(wf, *args, **kwargs) as *args.
        actually_filter : bool
            If False, then no changes are done to
            this waveform_set object. If True, then
            the waveforms which are filtered out
            are deleted from the self.__waveforms
            attribute of this waveform_set object.
            If so, the self.__runs,
            self.__record_numbers and the
            self.__available_channels attributes
            are updated accordingly, and the
            the self.__mean_adcs and the
            self.__mean_adcs_idcs are reset to None.
        return_the_staying_ones : bool
            If True (resp. False), then this method
            returns the indices of the waveforms which
            passed (resp. didn't pass) the filter, i.e.
            those for which the filter evaluated to
            True (resp. False).
        *kwargs
            For each waveform, wf, these are the
            keyword arguments which are given to
            wf_filter(wf, *args, **kwargs) as *kwargs

        Returns
        ----------
        output : list of int
            If return_the_staying_ones is True (resp.
            False), then this list contains the indices,
            with respect to the self.__waveforms list,
            for the waveforms, wf, for which
            wf_filter(wf, *args, **kwargs) evaluated to
            True (resp. False).
        """

        signature = inspect.signature(wf_filter)

        wuf.check_well_formedness_of_generic_waveform_function(signature)

        # Better fill the two lists during the waveform_set scan and then
        # return
        staying_ones, dumped_ones = [], []
        # the desired one, rather than filling just the dumped_ones one and
        # then computing its negative in case return_the_staying_ones is True
        for i in range(len(self.__waveforms)):
            if wf_filter(self.__waveforms[i], *args, **kwargs):
                staying_ones.append(i)
            else:
                dumped_ones.append(i)

        if actually_filter:

            # dumped_ones is increasingly ordered, so
            for idx in reversed(dumped_ones):
                # iterate in reverse order for waveform deletion
                del self.waveforms[idx]

            # If actually_filter, then we need to update
            self.__update_runs(other_runs=None)
            # the self.__runs, self.__record_numbers and
            self.__update_record_numbers(other_record_numbers=None)
            self.__update_available_channels(
                other_available_channels=None)   # self.__available_channels

            # We also need to reset the attributes regarding the mean
            self.__mean_adcs = None
            # waveform, for which some of the waveforms might have been removed
            self.__mean_adcs_idcs = None

        if return_the_staying_ones:
            return staying_ones
        else:
            return dumped_ones

    @classmethod
    def from_filtered_WaveformSet(
            cls, original_WaveformSet: 'waveform_set',
            wf_filter: Callable[..., bool],
            *args,
            **kwargs) -> 'waveform_set':
        """
        This method returns a new waveform_set object
        which contains only the waveforms from the
        given original_WaveformSet object which passed
        the given wf_filter callable, i.e. those Waveform
        objects, wf, for which
        wf_filter(wf, *args, **kwargs) evaluated to True.
        To do so, this method calls the waveform_set.filter()
        instance method of the waveform_set given to the
        'original_WaveformSet' parameter by setting the
        its 'actually_filter' parameter to True.

        Parameters
        ----------
        original_WaveformSet : waveform_set
            The waveform_set object which will be filtered
            so as to create the new waveform_set object
        wf_filter : callable
            It must be a callable whose first parameter
            must be called 'waveform' and its type
            annotation must match the Waveform class.
            Also, its return value must be annotated
            as a boolean. The well-formedness of
            the given callable is not checked by
            this method, but checked by the
            waveform_set.filter() instance method of
            the original_WaveformSet object, whose
            'wf_filter' parameter receives the input
            given to the 'wf_filter' parameter of this
            method. The waveforms which end up staying
            in the returned waveform_set object are those
            within the original_WaveformSet object,
            wf, for which wf_filter(wf, *args, **kwargs)
            evaluated to True.
        *args
            For each waveform, wf, these are the
            positional arguments which are given to
            wf_filter(wf, *args, **kwargs) as *args.
        **kwargs
            For each waveform, wf, these are the
            keyword arguments which are given to
            wf_filter(wf, *args, **kwargs) as **kwargs

        Returns
        ----------
        waveform_set
            A new waveform_set object which contains
            only the waveforms from the given
            original_WaveformSet object which passed
            the given wf_filter callable.
        """

        staying_wfs_idcs = original_WaveformSet.filter(
            wf_filter,
            *args,
            actually_filter=False,
            return_the_staying_ones=True,
            **kwargs)

        waveforms = [
            original_WaveformSet.waveforms[idx]
            for idx in staying_wfs_idcs]

        # About the waveforms that we will handle to the new
        # waveform_set object: Shall they be a deep copy? If
        # they are not, maybe some of the Waveform objects
        # that belong to both - the original and the filtered
        # waveform_set objects are not independent, but references
        # to the same Waveform objects in memory. This could be
        # an issue if we want, p.e. to run different analyses on
        # the different waveform_set objects. I.e. running an
        # analysis on the filtered waveform_set could modify the
        # analysis on the same waveform in the original
        # waveform_set. This would not be an issue, though, if we
        # want to partition the original waveform_set into disjoint
        # waveformsets, and never look back on the original
        # waveform_set, p.e. if we want to partition the original
        # waveform_set according to the endpoints. This needs to be
        # checked, because it might be an open issue.

        return cls(*waveforms)

    # waveform_set.plot_calibration_histogram() is not supported anymore,
    # and so, it may not work. ChannelWSGrid.plot() with its 'mode'
    # input parameter set to calibration already covers this feature.
    #  It is not worth the effort to fix this method for waveform_set,
    # it will be deleted soon.

    def plot_calibration_histogram(
        self, nrows: int = 1,
        # This is a quick solution a la waveform_set.plot_wfs()
        ncols: int = 1,  # which is useful to produce calibration plots in
        # self-trigger cases where a general integration window
        figure: Optional[pgo.Figure] = None,
        # can be defined. Eventually, a method like this should
        wfs_per_axes: Optional[int] = 100,
        # inspect the Analyses attribute of each waveform
        grid_of_wf_idcs: Optional[List[List[List[int]]]] = None,
        # in search for the spotted WfPeaks and their integrals.
        analysis_label: Optional[str] = None,
        bins: int = 250,
        # It's the regular range, but here it is called 'domain'
        domain: Tuple[float, float] = (-20000., 60000.),
        # to not collide with the 'range' reserved keyword
        share_x_scale: bool = False,
        share_y_scale: bool = False,
            detailed_label: bool = True) -> pgo.Figure:
        # Also, most of the code of this function is copied from
        # that of waveform_set.plot_wfs(). A way to avoid this is
        # to incorporate the histogram functionality into the
        # waveform_set.plot_wfs() method, but I don't think that's
        # a good idea, though. Maybe we should find a way to
        # encapsulate the shared code into an static method.
        """
        This method returns a plotly.graph_objects.Figure
        with a nrows x ncols grid of axes, with plots of
        the calibration histograms which include a subset
        of the waveforms in this waveform_set object.

        Parameters
        ----------
        nrows (resp. ncols) : int
            Number of rows (resp. columns) of the returned
            grid of axes.
        figure : plotly.graph_objects.Figure
            If it is not None, then it must have been
            generated using plotly.subplots.make_subplots()
            (even if nrows and ncols equal 1). It is the
            caller's responsibility to ensure this.
            If that's the case, then this method adds the
            plots to this figure and eventually returns
            it. In such case, the number of rows (resp.
            columns) in such figure must match the 'nrows'
            (resp. 'ncols') parameter.
        wfs_per_axes : int
            If it is not None, then the argument given to
            'grid_of_wf_idcs' will be ignored. In this case,
            the number of waveforms considered for each
            axes is wfs_per_axes. P.e. for wfs_per_axes
            equal to 100, the axes at the first row and
            first column contains a calibration histogram
            with 100 entries, each of which comes from the
            integral of the first 100 waveforms in this
            waveform_set object. The axes in the first
            row and second column will consider the
            following 100 waveforms, and so on.
        grid_of_wf_idcs : list of list of list of int
            This list must contain nrows lists, each of
            which must contain ncols lists of integers.
            grid_of_wf_idcs[i][j] gives the indices of the
            waveforms, with respect to this waveform_set, whose
            integrals will be part of the calibration
            histogram which is located at the i-th row
            and j-th column.
        analysis_label : str
            This parameter gives the key for the WfAna
            object within the Analyses attribute of each
            considered waveform from where to take the
            integral value to add to the calibration
            histogram. Namely, if such WfAna object is
            x, then x.Result.Integral is the considered
            integral. If 'analysis_label' is None,
            then the last analysis added to
            the Analyses attribute will be the used one.
        bins : int
            A positive integer giving the number of bins
            in each histogram
        domain : tuple of float
            It must contain two floats, so that domain[0]
            is smaller than domain[1]. It is the range
            of each histogram.
        share_x_scale (resp. share_y_scale) : bool
            If True, the x-axis (resp. y-axis) scale will be
            shared among all the subplots.
        detailed_label : bool
            Whether to show the iterator values of the two
            first available waveforms (which contribute to
            the calibration histogram) in the label of
            each histogram.

        Returns
        ----------
        figure : plotly.graph_objects.Figure
            The figure with the grid plot of the waveforms
        """

        if nrows < 1 or ncols < 1:
            raise Exception(generate_exception_message(
                1,
                'waveform_set.plot_calibration_histogram()',
                'The number of rows and columns must be positive.'))
        fFigureIsGiven = False
        if figure is not None:

            try:
                fig_rows, fig_cols = figure._get_subplot_rows_columns()
                # Returns two range objects
                fig_rows, fig_cols = list(fig_rows)[-1], list(fig_cols)[-1]

            except Exception:
                # Happens if figure was not created using
                # plotly.subplots.make_subplots

                raise Exception(generate_exception_message(
                    2,
                    'waveform_set.plot_calibration_histogram()',
                    'The given figure is not a subplot grid.'))
            if fig_rows != nrows or fig_cols != ncols:

                raise Exception(generate_exception_message(
                    3,
                    'waveform_set.plot_calibration_histogram()',
                    "The number of rows and columns in the given figure"
                    f" ({fig_rows}, {fig_cols}) must match the nrows"
                    f" ({nrows}) and ncols ({ncols}) parameters."))
            fFigureIsGiven = True

        grid_of_wf_idcs_ = None         # Logically useless

        if wfs_per_axes is not None:    # wfs_per_axes is defined

            if wfs_per_axes < 1:
                raise Exception(generate_exception_message(
                    4,
                    'waveform_set.plot_calibration_histogram()',
                    'The number of waveforms per axes must be positive.'))

            grid_of_wf_idcs_ = self.get_map_of_wf_idcs(
                nrows,
                ncols,
                wfs_per_axes=wfs_per_axes)

        elif grid_of_wf_idcs is None:   # Nor wf_per_axes, nor
            # grid_of_wf_idcs are defined

            raise Exception(generate_exception_message(
                5,
                'waveform_set.plot_calibration_histogram()',
                "The 'grid_of_wf_idcs' parameter must be defined"
                " if wfs_per_axes is not."))

        elif not map_.list_of_lists_is_well_formed(
                grid_of_wf_idcs,    # wf_per_axes is not defined,
                nrows,              # but grid_of_wf_idcs is, but
                ncols):             # it is not well-formed
            raise Exception(generate_exception_message(
                6,
                'waveform_set.plot_calibration_histogram()',
                "The given grid_of_wf_idcs is not well-formed according "
                f"to nrows ({nrows}) and ncols ({ncols})."))
        else:   # wf_per_axes is not defined,
            # but grid_of_wf_idcs is,
            # and it is well-formed

            grid_of_wf_idcs_ = grid_of_wf_idcs

        if bins < 1:
            raise Exception(generate_exception_message(
                7,
                'waveform_set.plot_calibration_histogram()',
                f"The given number of bins ({bins}) is not positive."))

        if domain[0] >= domain[1]:
            raise Exception(generate_exception_message(
                8,
                'waveform_set.plot_calibration_histogram()',
                f"The given domain ({domain}) is not well-formed."))
        if not fFigureIsGiven:

            figure_ = psu.make_subplots(rows=nrows,
                                        cols=ncols)
        else:
            figure_ = figure

        waveform_set.update_shared_axes_status(
            figure_,                    # An alternative way is to specify
            # shared_xaxes=True (or share_yaxes=True)
            share_x=share_x_scale,
            # in psu.make_subplots(), but, for us,
            share_y=share_y_scale)
        # that alternative is only doable for
        # the case where the given 'figure'
        # parameter is None.

        step = (domain[1] - domain[0]) / bins

        for i in range(nrows):
            for j in range(ncols):
                if len(grid_of_wf_idcs_[i][j]) > 0:

                    aux_name = f"{len(grid_of_wf_idcs_[i][j])} Wf(s)"
                    if detailed_label:
                        aux_name += f": [ {
                            waveform_set.get_string_of_first_n_integers_if_available(
                                grid_of_wf_idcs_[i][j], queried_no=2)}]"

                    data, _ = wun.histogram1d(
                        np.array([self.waveforms[idc].get_analysis(
                            analysis_label).Result.Integral
                            for idc in grid_of_wf_idcs_[i][j]]),
                        # Trying to grab the WfAna object
                        bins,  # waveform by waveform using
                        domain,  # waveform_adcs.get_analysis()
                        keep_track_of_idcs=False)
                    # might be slow. Find a different
                    # solution if this becomes a
                    # a problem at some point.
                    figure.add_trace(
                        pgo.Scatter(
                            x=np.linspace(
                                domain[0] + (step / 2.0),
                                domain[1] - (step / 2.0),
                                num=bins,
                                endpoint=True),
                            y=data,
                            mode='lines',
                            line=dict(
                                color='black',
                                width=0.5),
                            name=f"({i+1},{j+1}) - C. H. of " + aux_name,),
                        row=i + 1,
                        col=j + 1)
                else:

                    waveform_set.__add_no_data_annotation(
                        figure_,
                        i + 1,
                        j + 1)
        return figure_

    def merge(self, other: 'waveform_set') -> None:
        """
        This method merges the given other waveform_set
        object into this waveform_set object. For every
        waveform in the given other waveform_set object,
        it is appended to the list of waveforms of this
        waveform_set object. The self.__runs,
        self.__record_numbers and self.__available_channels
        are updated accordingly. The self.__mean_adcs and
        self.__mean_adcs_idcs are reset to None.

        Parameters
        ----------
        other : waveform_set
            The waveform_set object to be merged into this
            waveform_set object. The points_per_wf attribute
            of the given waveform_set object must be equal
            to the points_per_wf attribute of this waveform_set
            object. Otherwise, an exception is raised.

        Returns
        ----------
        None
        """

        if other.points_per_wf != self.points_per_wf:
            raise Exception(generate_exception_message(
                1,
                'waveform_set.merge()',
                "The given waveform_set object has waveforms with lengths"
                f" ({other.points_per_wf}) different to the ones in this"
                f" waveform_set object ({self.points_per_wf})."))
        for wf in other.waveforms:
            self.__waveforms.append(wf)

        self.__update_runs(other_runs=other.runs)
        self.__update_record_numbers(other_record_numbers=other.record_numbers)
        self.__update_available_channels(
            other_available_channels=other.available_channels)

        self.__mean_adcs = None
        self.__mean_adcs_idcs = None

        return
