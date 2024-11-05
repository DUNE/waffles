import inspect
import copy

import numpy as np
from tqdm import tqdm
from typing import List, Dict, Callable, Optional

from waffles.data_classes.WaveformAdcs import WaveformAdcs
from waffles.data_classes.WfAna import WfAna
from waffles.data_classes.IPDict import IPDict

import waffles.utils.filtering_utils as wuf

from waffles.Exceptions import GenerateExceptionMessage


class WaveformSet:
    """This class implements a set of waveforms.

    Attributes
    ----------
    waveforms: list of Waveform objects
        waveforms[i] gives the i-th Waveform in the set.
    points_per_wf: int
        Number of entries for the adcs attribute of
        each Waveform object in this WaveformSet object.
    runs: set of int
        It contains the run number of any run for which
        there is at least one Waveform in the set.
    record_numbers: dictionary of sets
        It is a dictionary whose keys are runs (int) and
        its values are sets of record numbers (set of int).
        If there is at least one Waveform object within
        this WaveformSet which was acquired during run n,
        then n belongs to record_numbers.keys(). record_numbers[n]
        is a set of record numbers for run n. If there is at
        least one Waveform acquired during run n whose
        record_number is m, then m belongs to record_numbers[n].
    available_channels: dictionary of dictionaries of sets
        It is a dictionary whose keys are run numbers (int),
        so that if there is at least one Waveform in the set
        which was acquired during run n, then n belongs to
        available_channels.keys(). available_channels[n] is a
        dictionary whose keys are endpoints (int) and its
        values are sets of channels (set of int). If there
        is at least one Waveform object within this WaveformSet
        which was acquired during run n and which comes from
        endpoint m, then m belongs to available_channels[n].keys().
        available_channels[n][m] is a set of channels for
        endpoint m during run n. If there is at least one
        Waveform for run n, endpoint m and channel p, then p
        belongs to available_channels[n][m].
    mean_adcs: WaveformAdcs
        The mean of the adcs arrays for every Waveform
        or a subset of waveforms in this WaveformSet. It
        is a WaveformAdcs object whose time_step_ns
        attribute is assumed to match that of the first
        Waveform which was used in the average sum.
        Its adcs attribute contains points_per_wf entries,
        so that mean_adcs.adcs[i] is the mean of
        self.waveforms[j].adcs[i] for every value
        of j or a subset of values of j, within
        [0, len(self.__waveforms) - 1]. It is not
        computed by default. I.e. if self.mean_adcs
        equals to None, it should be interpreted as
        unavailable data. Call the 'compute_mean_waveform'
        method of this WaveformSet to compute it.
    mean_adcs_idcs: tuple of int
        It is a tuple of integers which contains the indices
        of the waveforms, with respect to this WaveformSet,
        which were used to compute the mean_adcs.adcs
        attribute. By default, it is None. I.e. if
        self.mean_adcs_idcs equals to None, it should be
        interpreted as unavailable data. Call the
        'compute_mean_waveform' method of this WaveformSet
        to compute it.

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self, *waveforms):
        """WaveformSet class initializer

        Parameters
        ----------
        waveforms: unpacked list of Waveform objects
            The waveforms that will be added to the set
        """

        # Shall we add type checks here?

        if len(waveforms) == 0:
            raise Exception(GenerateExceptionMessage(
                1,
                'WaveformSet.__init__()',
                'There must be at least one Waveform in the set.'))
        
        self.__waveforms = list(waveforms)

        if not self.check_length_homogeneity():
            raise Exception(GenerateExceptionMessage(
                2,
                'WaveformSet.__init__()',
                'The length of the given waveforms is not homogeneous.'))

        self.__points_per_wf = len(self.__waveforms[0].adcs)

        self.__runs = set()
        self.__update_runs(other_runs=None)

        self.__record_numbers = {}
        self.__update_record_numbers(other_record_numbers=None)

        self.__available_channels = {}

        # Running on an Apple M2, it took
        # ~ 52 ms to run this line for a
        # WaveformSet with 1046223 waveforms
        self.__update_available_channels(other_available_channels=None)

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
        """This method returns a set which contains every endpoint
        for which there is at least one Waveform in this
        WaveformSet object.

        Returns
        ----------
        output: set of int
        """

        output = set()

        for run in self.__available_channels.keys():
            for endpoint in self.__available_channels[run].keys():
                output.add(endpoint)

        return output

    def get_run_collapsed_available_channels(self) -> dict:
        """This method returns a dictionary of sets of integers,
        say output, whose keys are endpoints. If there is
        at least one Waveform within this set that comes from
        endpoint n, then n belongs to output.keys(). output[n]
        is a set of integers, so that if there is at least a
        Waveform coming from endpoint n and channel m, then m
        belongs to output[n].

        Returns
        ----------
        output: dictionary of sets
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
        """This method returns True if the adcs attribute
        of every Waveform object in this WaveformSet
        has the same length. It returns False if else.
        In order to call this method, there must be at
        least one Waveform in the set.

        Returns
        ----------
        bool
        """

        if len(self.__waveforms) == 0:
            raise Exception(GenerateExceptionMessage(
                1,
                'WaveformSet.check_length_homogeneity()',
                'There must be at least one Waveform in the set.'))
        
        length = len(self.__waveforms[0].adcs)
        for i in range(1, len(self.__waveforms)):
            if len(self.__waveforms[i].adcs) != length:
                return False
        return True

    def __update_runs(self, other_runs: Optional[set] = None) -> None:
        """This method is not intended to be called by the user.
        This method updates the self.__runs attribute of this
        object. Its behaviour is different depending on whether
        the 'other_runs' parameter is None or not. Check its
        documentation for more information.

        Parameters
        ----------
        other_runs: set of int
            If it is None, then this method clears the self.__runs
            attribute of this object and then iterates through the
            whole WaveformSet to fill such attribute according to
            the waveforms which are currently present in this
            WaveformSet object. If the 'other_runs' parameter is
            defined, then it must be a set of integers, as expected
            for the runs attribute of a WaveformSet object. In this
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
        """This method is not intended for user usage.
        This method must only be called by the
        WaveformSet.__update_runs() method. It clears
        the self.__runs attribute of this object and
        then iterates through the whole WaveformSet to
        fill such attribute according to the waveforms
        which are currently present in this WaveformSet
        object.
        """

        self.__runs.clear()

        for wf in self.__waveforms:
            self.__runs.add(wf.run_number)

        return

    def __update_record_numbers(
        self,
        other_record_numbers: Optional[Dict[int, set]] = None
    ) -> None:
        """This method is not intended to be called by the user.
        This method updates the self.__record_numbers attribute
        of this object. Its behaviour is different depending on
        whether the 'other_record_numbers' parameter is None or
        not. Check its documentation for more information.

        Parameters
        ----------
        other_record_numbers: dictionary of sets of int
            If it is None, then this method clears the
            self.__record_numbers attribute of this object
            and then iterates through the whole WaveformSet
            to fill such attribute according to the waveforms
            which are currently present in this WaveformSet.
            If the 'other_record_numbers' parameter is defined,
            then it must be a dictionary of sets of integers,
            as expected for the record_numbers attribute of a
            WaveformSet object. In this case, the information
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
                    
                    # If this run is present in both, this WaveformSet and
                    # the incoming one, then carefully merge the information

                    self.__record_numbers[run] = self.__record_numbers[
                        run].union(other_record_numbers[run])

                else:
                    # If this run is present in the incoming WaveformSet but 
                    # not in self, then simply get the information from the
                    # incoming WaveformSet as a block

                    self.__record_numbers[run] = other_record_numbers[run]
        return

    def __reset_record_numbers(self) -> None:
        """This method is not intended for user usage.
        This method must only be called by the
        WaveformSet.__update_record_numbers() method. It clears
        the self.__record_numbers attribute of this object and
        then iterates through the whole WaveformSet to fill such
        attribute according to the waveforms which are currently
        present in this WaveformSet object.
        """

        self.__record_numbers.clear()

        for wf in self.__waveforms:
            try:
                self.__record_numbers[wf.run_number].add(wf.record_number)
            except KeyError:
                self.__record_numbers[wf.run_number] = set()
                self.__record_numbers[wf.run_number].add(wf.record_number)
        return

    def __update_available_channels(
        self,
        other_available_channels: Optional[Dict[int, Dict[int, set]]] = None
    ) -> None:
        """This method is not intended to be called by the user.
        This method updates the self.__available_channels
        attribute of this object. Its behaviour is different
        depending on whether the 'other_available_channels'
        parameter is None or not. Check its documentation for
        more information.

        Parameters
        ----------
        other_available_channels: dictionary of dictionaries of sets
            If it is None, then this method clears the
            self.__available_channels attribute of this object
            and then iterates through the whole WaveformSet
            to fill such attribute according to the waveforms
            which are currently present in this WaveformSet.
            If the 'other_available_channels' parameter is
            defined, then it must be a dictionary of dictionaries
            of sets of integers, as expected for the
            available_channels attribute of a WaveformSet object.
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
                    
                    # If this run is present in both, this WaveformSet and
                    # the incoming one, then carefully merge the information

                    for endpoint in other_available_channels[run].keys():
                        if endpoint in self.__available_channels[run].keys():

                            # If this endpoint for this run is present
                            # in both Waveform sets, then carefully
                            # merge the information.

                            self.__available_channels[run][
                                endpoint] = self.__available_channels[
                                    run][endpoint].union(
                                        other_available_channels[run][endpoint])
                        else:
                            
                            # If this endpoint for this run is present in the
                            # incoming WaveformSet but not in
                            # self, then simply get the information from the
                            # incoming WaveformSet as a block

                            self.__available_channels[run][
                                endpoint] = other_available_channels[run][
                                    endpoint]

                else:

                    # If this run is present in the incoming
                    # WaveformSet but not in self, then
                    # simply get the information from the
                    # incoming WaveformSet as a block

                    self.__available_channels[
                        run] = other_available_channels[run]
        return

    def __reset_available_channels(self) -> None:
        """This method is not intended for user usage.
        This method must only be called by the
        WaveformSet.__update_available_channels() method. It clears
        the self.__available_channels attribute of this object and
        then iterates through the whole WaveformSet to fill such
        attribute according to the waveforms which are currently
        present in this WaveformSet object.
        """

        self.__available_channels.clear()

        for wf in self.__waveforms:
            try:
                aux = self.__available_channels[wf.run_number]

                try:
                    aux[wf.endpoint].add(wf.channel)

                except KeyError:
                    aux[wf.endpoint] = set()
                    aux[wf.endpoint].add(wf.channel)

            except KeyError:
                self.__available_channels[wf.run_number] = {}
                self.__available_channels[wf.run_number][wf.endpoint] = set()
                self.__available_channels[wf.run_number][wf.endpoint].add(
                    wf.channel)
        return

    def analyse(
        self,
        label: str,
        analysis_class: type,
        input_parameters: IPDict,
        *args,
        analysis_kwargs: dict = {},
        checks_kwargs: dict = {},
        overwrite: bool = False
    ) -> dict:
        """For each Waveform in this WaveformSet, this method
        calls its 'analyse' method passing to it the parameters
        given to this method. In turn, Waveform.analyse()
        (actually WaveformAdcs.analyse()) creates an object
        of type analysis_class (which must be a class which
        inherits from the WfAna class) and runs its analyse()
        method on the current Waveform object. The created
        analysis object is stored in the analyses attribute
        of the Waveform object, using the given label parameter
        as its key. This method returns a dictionary, say x,
        where the keys are indices of the waveforms in this
        WaveformSet, so that x[i] is the output of the
        self.__waveforms[i].analyse() method.

        Parameters
        ----------
        label: str
            For every analysed Waveform, this is the key
            for the new WfAna (or derived) object within its
            analyses attribute.
        analysis_class: type
            Class (type) which must inherit from WfAna. The
            given class must have an analyse() method which
            takes a WaveformAdcs object as its first argument
            (after self).
        input_parameters: IPDict
            The input parameters which will be passed to the
            analysis_class initializer by the WaveformAdcs.analyse()
            method, for each analysed Waveform. It is the
            user's responsibility to ensure that
            input_parameters contain the required information
            to initialize the analysis_class object, and that
            it is well-defined.
        *args
            Additional positional arguments which are given
            to the Waveform.analyse() (actually WaveformAdcs.analyse())
            for each analysed Waveform, which in turn,
            are given to the analyse() method of analysis_class.
        analysis_kwargs: dict
            Additional keyword arguments which are given
            to the Waveform.analyse() (actually WaveformAdcs.analyse())
            for each analysed Waveform, which in turn,
            are given to the analyse() method of analysis_class.
        checks_kwargs: dict
            Additional keyword arguments which are given
            to the check_input_parameters() method of
            the analysis_class class.
        overwrite: bool
            If True, for every analysed Waveform, its
            'analyse' method will overwrite any existing
            WfAna (or derived) object with the same label
            (key) within its analyses attribute.

        Returns
        ----------
        output: dict
            output[i] gives the output of
            self.__waveforms[i].analyse(...), which is a
            dictionary containing any additional information
            of the analysis which was performed over the
            i-th Waveform of this WaveformSet. Such
            dictionary is empty if the analyser method gives
            no additional information.
        """

        if not issubclass(analysis_class, WfAna):
            raise Exception(GenerateExceptionMessage(
                1,
                'WaveformSet.analyse()',
                'The analysis class must be derived from the WfAna class.'))

        analysis_class.check_input_parameters(
            input_parameters,
            **checks_kwargs)

        # analysis_class may have not implemented an abstract method
        # of WfAna, p.e. analyse(), and still produce no errors until
        # an object of such class is instantiated. If that's the case,
        # 'signature' is actually the signature of WfAna.analyse() and
        # inspecting it is dumb, since we would not be checking the
        # signature of analysis_class.analyse(). This is not a big deal,
        # though, because the user will, anyway, encounter a descriptive
        # error when trying to run the analysis for the first Waveform,
        # where an object of analysis_class is instantiated.
        signature = inspect.signature(analysis_class.analyse)

        try:

            aux = list(signature.parameters.keys())[1]
            if aux != 'waveform':  # The first parameter is 'self'

                raise Exception(GenerateExceptionMessage(
                    2,
                    "WaveformSet.analyse()",
                    "The name of the first parameter of the 'analyse()'"
                    f" method ('{aux}') of the given analysis class"
                    f" ({analysis_class.__name__}) must be 'waveform'."))

            if signature.parameters['waveform'].annotation != WaveformAdcs:
                raise Exception(GenerateExceptionMessage(
                    3,
                    "WaveformSet.analyse()",
                    "The 'waveform' parameter of the 'analyse()' "
                    "method of the given analysis class"
                    f" ({analysis_class.__name__}) must be hinted as a "
                    "WaveformAdcs object."))
        except IndexError:
            raise Exception(GenerateExceptionMessage(
                4,
                "WaveformSet.analyse()",
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
        self, 
        *args,
        wf_idcs: Optional[List[int]] = None,
        wf_selector: Optional[Callable[..., bool]] = None,
        **kwargs
    ) -> WaveformAdcs:
        """If wf_idcs is None and wf_selector is None,
        then this method creates a WaveformAdcs
        object whose adcs attribute is the mean
        of the adcs arrays for every Waveform in
        this WaveformSet. If wf_idcs is not None,
        then such mean is computed using the adcs
        arrays of the waveforms whose iterator
        values, with respect to this WaveformSet,
        are given in wf_idcs. If wf_idcs is None
        but wf_selector is not None, then such
        mean is computed using the adcs arrays
        of the waveforms, wf, within this
        WaveformSet for which
        wf_selector(wf, *args, **kwargs) evaluates
        to True. In any case, the time_step_ns
        attribute of the newly created WaveformAdcs
        object assumed to match that of the first
        Waveform which was used in the average sum.

        In any case, the resulting WaveformAdcs
        object is assigned to the
        self.__mean_adcs attribute. The
        self.__mean_adcs_idcs attribute is also
        updated with a tuple of the indices of the
        waveforms which were used to compute the
        mean WaveformAdcs. Finally, this method
        returns the averaged WaveformAdcs object.

        Parameters
        ----------
        *args
            These arguments only make a difference if
            the 'wf_idcs' parameter is None and the
            'wf_selector' parameter is suitable defined.
            For each Waveform, wf, these are the
            positional arguments which are given to
            wf_selector(wf, *args, **kwargs) as *args.
        wf_idcs: list of int
            If it is not None, then it must be a list
            of integers which must be a valid iterator
            value for the __waveforms attribute of this
            WaveformSet. I.e. any integer i within such
            list must satisfy
            0 <= i <= len(self.__waveforms) - 1. Any
            integer which does not satisfy this condition
            is ignored. These integers give the waveforms
            which are averaged.
        wf_selector: callable
            This parameter only makes a difference if
            the 'wf_idcs' parameter is None. If that's
            the case, and 'wf_selector' is not None, then
            it must be a callable whose first parameter
            must be called 'waveform' and its type
            annotation must match the Waveform class.
            Its return value must be annotated as a
            boolean. In this case, the mean Waveform
            is averaged over those waveforms, wf, for
            which wf_selector(wf, *args, **kwargs)
            evaluates to True.
        *kwargs
            These keyword arguments only make a
            difference if the 'wf_idcs' parameter is
            None and the 'wf_selector' parameter is
            suitable defined. For each Waveform, wf,
            these are the keyword arguments which are
            given to wf_selector(wf, *args, **kwargs)
            as **kwargs.

        Returns
        ----------
        output: np.ndarray
            The averaged adcs array
        """

        if len(self.__waveforms) == 0:
            raise Exception(GenerateExceptionMessage(
                1,
                'WaveformSet.compute_mean_waveform()',
                'There are no waveforms in this WaveformSet object.'))
        if wf_idcs is None and wf_selector is None:

            # Average over every Waveform in this WaveformSet
            output = self.__compute_mean_waveform_of_every_waveform()

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

                    # Just make sure that there
                    # is at least one valid
                    # iterator value in the given list

                    fWfIdcsIsWellFormed = True
                    break                       
                
            if not fWfIdcsIsWellFormed:
                raise Exception(GenerateExceptionMessage(
                    2,
                    'WaveformSet.compute_mean_waveform()',
                    'The given list of Waveform indices is empty or it does '
                    'not contain even one valid iterator value in the given '
                    'list. I.e. there are no waveforms to average.'))

            # In this case we also need to remove indices
            # redundancy (if any) before giving wf_idcs to
            # WaveformSet.__compute_mean_waveform_of_given_waveforms.
            # This is a open issue for now.

            output = self.__compute_mean_waveform_of_given_waveforms(wf_idcs)

        return output

    def __compute_mean_waveform_of_every_waveform(self) -> WaveformAdcs:
        """This method should only be called by the
        WaveformSet.compute_mean_waveform() method,
        where any necessary well-formedness checks
        have already been performed. It is called by
        such method in the case where both the 'wf_idcs'
        and the 'wf_selector' input parameters are
        None. This method sets the self.__mean_adcs
        and self.__mean_adcs_idcs attributes according
        to the WaveformSet.compute_mean_waveform()
        method documentation. It also returns the
        averaged WaveformAdcs object. Refer to the
        WaveformSet.compute_mean_waveform() method
        documentation for more information.

        Returns
        ----------
        output: np.ndarray
            The averaged adcs array
        """

        # WaveformSet.compute_mean_waveform()
        # has already checked that there is at
        # least one Waveform in this WaveformSet
        aux = copy.deepcopy(self.waveforms[0].adcs)

        for i in range(1, len(self.__waveforms)):
            aux += self.waveforms[i].adcs

        output = WaveformAdcs(
            self.__waveforms[0].time_step_ns,
            aux/len(self.__waveforms),
            time_offset=0)

        self.__mean_adcs = output
        self.__mean_adcs_idcs = tuple(range(len(self.__waveforms)))

        return output

    def __compute_mean_waveform_with_selector(
        self, wf_selector: Callable[..., bool],
        *args,
        **kwargs
    ) -> WaveformAdcs:
        """This method should only be called by the
        WaveformSet.compute_mean_waveform() method,
        where any necessary well-formedness checks
        have already been performed. It is called by
        such method in the case where the 'wf_idcs'
        parameter is None and the 'wf_selector'
        parameter is suitably defined. This method
        sets the self.__mean_adcs and
        self.__mean_adcs_idcs attributes according
        to the WaveformSet.compute_mean_waveform()
        method documentation. It also returns the
        averaged WaveformAdcs object. Refer to the
        WaveformSet.compute_mean_waveform() method
        documentation for more information.

        Parameters
        ----------
        wf_selector: callable
        *args
        **kwargs

        Returns
        ----------
        output: np.ndarray
            The averaged adcs array
        """

        added_wvfs = []

        aux = np.zeros((self.__points_per_wf,))

        for i in range(len(self.__waveforms)):
            if wf_selector(self.__waveforms[i], *args, **kwargs):
                aux += self.__waveforms[i].adcs
                added_wvfs.append(i)

        if len(added_wvfs) == 0:
            raise Exception(GenerateExceptionMessage(
                1,
                'WaveformSet.__compute_mean_waveform_with_selector()',
                'No Waveform in this WaveformSet object '
                'passed the given selector.'))

        output = WaveformAdcs(
            self.__waveforms[added_wvfs[0]].time_step_ns,
            aux/len(added_wvfs),
            time_offset=0)

        self.__mean_adcs = output
        self.__mean_adcs_idcs = tuple(added_wvfs)

        return output

    def __compute_mean_waveform_of_given_waveforms(
        self, 
        wf_idcs: List[int]
    ) -> WaveformAdcs:
        """This method should only be called by the
        WaveformSet.compute_mean_waveform() method,
        where any necessary well-formedness checks
        have already been performed. It is called by
        such method in the case where the 'wf_idcs'
        parameter is not None, regardless the input
        given to the 'wf_selector' parameter. This
        method sets the self.__mean_adcs and
        self.__mean_adcs_idcs attributes according
        to the WaveformSet.compute_mean_waveform()
        method documentation. It also returns the
        averaged WaveformAdcs object. Refer to the
        WaveformSet.compute_mean_waveform() method
        documentation for more information.

        Parameters
        ----------
        wf_idcs: list of int

        Returns
        ----------
        output: np.ndarray
            The averaged adcs array
        """

        added_wvfs = []

        aux = np.zeros((self.__points_per_wf,))

        for idx in wf_idcs:
            try:
                # WaveformSet.compute_mean_waveform() only checked that there
                # is at least one valid iterator value, but we need to handle
                # the case where there are invalid iterator values

                aux += self.__waveforms[idx].adcs
            except IndexError:
                continue
                # Ignore the invalid iterator values as specified in the
                # WaveformSet.compute_mean_waveform() method documentation
            else:
                added_wvfs.append(idx)

        output = WaveformAdcs(
            self.__waveforms[added_wvfs[0]].time_step_ns,
            # len(added_wvfs) must be at least 1.
            # This was already checked by
            # WaveformSet.compute_mean_waveform()
            aux/len(added_wvfs),
            time_offset=0)
        
        self.__mean_adcs = output
        self.__mean_adcs_idcs = tuple(added_wvfs)

        return output

    def is_valid_iterator_value(
        self, 
        iterator_value: int
    ) -> bool:
        """This method returns True if
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
        self, 
        wf_filter: Callable[..., bool],
        *args,
        actually_filter: bool = False,
        return_the_staying_ones: bool = True,
        show_progress: bool = False,
        **kwargs
    ) -> List[int]:
        """This method filters the waveforms in this WaveformSet
        using the given wf_filter callable. I.e. for each
        Waveform object, wf, in this WaveformSet, it runs
        wf_filter(wf, *args, **kwargs). This method returns
        a list of indices for the waveforms which got the
        same result from the filter.

        Parameters
        ----------
        wf_filter: callable
            It must be a callable whose first parameter
            must be called 'waveform' and its type
            annotation must match the Waveform class.
            Its return value must be annotated as a
            boolean. The waveforms that are filtered
            out are those for which
            wf_filter(Waveform, *args, **kwargs)
            evaluates to False.
        *args
            For each Waveform, wf, these are the
            positional arguments which are given to
            wf_filter(wf, *args, **kwargs) as *args.
        actually_filter: bool
            If False, then no changes are done to
            this WaveformSet object. If True, then
            the waveforms which are filtered out
            are deleted from the self.__waveforms
            attribute of this WaveformSet object.
            If so, the self.__runs,
            self.__record_numbers and the
            self.__available_channels attributes
            are updated accordingly, and the
            the self.__mean_adcs and the
            self.__mean_adcs_idcs are reset to None.
        return_the_staying_ones: bool
            If True (resp. False), then this method
            returns the indices of the waveforms which
            passed (resp. didn't pass) the filter, i.e.
            those for which the filter evaluated to
            True (resp. False).
        show_progress: bool
            If True, will show tqdm progress bar
        *kwargs
            For each Waveform, wf, these are the
            keyword arguments which are given to
            wf_filter(wf, *args, **kwargs) as *kwargs

        Returns
        ----------
        output: list of int
            If return_the_staying_ones is True (resp.
            False), then this list contains the indices,
            with respect to the self.__waveforms list,
            for the waveforms, wf, for which
            wf_filter(wf, *args, **kwargs) evaluated to
            True (resp. False).
        """

        signature = inspect.signature(wf_filter)

        wuf.check_well_formedness_of_generic_waveform_function(signature)

        # Better fill the two lists during the WaveformSet scan and then return
        # the desired one, rather than filling just the dumped_ones one and then 
        # computing its negative in case return_the_staying_ones is True

        staying_ones, dumped_ones = [], []

        for i in tqdm(range(len(self.__waveforms)), disable=not show_progress):
            if wf_filter(self.__waveforms[i], *args, **kwargs):
                staying_ones.append(i)
            else:
                dumped_ones.append(i)

        if actually_filter:

            # dumped_ones is increasingly ordered, so
            # iterate in reverse order for Waveform deletion
            for idx in reversed(dumped_ones):    
                del self.waveforms[idx]

            # If actually_filter, then we need to update
            # the self.__runs, self.__record_numbers and
            # self.__available_channels
            self.__update_runs(other_runs=None)
            self.__update_record_numbers(other_record_numbers=None)
            self.__update_available_channels(other_available_channels=None)   

            # We also need to reset the attributes regarding the mean
            # Waveform, for which some of the waveforms might have been removed
            self.__mean_adcs = None
            self.__mean_adcs_idcs = None

        if return_the_staying_ones:
            return staying_ones
        else:
            return dumped_ones

    @classmethod
    def from_filtered_WaveformSet(
        cls, 
        original_WaveformSet: 'WaveformSet',
        wf_filter: Callable[..., bool],
        *args,
        **kwargs
    ) -> 'WaveformSet':
        """This method returns a new WaveformSet object
        which contains only the waveforms from the
        given original_WaveformSet object which passed
        the given wf_filter callable, i.e. those Waveform
        objects, wf, for which
        wf_filter(wf, *args, **kwargs) evaluated to True.
        To do so, this method calls the WaveformSet.filter()
        instance method of the WaveformSet given to the
        'original_WaveformSet' parameter by setting the
        its 'actually_filter' parameter to True.

        Parameters
        ----------
        original_WaveformSet: WaveformSet
            The WaveformSet object which will be filtered
            so as to create the new WaveformSet object
        wf_filter: callable
            It must be a callable whose first parameter
            must be called 'waveform' and its type
            annotation must match the Waveform class.
            Also, its return value must be annotated
            as a boolean. The well-formedness of
            the given callable is not checked by
            this method, but checked by the
            WaveformSet.filter() instance method of
            the original_WaveformSet object, whose
            'wf_filter' parameter receives the input
            given to the 'wf_filter' parameter of this
            method. The waveforms which end up staying
            in the returned WaveformSet object are those
            within the original_WaveformSet object,
            wf, for which wf_filter(wf, *args, **kwargs)
            evaluated to True.
        *args
            For each Waveform, wf, these are the
            positional arguments which are given to
            wf_filter(wf, *args, **kwargs) as *args.
        **kwargs
            For each Waveform, wf, these are the
            keyword arguments which are given to
            wf_filter(wf, *args, **kwargs) as **kwargs

        Returns
        ----------
        WaveformSet
            A new WaveformSet object which contains
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
        # WaveformSet object: Shall they be a deep copy? If
        # they are not, maybe some of the Waveform objects
        # that belong to both - the original and the filtered
        # WaveformSet objects are not independent, but references
        # to the same Waveform objects in memory. This could be
        # an issue if we want, p.e. to run different analyses on
        # the different WaveformSet objects. I.e. running an
        # analysis on the filtered WaveformSet could modify the
        # analysis on the same Waveform in the original
        # WaveformSet. This would not be an issue, though, if we
        # want to partition the original WaveformSet into disjoint
        # waveformsets, and never look back on the original
        # WaveformSet, p.e. if we want to partition the original
        # WaveformSet according to the endpoints. This needs to be
        # checked, because it might be an open issue.

        return cls(*waveforms)

    def merge(self, other: 'WaveformSet') -> None:
        """This method merges the given other WaveformSet
        object into this WaveformSet object. For every
        Waveform in the given other WaveformSet object,
        it is appended to the list of waveforms of this
        WaveformSet object. The self.__runs,
        self.__record_numbers and self.__available_channels
        are updated accordingly. The self.__mean_adcs and
        self.__mean_adcs_idcs are reset to None.

        Parameters
        ----------
        other: WaveformSet
            The WaveformSet object to be merged into this
            WaveformSet object. The points_per_wf attribute
            of the given WaveformSet object must be equal
            to the points_per_wf attribute of this WaveformSet
            object. Otherwise, an exception is raised.

        Returns
        ----------
        None
        """

        if other.points_per_wf != self.points_per_wf:
            raise Exception(GenerateExceptionMessage(
                1,
                'WaveformSet.merge()',
                "The given WaveformSet object has waveforms with lengths"
                f" ({other.points_per_wf}) different to the ones in this"
                f" WaveformSet object ({self.points_per_wf})."))
        
        for wf in other.waveforms:
            self.__waveforms.append(wf)

        self.__update_runs(other_runs=other.runs)
        self.__update_record_numbers(other_record_numbers=other.record_numbers)
        self.__update_available_channels(
            other_available_channels=other.available_channels)

        self.__mean_adcs = None
        self.__mean_adcs_idcs = None

        return
