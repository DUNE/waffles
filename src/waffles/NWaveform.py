import inspect
from typing import Tuple, List, Optional
from collections import OrderedDict

import numpy as np
from plotly import graph_objects as go

from src.waffles.WfAna import WfAna

from src.waffles.WfAnaResult import WfAnaResult
from src.waffles.Exceptions import generate_exception_message

class Waveform:

    """
    This class implements a waveform.

    Attributes
    ----------
    Timestamp : int
        The timestamp value for this waveform
    TimeStep_ns : float
        The time step (in nanoseconds) for this waveform
    Adcs : unidimensional numpy array of integers
        The readout for this waveform, in # of ADCs
    RunNumber : int
        Number of the run from which this waveform was
        acquired
    Endpoint : int
        Endpoint number from which this waveform was
        acquired
    Channel : int
        Channel number for this waveform
    Analyses : OrderedDict of WfAna objects. 

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  timestamp : int,
                        time_step_ns : float,
                        adcs : np.ndarray,
                        run_number : int,
                        endpoint : int,
                        channel : int):
        
        """
        Waveform class initializer
        
        Parameters
        ----------
        timestamp : int
        time_step_ns : float
        adcs : unidimensional numpy array of integers
        run_number : int
        endpoint : int
        channel : int
        """

        ## Shall we add add type checks here?
    
        self.__timestamp = timestamp
        self.__time_step_ns = time_step_ns
        self.__adcs = adcs
        self.__run_number = run_number
        self.__endpoint = endpoint
        self.__channel = channel
        self.__analyses = OrderedDict() # Initialize the analyses 
                                        # attribute as an empty 
                                        # OrderedDict.

        ## Do we need to add trigger primitives as attributes?
    
    #Getters
    @property
    def Timestamp(self):
        return self.__timestamp
    
    @property
    def TimeStep_ns(self):
        return self.__time_step_ns
    
    @property
    def Adcs(self):
        return self.__adcs
    
    @property
    def RunNumber(self):
        return self.__run_number
    
    @property
    def Endpoint(self):
        return self.__endpoint
    
    @property
    def Channel(self):
        return self.__channel
    
    @property
    def Analyses(self):
        return self.__analyses
    
#   #Setters                       # For the moment there are no setters for 
#   @Timestamp.setter              # the attributes of Waveform. I.e. you can
#   def Timestamp(self, input):    # only set the value of its attributes
#       self.__timestamp = input     # through Waveform.__init__. Here's an example
#       return                     # of what a setter would look like, though.

    def confine_iterator_value(self, input : int) -> int:

        """
        Confines the input integer to the range [0, len(self.__adcs)-1].
        I.e returns 0 if input is negative, returns input if input belongs
        to the range [0, len(self.__adcs)-1], and returns len(self.__adcs)-1
        in any other case.

        Parameters
        ----------
        input : int

        Returns
        ----------
        int
        """
    
        if input < 0:
            return 0
        elif input < len(self.__adcs):
            return input
        else:
            return len(self.__adcs)-1

    def analyse(self,   label : str,
                        analyser_name : str,
                        baseline_limits : List[int],
                        int_ll : int = 0,
                        int_ul : Optional[int] = None,
                        *args,
                        overwrite : bool = False,
                        **kwargs) -> None:

        """
        This method creates a WfAna object and adds it to the
        self.__analyses dictionary using label as its key.
        This method grabs the analyser method from the WfAna
        class, up to the given analyser_name, runs it on this 
        Waveform object, and adds its results to the 'Result' 
        and 'Passed' attributes of the newly created WfAna object.

        Parameters
        ----------
        label : str
            Key for the new WfAna object within the self.__analyses
            OrderedDict
        analyser_name : str
            It must match the name of a WfAna method whose first            
            argument must be called 'waveform' and whose type           # The only way to import the Waveform class in WfAna without having         # This would not be a problem (and we would not    
            annotation must match the Waveform class or the             # a circular import is to use the typing.TYPE_CHECKING variable, which      # need to grab the analyser method using an 
            'Waveform' string literal. Such method should also          # is only defined for type-checking runs. As a consequence, the type        # string and getattr) if the analyser methods were
            have a defined return-annotation which must match           # annotation should be an string, which the type-checking software          # defined as Waveform methods or in a separate module.
            Tuple[WfAnaResult, bool]. It is the caller's                # successfully associates to the class itself, but which is detected        # There might be other downsizes to it such as the
            responsibility to check such conditions for this            # as so (a string) by inspect.signature().                                  # accesibility to WfAna attributes.
            parameter. No checks are performed here for this
            input.
        baseline_limits : list of int                                   
            Given to the 'baseline_limits' parameter of                 
            WfAna.__init__. It must have an even number
            of integers which must meet 
            baseline_limits[i] < baseline_limits[i+1] for
            all i. The points which are used for 
            baseline calculation are 
            self.__adcs[baseline_limits[2*i]:baseline_limits[2*i+1]],
            with i = 0,1,...,(len(baseline_limits)/2)-1. 
            The upper limits are exclusive. It is the
            caller's responsibility to ensure the
            well-formedness of this input. No checks are
            performed here for 'baseline_limits'.
        int_ll (resp. int_ul): int
            Given to the 'int_ll' (resp. 'int_ul') parameter of
            WfAna.__init__. Iterator value for the first (resp. 
            last) point of self.Adcs that falls into the 
            integration window. int_ll must be smaller than 
            int_ul. These limits are inclusive. If they are 
            not defined, then the whole self.Adcs is considered.
            It is the caller's responsibility to ensure the 
            well-formedness of this input. No checks are
            performed here for this parameter.
        *args
            Positional arguments which are given to the 
            analyser method.
        overwrite : bool
            If True, the method will overwrite any existing
            WfAna object with the same label (key) within
            self.__analyses.
        **kwargs
            Keyword arguments which are given to the analyser
            method.

        Returns
        ----------
        None
        """

        if label in self.__analyses.keys() and not overwrite:
            raise Exception(generate_exception_message( 1,
                                                        'Waveform.analyse()',
                                                        f"There is already an analysis with label '{label}'. If you want to overwrite it, set the 'overwrite' parameter to True."))
        else:

            ## *DISCLAIMER: The following two 'if' statements might make the run time go 
            ## prohibitively high when running analyses sequentially over a large WaveformSet. 
            ## If that's the case, these checks might be implemented at the WaveformSet level, 
            ## or simply removed.

            aux = WfAna(baseline_limits,
                        int_ll,
                        int_ul)
            try:
                analyser = getattr(aux, analyser_name)
            except AttributeError:
                raise Exception(generate_exception_message( 4,
                                                            'Waveform.analyse()',
                                                            f"The analyser method '{analyser_name}' does not exist in the WfAna class."))
            try:
                signature = inspect.signature(analyser)
            except TypeError:
                raise Exception(generate_exception_message( 5,
                                                            'Waveform.analyse()',
                                                            f"'{analyser_name}' does not match a callable attribute of WfAna."))
            try:

                ## DISCLAIMER: Same problem here for the following
                ## three 'if' statements as for the disclaimer above.

                if list(signature.parameters.keys())[0] != 'waveform':
                    raise Exception(generate_exception_message( 6,
                                                                "Waveform.analyse",
                                                                "The name of the first parameter of the given analyser method must be 'waveform'."))
                
                if signature.parameters['waveform'].annotation not in ['Waveform', Waveform]:
                    raise Exception(generate_exception_message( 7,
                                                                "Waveform.analyse",
                                                                "The 'waveform' parameter of the analyser method must be hinted as a Waveform object."))
                
                if signature.return_annotation != Tuple[WfAnaResult, bool]:
                    raise Exception(generate_exception_message( 8,
                                                                "Waveform.analyse",
                                                                "The return type of the analyser method must be hinted as Tuple[WfAnaResult, bool]."))
            except IndexError:
                raise Exception(generate_exception_message( 9,
                                                            "Waveform.analyse",
                                                            "The given filter must take at least one parameter."))
            output_1, output_2 = analyser(self, *args, 
                                                **kwargs)

            aux.Result = output_1
            aux.Passed = output_2

            self.__analyses[label] = aux

            return
        
    def get_global_channel(self):

        """
        Returns
        ----------
        int
            An integer value for the readout channel with respect to a numbering 
            scheme which identifies the endpoint and the APA channel at the same
            time
        """

        pass

    def plot(self,  figure : go.Figure,
                    name : Optional[str] = None,
                    row : Optional[int] = None,
                    col : Optional[int] = None,
                    plot_analysis_markers : bool = False,
                    analysis_label : Optional[str] = None) -> None:

        """
        This method plots the waveform in the given figure.
        
        Parameters
        ----------
        figure : plotly.graph_objects.Figure
            The figure in which the waveform will be plotted
        name : str
            The name for the waveform trace which will be added
            to the given figure.
        row (resp. col) : int
            The row (resp. column) in which the waveform will
            be plottled. This parameter is directly handled to
            the 'row' (resp. 'col') parameter of
            plotly.graph_objects.Figure.add_trace() and
            plotly.graph_objects.Figure.add_shape(). It is the
            caller's responsibility to ensure two things:
                
                - if the given 'figure' parameter does not contain
                  a subplot grid (p.e. it was not created by
                  plotly.subplots.make_subplots()) then 'row' and
                  'col' must be None.
                   
                - if the given 'figure' parameter contains a subplot
                  grid, then 'row' and 'col' must be valid 1-indexed
                  integers.
        plot_analysis_markers : bool                                        ## Plotting every marker makes the call to
            If True, this method will also plot the analysis markers        ## WaveformSet.plot() very slow. We should
            for this waveform in the given figure. If False, it will        ## allow the user to choose a level of detail,
            not. By analysis markers we mean those which are                ## i.e. which markers want to use (not simply
            available among:                                                ## enabling all of them at once with plot_analysis_markers)

                - Vertical lines for the baseline limits
                - An horizontal line for the computed baseline
                - Two vertical lines for the integration limits
                - A triangle marker over each spotted peak
                - Two vertical lines framing each spotted peak
                  marking the integration limits for each peak.
        analysis_label : str
            This parameter only makes a difference if 
            'plot_analysis_markers' is set to True. In that case, 
            'analysis_label' is the key for the WfAna object within 
            self.__analyses from where to take the information for 
            the analysis markers plot. If 'analysis_label' is None,
            then the last analysis added to self.__analyses will be
            the used one.
        """

        x = np.arange(len(self.Adcs))

        wf_trace = go.Scatter(  x = x,      ## If we think x might match for every waveform, 
                                                                    ## it might be defined by the caller, so as not
                                                                    ## to recompute this array for each waveform.
                                y = self.Adcs,
                                mode = 'lines',
                                name = name)
        
        figure.add_trace(   wf_trace,
                            row = row,
                            col = col)

        if plot_analysis_markers:

            if analysis_label is None:
                try:
                    aux = next(reversed(self.__analyses.values()))  # Grabbing the last analysis
                except StopIteration:
                    raise Exception(generate_exception_message( 1,
                                                                'Waveform.plot()',
                                                                'The waveform has not been analysed yet.'))
            else:
                try:
                    aux = self.__analyses[analysis_label]
                except KeyError:
                    raise Exception(generate_exception_message( 2,
                                                                'Waveform.plot()',
                                                                f"There is no analysis with label '{analysis_label}'."))
            
            # Plot the markers for the baseline limits
            for i in range(len(aux.BaselineLimits)//2):

                figure.add_shape(   type = 'line',
                                    x0 = aux.BaselineLimits[2*i], y0 = 0,
                                    x1 = aux.BaselineLimits[2*i], y1 = 1,
                                    line = dict(color = 'grey',         # Properties for
                                                width = 1,              # the beginning of
                                                dash = 'dash'),         # a baseline chunk
                                    xref = 'x',
                                    yref = 'y domain',
                                    row = row,
                                    col = col)
                
                figure.add_shape(   type = 'line',
                                    x0 = aux.BaselineLimits[(2*i)+1], y0 = 0,
                                    x1 = aux.BaselineLimits[(2*i)+1], y1 = 1,
                                    line = dict(color = 'grey',         # Properties for
                                                width = 1,              # the end of a
                                                dash = 'dashdot'),      # baseline chunk
                                    xref = 'x',
                                    yref = 'y domain',
                                    row = row,
                                    col = col)

            # Plot the marker for the baseline 
            figure.add_shape(   type = "line",
                                x0 = 0, y0 = aux.Result.Baseline,
                                x1 = 1, y1 = aux.Result.Baseline,
                                line = dict(color = 'grey',             # Properties for
                                            width = 1,                  # the computed
                                            dash = 'dot'),              # baseline
                                xref = 'x domain',
                                yref = 'y',
                                row = row,
                                col = col)
            
            # Plot the markers for the peaks positions
            for peak in aux.Result.Peaks:

                aux = x[peak.Position]

                figure.add_shape(   type = 'line',
                                    x0 = aux, y0 = 0,
                                    x1 = aux, y1 = 1,
                                    line = dict(color = 'red',      # Properties for
                                                width = 1,          # the peaks markers
                                                dash = 'dot'),
                                    xref = 'x',
                                    yref = 'y domain',
                                    row = row,
                                    col = col)
            
            ## The rest of the markers are not implemented yet
            ##  - General integration limits
            ##  - Peaks integration limits 

        return