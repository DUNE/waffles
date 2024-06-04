from typing import List, Optional
from collections import OrderedDict

import numpy as np
from plotly import graph_objects as pgo

from src.waffles.WfAna import WfAna

from src.waffles.Exceptions import generate_exception_message

class WaveformAdcs:

    """
    This class implements the Adcs array of a waveform.     # It is useful to have such a class so that
                                                            # tools which only need the Adcs information
                                                            # can be run even in situations where a 
                                                            # waveform does not have a defined timestamp,
                                                            # endpoint or any other attribute which could
                                                            # be used to identify a waveform at a higher 
                                                            # level. For example, the waveform which is the
                                                            # result of a averaging over every waveform
                                                            # for a certain channel could be analyzed so
                                                            # as to compute its baseline, but its timestamp
                                                            # is not defined, i.e. it makes no sense.                                           
    Attributes
    ----------
    TimeStep_ns : float
        The time step (in nanoseconds) for this waveform
    Adcs : unidimensional numpy array of integers
        The readout for this waveform, in # of ADCs
    Analyses : OrderedDict of WfAna objects

    Methods
    ----------
    ## Add the list of methods and a summary for each one here
    """

    def __init__(self,  time_step_ns : float,
                        adcs : np.ndarray):
        
        """
        WaveformAdcs class initializer
        
        Parameters
        ----------
        time_step_ns : float
        adcs : unidimensional numpy array of integers
        """

        ## Shall we add add type checks here?

        self.__time_step_ns = time_step_ns
        self.__adcs = adcs
        self.__analyses = OrderedDict() # Initialize the analyses 
                                        # attribute as an empty 
                                        # OrderedDict.

        ## Do we need to add trigger primitives as attributes?
    
    #Getters
    
    @property
    def TimeStep_ns(self):
        return self.__time_step_ns
    
    @property
    def Adcs(self):
        return self.__adcs
    
    @property
    def Analyses(self):
        return self.__analyses
    
#   #Setters                            # For the moment there are no setters for 
#   @TimeStep_ns.setter                 # the attributes of WaveformAdcs. I.e. you
#   def TimeStep_ns(self, input):       # can only set the value of its attributes
#       self.__time_step_ns = input     # through WaveformAdcs.__init__. Here's an
#       return                          # example of what a setter would look like, though.

    def confine_iterator_value(self, input : int) -> int:

        """
        Confines the input integer to the range [0, len(self.__adcs) - 1].
        I.e returns 0 if input is negative, returns input if input belongs
        to the range [0, len(self.__adcs) - 1], and returns len(self.__adcs) - 1
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
            return len(self.__adcs) - 1

    def analyse(self,   label : str,
                        analyser_name : str,
                        baseline_limits : List[int],
                        int_ll : int = 0,
                        int_ul : Optional[int] = None,
                        *args,
                        overwrite : bool = False,
                        **kwargs) -> dict:

        """
        This method creates a WfAna object and adds it to the
        self.__analyses dictionary using label as its key.
        To do so, it grabs the WfAna instance method whose name
        matches analyser_name and runs it on this WaveformAdcs
        object. Then, this method does two things:
        
            -   first, it adds the two first outputs of such
                analyser method to the 'Result' and 'Passed' 
                attributes of the newly created WfAna object,
                respectively.
            -   second, it returns the third output of the
                analyser method, which should be a dictionary
                containing any additional information that the
                analyser method wants to return. Such dictionary
                is empty if no additional information is
                provided by the analyser method.

        Parameters
        ----------
        label : str
            Key for the new WfAna object within the self.__analyses
            OrderedDict
        analyser_name : str
            It must match the name of a WfAna method whose first
            argument must be called 'waveform' and whose type   
            annotation must match the WaveformAdcs class or the     
            'WaveformAdcs' string literal. Such method should also  
            have a defined return-annotation which must match   
            Tuple[WfAnaResult, bool, dict]. It is the caller's
            responsibility to check such conditions for this
            parameter. No checks are performed here for this
            input.
        baseline_limits : list of int                                   
            Given to the 'baseline_limits' parameter of                 
            WfAna.__init__. It must have an even number
            of integers which must meet 
            baseline_limits[i] < baseline_limits[i + 1] for
            all i. The points which are used for 
            baseline calculation are 
            self.__adcs[baseline_limits[2*i]:baseline_limits[(2*i) + 1]],
            with i = 0,1,...,(len(baseline_limits)/2) - 1. 
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
        output_3 : dict
            The third output of the analyser method, which
            should be a dictionary containing any additional
            information that the analyser method wants to
            return. Note that the analyser method must return
            a dictionary as its third output, even it its
            an empty one.
        """

        if label in self.__analyses.keys() and not overwrite:
            raise Exception(generate_exception_message( 1,
                                                        'WaveformAdcs.analyse()',
                                                        f"There is already an analysis with label '{label}'. If you want to overwrite it, set the 'overwrite' parameter to True."))
        else:

            aux = WfAna(baseline_limits,
                        int_ll,
                        int_ul)
            
            analyser = getattr(aux, analyser_name)

            output_1, output_2, output_3 = analyser(self,   *args, 
                                                            **kwargs)
            aux.Result = output_1
            aux.Passed = output_2

            self.__analyses[label] = aux

            return output_3

    def plot(self,  figure : pgo.Figure,
                    name : Optional[str] = None,
                    row : Optional[int] = None,
                    col : Optional[int] = None,
                    plot_analysis_markers : bool = False,
                    show_baseline_limits : bool = False, 
                    show_baseline : bool = True,
                    show_general_integration_limits : bool = False,
                    show_spotted_peaks : bool = True,
                    show_peaks_integration_limits : bool = False,
                    analysis_label : Optional[str] = None) -> None:

        """
        This method plots this waveform in the given figure.
        
        Parameters
        ----------
        figure : plotly.graph_objects.Figure
            The figure in which the waveform will be plotted
        name : str
            The name for the waveform trace which will be added
            to the given figure.
        row (resp. col) : int
            The row (resp. column) in which the waveform will
            be plotted. This parameter is directly handled to
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
        plot_analysis_markers : bool
            If True, this method will potentially plot the
            analysis markers for this waveform in the given
            figure. If False, it will not. By analysis markers
            we mean those which are available among:

                - Vertical lines for the baseline limits
                - An horizontal line for the computed baseline
                - Two vertical lines for the integration limits
                - A triangle marker over each spotted peak                  ## This is not true yet. Change the 
                - Two vertical lines framing each spotted peak              ## vertical lines for triangle markers.
                  marking the integration limits for each peak.
        show_baseline_limits : bool
            This parameter only makes a difference if
            'plot_analysis_markers' is set to True. In that case,
            this parameter means whether to plot vertical lines
            framing the intervals which were used to compute
            the baseline.
        show_baseline : bool
            This parameter only makes a difference if
            'plot_analysis_markers' is set to True. In that case,
            this parameter means whether to plot an horizontal
            line matching the computed baseline
        show_general_integration_limits : bool
            This parameter only makes a difference if
            'plot_analysis_markers' is set to True. In that case,
            this parameter means whether to plot vertical lines
            framing the general integration interval.
        show_spotted_peaks : bool
            This parameter only makes a difference if
            'plot_analysis_markers' is set to True. In that case,
            this parameter means whether to plot a triangle
            marker over each spotted peak.
        show_peaks_integration_limits : bool
            This parameter only makes a difference if
            'plot_analysis_markers' is set to True. In that case,
            this parameter means whether to plot two vertical
            lines framing the integration interval for each
            spotted peak.
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

        wf_trace = pgo.Scatter( x = x,                  ## If we think x might match for every waveform, in
                                                        ## a certain WaveformSet object, it might be more
                                                        ## efficient to let the caller define it, so as 
                                                        ## not to recompute this array for each waveform.
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
                                                                'WaveformAdcs.plot()',
                                                                'The waveform has not been analysed yet.'))
            else:
                try:
                    aux = self.__analyses[analysis_label]
                except KeyError:
                    raise Exception(generate_exception_message( 2,
                                                                'WaveformAdcs.plot()',
                                                                f"There is no analysis with label '{analysis_label}'."))
            
            if show_baseline_limits:    # Plot the markers for the baseline limits

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
                                        x0 = aux.BaselineLimits[(2*i) + 1], y0 = 0,
                                        x1 = aux.BaselineLimits[(2*i) + 1], y1 = 1,
                                        line = dict(color = 'grey',         # Properties for
                                                    width = 1,              # the end of a
                                                    dash = 'dashdot'),      # baseline chunk
                                        xref = 'x',
                                        yref = 'y domain',
                                        row = row,
                                        col = col)

            if show_baseline:       # Plot the baseline
            
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
                
            if show_general_integration_limits:  # Plot the markers for the general integration limits

                    figure.add_shape(   type = 'line',
                                        x0 = x[aux.IntLl], y0 = 0,
                                        x1 = x[aux.IntLl], y1 = 1,
                                        line = dict(color = 'black',        # Properties for
                                                    width = 1,              # the beginning of
                                                    dash = 'solid'),        # a baseline chunk
                                        xref = 'x',
                                        yref = 'y domain',
                                        row = row,
                                        col = col)
                    
                    figure.add_shape(   type = 'line',
                                        x0 = x[aux.IntUl], y0 = 0,
                                        x1 = x[aux.IntUl], y1 = 1,
                                        line = dict(color = 'black',        # Properties for
                                                    width = 1,              # the beginning of
                                                    dash = 'solid'),        # a baseline chunk
                                        xref = 'x',
                                        yref = 'y domain',
                                        row = row,
                                        col = col)
            
            if show_spotted_peaks:      # Plot the markers for the spotted peaks

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
                    
            if show_peaks_integration_limits:   # Plot the markers for the peaks integration limits
                pass    ## To be implemented

        return