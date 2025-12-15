import os
import pickle
import yaml
import numpy as np
import pandas as pd
from plotly import graph_objects as pgo
from typing import Tuple, Dict, Optional
from numbers import Number

from waffles.data_classes.WafflesAnalysis import BaseInputParams
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWs import ChannelWs
from waffles.utils.integral.WindowIntegrator import WindowIntegrator
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.Map import Map
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.CalibrationHistogram import CalibrationHistogram

from waffles.utils.integral.integral_utils import get_pulse_window_limits
from waffles.plotting.plot import plot_ChannelWsGrid
from waffles.np04_utils.utils import get_channel_iterator
import waffles.Exceptions as we

def get_and_check_channels_per_run(
    channels_per_run_filepath: str
) -> pd.DataFrame:
    """This function reads a CSV file containing the
    channels-per-run database and checks that it contains
    the required columns, i.e. 'run', 'batch',
    'acquired_apas', 'aimed_channels' and 'pde'. It raises
    a we.MissingColumnsInDataFrame exception if any of
    these columns is missing.
    """

    channels_per_run = pd.read_csv(
        channels_per_run_filepath
    )

    if not set([
        'run',
        'batch',
        'acquired_apas',
        'aimed_channels',
        'pde'
    ]).issubset(channels_per_run.columns):

        raise we.MissingColumnsInDataFrame(
            we.GenerateExceptionMessage(
                1,
                'get_and_check_channels_per_run()',
                f"The file {channels_per_run_filepath} is missing "
                "some of the required columns. It must contain at least "
                "'run', 'batch', 'acquired_apas', 'aimed_channels' "
                "and 'pde'."
            )
        )

    return channels_per_run

def get_check_and_filter_excluded_channels(
    excluded_channels_filepath: str,
    integrate_entire_pulse: bool,
    deviation_from_baseline: float,
    verbose: bool = True
) -> pd.DataFrame:
    """This function reads a CSV file containing excluded
    channels for different integration configurations and
    filters it based on the specified integration parameters.
    The excluded channels are organized by integration
    deviation-from-baseline (integration_dfb), batch, APA,
    and PDE values.
    
    Parameters
    ----------
    excluded_channels_filepath: str
        Path to the CSV file containing the excluded channels
        database. The file must contain at least the columns
        'integration_dfb', 'batch', 'apa', 'pde' and
        'excluded_channels'
    integrate_entire_pulse: bool
        If True, selects excluded channels for the configuration
        closest to full-pulse integration (smallest
        integration_dfb). If False, filters for the exact
        deviation_from_baseline value.
    deviation_from_baseline: float
        The specific deviation-from-baseline value to filter
        for when integrate_entire_pulse is False. Values are
        rounded to 2 decimal places to avoid floating-point
        precision issues.
    verbose: bool, default True
        Whether to print informational messages about the
        filtering process.
        
    Returns
    -------
    excluded_channels: pd.DataFrame
        Filtered dataframe containing only the excluded
        channels entries that match the specified integration
        configuration. Contains the same columns as the input
        file but filtered to the relevant rows.
    """

    excluded_channels = pd.read_csv(
        excluded_channels_filepath
    )

    if not set([
        'integration_dfb',
        'batch',
        'apa',
        'pde',
        'excluded_channels'
    ]).issubset(excluded_channels.columns):

        raise we.MissingColumnsInDataFrame(
            we.GenerateExceptionMessage(
                1,
                'get_check_and_filter_excluded_channels()',
                f"The file {excluded_channels_filepath} is missing "
                "some of the required columns. It must contain at least "
                "'integration_dfb', 'batch', 'apa', 'pde' and "
                "'excluded_channels'."
            )
        )

    if integrate_entire_pulse:
        aux = min(excluded_channels['integration_dfb'])

        excluded_channels = excluded_channels[
            # Get the list of excluded channels which is
            # the closest to the full-pulse integration,
            # i.e. the one with the smallest integration_dfb
            excluded_channels['integration_dfb'] == aux
        ]

        if verbose:
            print(
                "In function get_check_and_filter_excluded_channels(): "
                "Since entire pulse integration was "
                "enabled, the excluded channels list for "
                f"integration_dfb = {aux} was chosen."
            )
    else:
        excluded_channels = excluded_channels[
            # Avoid problems due to floating-point precision
            # Differences in the integration deviation-from-baseline
            # below 1% are irrelevant anyways
            excluded_channels['integration_dfb'] == round(
                deviation_from_baseline,
                2
            )
        ]

        if len(excluded_channels) == 0:
            print(
                "In function get_check_and_filter_excluded_channels(): "
                "WARNING: No excluded channels found for "
                "deviation_from_baseline = "
                f"{deviation_from_baseline}. "
                "No channels will be excluded from the "
                "calibration.",
                end=''
            )

    return excluded_channels

def get_batches_dates_mapping(
    filepath_to_batches_dates_csv: str
) -> dict[int, str]:
    """
    Create a mapping from batch identifiers to date strings by
    reading a CSV file.

    Parameters
    ----------
    filepath_to_batches_dates_csv: str
        Path to a CSV file that must contain the columns:
        'batch', 'date_day', 'date_month', 'date_year'. Each row
        represents one batch and its associated day, month and year
        components.

    Returns
    -------
    dict[int, str]
        A dictionary mapping each batch (int) to a date string in
        the format "YYYY/MM/DD" (year/month/day).
    """

    try:
        df = pd.read_csv(filepath_to_batches_dates_csv)
    except FileNotFoundError:
        raise FileNotFoundError(
            "In function get_batches_dates_mapping(): "
            f"The file {filepath_to_batches_dates_csv} was not found."
        )

    required_cols = {'batch', 'date_day', 'date_month', 'date_year'}
    if not required_cols.issubset(df.columns):
        raise we.MissingColumnsInDataFrame(
            we.GenerateExceptionMessage(
                1,
                'get_batches_dates_mapping()',
                f"The file {filepath_to_batches_dates_csv} is missing "
                "some of the required columns. It must contain at least "
                "'batch', 'date_day', 'date_month' and 'date_year'."
            )
        )

    mapping = {}

    for _, row in df.iterrows():
        try:
            batch = int(row['batch'])
            day = int(row['date_day'])
            month = int(row['date_month'])
            year = int(row['date_year'])
        except Exception as e:
            raise ValueError(
                "In function get_batches_dates_mapping(): "
                "Could not parse numeric values for batch/date in the "
                f"file {filepath_to_batches_dates_csv}: {e}"
            )

        if batch in mapping:
            raise Exception(
                "In function get_batches_dates_mapping(): "
                f"Duplicate entry for batch {batch} found in "
                f"{filepath_to_batches_dates_csv}. Each batch must "
                "appear only once."
            )

        mapping[batch] = f"{year:04d}-{month:02d}-{day:02d}"

    return mapping

def get_sipm_vendor_dataframe(
    filepath: str | None
    ) -> pd.DataFrame:
    """This function reads the given CSV file and
    checks that it contains the required columns:
    'endpoint', 'daphne_ch', and 'sipm'. If filepath
    is None, a we.IncompatibleInput exception is raised.

    Parameters
    ----------
    filepath: str | None
        Path to the CSV file containing the channel-wise
        SiPM vendor information

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the required information
    """

    if filepath is None:
        raise we.IncompatibleInput(
            we.GenerateExceptionMessage(
                1,
                'get_sipm_vendor_dataframe()',
                'The filepath parameter must be provided.'
            )
        )

    sipm_vendor_df = pd.read_csv(filepath)

    required_cols = {
        'endpoint',
        'daphne_ch',
        'sipm'
    }

    if not required_cols.issubset(sipm_vendor_df.columns):
        raise we.MissingColumnsInDataFrame(
            we.GenerateExceptionMessage(
                2,
                'get_sipm_vendor_dataframe()',
                f"The file {filepath} is missing some of the required "
                "columns. It must contain at least 'endpoint', "
                "'daphne_ch', and 'sipm'."
            )
        )

    return sipm_vendor_df

def get_alignment_seeds_dataframe(
    filepath: str | None
    ) -> pd.DataFrame:
    """This function reads the given CSV file and
    checks that it contains the required columns
    ('batch', 'APA', 'PDE', 'endpoint', 'channel',
    'vendor', 'center_0', 'center_1' and
    'SPE_mean_adcs'). If filepath is None,
    a we.IncompatibleInput exception is raised.

    Parameters
    ----------
    filepath: str | None
        Path to the CSV file containing the seeds
        required for correlation-based alignment

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the required information
    """

    if filepath is None:
        raise we.IncompatibleInput(
            we.GenerateExceptionMessage(
                1,
                'get_alignment_seeds_dataframe()',
                'The filepath parameter must be provided.'
            )
        )

    alignment_seeds_df = pd.read_csv(filepath)

    required_cols = {
        'batch',
        'APA',
        'PDE',
        'endpoint',
        'channel',
        'vendor',
        'center_0',
        'center_1',
        'SPE_mean_adcs'
    }

    if not required_cols.issubset(alignment_seeds_df.columns):
        raise we.MissingColumnsInDataFrame(
            we.GenerateExceptionMessage(
                2,
                'get_alignment_seeds_dataframe()',
                f"The file {filepath} is missing some of the required "
                "columns. It must contain at least 'batch', 'APA', "
                "'PDE', 'endpoint', 'channel', 'vendor', 'center_0', "
                "'center_1' and 'SPE_mean_adcs'."
            )
        )

    return alignment_seeds_df

def __got_well_formed_alignment_seeds(
    filtered_df: pd.DataFrame
) -> bool:
    """This is a helper function which should only be called by
    get_alignment_seeds(). It checks whether the given
    filtered_df contains well-formed data for the correlation-based
    alignment. It returns False if any of the following conditions
    are met:

        - filtered_df is empty
        - the first entry of the 'center_0' column is not a float
        - the first entry of the 'center_1' column is not a float
        - the first entry of the 'SPE_mean_adcs' column is not a str

    It returns True if else.

    Parameters
    ----------
    filtered_df: pandas.DataFrame
        DataFrame to check

    Returns
    -------
    bool
        True if the data is well-formed, False otherwise
    """

    # isinstance(..., Number) don't distinguish
    # between int, np.int, float, np.float etc.
    if len(filtered_df) == 0 or \
        not isinstance(filtered_df.iloc[0]['center_0'], Number) or \
        not isinstance(filtered_df.iloc[0]['center_1'], Number) or \
        not isinstance(filtered_df.iloc[0]['SPE_mean_adcs'], str):

        return False

    return True

def get_alignment_seeds(
    alignment_seeds_dataframe: pd.DataFrame,
    batch: int,
    apa: int,
    pde: float,
    endpoint: int,
    channel: int,
    sipm_vendor_df: Optional[pd.DataFrame] = None,
    same_endpoint_fallback: bool = True,
    same_batch_apa_and_pde_fallback: bool = True
) -> np.ndarray:
    """This function retrieves the following entries: 'center_0', 
    'center_1' and 'SPE_mean_adcs'; from the given dataframe,
    based on the specified batch, APA, PDE, endpoint and channel
    values. If no exact match is found, it can optionally fall
    back to searching for such information
    
        1) from another channel at the same endpoint or
        2) from another channel within the same APA (regardless of the endpoint)

    In any fallback case, the batch, APA and PDE values are
    always kept fixed. Additionally, if sipm_vendor_df is provided,
    in the standard case and in any fallback case, the function
    only considers rows whose 'vendor' column matches the
    vendor retrieved from sipm_vendor_df for the given
    endpoint and channel.

    Parameters
    ----------
    alignment_seeds_dataframe: pandas.DataFrame
        DataFrame containing, at least, the following columns:
            - 'batch',
            - 'APA',
            - 'PDE',
            - 'endpoint',
            - 'channel',
            - 'vendor',
            - 'center_0',
            - 'center_1' and
            - 'SPE_mean_adcs'.
    batch: int
        The batch number to look for
    apa: int
        The APA number to look for
    pde: float
        The PDE value to look for
    endpoint: int
        The endpoint number to look for
    channel: int
        The channel number to look for
    sipm_vendor_df: pd.DataFrame | None
        If it is defined, it must be a DataFrame
        containing the SiPM vendor information for each
        endpoint-channel pair. In that case, it must
        contain at least the columns 'endpoint',
        'daphne_ch' and 'sipm', and the seeds returned
        by this function come from a row whose 'vendor'
        column matches that of the given endpoint and
        channel.
    same_endpoint_fallback: bool, default True
        If True, and no exact match is found, the function
        will search for the required information from the
        same endpoint, regardless of channel.
    same_batch_apa_and_pde_fallback: bool, default True
        If True, and no match is found even after applying
        the same_endpoint_fallback, the function will
        search for the required information from the
        same batch, APA and PDE, regardless of endpoint
        and channel.

    Returns
    -------
    dict
        The returned dictionary has the following structure:
        {
            'center_0': float,
            'center_1': float,
            'SPE_mean_adcs': np.ndarray
        }
    """

    aux = (alignment_seeds_dataframe['batch'] == batch) & \
        (alignment_seeds_dataframe['APA'] == apa) & \
        (alignment_seeds_dataframe['PDE'] == pde) & \
        (alignment_seeds_dataframe['endpoint'] == endpoint) & \
        (alignment_seeds_dataframe['channel'] == channel)
    
    vendor = get_vendor(
        endpoint,
        channel,
        sipm_vendor_df=sipm_vendor_df
    )

    if vendor != 'unavailable':
        aux = aux & (alignment_seeds_dataframe['vendor'] == vendor)
    
    filtered_df = alignment_seeds_dataframe[aux]
    fGotWellFormedData = __got_well_formed_alignment_seeds(filtered_df)

    if not fGotWellFormedData:
        if not same_endpoint_fallback:
            raise Exception(
                "In function get_alignment_seeds(): "
                "Could not find the required information (or it "
                "was found but its type is not as expected) for "
                f"batch {batch}, APA {apa}, PDE {pde}, endpoint "
                f"{endpoint}, channel {channel} and vendor "
                f"{vendor}, and same_endpoint_fallback is set to "
                "False."
            )

        aux = (alignment_seeds_dataframe['batch'] == batch) & \
            (alignment_seeds_dataframe['APA'] == apa) & \
            (alignment_seeds_dataframe['PDE'] == pde) & \
            (alignment_seeds_dataframe['endpoint'] == endpoint)
        
        if vendor != 'unavailable':
            aux = aux & (alignment_seeds_dataframe['vendor'] == vendor)

        filtered_df = alignment_seeds_dataframe[aux]
        fGotWellFormedData = __got_well_formed_alignment_seeds(filtered_df)

        if not fGotWellFormedData:
            if not same_batch_apa_and_pde_fallback:
                raise Exception(
                    "In function get_alignment_seeds(): "
                    "Even after extending the search to the whole "
                    "endpoint, could not find the required information "
                    "(or it was found but its type is not as expected) "
                    f"for batch {batch}, APA {apa}, PDE {pde}, "
                    f"endpoint {endpoint} and vendor {vendor}. Enable "
                    "same_batch_apa_and_pde_fallback to True to allow "
                    "fallback to other endpoints within the same batch, "
                    "APA, PDE and vendor."
                )
            
            aux = (alignment_seeds_dataframe['batch'] == batch) & \
                (alignment_seeds_dataframe['APA'] == apa) & \
                (alignment_seeds_dataframe['PDE'] == pde)
            
            if vendor != 'unavailable':
                aux = aux & (alignment_seeds_dataframe['vendor'] == vendor)

            filtered_df = alignment_seeds_dataframe[aux]
            fGotWellFormedData = __got_well_formed_alignment_seeds(filtered_df)

            if not fGotWellFormedData:
                raise Exception(
                    "In function get_alignment_seeds(): "
                    "Even after extending the search to the whole "
                    f"(batch {batch}, APA {apa}, PDE {pde}, vendor "
                    f"{vendor}) set, could not retrieve the required "
                    "information (or it was retrieved but its type "
                    "is not as expected)."
                )
        
    used_row = filtered_df.iloc[0]
    if endpoint != used_row['endpoint'] or \
        channel != used_row['channel']:
        print(
            "In function get_alignment_seeds(): "
            f"WARNING: For batch {batch}, APA {apa}, PDE {pde}, "
            f"channel {endpoint}-{channel}, using alignment seeds"
            f" from a different channel ({used_row['endpoint']}-"
            f"{used_row['channel']})"
        )
            
    # At this point, filtered_df must contain at least one entry
    SPE_template_str = used_row['SPE_mean_adcs']
    SPE_template_str = SPE_template_str.strip()[1:-1]
    
    SPE_template = np.array(
        [float(x.strip()) for x in SPE_template_str.split(',')]
    )

    return {
            'center_0': float(used_row['center_0']),
            'center_1': float(used_row['center_1']),
            'SPE_mean_adcs': SPE_template
        }

def align_waveforms_by_correlation(
    input_ChannelWs: ChannelWs,
    center_0: float,
    center_1: float,
    SPE_template_array: np.ndarray,
    integrate_entire_pulse: bool,
    baseline_analysis_label: str,
    SPE_template_lower_limit_wrt_pulse: int, 
    SPE_template_upper_limit_wrt_pulse: int,
    maximum_allowed_shift: int,
    deviation_from_baseline: float = 0.3,
    lower_limit_correction: int = 0,
    upper_limit_correction: int = 0,
    minimum_sliced_template_points: int = 10
) -> None:
    """This function aligns the waveforms in the given
    ChannelWs object based on their correlation with a
    provided SPE template. To do so, the function first
    computes the average waveform across all waveforms
    in the input ChannelWs, and uses it to determine
    the average pulse position and the integration limits
    around the pulse. The function then integrates each
    waveform using the inferred limits and ignores or
    aligns them based on their integral values. Namely,
    waveforms whose integral evaluates to less than
    (center_0 + center_1)/2 are considered as baselines
    and are not aligned. This cut is meant to avoid
    biasing the pedestal position upwards: aligning
    baseline samples to the SPE template would find
    noisy regions which are systematically positive,
    artificially increasing its integral on average. The
    waveforms are finally trimmed, so that they are
    aligned amongst each other, according to some limits
    which are defined by the input parameters. These
    changes to the input ChannelWs object are done in
    place.

    Parameters
    ----------
    input_ChannelWs: ChannelWs
        The ChannelWs object containing the waveforms
        to be aligned.
    center_0 (resp. center_1): float
        The estimated mean of the baselines (resp. SPE)
        integrals, used to decide if a waveform is a
        baseline or not.
    SPE_template_array: np.ndarray
        The SPE template array used for correlation
    integrate_entire_pulse: bool
        Whether the integration to asess if a waveform
        is a baseline or not should consider the
        entire pulse (True) or only a window around
        the pulse (False). If True, the
        deviation_from_baseline, lower_limit_correction
        and upper_limit_correction parameters are
        ignored for this integration, and the entire
        pulse is considered. If False, these
        parameters are used to define the integration
        window around the pulse.
    baseline_analysis_label: str
        The WindowIntegrator analysis needs a baseline
        value to compute the waveform integrals. This
        parameter gives the label for the WfAna object
        which contains such baseline value. Namely, it
        is the key for the analyses attribute dictionary
        of each Waveform object within the input
        ChannelWs.
    SPE_template_lower_limit_wrt_pulse (resp. SPE_template_upper_limit_wrt_pulse): int
        The length of the slice of the SPE template
        which is used for the correlation computation
        is defined by these two parameters. Namely, 
        they give the lower and upper limits (in
        number of time ticks) with respect to the
        pulse position (minimum of the SPE template).
    maximum_allowed_shift: int
        For each waveform in the input ChannelWs,
        the length of the slice which is used
        for the correlation computation is defined
        by this parameter. Given the average pulse
        position (inferred from the average waveform
        across all waveforms in the input ChannelWs),
        the lower (resp. upper) slicing index for
        each waveform are defined as

            1)  average_pulse_position
                - SPE_template_lower_limit_wrt_pulse
                - maximum_allowed_shift

            2)  average_pulse_position
                + SPE_template_upper_limit_wrt_pulse
                + maximum_allowed_shift
        
        The body of this function makes sure that
        the number of points in the sliced waveform
        is bigger or equal to the number of points
        in the sliced SPE template. Hence, when
        calling numpy.correlate() with mode='valid'
        (i.e. only positions where the two input
        arrays fully overlap are considered),
        it is ensured that there is at least one
        output value for which a complete overlap
        with the full SPE template was considered.
    deviation_from_baseline: float
        It is only used if integrate_entire_pulse is
        False. In that case, it is given to the
        deviation_from_baseline parameter of the
        get_pulse_window_limits() function. It must be
        greater than 0.0 and smaller than 1.0. It controls
        the width of the integration window around the
        pulse. For more information, check
        the get_pulse_window_limits() function docstring.
    lower_limit_correction (resp. upper_limit_correction): int
        It is only used if integrate_entire_pulse is
        False. In that case, it is given to the
        lower_limit_correction (resp.
        upper_limit_correction) parameter of the
        get_pulse_window_limits() function. It applies
        a correction to the lower (resp. upper) limit
        of the pulse window. For more information, check
        the get_pulse_window_limits() function docstring.
    minimum_sliced_template_points: int
        This parameter makes a difference only if the
        sliced waveforms end up having less points than
        the sliced SPE template. In that case, the
        function trims the sliced SPE template further,
        so that its number of points is smaller than the
        number of points in the sliced waveforms by
        2 * maximum_allowed_shift points. If, after this
        reduction, the number of points in the sliced
        SPE template is less than minimum_sliced_template_points,
        the function raises an Exception.

    Returns
    -------
    None
    """

    # Find the pulse in the template
    template_pulse_idx = np.argmin(SPE_template_array)

    # Find the slicing indices for the template
    template_slice_i_low = max(
        0,
        template_pulse_idx - abs(int(SPE_template_lower_limit_wrt_pulse))
    )

    template_slice_i_up = min(
        len(SPE_template_array),
        template_pulse_idx + abs(int(SPE_template_upper_limit_wrt_pulse))
    )

    # Number of points in the sliced template, which match
    # the number of points in the waveforms for the resulting
    # aligned and filtered WaveformSet
    template_n_points = template_slice_i_up - template_slice_i_low

    # Compute the average waveform of the input ChannelWs
    mean_WaveformAdcs = input_ChannelWs.compute_mean_waveform()

    # Find the average pulse position in the input ChannelWs
    average_idx_pulse = np.argmin(mean_WaveformAdcs.adcs)

    # Find the slicing indices for the waveforms
    wfs_slice_i_low = max(
        0, 
        average_idx_pulse \
            - abs(int(SPE_template_lower_limit_wrt_pulse)) \
                - abs(maximum_allowed_shift)
    )

    wfs_slice_i_up = min(
        input_ChannelWs.points_per_wf,
        average_idx_pulse \
            + abs(int(SPE_template_upper_limit_wrt_pulse)) \
                + abs(maximum_allowed_shift)
    )

    waveforms_n_points = wfs_slice_i_up - wfs_slice_i_low

    if waveforms_n_points < template_n_points:
        print(
            "In function align_waveforms_by_correlation(): "
            "WARNING: The given SPE template peaks at time tick "
            f"{template_pulse_idx}, resulting in the following "
            f"slicing indices: [{template_slice_i_low}, "
            f"{template_slice_i_up}). The data for the current "
            "ChannelWs object peaks, on average, at time tick "
            f"{average_idx_pulse}, resulting in the following "
            f"slicing indices: [{wfs_slice_i_low}, "
            f"{wfs_slice_i_up}). This interval contains "
            f"{waveforms_n_points} points for the waveform "
            "samples, which is smaller than the number of "
            f"points ({template_n_points}) for the template "
            "slice. The number of points in the SPE template "
            f"will be reduced from {template_n_points} to "
            f"waveforms_n_points - (2 * {maximum_allowed_shift}) = ",
            end=''
        )

        # Compute left_fraction before updating template_n_points
        left_fraction = round(
            (template_pulse_idx - template_slice_i_low) / template_n_points
        )

        # Consider an smaller template slice, so
        # that shifts of maximum_allowed_shift
        # time ticks can still be considered
        template_n_points = waveforms_n_points - (2 * maximum_allowed_shift)
        print(f"{template_n_points} points.")

        if minimum_sliced_template_points <= 0:
            raise Exception(
                "In function align_waveforms_by_correlation(): "
                "The minimum_sliced_template_points parameter must "
                "be greater than 0."
            )

        elif template_n_points <= minimum_sliced_template_points:
            raise Exception(
                "In function align_waveforms_by_correlation(): "
                "After reducing the number of points in the template, "
                "they fell below the minimum allowed value."
            )

        template_slice_i_low = \
            template_pulse_idx - round(left_fraction * template_n_points)
        
        template_slice_i_up = \
            template_slice_i_low + template_n_points

    # As of here, the number of points in the sliced waveforms
    # is bigger or equal to the number of points in the sliced
    # template. Hence, when we call np.correlate() specifying
    # mode='valid', we will always get an output which contains at
    # least one value for which a complete overlap with the full
    # SPE template was considered.

    # Use the same normalization for the SPE template
    # and for the sample waveforms later on
    normalizer = lambda input_array: \
        input_array / np.max(np.abs(input_array))

    # Slice and normalize the template only once
    sliced_template_array = SPE_template_array[
        template_slice_i_low: template_slice_i_up
    ]

    normalized_sliced_template_array = normalizer(
        sliced_template_array
    )
    
    # Integrate the waveforms to decide which ones
    # are baselines and which ones are not
    label = 'integration_for_baseline_identification'

    aux_limits = get_pulse_window_limits(
        mean_WaveformAdcs.adcs,
        0,
        0.1 if integrate_entire_pulse else \
            deviation_from_baseline,
        lower_limit_correction=-1 if \
            integrate_entire_pulse else \
            lower_limit_correction,
        upper_limit_correction=0 if \
            integrate_entire_pulse else \
            upper_limit_correction,
        get_zero_crossing_upper_limit=True if \
            integrate_entire_pulse else False
    )

    integrator_input_parameters = IPDict({
        'baseline_analysis': baseline_analysis_label,
        'inversion': True,
        'int_ll': aux_limits[0],
        'int_ul': aux_limits[1],
        'amp_ll': aux_limits[0],
        'amp_ul': aux_limits[1]
    })

    checks_kwargs = IPDict({
        'points_no': input_ChannelWs.points_per_wf
    })

    _ = input_ChannelWs.analyse(
        label,
        WindowIntegrator,
        integrator_input_parameters,
        checks_kwargs=checks_kwargs,
        overwrite=True
    )
    
    is_baseline = []
    optimal_shift_in_time_ticks = []

    # Compute the baseline charge threshold only once
    baseline_charge_threshold = (center_0 + center_1) / 2.

    for waveform in input_ChannelWs.waveforms:

        # Do not align baselines (i.e. do not bias
        # the pedestal position upwards)
        if waveform.analyses[label].result['integral'] < \
            baseline_charge_threshold:

            is_baseline.append(True)
            optimal_shift_in_time_ticks.append(None)

        else:
            sliced_waveform_adcs = waveform.adcs[
                wfs_slice_i_low: wfs_slice_i_up
            ]

            correlation_coefficients = np.correlate(
                normalizer(sliced_waveform_adcs),
                normalized_sliced_template_array,
                mode='valid'
            )

            is_baseline.append(False)
            optimal_shift_in_time_ticks.append(
                np.argmax(correlation_coefficients)
            )

    # In this loop, we truncate the waveforms (in place) to the
    # template limits, applying the optimal shift when needed
    for i in range(len(input_ChannelWs.waveforms)):

        if is_baseline[i]:
            # If this waveform has been identified as a baseline,
            # we just slice the waveform to the template limits
            # without applying any shift.
            input_ChannelWs.waveforms[i]._Waveform__slice_adcs(
                template_slice_i_low,
                # Matches template_slice_i_low + template_n_points, by definition
                template_slice_i_up
            )

        else:
            input_ChannelWs.waveforms[i]._Waveform__slice_adcs(
                # Note that the lag index (the shifts) returned by
                # np.correlate() are relative to the start of the
                # longest input signal, which in this case is the
                # sliced sample (not the template)
                wfs_slice_i_low + optimal_shift_in_time_ticks[i],
                wfs_slice_i_low + optimal_shift_in_time_ticks[i] + template_n_points
            )

    # Update the points_per_wf attribute
    input_ChannelWs._WaveformSet__points_per_wf = template_n_points
    input_ChannelWs.reset_mean_waveform()

    return

def get_fine_selection_regions(
    input_ChannelWs: ChannelWs,
    baseline_region_points: int,
    signal_region_half_points: int
) -> tuple[int, int, int]:
    """This function computes the parameters baseline_i_low,
    baseline_i_up and signal_i_up as required by the
    fine_selection_for_led_calibration() function. To do so,
    this function first estimates the iterator value for
    which a pulse typically happens in the waveform.adcs
    arrays of the given ChannelWs object.

    Parameters
    ----------
    baseline_region_points: int
        The total number of points in the baseline region.
        Namely, the baseline region is defined, before the
        signal region, as the
            [
                estimated_pulse_idx - signal_region_half_points - baseline_region_points:
                estimated_pulse_idx - signal_region_half_points
            ]
        interval in the waveform.adcs array.
    signal_region_half_points: int
        The total number of points in the signal region
        equals to (2 * signal_region_half_points) + 1. 
        I.e. the signal region is defined, around the
        estimated pulse index, as the
            [
                estimated_pulse_idx - signal_region_half_points:
                estimated_pulse_idx + signal_region_half_points + 1
            ]
        interval in the waveform.adcs array.

    Returns
    -------
    tuple[int, int, int]
        A tuple containing the following three integers:
            - baseline_i_low: int
                The lower limit index for the baseline region.
            - baseline_i_up: int
                The upper limit index for the baseline region,
                which matches the lower limit index for the
                signal region.
            - signal_i_up: int
                The upper limit index for the signal region.
    """

    estimated_pulse_idx = np.argmin(
        input_ChannelWs.compute_mean_waveform().adcs
    )

    baseline_i_low = max(
        0,
        estimated_pulse_idx \
            - signal_region_half_points \
                - baseline_region_points
    )

    baseline_i_up = max(
        0,
        estimated_pulse_idx - signal_region_half_points
    )

    if baseline_i_up == 0: 
        # Then baseline_i_low is also 0
        print(
            "In function get_fine_selection_regions(): "
            f"WARNING: For channel {input_ChannelWs.endpoint}-"
            f"{input_ChannelWs.channel}, there are not "
            "enough points before the estimated pulse "
            "position to define a baseline region, since "
            f"estimated_pulse_idx ({estimated_pulse_idx}) - "
            f"signal_region_half_points ({signal_region_half_points}) "
            f"< 0. Both baseline_i_low and baseline_i_up "
            "will be set to 0."
        )
    
    signal_i_up = min(
        input_ChannelWs.points_per_wf,
        estimated_pulse_idx + signal_region_half_points + 1
    )

    return (
        baseline_i_low,
        baseline_i_up,
        signal_i_up
    )

def regions_limits_are_consistent(
    baseline_i_low: int,
    baseline_i_up: int,
    signal_i_up: int,
    points_per_wf: int,
) -> bool:
    """This function checks that the given baseline
    and signal region indices are consistent with
    each other and with the given points_per_wf
    value. This function returns True if the
    following conditions are met:

        0 <= baseline_i_low < baseline_i_up < signal_i_up <= points_per_wf

    It returns False otherwise.

    Parameters
    ----------
    baseline_i_low: int
    baseline_i_up: int
    signal_i_up: int
    points_per_wf: int
        The total number of points in each waveform

    Returns
    -------
    None
    """

    return (0 <= baseline_i_low < baseline_i_up < signal_i_up <= points_per_wf)

def backup_input_parameters(
    params: BaseInputParams,
    output_folderpath: str
) -> None:
    """Backup the input parameters to a YAML file
    in the given output folder.

    Parameters
    ----------
    params: BaseInputParams
        The input parameters to backup.
    output_folderpath: str
        The path to the output folder.
    """
    with open(
        os.path.join(
            output_folderpath,
            'input_parameters_backup.yml'
        ),
        'w'

    ) as f:
        yaml.dump(params.dict(), f)

    return

def get_input_filepaths_for_run(
    base_folderpath: str,
    batch: int,
    pde: float,
    run: int
) -> list[str]:
    """Get the list of input file paths for a specific run
    within a batch and PDE setting.

    Parameters
    ----------
    base_folderpath: str
        The base directory containing batch and PDE
        subdirectories
    batch: int
        The batch number to look for
    pde: float
        The PDE (Photon Detection Efficiency) value
        to look for
    run: int
        The run number to filter files by

    Returns
    -------
    input_filepaths: list of str
        A list of file paths corresponding to the
        specified run. Files are selected if their
        filenames contain the substring "run_{run}_".
        The function expects the directory structure
        to be organized as
        f"{base_folderpath}/batch_{batch}/pde_{pde}/"

    Raises
    ------
    we.NonExistentDirectory
        If the expected folder path, based on the 'base_folderpath',
        'batch' and 'pde' parameters, does not exist or is not a
        directory.
    """

    if pde < 1.:
        # Assume that the PDE is given as a fraction
        pde_str = str(int(100. * pde))
    else:
        # Assume that the PDE is given as a percentage
        pde_str = str(int(pde))

    candidate_folderpath = os.path.join(
        base_folderpath,
        f"batch_{batch}/pde_{pde_str}"
    )

    # Check that the candidate folderpath exists
    # and that it is a directory
    if not os.path.exists(candidate_folderpath):
        raise we.NonExistentDirectory(
            we.GenerateExceptionMessage(
                1,
                'get_input_filepaths_for_run()',
                f"The path {candidate_folderpath} does not exist."
            )
        )
    elif not os.path.isdir(candidate_folderpath):
        raise we.NonExistentDirectory(
            we.GenerateExceptionMessage(
                1,
                'get_input_filepaths_for_run()',
                f"The path {candidate_folderpath} is not a directory."
            )
        )

    input_filepaths = []

    for filename in os.listdir(candidate_folderpath):
        if f"run_{str(run)}_" in filename:
            input_filepaths.append(
                os.path.join(
                    candidate_folderpath,
                    filename
                )
            )

    return input_filepaths

def join_channel_number(
    endpoint: int,
    channel: int,
) -> int:
    """This function concatenates the given endpoint and
    channel value. The second one is assumed to be a
    two-digit number. For example, if the endpoint is
    113 and the channel is 7, the returned channel number
    is 11307."""

    return (endpoint * 100) + int(channel)

def split_channel_number(
    joint_channel: int,
) -> tuple:
    """This function splits the given joint channel number
    into its constituent endpoint and channel values.
    For example, if the joint channel is 11307, the
    returned values will be (113, 7)."""

    endpoint = joint_channel // 100
    channel = joint_channel % 100

    return (endpoint, channel)

def arrange_dictionary_of_endpoints_and_channels(
    list_of_joint_channels: list[int]
) -> dict[int, list[int]]:
    """This function takes a list of joint channel numbers
    and arranges them into a dictionary where the keys are
    endpoint numbers and the values are lists of channel
    numbers.

    Example:
    - [11102, 11107, 11307, 10931, 11315, 11317] -> \
        {111: [2, 7], 113: [7, 15, 17], 109: [31]}
    """

    result = {}

    for joint_channel in list_of_joint_channels:
        endpoint, channel = split_channel_number(joint_channel)

        try:
            result[endpoint].append(channel)

        except KeyError:
            # Happens if it is the first
            # occurrence of this endpoint
            result[endpoint] = [channel]

    for endpoint in result.keys():
        # Prevent duplicates
        result[endpoint] = list(set(result[endpoint]))

    return result

def comes_from_channel( 
    waveform: Waveform, 
    channels: list,
) -> bool:
    """The channels list should contain integers which
    are the concatenation of the endpoint and a channel
    number, as returned by the join_channel_number()
    function. This function returns true if the
    given waveform comes from one of the channels in the
    list. Otherwise, it returns false."""

    aux = join_channel_number(
        waveform.endpoint,
        waveform.channel
    )

    if aux in channels:
        return True
    
    return False

def parse_numeric_list(input_string: str) -> list:
    """Converts the string representation of a list of
    numbers into the list of numbers itself. If at least
    one decimal point (i.e. '.') is present for any of the
    numbers, the input will be interpreted as a list of
    floats. If no decimal points are present, the input
    will be interpreted as a list of integers.
    
    Examples:
    - "[1, 2, 3]" -> [1, 2, 3]
    - "[1.2, 3, 5.]" -> [1.2, 3.0, 5.0]
    """

    if len(input_string) < 2:
        raise ValueError(
            "In function parse_numeric_list():"
            "The input string must contain at least 2 characters"
        )
    else:
        if input_string[0] != '[' or input_string[-1] != ']':
            raise ValueError(
                "In function parse_numeric_list():"
                "The input string must start with '[' and end with ']'"
            )

    # Remove the brackets
    input_string = input_string.strip()[1:-1]

    if len(input_string) == 0:
        return []

    # Split the string by commas
    items = input_string.split(',')

    # Remove whitespace around each element
    items = [item.strip() for item in items]

    # Type inference: if any item has a decimal point, 
    # we assume float for every item, otherwise int.
    fThereIsAFloat = any(
        ['.' in item for item in items]
    )

    cast = float if fThereIsAFloat else int

    return [cast(item) for item in items]

def compute_average_baseline_std(
    waveform_set: WaveformSet,
    baseline_analysis_label: str
) -> float:
    """For the waveforms in a given WaveformSet, this function computes
    the average of the signal standard deviation in the baseline region.

    Parameters
    ----------
    waveform_set: WaveformSet
        The WaveformSet object containing the waveforms of interest
    baseline_analysis_label: str
        The label of the analysis which, for each waveform, should
        contain the baseline standard deviation under the 'baseline_std'
        key in the analysis results.

    Returns
    ----------
    float
    """
    
    try:
        samples = [
            wf.analyses[baseline_analysis_label].result['baseline_std']
            for wf in waveform_set.waveforms
        ]

    except KeyError:
        raise KeyError(
            f"The analysis label '{baseline_analysis_label}' "
            "is not present in the analyses of the waveforms "
            "in the given WaveformSet, or it is, but it does "
            "not contain the 'baseline_std' key in its result."
        )
    
    return np.mean(np.array(samples))

def get_number_of_fitted_peaks(
    a_CalibrationHistogram: CalibrationHistogram
) -> int:
    """This function retrieves the number of fitted peaks
    from the given CalibrationHistogram object.

    Parameters
    ----------
    a_CalibrationHistogram: CalibrationHistogram
        The CalibrationHistogram object from which to
        retrieve the number of fitted peaks.

    Returns
    -------
    int
        The number of fitted peaks in the given
        CalibrationHistogram.
    """
    
    fit_parameters = a_CalibrationHistogram.gaussian_fits_parameters

    n_peaks = len(fit_parameters['scale'])

    if n_peaks != len(fit_parameters['mean']) or \
        n_peaks != len(fit_parameters['std']):

        raise Exception(
            "In function get_number_of_fitted_peaks(): "
            "Inconsistent number of fitted peaks found "
            "in the given CalibrationHistogram."
        )

    return n_peaks

def get_gain_snr_and_fit_parameters(
    grid_apa: ChannelWsGrid,
    excluded_channels: list,
    reset_excluded_channels: bool = False
) -> dict:

    data = {}

    for i in range(grid_apa.ch_map.rows):
        for j in range(grid_apa.ch_map.columns):

            endpoint = grid_apa.ch_map.data[i][j].endpoint
            channel = grid_apa.ch_map.data[i][j].channel

            if join_channel_number(
                endpoint,
                channel
            ) in excluded_channels:

                print(
                    "In function get_gain_snr_and_fit_parameters(): "
                    f"Excluding channel {endpoint}-{channel} ..."
                )
                if reset_excluded_channels:
                    try:
                        calibration_histogram = \
                            grid_apa.ch_wf_sets[endpoint][channel].calib_histo
                    except KeyError:
                        continue

                    if calibration_histogram is not None:
                        calibration_histogram._CalibrationHistogram__reset_gaussian_fit_parameters()

                continue

            try:
                fit_params = grid_apa.ch_wf_sets[endpoint][channel].\
                    calib_histo.gaussian_fits_parameters

            except KeyError:
                print(
                    "In function get_gain_snr_and_fit_parameters(): "
                    f"Skipping channel {endpoint}-{channel} "
                    "since it was not found in data."
                )
                continue

            fitted_peaks = get_number_of_fitted_peaks(
                grid_apa.ch_wf_sets[endpoint][channel].calib_histo
            )

            if fitted_peaks == 0:
                print(
                    "In function get_gain_snr_and_fit_parameters(): "
                    "No fitted peaks found for channel "
                    f"{endpoint}-{channel}. All of the "
                    "output entries (namely 'gain', 'snr', "
                    "'center_0', 'center_0_error', "
                    "'center_1', 'center_1_error', 'std_0', "
                    "'std_0_error', 'std_1', 'std_1_error') "
                    "will be set to NaN."
                )
                
                aux = {
                    'gain': np.nan,
                    'snr': np.nan,
                    'center_0': np.nan,
                    'center_0_error': np.nan,
                    'center_1': np.nan,
                    'center_1_error': np.nan,
                    'std_0': np.nan,
                    'std_0_error': np.nan,
                    'std_1': np.nan,
                    'std_1_error': np.nan
                }

            elif fitted_peaks == 1:
                print(
                    "In function get_gain_snr_and_fit_parameters(): "
                    "Only one fitted peak found for channel "
                    f"{endpoint}-{channel}. Since the gain and "
                    "the SNR cannot be computed, some of the "
                    "entries (namely 'gain', 'snr', 'center_1', "
                    "'center_1_error', 'std_1', 'std_1_error') "
                    "will be set to NaN."
                )

                aux = {
                    'gain': np.nan,
                    'snr': np.nan,
                    'center_0': fit_params['mean'][0][0],
                    'center_0_error': fit_params['mean'][0][1],
                    'center_1': np.nan,
                    'center_1_error': np.nan,
                    'std_0': fit_params['std'][0][0],
                    'std_0_error': fit_params['std'][0][1],
                    'std_1': np.nan,
                    'std_1_error': np.nan
                }

            else: # fitted_peaks >= 2:

                aux_gain = fit_params['mean'][1][0] - fit_params['mean'][0][0]
                aux = {
                    'gain': aux_gain,
                    'snr': aux_gain / fit_params['std'][0][0],
                    'center_0': fit_params['mean'][0][0],
                    'center_0_error': fit_params['mean'][0][1],
                    'center_1': fit_params['mean'][1][0],
                    'center_1_error': fit_params['mean'][1][1],
                    'std_0': fit_params['std'][0][0],
                    'std_0_error': fit_params['std'][0][1],
                    'std_1': fit_params['std'][1][0],
                    'std_1_error': fit_params['std'][1][1]
                }

            if endpoint not in data.keys():
                data[endpoint] = {}

            elif channel in data[endpoint].keys():
                raise Exception(
                    "In function get_gain_snr_and_fit_parameters(): "
                    f"An entry for channel {endpoint}-{channel} "
                    f"was already found when trying to save "
                    "the gain and SNR for this channel. Something "
                    "went wrong."
                )

            data[endpoint][channel] = aux

    return data

def get_vendor(
    endpoint: int,
    channel: int,
    sipm_vendor_df: Optional[pd.DataFrame] = None
) -> str:
    """This function retrieves the vendor name (which
    must match either 'HPK' or 'FBK') for the specified
    endpoint and channel from the given sipm_vendor_df
    DataFrame.

    Parameters
    ----------
    endpoint: int
    channel: int
    sipm_vendor_df: pd.DataFrame | None
        If defined, it must be a DataFrame containing
        the vendor information for SiPMs, with columns
        'endpoint', 'daphne_ch' and 'sipm'. This check
        is not done here.

    Returns
    -------
    str
        The vendor name (either 'HPK' or 'FBK') for the
        specified endpoint and channel. If the dataframe
        is not given, if the vendor is not recognized or
        if the endpoint-channel pair is not found in the
        DataFrame, the function returns 'unavailable'.
    """

    if sipm_vendor_df is None:
        vendor = 'unavailable'
        print(
            "In function get_vendor(): "
            "The vendor dataframe was not provided. "
            "The vendor returned by this function "
            "will be set to 'unavailable'."
        )
    else:
        try:
            vendor = sipm_vendor_df[
                (sipm_vendor_df['endpoint'] == endpoint) &
                (sipm_vendor_df['daphne_ch'] == channel)
            ]['sipm'].values[0]

            if vendor not in ('HPK', 'FBK'):
                print(
                    "In function get_vendor(): "
                    f"Channel {endpoint}-{channel} has an "
                    f"unrecognized vendor '{vendor}' in the "
                    "given vendor dataframe. The vendor "
                    "returned by this function will be "
                    "set to 'unavailable'."
                )
                vendor = 'unavailable'

        # Happens if the current endpoint-channel
        # pair is not found in the vendor_df dataframe
        except IndexError:
            print(
                "In function get_vendor(): "
                f"Channel {endpoint}-{channel} was not "
                "found in the given vendor dataframe. "
                "The vendor returned by this function "
                "will be set to 'unavailable'."
            )
            vendor = 'unavailable'

    return vendor

def save_data_to_dataframe(
    date: str,
    batch: int,
    apa: int,
    pde: float,
    packed_gain_snr_and_SPE_info: dict,
    packed_limits: dict,
    path_to_output_file: str,
    sipm_vendor_df: Optional[pd.DataFrame] = None,
    actually_save: bool = True,
    overwrite: bool = False
) -> None:
    """This function saves the given data to a CSV file
    located at path_to_output_file. If the file does not
    exist, it is created. If it exists, the new data is
    appended to it. The output CSV file will contain the
    following columns:

        - date
        - batch
        - APA
        - PDE
        - endpoint
        - channel
        - channel_iterator
        - vendor
        - OV#
        - OV_V
        - gain
        - snr
        - center_0
        - center_0_error
        - center_1
        - center_1_error
        - std_0
        - std_0_error
        - std_1
        - std_1_error
        - SPE_mean_amplitude
        - SPE_mean_adcs
        - fine_selection_baseline_i_low
        - fine_selection_baseline_i_up
        - fine_selection_signal_i_up
        - integration_lower_limit
        - integration_upper_limit

    Parameters
    ----------
    date: str
        The date when the data for the specified batch was
        obtained. This date value will appear in every
        row of the 'date' column of the output dataframe.
    batch: int
        The batch with which the data was obtained. This
        batch value will appear in every row of the 'batch'
        column of the output dataframe.
    apa: int
        The APA with which the data was obtained. This
        APA value will appear in every row of the 'APA'
        column of the output dataframe.
    pde: float
        The PDE with which the data was obtained. This
        PDE value will appear in every row of the 'PDE'
        column of the output dataframe.
    packed_gain_snr_and_SPE_info: dict
        It is expected to have the following form:
            {
                endpoint1: {
                    channel1: {
                        'gain': gain_value_11,
                        'snr': ...,
                        'center_0': ...,
                        'center_0_error': ...,
                        'center_1': ...,
                        'center_1_error': ...,
                        'std_0': ...,
                        'std_0_error': ...,
                        'std_1': ...,
                        'std_1_error': ...,
                        'SPE_mean_amplitude': ...,
                        'SPE_mean_adcs': SPE_mean_adcs_value_11
                    },
                    channel2: {
                        'gain': gain_value_12,
                        'snr': ...,
                        'center_0': ...,
                        'center_0_error': ...,
                        'center_1': ...,
                        'center_1_error': ...,
                        'std_0': ...,
                        'std_0_error': ...,
                        'std_1': ...,
                        'std_1_error': ...,
                        'SPE_mean_amplitude': ...,
                        'SPE_mean_adcs': SPE_mean_adcs_value_12
                    },
                    ...
                },
                endpoint2: {
                    channel1: {
                        'gain': gain_value_21,
                        'snr': ...,
                        'center_0': ...,
                        'center_0_error': ...,
                        'center_1': ...,
                        'center_1_error': ...,
                        'std_0': ...,
                        'std_0_error': ...,
                        'std_1': ...,
                        'std_1_error': ...,
                        'SPE_mean_amplitude': ...,
                        'SPE_mean_adcs': SPE_mean_adcs_value_21
                    },
                    channel2: {
                        'gain': gain_value_22,
                        'snr': ...,
                        'center_0': ...,
                        'center_0_error': ...,
                        'center_1': ...,
                        'center_1_error': ...,
                        'std_0': ...,
                        'std_0_error': ...,
                        'std_1': ...,
                        'std_1_error': ...,
                        'SPE_mean_amplitude': ...,
                        'SPE_mean_adcs': SPE_mean_adcs_value_22
                    },
                    ...
                },
                ...
            }
        where the endpoint and channel values are integers.
    packed_limits: dict
        It is expected to have the following form:
            {
                endpoint1: {
                    channel1: {
                        'fine_selection': (
                            baseline_i_low11,
                            baseline_i_up11,
                            signal_i_up11,
                            ...
                        ),
                        'integration': (
                            integration_lower_limit11,
                            integration_upper_limit11
                        )
                    },
                    channel2: {
                        'fine_selection': (
                            baseline_i_low12,
                            baseline_i_up12,
                            signal_i_up12,
                            ...
                        ),
                        'integration': (
                            integration_lower_limit12,
                            integration_upper_limit12
                        )
                    },
                    ...
                },
                endpoint2: {
                    channel1: {
                        'fine_selection': (
                            baseline_i_low21,
                            baseline_i_up21,
                            signal_i_up21,
                            ...
                        ),
                        'integration': (
                            integration_lower_limit21,
                            integration_upper_limit21
                        )
                    },
                    channel2: {
                        'fine_selection': (
                            baseline_i_low22,
                            baseline_i_up22,
                            signal_i_up22,
                            ...
                        ),
                        'integration': (
                            integration_lower_limit22,
                            integration_upper_limit22
                        )
                    },
                    ...
                },
                ...
            }
        where the endpoint, channel and deepest values are
        integers.
    path_to_output_file: str
        The path to the output CSV file
    sipm_vendor_df: pd.DataFrame, optional
        If None, the 'vendor' column of the output CSV file will
        be filled with strings which match 'unavailable'. If it
        is defined, then it is a pandas.DataFrame which should
        contain the columns 'endpoint', 'daphne_ch', and 'sipm',
        from which the endpoint, the channel and the vendor
        associated to each channel can be retrieved, respectively.
        In this case, the 'vendor' column of the output CSV file
        will be filled with the vendor information retrieved from
        this DataFrame.
    actually_save: bool
        If True, the data will actually be saved to the output
        CSV file. If False, the function will run as usual, but
        the data will not be saved to the output CSV file. It is
        useful for testing cases where one does not want to
        potentially include spurious data to a running dataframe
        by mistake.
    overwrite: bool
        Whether to potentially overwrite existing rows in the
        output dataframe
    """

    # PDE-to-OV mapping for HPK sipms
    hpk_ov = {
        0.4: 2.0,
        0.45: 2.5,
        0.50: 3.0
    }

    # PDE-to-OV mapping for FBK sipms
    fbk_ov = {
        0.4: 3.5,
        0.45: 4.5,
        0.50: 7.0
    }

    # Enumeration of PDE values
    ov_no = {
        0.4: 1,
        0.45: 2,
        0.50: 3
    }

    hpk_ov = hpk_ov[pde]
    fbk_ov = fbk_ov[pde]
    ov_no = ov_no[pde]

    expected_columns = {
        "date": [],
        "batch": [],
        "APA": [],
        "PDE": [],
        "endpoint": [],
        "channel": [],
        "channel_iterator": [],
        "vendor": [],
        "OV#": [],
        "OV_V": [],
        "gain": [],
        "snr": [],
        "center_0": [],
        "center_0_error": [],
        "center_1": [],
        "center_1_error": [],
        "std_0": [],
        "std_0_error": [],
        "std_1": [],
        "std_1_error": [],
        "SPE_mean_amplitude": [],
        "SPE_mean_adcs": [],
        "fine_selection_baseline_i_low": [],
        "fine_selection_baseline_i_up": [],
        "fine_selection_signal_i_up": [],
        "integration_lower_limit": [],
        "integration_upper_limit": []
    }

    # If the file does not exist, create it
    if not os.path.exists(path_to_output_file):
        df = pd.DataFrame(expected_columns)

        # Force column-wise types
        df['date'] = df['date'].astype(str)
        df['batch'] = df['batch'].astype(int)
        df['APA'] = df['APA'].astype(int)
        df['PDE'] = df['PDE'].astype(float)
        df['endpoint'] = df['endpoint'].astype(int)
        df['channel'] = df['channel'].astype(int)
        df['channel_iterator'] = df['channel_iterator'].astype(int)
        df['vendor'] = df['vendor'].astype(str)
        df['OV#'] = df['OV#'].astype(int)
        df['OV_V'] = df['OV_V'].astype(float)
        df['gain'] = df['gain'].astype(float)
        df['snr'] = df['snr'].astype(float)
        df['center_0'] = df['center_0'].astype(float)
        df['center_0_error'] = df['center_0_error'].astype(float)
        df['center_1'] = df['center_1'].astype(float)
        df['center_1_error'] = df['center_1_error'].astype(float)
        df['std_0'] = df['std_0'].astype(float)
        df['std_0_error'] = df['std_0_error'].astype(float)
        df['std_1'] = df['std_1'].astype(float)
        df['std_1_error'] = df['std_1_error'].astype(float)
        df['SPE_mean_amplitude'] = df['SPE_mean_amplitude'].astype(float)
        df['SPE_mean_adcs'] = df['SPE_mean_adcs'].astype(object) # cannot specify list[float] or np.ndarray
        df['fine_selection_baseline_i_low'] = df['fine_selection_baseline_i_low'].astype(int)
        df['fine_selection_baseline_i_up'] = df['fine_selection_baseline_i_up'].astype(int)
        df['fine_selection_signal_i_up'] = df['fine_selection_signal_i_up'].astype(int)
        df['integration_lower_limit'] = df['integration_lower_limit'].astype(int)
        df['integration_upper_limit'] = df['integration_upper_limit'].astype(int)

        df.to_csv(
            path_to_output_file,
            index=False
        )

    df = pd.read_csv(path_to_output_file)

    if set(df.columns) != expected_columns.keys():
        raise Exception(
            "In function save_data_to_dataframe(): "
            "The columns of the found dataframe do not "
            "match the expected ones. Something went wrong."
        )
    else:
        for endpoint in packed_gain_snr_and_SPE_info.keys():
            for channel in packed_gain_snr_and_SPE_info[endpoint]:

                vendor = get_vendor(
                    endpoint,
                    channel,
                    sipm_vendor_df=sipm_vendor_df
                )

                try:
                    fine_selection_limits = \
                        packed_limits[endpoint][channel]['fine_selection']
                except KeyError:
                    print(
                        "In function save_data_to_dataframe(): "
                        f"Fine-selection limits for channel {endpoint}-"
                        f"{channel} were not found. Setting them to NaN."
                    )
                    fine_selection_limits = (np.nan, np.nan, np.nan)

                try:
                    integration_limits = \
                        packed_limits[endpoint][channel]['integration']

                except KeyError:
                    print(
                        "In function save_data_to_dataframe(): "
                        "Integration limits for channel "
                        f"{endpoint}-{channel} were not found. "
                        "Setting them to NaN."
                    )
                    integration_limits = (np.nan, np.nan)

                try:
                    aux_SPE_mean_amplitude = \
                        abs(packed_gain_snr_and_SPE_info[endpoint][channel]["SPE_mean_amplitude"])
                    
                    aux_SPE_mean_adcs = [
                        round(float(x), 4) for x in \
                        packed_gain_snr_and_SPE_info[endpoint][channel]["SPE_mean_adcs"]
                    ]
                
                except KeyError:
                    print(
                        "In function save_data_to_dataframe(): "
                        "SPE mean amplitude or SPE mean adcs for "
                        f"channel {endpoint}-{channel} were not found. "
                        "Setting them to NaN."
                    )
                    aux_SPE_mean_amplitude = np.nan
                    aux_SPE_mean_adcs = np.nan

                # Assemble the new row
                new_row = {
                    "date": [date],
                    "batch": [int(batch)],
                    "APA": [int(apa)],
                    "PDE": [pde],
                    "endpoint": [endpoint],
                    "channel": [channel],
                    "channel_iterator": [get_channel_iterator(
                        apa,
                        endpoint,
                        channel
                    )],
                    "vendor": [vendor],
                    "OV#": [ov_no],
                    # We've made sure that vendor is either
                    # 'HPK', 'FBK' or 'unavailable'
                    "OV_V": [
                        {
                            'HPK': hpk_ov,
                            'FBK': fbk_ov,
                            'unavailable': np.nan
                        }[vendor]
                    ],
                    "gain": [packed_gain_snr_and_SPE_info[endpoint][channel]["gain"]],
                    "snr": [packed_gain_snr_and_SPE_info[endpoint][channel]["snr"]],
                    "center_0": [packed_gain_snr_and_SPE_info[endpoint][channel]["center_0"]],
                    "center_0_error": [packed_gain_snr_and_SPE_info[endpoint][channel]["center_0_error"]],
                    "center_1": [packed_gain_snr_and_SPE_info[endpoint][channel]["center_1"]],
                    "center_1_error": [packed_gain_snr_and_SPE_info[endpoint][channel]["center_1_error"]],
                    "std_0": [packed_gain_snr_and_SPE_info[endpoint][channel]["std_0"]],
                    "std_0_error": [packed_gain_snr_and_SPE_info[endpoint][channel]["std_0_error"]],
                    "std_1": [packed_gain_snr_and_SPE_info[endpoint][channel]["std_1"]],
                    "std_1_error": [packed_gain_snr_and_SPE_info[endpoint][channel]["std_1_error"]],
                    "SPE_mean_amplitude": [aux_SPE_mean_amplitude],
                    "SPE_mean_adcs": [aux_SPE_mean_adcs],
                    "fine_selection_baseline_i_low": [fine_selection_limits[0]],
                    "fine_selection_baseline_i_up": [fine_selection_limits[1]],
                    "fine_selection_signal_i_up": [fine_selection_limits[2]],
                    "integration_lower_limit": [integration_limits[0]],
                    "integration_upper_limit": [integration_limits[1]]
                }

                # Check if there is already an entry for the
                # given endpoint and channel for this OV and batch
                matching_rows_indices = df[
                    (df['batch'] == batch) &
                    (df['endpoint'] == endpoint) &       
                    (df['channel'] == channel) &
                    (df['OV#'] == ov_no)
                ].index          

                if len(matching_rows_indices) > 1:
                    raise Exception(
                        "In function save_data_to_dataframe(): "
                        "There are already more than one rows "
                        f"for the given channel ({endpoint}-{channel}"
                        f"), batch ({batch}) and OV# ({ov_no})"
                        ". Something went wrong."
                    )

                elif len(matching_rows_indices) == 1:
                    if overwrite:

                        row_index = matching_rows_indices[0]

                        new_row = {key: new_row[key][0] for key in new_row.keys()}  

                        if actually_save:
                            df.loc[row_index, :] = new_row

                    else:
                        print(
                            "In function save_data_to_dataframe(): "
                            f"Since overwrite is set to False, "
                            f"and an entry for batch {batch}, "
                            f"channel {endpoint}-{channel} at OV#"
                            f" {ov_no} already exists, the new "
                            "entry for this channel will not be saved."
                        )

                else: # len(matching_rows_indices) == 0
                    if actually_save:
                        df = pd.concat(
                            [df, pd.DataFrame(new_row)],
                            axis=0,
                            ignore_index=True
                        )
                        df.reset_index()
        df.to_csv(
            path_to_output_file,
            index=False,
            na_rep="nan"
        )

    return

def dump_object_to_pickle(
    object, 
    saving_folderpath : str,
    output_filename : str,
    verbose : bool = True
) -> None:
    """This function gets the following positional argument:

    - object
    - saving_folderpath (str): Path to the folder
    where to save the file.
    - output_filename (str): Name of the output 
    pickle file.

    And the following keyword argument:

    - verbose (bool): Whether to print functioning
    related messages.
    
    It saves the given object, object, to a pickle file 
    which is stored in the path given by saving_filepath
    """

    # If the saving folder does not exist, create it
    if not os.path.exists(saving_folderpath):

        if verbose:
            print(
                "In function dump_object_to_pickle(): Folder "
                f"{saving_folderpath} does not exist. It will "
                "be created."
            )

        os.makedirs(saving_folderpath)

    # Create the output filepath
    output_filepath = os.path.join(
        saving_folderpath, 
        output_filename
    )
    
    with open(
        output_filepath, 
        "wb"
    ) as output_file:

        pickle.dump(object, output_file)

        return

def next_subsample(
    current_subsample: int,
    read_quantity: int,
    required_quantity: int
) -> int:
    """In the context of a reading process which uses a certain
    subsampling rate, this function gives an estimation of the
    subsampling rate which should be used next to get the required
    number of elements.

    Parameters
    ----------
    current_subsample: int
        The subsampling rate which yielded the number of elements
        given by read_quantity
    read_quantity: int
        The number of elements yielded by the last reading process
        which used the subsampling rate given by current_subsample
    required_quantity: int
        The required number of elements

    Returns
    ----------
    proposed_subsample: int
        The subsampling rate which should be used next to get
        the required number of elements
    """

    if current_subsample <= 1:
        return 1
    
    else:
        estimated_available_quantity = \
            current_subsample * read_quantity

        for proposed_subsample in reversed(range(1, current_subsample+1)):
            # int() always truncates
            if int(estimated_available_quantity / proposed_subsample) \
                >= required_quantity:
                break

        # If not even proposed_subsample = 1 reaches the desired
        # quantity, then return the best-case scenario which is
        # still proposed_subsample = 1

        return proposed_subsample
    
def get_nbins_and_channel_wise_domain(
    batch: int,
    apa: int,
    pde: float,
    gain_seeds_filepath: str,
    bins_per_charge_peak: int,
    domain_ll_in_gain_seeds: float = -1.,
    domain_ul_in_gain_seeds: float = 5.,
    verbose: bool = True
) -> tuple[dict, dict]:
    """This function reads the gain seeds, for the specified
    batch and PDE, from the given CSV file and computes the
    number of bins and domain for the charge calibration
    histograms for each channel of the specified APA which
    appear in the gain seeds file.

    Parameters
    ----------
    batch: int
        The batch number to look for
    apa: int
        The APA number to look for
    pde: float
        The PDE (Photon Detection Efficiency) value
        to look for
    gain_seeds_filepath: str
        The path to the CSV file which must contain, at
        least, the following columns:
            - 'batch'
            - 'APA'
            - 'PDE'
            - 'endpoint'
            - 'channel'
            - 'gain'
        The 'gain' column contains the gain seeds for
        the charge calibration histograms.
    bins_per_charge_peak: int
        The number of bins per charge peak to use when
        computing the number of bins. It is assumed
        that the gain seed is a good estimate of the
        width (i.e. valley to valley distance) of each
        charge peak.
    domain_ll_in_gain_seeds: float
        The lower limit of the domain, in units of the
        gain seed, to use when computing the domain
    domain_ul_in_gain_seeds: float
        The upper limit of the domain, in units of the
        gain seed, to use when computing the domain
    verbose: bool
        Whether to print functioning related messages

    Returns
    -------
    nbins: int
        The number of bins to use for the charge calibration
        histogram of every channel. It is computed as
        round(
            bins_per_charge_peak * \
                (domain_ul_in_gain_seeds - domain_ll_in_gain_seeds)
        )
    domain: Dict[int, Dict[int, np.ndarray]]
        A dictionary where the keys are endpoint numbers
        and the values are dictionaries where the keys
        are channel numbers and the values are 2x1 numpy
        arrays where (domain[0], domain[1]) gives the
        range to consider for the charge calibration
        histogram.
    """

    if verbose:
        print(
            "In function get_nbins_and_channel_wise_domain(): "
            f"Reading gain seeds from {gain_seeds_filepath}"
        )

    try:
        df = pd.read_csv(gain_seeds_filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"In function get_nbins_and_channel_wise_domain(): "
            f"The file {gain_seeds_filepath} was not found."
        )
    
    required_columns = [
        'batch',
        'APA',
        'PDE',
        'endpoint',
        'channel',
        'gain'
    ]

    for col in required_columns:
        if col not in df.columns:
            raise Exception(
                f"In function get_nbins_and_channel_wise_domain(): "
                f"The column '{col}' was not found in the given "
                f"gain seeds file, {gain_seeds_filepath}. "
                f"Make sure that the file contains at least the "
                f"following columns: {required_columns}."
            )
        
    # Filter the dataframe by the given batch, APA and PDE
    df = df[
        (df['batch'] == batch) &
        (df['APA'] == apa) &
        (df['PDE'] == pde)
    ]

    if df.empty:
        raise Exception(
            f"In function get_nbins_and_channel_wise_domain(): "
            f"No entries were found in the given gain seeds file, "
            f"{gain_seeds_filepath}, for batch {batch}, APA {apa} "
            f"and PDE {pde}."
        )
    
    domain = {}

    for _, row in df.iterrows():
        endpoint = int(row['endpoint'])
        channel = int(row['channel'])
        gain_seed = float(row['gain'])

        if np.isnan(gain_seed):
            print(
                "In function get_nbins_and_channel_wise_domain(): "
                f"WARNING: For channel {endpoint}-{channel}, found a "
                "NaN gain seed. A domain for this channel will not be "
                "added to the output dictionary."
            )
            continue

        if endpoint not in domain.keys():
            domain[endpoint] = {}

        if channel in domain[endpoint].keys():
            raise Exception(
                "In function get_nbins_and_channel_wise_domain(): "
                f"Duplicate entry for channel {endpoint}-{channel} "
                f"found in the given gain seeds file, {gain_seeds_filepath},"
                f" for batch {batch}, APA {apa} and PDE {pde}. Each "
                "endpoint-channel pair should appear only once."
            )

        domain[endpoint][channel] = np.array([
            domain_ll_in_gain_seeds * gain_seed,
            domain_ul_in_gain_seeds * gain_seed
        ])

    nbins = round(
        bins_per_charge_peak * (
            domain_ul_in_gain_seeds - domain_ll_in_gain_seeds
        )
    )

    return nbins, domain

def add_integration_and_fine_selection_limits_to_persistence_heatmaps(
    persistence_figure: pgo.Figure,
    grid_apa: ChannelWsGrid,
    current_excluded_channels: list,
    limits: Dict[int, Dict[int, Dict[str, Tuple[int]]]]
) -> None:
    """This function adds the integration limits and
    the fine-selection thresholds to the persistence
    heatmaps, channel by channel. The style parameters
    of the lines used to draw the limits are hardcoded
    in the body of this function.

    Parameters
    ----------
    persistence_figure: plotly.graph_objects.Figure
        The figure containing the persistence heatmaps
    grid_apa: ChannelWsGrid
        The ChannelWsGrid object containing the
        channel-waveform sets
    current_excluded_channels: list of int
        A list of joint channel numbers (as returned
        by the join_channel_number() function) which
        should be excluded from having their integration
        limits drawn in the persistence heatmaps
    limits: Dict[int, Dict[int, Tuple[int, int]]]
        A nested dictionary with the following format:
            {
                endpoint1: {
                    channel1: {
                        'fine_selection': (
                            baseline_i_low11,
                            baseline_i_up11,
                            signal_i_up11,
                            baseline_threshold11,
                            signal_threshold11
                        ),
                        'integration': (
                            integration_lower_limit11,
                            integration_upper_limit11
                        )
                    },
                    channel2: {
                        'fine_selection': (
                            baseline_i_low12,
                            baseline_i_up12,
                            signal_i_up12,
                            baseline_threshold12,
                            signal_threshold12
                        ),
                        'integration': (
                            integration_lower_limit12,
                            integration_upper_limit12
                        )
                    },
                    ...
                },
                endpoint2: {
                    channel1: {
                        'fine_selection': (
                            baseline_i_low21,
                            baseline_i_up21,
                            signal_i_up21,
                            baseline_threshold21,
                            signal_threshold21
                        ),
                        'integration': (
                            integration_lower_limit21,
                            integration_upper_limit21
                        )
                    },
                    channel2: {
                        'fine_selection': (
                            baseline_i_low22,
                            baseline_i_up22,
                            signal_i_up22,
                            baseline_threshold22,
                            signal_threshold22
                        ),
                        'integration': (
                            integration_lower_limit22,
                            integration_upper_limit22
                        )
                    },
                    ...
                },
                ...
            }
        where the endpoint, channel and deepest values are
        integers.

    Returns
    -------
    None
    """

    # Add the integration limits to the
    # persistence heatmaps, channel by channel
    for endpoint in grid_apa.ch_wf_sets.keys():
        for channel in grid_apa.ch_wf_sets[endpoint].keys():

            if join_channel_number(
                endpoint,
                channel,
            ) in current_excluded_channels:
                continue

            found_it, channel_position = grid_apa.ch_map.find_channel(
                UniqueChannel(
                    endpoint,
                    channel
                )
            )

            if not found_it:
                print(
                    "In function add_integration_and_fine_selection_"
                    "limits_to_persistence_heatmaps(): "
                    "WARNING: Something went wrong. Channel "
                    f"{endpoint}-{channel} retrieved from the "
                    "ch_wf_sets attribute of the current "
                    "ChannelWsGrid object, was not found in "
                    "its own ch_map attribute. Skipping this "
                    "channel."
                )
                continue

            # Unpack the channel position
            i, j = channel_position

            try:
                aux_integration_limits = \
                    limits[endpoint][channel]['integration']

            except KeyError:
                print(
                    "In function add_integration_and_fine_selection_"
                    "limits_to_persistence_heatmaps(): "
                    "Could not find the integration limits "
                    f"for channel {endpoint}-{channel}. They "
                    "will not be drawn in the persistence "
                    "heatmap."
                )
            else:
                # Integration lower limit
                persistence_figure.add_shape(
                    type="line",
                    x0=aux_integration_limits[0],
                    x1=aux_integration_limits[0],
                    y0=0,
                    y1=1,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash"
                    ),
                    xref='x',
                    yref='y domain',
                    row=i + 1,
                    col=j + 1,
                )

                # Integration upper limit
                persistence_figure.add_shape(
                    type="line",
                    x0=aux_integration_limits[1],
                    x1=aux_integration_limits[1],
                    y0=0,
                    y1=1,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash"
                    ),
                    xref='x',
                    yref='y domain',
                    row=i + 1,
                    col=j + 1,
                )

            try:
                aux_fine_selection_limits = \
                    limits[endpoint][channel]['fine_selection']

            except KeyError:
                print(
                    "In function add_integration_and_fine_selection_"
                    "limits_to_persistence_heatmaps(): "
                    "Could not find the fine-selection limits "
                    f"for channel {endpoint}-{channel}. They "
                    "will not be drawn in the persistence "
                    "heatmap."
                )
            else:
                # Positive baseline threshold
                persistence_figure.add_shape(
                    type="line",
                    x0=aux_fine_selection_limits[0],
                    x1=aux_fine_selection_limits[1],
                    y0=aux_fine_selection_limits[3],
                    y1=aux_fine_selection_limits[3],
                    line=dict(
                        color="red",
                        width=2,
                        dash="solid"
                    ),
                    xref='x',
                    yref='y',
                    row=channel_position[0] + 1,
                    col=channel_position[1] + 1,
                )

                # Negative baseline threshold
                persistence_figure.add_shape(
                    type="line",
                    x0=aux_fine_selection_limits[0],
                    x1=aux_fine_selection_limits[1],
                    y0=-1 * aux_fine_selection_limits[3],
                    y1=-1 * aux_fine_selection_limits[3],
                    line=dict(
                        color="red",
                        width=2,
                        dash="solid"
                    ),
                    xref='x',
                    yref='y',
                    row=channel_position[0] + 1,
                    col=channel_position[1] + 1,
                )

                # Artificial vertical line connecting the negative
                # baseline threshold with the signal threshold
                persistence_figure.add_shape(
                    type="line",
                    x0=aux_fine_selection_limits[1],
                    x1=aux_fine_selection_limits[1],
                    y0=-1 * aux_fine_selection_limits[3],
                    y1=aux_fine_selection_limits[4],
                    line=dict(
                        color="red",
                        width=2,
                        dash="solid"
                    ),
                    xref='x',
                    yref='y',
                    row=channel_position[0] + 1,
                    col=channel_position[1] + 1,
                )

                # Signal threshold
                persistence_figure.add_shape(
                    type="line",
                    x0=aux_fine_selection_limits[1],
                    x1=aux_fine_selection_limits[2],
                    y0=aux_fine_selection_limits[4],
                    y1=aux_fine_selection_limits[4],
                    line=dict(
                        color="red",
                        width=2,
                        dash="solid"
                    ),
                    xref='x',
                    yref='y',
                    row=channel_position[0] + 1,
                    col=channel_position[1] + 1,
                )
    
    return

def __infer_SPE_peak_index(
    gaussian_fits_parameters: dict,
    gain: float
) -> int:
    """This helper function should only be called from
    add_SPE_info_to_output_dictionary(), where the proper
    well-formedness checks have already been done. It
    infers the index of the Single Photo-Electron (SPE,
    1-PE) peak from the fitted Gaussian parameters of
    a calibration histogram, using the provided gain
    value.

    Parameters
    ----------
    gaussian_fits_parameters: dict
        A dictionary containing the fitted Gaussian
        parameters of a calibration histogram.
    gain: float
        The gain value used to infer the number of
        photo-electrons corresponding to each peak
        in the fitted Gaussian parameters.

    Returns
    -------
    int
        The index of the SPE (1-PE) peak in the
        fitted Gaussian parameters.

    Raises
    ------
    Exception
        If the closest peak to 1-PE found in the
        fitted Gaussian parameters is more compatible
        with 0-PE or 2-PE than it is with 1-PE.
    """

    # inferred_number_of_PEs[i] gives the (inferred) number
    # of photo-electrons corresponding to the i-th peak
    # in the fitted Gaussian parameters
    inferred_number_of_PEs = []

    for center_pack in gaussian_fits_parameters['mean']:
        inferred_number_of_PEs.append(
            # center_pack[1] gives the error
            # on the center, not the center itself
            center_pack[0] / gain
        )

    # Find the peak whose inferred number of PEs
    # is closest to 1
    targeted_peak_idx = int(
        np.argmin(
            np.abs(
                np.array(inferred_number_of_PEs) - 1.0
            )
        )
    )

    if inferred_number_of_PEs[targeted_peak_idx] < 0.5:
        raise Exception(
            "In function __infer_SPE_peak_index(): "
            "The closest peak to 1-PE found in the fitted "
            "Gaussian parameters is more compatible with "
            f"0-PE than it is with 1-PE."
        )
    elif inferred_number_of_PEs[targeted_peak_idx] > 1.5:
        raise Exception(
            "In function __infer_SPE_peak_index(): "
            "The closest peak to 1-PE found in the fitted "
            "Gaussian parameters is more compatible with "
            f"2-PE than it is with 1-PE."
        )

    return targeted_peak_idx

def add_SPE_info_to_output_dictionary(
    grid_apa: ChannelWsGrid,
    excluded_channels: list,
    output_data: Dict[int, Dict[int, ...]],
    half_width_around_peak_in_stds: float = 1.0
) -> Dict[int, Dict[int, ...]]:
    """This function processes each channel in the given
    ChannelWsGrid to extract information about the Single
    Photo-Electron (SPE, 1-PE) peak from the fitted calibration
    histograms. For each channel, it identifies waveforms that
    contribute to the SPE peak based on their integrated charge
    values and computes the mean SPE waveform. The mean SPE
    adcs, as well as the mean SPE amplitude (it is assumed
    that the pulse is negative) and the indices of the waveforms
    which contribute to the mean SPE (relative to the waveforms
    attribute of each ChannelWs object) are added to the output
    dictionary.
    
    To this end, the function identifies the SPE peak as the
    second peak (index 1) in the fitted Gaussian parameters,
    then selects waveforms whose integrated charge falls within
    a specified range around this peak.
    
    Parameters
    ----------
    grid_apa: ChannelWsGrid
        The ChannelWsGrid object containing the channel-waveform
        sets with fitted calibration histograms and Gaussian
        fit parameters
    excluded_channels: list of int
        A list of joint channel numbers (as returned by
        join_channel_number()) which should be excluded from SPE
        processing
    output_data: Dict[int, Dict[int, ...]]
        Dictionary where the keys are endpoint numbers and values
        are dictionaries where the keys are channel numbers and
        values contain channel data, p.e. the gain and the S/N.
        This dictionary will be modified in-place to include SPE
        information.
    half_width_around_peak_in_stds: float, default 1.0
        The half-width around the SPE peak mean, expressed in
        units of the peak's standard deviation, used to select
        waveforms for SPE characterization. Waveforms with
        integrated charge in the range [mean - half_width*std, 
        mean + half_width*std] are included in the SPE computation.
        
    Returns
    -------
    Dict[int, Dict[int, ...]]
        The modified output_data dictionary with added SPE
        information for each processed channel. For each channel,
        the following keys are added:
        - 'SPE_mean_amplitude': float
            The minimum ADC value of the mean SPE waveform (peak amplitude)
        - 'SPE_mean_adcs': list of float
            The complete mean SPE waveform as a list of ADC values
        - 'SPE_idcs': list of int
            Indices of the waveforms that contributed to the SPE
            characterization (relative to the waveforms attribute
            of each ChannelWs object)
    """

    for endpoint in grid_apa.ch_wf_sets.keys():
        for channel in grid_apa.ch_wf_sets[endpoint].keys():

            if join_channel_number(
                endpoint,
                channel
            ) in excluded_channels:
                print(
                    "In function add_SPE_info_to_output_dictionary(): "
                    f"Excluding channel {endpoint}-{channel} ..."
                )
                continue

            try:
                current_channel_output_data = output_data[endpoint][channel]
            except KeyError:
                print(
                    "In function add_SPE_info_to_output_dictionary(): "
                    f"WARNING: Although channel {endpoint}-{channel} "
                    "is not excluded, it was not found in the given "
                    "output data dictionary. The SPE information for "
                    "this channel will not be computed."
                )
                continue
            
            # Some convenient definitions
            current_ChannelWs = grid_apa.ch_wf_sets[endpoint][channel]
            current_CalibrationHistogram = current_ChannelWs.calib_histo
            aux_fit_params = current_CalibrationHistogram.gaussian_fits_parameters

            aux_gain = current_channel_output_data['gain']
            if np.isnan(aux_gain):
                print(
                    "In function add_SPE_info_to_output_dictionary(): "
                    f"WARNING: The gain for channel {endpoint}-{channel} "
                    "is NaN, meaning that two Gaussian peaks were not "
                    "successfully fitted to its calibration histogram. "
                    "The SPE information for this channel won't be computed."
                )
                continue

            # If the 'gain' entry is available, it means that
            # get_gain_snr_and_fit_parameters() made sure that the number
            # of fitted peaks is at least 2
            try:
                targeted_peak_idx = __infer_SPE_peak_index(
                    aux_fit_params,
                    aux_gain
                )
            except Exception as e:
                print(
                    "In function add_SPE_info_to_output_dictionary(): "
                    "WARNING: Could not infer the SPE peak index for "
                    f"channel {endpoint}-{channel}. The SPE information "
                    f"for this channel won't be computed. Details: {e}"
                )
                continue

            try:
                SPE_peak_mean = aux_fit_params['mean'][targeted_peak_idx][0]
                SPE_peak_std = aux_fit_params['std'][targeted_peak_idx][0]

            except IndexError:
                print(
                    "In function add_SPE_info_to_output_dictionary(): "
                    "The information of the 1-PE peak was not found "
                    f"for channel {endpoint}-{channel}. The SPE "
                    "information for this channel won't be computed."
                )
                continue

            # Find the lowest bin whose entries will contribute to the mean SPE
            for i in range(len(current_CalibrationHistogram.edges)):
                if current_CalibrationHistogram.edges[i] > \
                    SPE_peak_mean - (half_width_around_peak_in_stds * SPE_peak_std):

                    inclusive_lower_limit_idx = i
                    break

            # Find the highest bin whose entries will contribute to the mean SPE
            for i in range(
                inclusive_lower_limit_idx + 1,
                len(current_CalibrationHistogram.edges)
            ):
                if current_CalibrationHistogram.edges[i] > \
                    SPE_peak_mean + (half_width_around_peak_in_stds * SPE_peak_std):
                    
                    inclusive_upper_limit_idx = i - 1
                    break

            if not (0 < inclusive_lower_limit_idx <= \
                    inclusive_upper_limit_idx < \
                        len(current_CalibrationHistogram.edges)):
                print(
                    "In function add_SPE_info_to_output_dictionary(): "
                    f"WARNING: For channel {endpoint}-{channel}, "
                    "the resulting limits for the SPE characterization "
                    "are not valid. They must comply with 0 < "
                    f"inclusive_lower_limit_idx ({inclusive_lower_limit_idx}) "
                    f"<= inclusive_upper_limit_idx ({inclusive_upper_limit_idx})"
                    f" < {len(current_CalibrationHistogram.edges)}. The SPE "
                    "information for this channel won't be computed."
                )
                continue

            contributing_Waveform_idcs = [
                j
                for i in range(inclusive_lower_limit_idx, inclusive_upper_limit_idx + 1)
                for j in current_CalibrationHistogram.indices[i]
            ]

            mean_SPE_adcs = current_ChannelWs.waveforms[
                contributing_Waveform_idcs[0]
                ].adcs

            for i in contributing_Waveform_idcs[1:]:
                mean_SPE_adcs += current_ChannelWs.waveforms[i].adcs

            mean_SPE_adcs *= 1. / len(contributing_Waveform_idcs)

            output_data[endpoint][channel]['SPE_mean_amplitude'] = mean_SPE_adcs.min()
            output_data[endpoint][channel]['SPE_mean_adcs'] = list(mean_SPE_adcs)
            output_data[endpoint][channel]['SPE_idcs'] = contributing_Waveform_idcs

    return output_data

def get_SPE_grid_plot(
    grid_apa: ChannelWsGrid,
    output_data: Dict[int, Dict[int, ...]],
    null_baseline_analysis_label: str,
    time_bins: int = 512,
    adc_bins: int = 50,
    verbose: bool = True
) -> pgo.Figure:
    """This function creates a comprehensive visualization of
    SPE characteristics across all channels in a ChannelWsGrid.
    For each channel, it displays a persistence heatmap of all
    waveforms identified as contributing to the SPE peak,
    overlaid with the computed mean SPE waveform as a black line
    trace.
    
    The function first creates a map matching the physical
    arrangement of the ChannelWsGrid, populated with the indices
    of waveforms that contribute to each channel's SPE
    characterization. It then generates persistence heatmaps using
    these waveform indices and overlays the mean SPE waveform for
    visual comparison and validation.
    
    Parameters
    ----------
    grid_apa: ChannelWsGrid
        The ChannelWsGrid object containing the waveforms that
        will be potentially plotted
    output_data: Dict[int, Dict[int, ...]]
        Dictionary containing SPE analysis results where keys
        are endpoint numbers and values are dictionaries with
        channel numbers as keys. Each channel entry must contain:
        - 'SPE_idcs': list of int
            Indices of waveforms contributing to SPE characterization
        - 'SPE_mean_adcs': list of float
            The mean SPE waveform as ADC values
        - 'SPE_mean_amplitude': float or
            Peak amplitude of the mean SPE waveform
    null_baseline_analysis_label: str
        Label for the baseline analysis used in heatmap generation.
        This should correspond to a baseline analysis where the
        baseline has been artificially set to zero (note that the
        baseline should have been subtracted).
    time_bins: int, default 512
        Number of time bins for the persistence heatmap. Controls
        the temporal resolution of the heatmap visualization.
    adc_bins: int, default 50
        Number of ADC bins for the persistence heatmap. Controls
        the amplitude resolution of the heatmap visualization.
    verbose: bool, default True
        Whether to print informational messages during processing,
        including warnings about missing channels or data.
        
    Returns
    -------
    plotly.graph_objects.Figure
        A plotly Figure object containing the grid of SPE heatmaps
        with mean SPE overlays. The figure layout matches the physical
        arrangement of channels in the ChannelWsGrid.
        
    Notes
    -----
    - The ADC range for heatmaps is automatically scaled based on the
    maximum SPE amplitude found across all channels
    - The function uses the same Map structure as the ChannelWsGrid
    for consistent spatial representation
    
    See Also
    --------
    add_SPE_info_to_output_dictionary: Function that computes the SPE
    data used as input
    plot_ChannelWsGrid: Underlying plotting function used for heatmap
    generation
    """

    # Create a map with the shape of the current
    # ChannelWsGrid which only contains empty lists
    map_of_SPEs_indices = Map.from_unique_value(
        grid_apa.ch_map.rows,
        grid_apa.ch_map.columns,
        list,
        [],
        independent_copies=False
    )

    running_max_SPE_amplitude = 0.

    # Fill the map, using the physical arrangement of
    # the current ChannelWsGrid, with the indices of
    # the waveforms which are SPEs
    for i in range(grid_apa.ch_map.rows):
        for j in range(grid_apa.ch_map.columns):

            endpoint = grid_apa.ch_map.data[i][j].endpoint
            channel = grid_apa.ch_map.data[i][j].channel

            try:
                aux_SPE_idcs = \
                    output_data[endpoint][channel]['SPE_idcs']
                aux_SPE_amplitude = \
                    output_data[endpoint][channel]['SPE_mean_amplitude']

            except KeyError:
                print(
                    "In function get_SPE_grid_plot(): "
                    f"WARNING: Channel {endpoint}-{channel}, "
                    "listed in the map of the current ChannelWsGrid "
                    "object, was not found in the given dictionary "
                    "containing the indices of the SPE waveforms. "
                    "No waveforms will be plotted for such channel."
                )

                aux_SPE_idcs = []
                aux_SPE_amplitude = 0.

            map_of_SPEs_indices.data[i][j] = aux_SPE_idcs

            if abs(aux_SPE_amplitude) > running_max_SPE_amplitude:
                running_max_SPE_amplitude = abs(aux_SPE_amplitude)

    persistence_figure = plot_ChannelWsGrid(
        grid_apa,
        figure=None,
        share_x_scale=False,
        share_y_scale=False,
        mode='heatmap',
        wfs_per_axes=map_of_SPEs_indices,
        analysis_label=null_baseline_analysis_label,
        time_bins=time_bins,
        adc_bins=adc_bins,
        time_range_lower_limit=None,
        time_range_upper_limit=None,
        adc_range_above_baseline=1.5*running_max_SPE_amplitude,
        adc_range_below_baseline=1.5*running_max_SPE_amplitude,
        detailed_label=True,
        verbose=verbose,
    )

    # Now add, channel by channel, the mean SPE adcs
    # on top of the heatmap of all of the SPEs
    for i in range(grid_apa.ch_map.rows):
        for j in range(grid_apa.ch_map.columns):

            endpoint = grid_apa.ch_map.data[i][j].endpoint
            channel = grid_apa.ch_map.data[i][j].channel

            try:
                aux_SPE_adcs = \
                    output_data[endpoint][channel]['SPE_mean_adcs']

            except KeyError:
                print(
                    "In function get_SPE_grid_plot(): "
                    f"WARNING: Channel {endpoint}-{channel}, "
                    "listed in the map of the current ChannelWsGrid "
                    "object, was not found in the given dictionary "
                    "containing the indices of the SPE waveforms. "
                    "The mean SPE will not be plotted for such channel."
                )
                continue

            persistence_figure.add_trace(
                pgo.Scatter(
                    x=np.arange(  
                        len(grid_apa.ch_wf_sets[endpoint][channel].waveforms[0].adcs),
                        dtype=np.float32
                    ),
                    y=aux_SPE_adcs,
                    mode='lines',
                    line=dict(
                        # Good contrast with the (positive) extremal color of the
                        # default color palette of plotly.graph_objects.Heatmap()
                        color='mediumseagreen', 
                        width=1.0
                    ),
                    name=f"Channel {endpoint}-{channel} mean SPE"
                ),
                row=i + 1,
                col=j + 1
            )

    return persistence_figure