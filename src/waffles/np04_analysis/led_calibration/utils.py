import os
import pickle
import yaml
import numpy as np
import pandas as pd
from plotly import graph_objects as pgo
from typing import Tuple, Dict, Optional

from waffles.data_classes.WafflesAnalysis import BaseInputParams
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.Map import Map
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid

from waffles.input_output.raw_root_reader import WaveformSet_from_root_files
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_files
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

def save_data_to_dataframe(
    date: str,
    batch: int,
    apa: int,
    pde: float,
    packed_gain_snr_and_SPE_info: dict,
    packed_integration_limits: dict,
    path_to_output_file: str,
    sipm_vendor_filepath: Optional[str] = None,
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
    packed_integration_limits: dict
        It is expected to have the following form:
            {
                endpoint1: {
                    channel1: (
                        integration_lower_limit11,
                        integration_upper_limit11
                    ),
                    channel2: (
                        integration_lower_limit12,
                        integration_upper_limit12
                    ),
                    ...
                },
                endpoint2: {
                    channel1: (
                        integration_lower_limit21,
                        integration_upper_limit21
                    ),
                    channel2: (
                        integration_lower_limit22,
                        integration_upper_limit22
                    ),
                    ...
                },
                ...
            }
        where the endpoint, channel and integration limit
        values are integers.
    path_to_output_file: str
        The path to the output CSV file
    sipm_vendor_filepath: str, optional
        If None, the 'vendor' column of the output CSV file will
        be filled with strings which match 'unavailable'. If it
        is defined, then it is the path to a CSV file which must
        contain the columns 'endpoint', 'daphne_ch', and 'sipm',
        from which the endpoint, the channel and the vendor
        associated to each channel can be retrieved, respectively.
        In this case, the 'vendor' column of the output CSV file
        will be filled with the vendor information retrieved from
        this file.
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

    fVendorAvailable = False
    if sipm_vendor_filepath is not None:
        try:
            vendor_df = pd.read_csv(sipm_vendor_filepath)

            if all(
                [
                    col in vendor_df.columns for col in ['endpoint', 'daphne_ch', 'sipm']
                ]
            ):
                fVendorAvailable = True

            else:
                print(
                    "In function save_data_to_dataframe(): "
                    f"The file {sipm_vendor_filepath} does not "
                    "contain the required columns: 'endpoint', "
                    "'daphne_ch' and 'sipm'. The vendor column "
                    "in the output dataframe will be filled with "
                    "'unavailable'."
                )

        except FileNotFoundError:
            print(
                "In function save_data_to_dataframe(): "
                f"The file {sipm_vendor_filepath} was not found. "
                "The vendor column in the output dataframe will "
                "be filled with 'unavailable'."
            )

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

                if fVendorAvailable:
                    try:
                        vendor = vendor_df[
                            (vendor_df['endpoint'] == endpoint) &
                            (vendor_df['daphne_ch'] == channel)
                        ]['sipm'].values[0]

                        if vendor not in ('HPK', 'FBK'):
                            print(
                                "In function save_data_to_dataframe(): "
                                f"Channel {endpoint}-{channel} has an "
                                f"unrecognized vendor '{vendor}' in the "
                                "vendor dataframe read from "
                                f"{sipm_vendor_filepath}. The vendor "
                                "column in the output dataframe will be "
                                "filled with 'unavailable'."
                            )
                            vendor = 'unavailable'

                    # Happens if the current endpoint-channel
                    # pair is not found in the vendor_df dataframe
                    except IndexError:
                        print(
                            "In function save_data_to_dataframe(): "
                            f"Channel {endpoint}-{channel} was not "
                            "found in the vendor dataframe read from "
                            f"{sipm_vendor_filepath}. The vendor "
                            "column in the output dataframe will be "
                            "filled with 'unavailable'."
                        )
                        vendor = 'unavailable'
                else:
                    vendor = 'unavailable'

                try:
                    integration_limits = \
                        packed_integration_limits[endpoint][channel]

                except KeyError:
                    print(
                        "In function save_data_to_dataframe(): "
                        "Integration limits for channel "
                        f"{endpoint}-{channel} were not found. "
                        "Setting them to NaN."
                    )
                    
                    integration_limits = (np.nan, np.nan)

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
                    "SPE_mean_amplitude": [abs(packed_gain_snr_and_SPE_info[endpoint][channel]["SPE_mean_amplitude"])],
                    "SPE_mean_adcs": [
                        [round(float(x), 4) for x in \
                        packed_gain_snr_and_SPE_info[endpoint][channel]["SPE_mean_adcs"]]
                    ],
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
            index=False
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

def add_integration_limits_to_persistence_heatmaps(
    persistence_figure: pgo.Figure,
    grid_apa: ChannelWsGrid,
    current_excluded_channels: list,
    integration_limits: Dict[int, Dict[int, Tuple[int, int]]]
) -> None:
    """This function adds the integration limits to the
    persistence heatmaps, channel by channel. The style
    parameters of the lines used to draw the limits
    are hardcoded in the body of this function.

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
    integration_limits: Dict[int, Dict[int, Tuple[int, int]]]
        A dictionary where the keys are endpoint numbers
        and the values are dictionaries where the keys
        are channel numbers and the values are tuples
        containing the (lower_limit, upper_limit) for
        each channel

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
                    "In function add_integration_limits_to_persistence_heatmaps(): "
                    "WARNING: Something went wrong. Channel "
                    f"{endpoint}-{channel} retrieved from the "
                    "ch_wf_sets attribute of the current "
                    "ChannelWsGrid object, was not found in "
                    "its own ch_map attribute. Skipping this "
                    "channel."
                )
                continue

            try:
                aux_integration_limits = \
                    integration_limits[endpoint][channel]

            except KeyError:
                print(
                    "In function add_integration_limits_to_persistence_heatmaps(): "
                    "Could not find the integration limits "
                    f"for channel {endpoint}-{channel}. They "
                    "will not be drawn in the persistence "
                    "heatmap."
                )
                continue

            # Unpack the channel position
            i, j = channel_position

            aux_ncols = grid_apa.ch_map.columns

            # Lower limit
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

            # Upper limit
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
    
    return

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

    # The 1-PE (SPE) peak is that of index equal to 1
    targeted_peak_idx = 1

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
                _ = output_data[endpoint][channel]
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
            aux = current_CalibrationHistogram.gaussian_fits_parameters
            
            try:
                SPE_peak_mean = aux['mean'][targeted_peak_idx][0]
                SPE_peak_std = aux['std'][targeted_peak_idx][0]

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