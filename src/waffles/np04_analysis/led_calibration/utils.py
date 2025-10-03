import os
import pickle
import numpy as np
import pandas as pd

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid

from waffles.input_output.raw_root_reader import WaveformSet_from_root_files
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_files
from waffles.np04_utils.utils import get_channel_iterator
import waffles.Exceptions as we

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

def get_gain_and_snr(
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
                    "In function get_gain_and_snr(): "
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
                    "In function get_gain_and_snr(): "
                    f"Skipping channel {endpoint}-{channel} "
                    "since it was not found in data."
                )
                continue
 
            # Handle a KeyError the first time we access a certain endpoint
            try:
                aux = data[endpoint]

            except KeyError:
                data[endpoint] = {}
                aux = data[endpoint]

            # Compute the gain
            try:
                aux_gain = fit_params['mean'][1][0] - fit_params['mean'][0][0]

            except IndexError:
                print(
                    "In function get_gain_and_snr(): "
                    "Could not compute the gain for channel "
                    f"{endpoint}-{channel} since two-peaks "
                    "data was not found. Skipping this channel."
                )
                continue
            
            # Handle a KeyError the first time we access a certain channel
            try:
                aux_2 = aux[channel]
            except KeyError:
                aux[channel] = {}
                aux_2 = aux[channel]

            aux_2['gain'] = aux_gain

            # Compute the signal to noise ratio
            aux_2['snr'] = aux_gain / \
                np.sqrt(fit_params['std'][0][0]**2 + fit_params['std'][1][0]**2)

    return data

def save_data_to_dataframe(
    batch: int,
    apa: int,
    pde: float,
    data: list,
    path_to_output_file: str
):
    
    # PDE-to-OV mapping for HPK sipms
    hpk_ov = {
        0.4: 2.0,
        0.45: 3.5,
        0.50: 4.0
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

    # Warning: Settings this variable to True will save
    # changes to the output dataframe, potentially introducing
    # spurious data. Only set it to True if you are sure of what
    # you are saving.
    actually_save = True   

    # Do you want to potentially overwrite existing rows of the dataframe?
    overwrite = False

    expected_columns = {
        "batch": [],
        "APA": [],
        "endpoint": [],
        "channel": [],
        "channel_iterator": [],
        "PDE": [],
        "gain": [],
        "snr": [],
        "OV#": [],
        "HPK_OV_V": [],
        "FBK_OV_V": [],
    }

    # If the file does not exist, create it
    if not os.path.exists(path_to_output_file):
        df = pd.DataFrame(expected_columns)

        # Force column-wise types
        df['batch'] = df['batch'].astype(int)
        df['APA'] = df['APA'].astype(int)
        df['endpoint'] = df['endpoint'].astype(int)
        df['channel'] = df['channel'].astype(int)
        df['channel_iterator'] = df['channel_iterator'].astype(int)
        df['PDE'] = df['PDE'].astype(float)
        df['gain'] = df['gain'].astype(float)
        df['snr'] = df['snr'].astype(float)
        df['OV#'] = df['OV#'].astype(int)
        df['HPK_OV_V'] = df['HPK_OV_V'].astype(float)
        df['FBK_OV_V'] = df['FBK_OV_V'].astype(float)

        df.to_csv(
            path_to_output_file,
            index=False
        )

    df = pd.read_csv(path_to_output_file)

    if len(df.columns) != len(expected_columns):
        raise Exception(
            "In function save_data_to_dataframe(): "
            "The columns of the found dataframe do not "
            "match the expected ones. Something went wrong."
        )

    elif not bool(np.prod(df.columns == pd.Index(data = expected_columns))):
        raise Exception(
            "In function save_data_to_dataframe(): "
            "The columns of the found dataframe do not "
            "match the expected ones. Something went wrong."
        )

    else:
        for endpoint in data.keys():
            for channel in data[endpoint]:
                # Assemble the new row
                new_row = {
                    "batch": [int(batch)],
                    "APA": [int(apa)],
                    "endpoint": [endpoint],
                    "channel": [channel],
                    "channel_iterator": [get_channel_iterator(
                        apa,
                        endpoint,
                        channel
                    )],
                    "PDE": [pde],
                    "gain": [data[endpoint][channel]["gain"]],
                    "snr": [data[endpoint][channel]["snr"]],
                    "OV#": [ov_no],
                    "HPK_OV_V": [hpk_ov],
                    "FBK_OV_V": [fbk_ov],
                }

                # Check if there is already an entry for the
                # given endpoint and channel for this OV and batch
                matching_rows_indices = df[
                    (df['batch'] == batch) &
                    (df['endpoint'] == endpoint) &       
                    (df['channel'] == channel) &
                    (df['OV#'] == ov_no)].index          

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