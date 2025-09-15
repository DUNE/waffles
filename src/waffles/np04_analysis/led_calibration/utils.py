import os
<<<<<<< HEAD
import numpy as np
import pandas as pd

from waffles.core.utils import build_parameters_dictionary
from waffles.input_output.raw_root_reader import WaveformSet_from_root_files
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_files
from waffles.input_output.raw_root_reader import WaveformSet_from_root_file
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_file
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.IPDict import IPDict
from waffles.np04_utils.utils import get_channel_iterator

input_parameters = build_parameters_dictionary('params.yml')

def get_input_filepath(
=======
import pickle
import numpy as np
import pandas as pd

from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid

from waffles.input_output.raw_root_reader import WaveformSet_from_root_files
from waffles.input_output.pickle_file_reader import WaveformSet_from_pickle_files
from waffles.np04_utils.utils import get_channel_iterator

def get_input_folderpath(
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
        base_folderpath: str,
        batch: int,
        apa: int,
        pde: float,
        run: int
    ) -> str:
<<<<<<< HEAD

    aux = get_input_folderpath(
        base_folderpath,
        batch,
        apa,
        pde
    )
    return  f"{aux}/run_0{run}/run_{run}_chunk_0.pkl"

def get_input_folderpath(
        base_folderpath: str,
        batch: int,
        apa: int,
        pde: float):
=======
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
    
    aux = get_apa_foldername(
        batch,
        apa
    )
<<<<<<< HEAD
    return  f"{base_folderpath}/batch_{batch}/{aux}/pde_{pde}/data/"

def get_apa_foldername(measurements_batch, apa_no):
=======

    return  f"{base_folderpath}/batch_{batch}/{aux}/pde_{pde}/data/run_0{run}/"

def get_apa_foldername(
        measurements_batch,
        apa_no
    ) -> str:
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
    """This function encapsulates the non-homogeneous 
    naming convention of the APA folders depending 
    on the measurements batch.""" 

    if measurements_batch not in [1, 2, 3]:
        raise ValueError(
            f"Measurements batch {measurements_batch} is not valid"
        )
    
    if apa_no not in [1, 2, 3, 4]:
<<<<<<< HEAD
        raise ValueError(f"APA number {apa_no} is not valid")
                         
=======
        raise ValueError(
            f"APA number {apa_no} is not valid"
        )

>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
    if measurements_batch == 1:
        if apa_no in [1, 2]:
            return 'apas_12'
        else:
            return 'apas_34'
        
    if measurements_batch in [2, 3]:
        if apa_no == 1:
            return 'apa_1'
        elif apa_no == 2:
            return 'apa_2'
        else:
            return 'apas_34'

<<<<<<< HEAD
def comes_from_channel(
        waveform: Waveform, 
        endpoint, 
        channels
    ) -> bool:

    if waveform.endpoint == endpoint:
        if waveform.channel in channels:
            return True
    return False

def get_analysis_params(
        apa_no: int,
        run: int = None
    ):

    if apa_no == 1:
        if run is None:
            raise Exception(
                "In get_analysis_params(): A run number "
                "must be specified for APA 1"
            )
        else:
            int_ll = input_parameters['starting_tick'][1][run]
    else:
        int_ll = input_parameters['starting_tick'][apa_no]

    analysis_input_parameters = IPDict(
        baseline_limits=\
            input_parameters['baseline_limits'][apa_no]
    )
    analysis_input_parameters['int_ll'] = int_ll
    analysis_input_parameters['int_ul'] = \
        int_ll + input_parameters['integ_window']
    analysis_input_parameters['amp_ll'] = int_ll
    analysis_input_parameters['amp_ul'] = \
        int_ll + input_parameters['integ_window']

    return analysis_input_parameters
=======
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

    if input_string[0] != '[' or input_string[-1] != ']':
        raise ValueError(
            "In function parse_numeric_list():"
            "Input string must start with '[' and end with ']'"
        )

    # Remove the brackets
    input_string = input_string.strip()[1:-1]

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
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107

def read_data(
        input_path: str,
        batch: int,
        apa_no: int,
<<<<<<< HEAD
        is_folder: bool = True,
        stop_fraction: float = 1.
    ):

    fProcessRootNotPickles = True if batch == 1 else False

    if is_folder: 
        if fProcessRootNotPickles:
            new_wfset = WaveformSet_from_root_files(
                "pyroot",
                folderpath=input_path,
                bulk_data_tree_name="raw_waveforms",
                meta_data_tree_name="metadata",
                set_offset_wrt_daq_window=True if apa_no == 1 else False,
                read_full_streaming_data=True if apa_no == 1 else False,
                truncate_wfs_to_minimum=True if apa_no == 1 else False,
                start_fraction=0.0,
                stop_fraction=stop_fraction,
                subsample=1,
            )
        else:
            new_wfset = WaveformSet_from_pickle_files(                
                folderpath=input_path,
                target_extension=".pkl",
                verbose=True,
            )
    else:
        if fProcessRootNotPickles:
            new_wfset = WaveformSet_from_root_file(
                "pyroot",
                filepath=input_path,
                bulk_data_tree_name="raw_waveforms",
                meta_data_tree_name="metadata",
                set_offset_wrt_daq_window=True if apa_no == 1 else False,
                read_full_streaming_data=True if apa_no == 1 else False,
                truncate_wfs_to_minimum=True if apa_no == 1 else False,
                start_fraction=0.0,
                stop_fraction=stop_fraction,
                subsample=1,
            )
        else:
            new_wfset = WaveformSet_from_pickle_file(input_path)

    return new_wfset

def get_gain_and_snr(
        grid_apa: ChannelWsGrid,
        excluded_channels: list
    ):
=======
        stop_fraction: float = 1.,
        verbose: bool = True
    ):
    """It is assumed that the input_path is a folder."""

    fProcessRootNotPickles = True if batch == 1 else False

    if fProcessRootNotPickles:
        new_wfset = WaveformSet_from_root_files(
            "pyroot",
            folderpath=input_path,
            bulk_data_tree_name="raw_waveforms",
            meta_data_tree_name="metadata",
            set_offset_wrt_daq_window=True if apa_no == 1 else False,
            read_full_streaming_data=True if apa_no == 1 else False,
            truncate_wfs_to_minimum=True if apa_no == 1 else False,
            start_fraction=0.0,
            stop_fraction=stop_fraction,
            subsample=1,
        )
    else:
        new_wfset = WaveformSet_from_pickle_files(                
            folderpath=input_path,
            target_extension=".pkl",
            verbose=verbose,
        )

    return new_wfset

def get_average_baseline_std(
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
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107

    data = {}

    for i in range(grid_apa.ch_map.rows):
        for j in range(grid_apa.ch_map.columns):

            endpoint = grid_apa.ch_map.data[i][j].endpoint
            channel = grid_apa.ch_map.data[i][j].channel

<<<<<<< HEAD
            if endpoint in excluded_channels.keys():
                if channel in excluded_channels[endpoint]:
                    print(f"    - Excluding channel {channel} from endpoint {endpoint}...")
                    continue

            try:
                fit_params = grid_apa.ch_wf_sets[endpoint][channel].calib_histo.gaussian_fits_parameters
            except KeyError:
                print(f"Endpoint {endpoint}, channel {channel} not found in data. Continuing...")
=======
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
                        grid_apa.ch_wf_sets[endpoint][channel].\
                            calib_histo._CalibrationHistogram__reset_gaussian_fit_parameters()
                    except KeyError:
                        pass
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
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
                continue
 
            # Handle a KeyError the first time we access a certain endpoint
            try:
                aux = data[endpoint]
<<<<<<< HEAD
=======

>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
            except KeyError:
                data[endpoint] = {}
                aux = data[endpoint]

<<<<<<< HEAD
            # compute the gain
            try:
                aux_gain = fit_params['mean'][1][0] - fit_params['mean'][0][0]
            except IndexError:
                print(f"Endpoint {endpoint}, channel {channel} not found in data. Continuing...")
                continue
            
            # this is to avoid a problem the first time ch is used
=======
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
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
            try:
                aux_2 = aux[channel]
            except KeyError:
                aux[channel] = {}
                aux_2 = aux[channel]

            aux_2['gain'] = aux_gain

<<<<<<< HEAD
            # compute the signal to noise ratio
            aux_2['snr'] = aux_gain/np.sqrt(fit_params['std'][0][0]**2 + fit_params['std'][1][0]**2)

    return data

def get_nbins_for_charge_histo(
        pde: float,
        apa_no: int
    ):

    if apa_no in [2, 3, 4]:
        if pde == 0.4:
            bins_number = 125
        elif pde == 0.45:
            bins_number = 110 # [100-110]
        else:
            bins_number = 90
    else:
        # It is still required to
        # do this tuning for APA 1
        bins_number = 125

    return bins_number

def save_data_to_dataframe(
    Analysis1_object,
=======
            # Compute the signal to noise ratio
            aux_2['snr'] = aux_gain / \
                np.sqrt(fit_params['std'][0][0]**2 + fit_params['std'][1][0]**2)

    return data

def save_data_to_dataframe(
    batch: int,
    apa: int,
    pde: float,
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
    data: list,
    path_to_output_file: str
):
    
<<<<<<< HEAD
    hpk_ov = input_parameters['hpk_ov'][Analysis1_object.pde]
    fbk_ov = input_parameters['fbk_ov'][Analysis1_object.pde]
    ov_no = input_parameters['ov_no'][Analysis1_object.pde]
    
=======
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

>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
    # Warning: Settings this variable to True will save
    # changes to the output dataframe, potentially introducing
    # spurious data. Only set it to True if you are sure of what
    # you are saving.
    actually_save = True   

    # Do you want to potentially overwrite existing rows of the dataframe?
    overwrite = False

    expected_columns = {
<<<<<<< HEAD
=======
        "batch": [],
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
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
<<<<<<< HEAD
=======
        df['batch'] = df['batch'].astype(int)
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
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

<<<<<<< HEAD
        df.to_pickle(path_to_output_file)

    df = pd.read_pickle(path_to_output_file)

    if len(df.columns) != len(expected_columns):
        raise Exception(
            f"The columns of the found dataframe do not "
            "match the expected ones. Something went wrong.")

    elif not bool(np.prod(df.columns == pd.Index(data = expected_columns))):
        raise Exception(
            f"The columns of the found dataframe do not "
            "match the expected ones. Something went wrong.")
=======
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
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107

    else:
        for endpoint in data.keys():
            for channel in data[endpoint]:
                # Assemble the new row
                new_row = {
<<<<<<< HEAD
                    "APA": [int(Analysis1_object.apa)],
                    "endpoint": [endpoint],
                    "channel": [channel],
                    "channel_iterator": [get_channel_iterator(
                        Analysis1_object.apa,
                        endpoint,
                        channel
                    )],
                    "PDE": [Analysis1_object.pde],
=======
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
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
                    "gain": [data[endpoint][channel]["gain"]],
                    "snr": [data[endpoint][channel]["snr"]],
                    "OV#": [ov_no],
                    "HPK_OV_V": [hpk_ov],
                    "FBK_OV_V": [fbk_ov],
                }

                # Check if there is already an entry for the
<<<<<<< HEAD
                # given endpoint and channel for this OV
                matching_rows_indices = df[
=======
                # given endpoint and channel for this OV and batch
                matching_rows_indices = df[
                    (df['batch'] == batch) &
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
                    (df['endpoint'] == endpoint) &       
                    (df['channel'] == channel) &
                    (df['OV#'] == ov_no)].index          

                if len(matching_rows_indices) > 1:
                    raise Exception(
<<<<<<< HEAD
                        f"There are already more than one rows "
                        f"for the given endpoint ({endpoint}), "
                        f"channel ({channel}) and OV# ({ov_no})"
=======
                        "In function save_data_to_dataframe(): "
                        "There are already more than one rows "
                        f"for the given channel ({endpoint}-{channel}"
                        f"), batch ({batch}) and OV# ({ov_no})"
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
                        ". Something went wrong."
                    )

                elif len(matching_rows_indices) == 1:
                    if overwrite:

                        row_index = matching_rows_indices[0]

<<<<<<< HEAD
                        new_row = { key : new_row[key][0] for key in new_row.keys() }  
=======
                        new_row = {key: new_row[key][0] for key in new_row.keys()}  
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107

                        if actually_save:
                            df.loc[row_index, :] = new_row

                    else:
                        print(
<<<<<<< HEAD
                            f"Skipping the entry for endpoint "
                            f"{endpoint}, channel {channel} and"
                            f" OV# {ov_no} ...")

                else: # len(matching_rows_indices) == 0
                    if actually_save:
                        df = pd.concat([df, pd.DataFrame(new_row)], axis = 0, ignore_index = True)
                        df.reset_index()

        df.to_pickle(path_to_output_file)
=======
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
>>>>>>> 264bdce2c6b35b5dd071455c3cbe62221217a107
