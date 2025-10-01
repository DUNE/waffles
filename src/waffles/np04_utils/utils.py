
import waffles
from waffles.np04_data.ProtoDUNE_HD_APA_maps import flat_APA_map
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.Exceptions import GenerateExceptionMessage
from pathlib import Path
import pandas as pd
from pathlib import Path
import json
from functools import lru_cache

def get_channel_iterator(   
    apa_no: int,
    endpoint: int,
    channel: int) -> int:
    """
    This function returns the iterator value of the given
    channel in the flattened map of the specified APA.
    If the given channel does not exist in the specified
    APA, or the given APA number is not recognised, an
    exception is raised.

    Parameters
    ----------
    apa_no: int
        The APA number
    endpoint: int
        The endpoint number
    channel: int
        The channel number

    Returns
    -------
    iterator: int
        The iterator value of the given channel in the
        flattened map of the specified APA
    """

    try:
        flat_map = flat_APA_map[apa_no]

    except KeyError:
        raise Exception(GenerateExceptionMessage( 
            1,
            'get_channel_iterator()',
            f"The given APA number ({apa_no}) is not recognised."))
    
    iterator = 0
    
    if flat_map.rows == 1:
        for j in range(flat_map.columns):

            if flat_map.data[0][j].endpoint == endpoint and \
                flat_map.data[0][j].channel == channel:
                    
                    return iterator
            else:
                iterator += 1

    elif flat_map.columns == 1:
         for i in range(flat_map.rows):

            if flat_map.data[i][0].endpoint == endpoint and \
                flat_map.data[i][0].channel == channel:
                    
                    return iterator
            else:
                iterator += 1
    else:
        raise Exception(GenerateExceptionMessage(
            2,
            'get_channel_iterator()',
            f"The retrieved map is not flat."))
    
    raise Exception(GenerateExceptionMessage(
        3,
        'get_channel_iterator()',
        f"The given channel ({endpoint}-{channel}) "
        "is not present in the specified APA."))

def get_endpoint_and_channel(
    apa_no: int,
    channel_iterator: int) -> UniqueChannel:
    """
    This function is the inverse of get_channel_iterator().
    It returns the endpoint and channel numbers given an
    iterator value of a flattened map. The used flattened
    map is that of the specified APA. The output format of
    this function is an UniqueChannel object. If the given 
    channel iterator does not exist in the specified APA, 
    or the given APA number is not recognised, an exception 
    is raised.

    Parameters
    ----------
    apa_no: int
        The APA number
    channel_iterator: int
        The iterator value of the channel in the flattened
        map of the specified APA

    Returns
    -------
    UniqueChannel
        An UniqueChannel object which encapsulates the
        endpoint and channel numbers of the channel with
        the given iterator value in the flattened map of
        the specified APA
    """

    try:
        flat_map = flat_APA_map[apa_no]

    except KeyError:
        raise Exception(GenerateExceptionMessage( 
            1,
            'get_endpoint_and_channel()',
            f"The given APA number ({apa_no}) is not recognised."))
    
    if channel_iterator < 0 or \
        channel_iterator >= flat_map.rows * flat_map.columns:

        raise Exception(GenerateExceptionMessage(
            2,
            'get_endpoint_and_channel()',
            f"The given channel iterator value ({channel_iterator}) "
            "is out of range."))
    
    if flat_map.rows == 1:
        return flat_map.data[0][channel_iterator]
    
    elif flat_map.columns == 1:
        return flat_map.data[channel_iterator][0]
    
    else:
        raise Exception(GenerateExceptionMessage(
            3,
            'get_endpoint_and_channel()',
            f"The retrieved map is not flat."))

@lru_cache(maxsize=1)
def get_np04_channel_mapping(version: str="", run: int=-1) -> pd.DataFrame:
    """
    This function reads the NP04 channel mapping file and returns it as a pandas DataFrame.
    The dataframe contains the following columns:
    endpoint, daphne_ch, offline_ch, APA, sipm
    Version "old" refers to the channel mapping used before the change on the 24th of September 2024.
    We consider run 29297 as the first run with the new channel mapping.
    Parameters
    ----------
    - version: str, optional ("old" or "new")
    - run: int, optional

    Returns
    -------
    - df: pd.DataFrame
        The channel mapping as a pandas DataFrame
    """
    wafflesdir = Path(waffles.__file__).parent
    if not Path(wafflesdir / "np04_utils" / "PDHD_PDS_ChannelMap.csv").exists() :
        raise FileNotFoundError(
            "The channel mapping was not found. You probably need to install waffles with -e option:\n`python3 -m pip install -e .`")

    if version == "":
        version = "old"

    if run != -1:
        if run >= 29297:
            version = "new"
        else:
            version = "old"

    if version == "old":
        mapping_file = wafflesdir / "np04_utils" / "PDHD_PDS_ChannelMap.csv"
    elif version == "new":
        mapping_file = wafflesdir / "np04_utils" / "PDHD_PDS_NewChannelMap.csv"
    else:
        raise ValueError("Invalid version specified. Use 'old' or 'new'.")

    df = pd.read_csv(mapping_file, sep=",")

    return df

@lru_cache(maxsize=1)
def get_np04_daphne_to_offline_channel_dict(version: str="", run: int=-1) -> dict:
    """
    This function returns a dictionary that maps daphne channels to offline channels.
    Version "old" refers to the channel mapping used before the change on the 24th of September 2024.
    We consider run 29297 as the first run with the new channel mapping.
    Daphne channels are numbered as endpoint*100 + daphne_ch, where daphne_ch is from 0->7, 10->17, 20->27, 30->37.
    Offline channels are numbered from 0 to 159.
    Parameters
    ----------
    - version: str, optional ("old" or "new")
    - run: int, optional

    Returns
    -------
    - daphne_to_offline: dict
        A dictionary that maps daphne channels to offline channels
    """
    df = get_np04_channel_mapping(version=version, run=run)
    daphne_channels = df['daphne_ch'].values[0] + 100*df['endpoint'].values[0]
    daphne_to_offline = dict(zip(daphne_channels, df['offline_ch']))
    return daphne_to_offline

@lru_cache(maxsize=1)
def get_np04_offline_to_daphne_channel_dict(version: str="", run: int=-1) -> dict:
    """
    This function returns a dictionary that maps offline channels to daphne channels.
    Version "old" refers to the channel mapping used before the change on the 24th of September 2024.
    We consider run 29297 as the first run with the new channel mapping.
    Offline channels are numbered from 0 to 159.
    Daphne channels are numbered as endpoint*100 + daphne_ch, where daphne_ch is from 0->7, 10->17, 20->27, 30->37.
    Parameters
    ----------
    - version: str, optional ("old" or "new")
    - run: int, optional

    Returns
    -------
    - offline_to_daphne: dict
        A dictionary that maps offline channels to daphne channels
    """
    df = get_np04_channel_mapping(version=version, run=run)
    daphne_channels = df['daphne_ch'].values[0] + 100*df['endpoint'].values[0]
    offline_to_daphne = dict(zip(df['offline_ch'], daphne_channels))
    return offline_to_daphne


@lru_cache(maxsize=1)
def load_configs(daphne_config_file = "") -> dict:
    """Load and cache Daphne configuration JSON."""
    if not daphne_config_file:
        wafflesdir = Path(waffles.__file__).parent
        daphne_config_file = wafflesdir / "np04_utils" / "DaphneConfigs.json"

    if not Path(daphne_config_file).exists():
        raise FileNotFoundError(
            "DaphneConfigs.json not found. Install waffles with -e option:\n"
            "`python3 -m pip install -e .`"
        )

    with open(daphne_config_file, "r") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_noise_results(noise_results_file = "") -> pd.DataFrame:
    """Load and cache noise results CSV."""
    if not noise_results_file:
        noise_results_file = Path("OfflineCh_RMS_Config_all.csv")

    if not Path(noise_results_file).exists():
        raise FileNotFoundError(
            "Noise results file not found. Install waffles with -e option:\n"
            "`python3 -m pip install -e .`"
        )

    return pd.read_csv(noise_results_file)


def get_average_baseline_std_from_file(
    run: int,
    endpoint: int = -1,
    channel: int = -1,
    daphne_channel: int = -1,
    offline_channel: int = -1,
    daphne_configuration_database_filepath: str = "",
    noise_results_dataframe_filepath: str = "",
) -> float:
    """
    Get the average baseline RMS for a given run/channel.
    Parameters
    ----------
    run: int
    endpoint: int, optional
    channel: int, optional (if endpoint is given). Channel number within the endpoint (0-7, 10-17, 20-27, 30-37)
    daphne_channel: int, optional. Daphne channel number (endpoint*100 + daphne_ch)
    offline_channel: int, optional. Offline channel number (0-159)
    daphne_configuration_database_filepath: str, optional
        Path to Daphne configuration JSON file.
    noise_results_dataframe_filepath: str, optional
        Path to noise results CSV file. It must contain columns: ConfigNumber, OfflineCh, RMS

    Returns
    -------
    rms of the baseline: float
        The expected baseline RMS for the specified run and channel.
    """
    # --- Load configs + find config ---
    configs = load_configs(daphne_configuration_database_filepath)

    this_config = next(
        (int(k) for k, v in configs.items() if v["first_run"] <= run <= v["last_run"]),
        -1,
    )
    if this_config == -1:
        raise ValueError(f"No config found for run {run}")

    print(
        f"Run {run} is in config {this_config} "
        f"(first_run={configs[str(this_config)]['first_run']}, "
        f"last_run={configs[str(this_config)]['last_run']})"
    )

    # --- Resolve offline channel ---
    if offline_channel == -1:
        if daphne_channel != -1:
            offline_channel = get_np04_daphne_to_offline_channel_dict()[daphne_channel]
        elif endpoint != -1 and channel != -1:
            offline_channel = get_np04_daphne_to_offline_channel_dict()[endpoint * 100 + channel]
        else:
            raise ValueError(
                "You must provide at least one of: endpoint+channel, daphne_channel, offline_channel"
            )

    # --- Get noise results ---
    df = load_noise_results(noise_results_dataframe_filepath)
    channel_rms = df.loc[
        (df["ConfigNumber"] == this_config) & (df["OfflineCh"] == offline_channel),
        "RMS",
    ].values

    if len(channel_rms) == 0:
        raise ValueError(
            f"No RMS found for run {run}, config {this_config}, offline_channel {offline_channel}"
        )

    return float(channel_rms[0])
