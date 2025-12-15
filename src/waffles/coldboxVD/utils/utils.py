import waffles
import pandas as pd
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def get_cb25_channel_mapping() -> pd.DataFrame:
    """
    This function reads the Coldbox-25 channel mapping file and returns it as a pandas DataFrame.
    The dataframe contains the following columns:
    ConfigCh, DAQCh, Module, SiPM, LN2Vbr
    Parameters
    ----------

    Returns
    -------
    - df: pd.DataFrame
        The channel mapping as a pandas DataFrame
    """
    wafflesdir = Path(waffles.__file__).parent
    mapping_file = wafflesdir / "coldboxVD" / "utils" / "CB25_ChannelMap.csv"
    if not Path(mapping_file).exists() :
        print(mapping_file)
        raise FileNotFoundError(
            "The channel mapping was not found. You probably need to install waffles with -e option:\n`python3 -m pip install -e .`")
    
    df = pd.read_csv(mapping_file, sep=",")

    return df

