from typing import List

import numpy as np
import glob
import pandas as pd



###########################
def read_templates(foldername) -> List:

    """ Read all channel templates from a folder """
    
    files = glob.glob(f'{foldername}/*.txt')

    templates = []
    for f in files:    
        templates.append(np.float32(open(f, 'r').read().split()))

    """        
    temp = {}
        
    for ep in [104,105,107,109,111,112]:
        configs[ep] = {}

        for ch in range(0,40):
            configs[ep][ch] = {}

    configs[1][0.40] = {
   """     
    return templates



###########################
def read_mapping(filename: str = '../../np04_data/daphne_larsoft_channel_mapping.csv') -> pd.core.frame.DataFrame:

    """ read the daphe-larsoft matching """
    
    df = pd.read_csv(filename)
    
    return df
