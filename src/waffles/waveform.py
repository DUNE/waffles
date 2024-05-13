


#import libraries
import os
import daqdataformats
import matplotlib.pyplot as plt
from daqdataformats import FragmentType
from rawdatautils.unpack.daphne import *
from hdf5libs import HDF5RawDataFile


#define waveform class
class waveform:
  def __init__(self, fragment):
    frag_id = str(frag).split(' ')[3][:-1]
    fragType = frag.get_header().fragment_type
    
    if fragType == FragmentType.kDAPHNE.value:  # For self trigger
        trigger = 'self_trigger'
        timestamps = np_array_timestamp(frag)
        adcs = np_array_adc(frag)
        channels = np_array_channels(frag)
    elif fragType == 13:  # For full_stream
        trigger = 'full_stream'
        timestamps = np_array_timestamp_stream(frag)
        adcs = np_array_adc_stream(frag)
        channels = np_array_channels_stream(frag)[0]
    
    return trigger, frag_id, channels, adcs, timestamps

