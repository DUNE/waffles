from waffles.data_classes.ChannelMap import ChannelMap

from waffles.np04_data.ProtoDUNE_HD_APA_maps import flat_APA_map
from waffles.Exceptions import GenerateExceptionMessage

def get_channel_iterator(   
    apa_no: ChannelMap,
    endpoint: int,
    channel: int):
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