import numpy as np
from waffles.data_classes.UniqueChannel import UniqueChannel
from waffles.data_classes.ChannelMap import ChannelMap

def createChannelMaps(endpoint_list):
    map_list = []
    for endpoint in endpoint_list:
        for i in range(0,5):
            channel_list = []
            for j in range(0,8):
                channel_list.append(UniqueChannel(endpoint,10*i+j))
            map_list.append(channel_list)
    print(map_list)
    return ChannelMap(len(map_list),8,map_list)

def getEndpointList(channelsList):
    endpoints = []
    channelsList = convertChannelStrToList(channelsList)
    for channel in channelsList:
        endpoints.append(int(float(channel)/100.0))
    return np.unique(np.array(endpoints))

def getUniqueChannelList(channelsList):
    uniqueChannelList = []
    channelsList = convertChannelStrToList(channelsList)
    for channel in channelsList:
        endpoint = int(float(channel)/100.0)
        channel = channel - endpoint*100
        uniqueChannelList.append(UniqueChannel(endpoint,int(channel)))
    return uniqueChannelList

def convertChannelStrToList(strList):
    strList = strList[1:len(strList)-1]
    strList = strList.split(',')
    return [float(str_) for str_ in strList]