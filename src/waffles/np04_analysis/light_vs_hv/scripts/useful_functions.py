from waffles.np04_analysis.light_vs_hv.imports import *

def check_endpoint_and_channel(endpoint,channel):
    for APA in range(1,5,1):
        for i in range(10):
            for j in range(4):
                endpoint_now=int(APA_map[APA].data[i][j].endpoint)
                ch_now=int(APA_map[APA].data[i][j].channel)
                
                if endpoint==endpoint_now and channel==ch_now:
                    return True
    return False
    