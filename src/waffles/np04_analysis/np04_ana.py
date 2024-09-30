from waffles.data_classes.Waveform import Waveform
# from waffles.data_classes.WaveformSet import WaveformSet

def get_run_folderpath(run, base_folderpath):
    return f"{base_folderpath}/run_0{run}"

def comes_from_channel( waveform : Waveform, 
                        endpoint, 
                        channels) -> bool:

    if waveform.endpoint == endpoint:
        if waveform.channel in channels:
            return True
    return False

def allow_certain_endpoints(waveform : Waveform, my_endpoints: list) -> bool:
    if waveform.endpoint in my_endpoints :
        return True
    else:
        return False