from waffles.data_classes.WaveformSet import *
from waffles.data_classes.Waveform import *
import pickle
import numpy as np
from waffles.np04_analysis.lightyield_vs_energy.scripts.myWfAna import MyWfAna
import pandas as pd
from pathlib import Path

'''

def spe_charge(df: pd.DataFrame, endpoint: int, channel: int, hpk_ov: float = 3.0, fbk_ov = 4.5):
    if fbk_or_hpk(endpoint, channel) == 'FBK':
        ov_column = 'FBK_OV_V'
        ov = fbk_ov
    else:
        ov_column = 'HPK_OV_V'
        ov = hpk_ov
    result = df[(df['endpoint'] == endpoint) & (df['channel'] == channel) & (df[ov_column] == ov)]['gain']
    return result.iloc[0] if not result.empty else None

def fbk_or_hpk(endpoint: int, channel: int):
    channel_vendor_map = {
    104: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK"},
    105: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 12: "FBK", 15: "FBK", 17: "FBK", 21: "HPK", 23: "HPK", 24: "HPK", 26: "HPK"},
    107: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK",
          10: "HPK", 12: "HPK", 15: "HPK", 17: "HPK"},
    109: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
          20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
    111: {0: "FBK", 1: "FBK", 2: "FBK", 3: "FBK", 4: "FBK", 5: "FBK", 6: "FBK", 7: "FBK",
          10: "FBK", 11: "FBK", 12: "FBK", 13: "FBK", 14: "FBK", 15: "FBK", 16: "FBK", 17: "FBK",
          20: "FBK", 21: "FBK", 22: "FBK", 23: "FBK", 24: "FBK", 25: "FBK", 26: "FBK", 27: "FBK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 41: "HPK", 42: "HPK", 43: "HPK", 44: "HPK", 45: "HPK", 46: "HPK", 47: "HPK"},
    112: {0: "HPK", 1: "HPK", 2: "HPK", 3: "HPK", 4: "HPK", 5: "HPK", 6: "HPK", 7: "HPK",
          10: "HPK", 11: "HPK", 12: "HPK", 13: "HPK", 14: "HPK", 15: "HPK", 16: "HPK", 17: "HPK",
          20: "HPK", 21: "HPK", 22: "HPK", 23: "HPK", 24: "HPK", 25: "HPK", 26: "HPK", 27: "HPK",
          30: "HPK", 31: "HPK", 32: "HPK", 33: "HPK", 34: "HPK", 35: "HPK", 36: "HPK", 37: "HPK",
          40: "HPK", 42: "HPK", 45: "HPK", 47: "HPK"},
    113: {0: "FBK", 2: "FBK", 5: "FBK", 7: "FBK"}}

    return channel_vendor_map[endpoint][channel]

print('\nReading led calibration info...')
LED_calibration_info = pd.read_pickle(f"/afs/cern.ch/user/a/anbalbon/waffles/src/waffles/np04_analysis/lightyield_vs_energy/data/batch_1_LED_calibration_data.pkl")
print(spe_charge(LED_calibration_info,112,0))        

'''

def channel_filter(waveform : Waveform, end : int, ch : int) -> bool:
    if (waveform.channel == ch) and (waveform.endpoint == end) :
        return True
    else:
        return False
    
def beam_self_trigger_filter(waveform : Waveform, timeoffset_min : int = -120, timeoffset_max : int = -90) -> bool:
    daq_pds_timeoffset = np.float32(np.int64(waveform.timestamp)-np.int64(waveform.daq_window_timestamp))
    if (daq_pds_timeoffset < timeoffset_max) and (daq_pds_timeoffset > timeoffset_min) :
        return True
    else:
        return False

def run_filter(waveform : Waveform, run : int) -> bool:
    if (waveform.run_number == run):
        return True
    else:
        return False

def searching_maritza_template(apa, endpoint, daq_channel, folder = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft', map_path = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft/PDHD_PDS_ChannelMap.csv'):
      df = pd.read_csv(map_path, sep=",")
      daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
      daphne_to_offline = dict(zip(daphne_channels, df['offline_ch']))
      offline_to_daphne = dict(zip(df['offline_ch'],daphne_channels))
      daphne_channel = daq_channel + 100*endpoint
      apa_template_folder  = next((f for f in Path(maritza_template_folder).glob("*APA2*") if f.is_dir()), None)
      martiza_template_file = next(apa_template_folder.glob(f"*APA{apa}_CH{daphne_to_offline[daphne_channel]}*.txt"), None)
      with open(martiza_template_file, "r") as file:
            maritza_values = [float(line.strip()) for line in file]
      maritza_template = np.array(maritza_values)
      return     maritza_template

apa = 2
end = 109
ch = 7 
run = 27355 #2 GeV
NP04_wfset_filepath = f"/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles/set_A/set_A_self_15files109.pkl"
maritza_template_folder = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft'
Larsoft_daphne_channel_map_path = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft/PDHD_PDS_ChannelMap.csv' #By federico

with open(NP04_wfset_filepath, 'rb') as f:
    wfset = pickle.load(f) 
    
wfset_ch_beam = WaveformSet.from_filtered_WaveformSet(WaveformSet.from_filtered_WaveformSet(wfset, beam_self_trigger_filter), channel_filter, end=end, ch=ch)
wfset_ch_beam_run = WaveformSet.from_filtered_WaveformSet(wfset_ch_beam, run_filter,run)


print('\nAnalysis... ')
analysis_label = 'mine'
baseline_start = 0 #before dec
baseline_stop = 45 #before dec
peak_finding_kwargs = dict(prominence = 20,rel_height=0.5,width=[0,75])
gauss_cutoff = 2 #MHz
template = searching_maritza_template(apa,end,ch)

int_ll = 50 #before
int_ul =100 #before

ip = IPDict(baseline_start=baseline_start,
            baseline_stop=baseline_stop,
            int_ll=int_ll,int_ul=int_ul,
            gauss_cutoff= gauss_cutoff,
            template= template)
analysis_kwargs = dict(  return_peaks_properties = False)
checks_kwargs   = dict( points_no = wfset_ch_beam_run.points_per_wf )
a = wfset_ch_beam_run.analyse(label=analysis_label ,analysis_class=MyWfAna, input_parameters=ip, checks_kwargs = checks_kwargs, overwrite=True)


print(wfset_ch_beam_run.waveforms[0].analyses[analysis_label].result['integral_before_deconvolution'])
print(wfset_ch_beam_run.waveforms[0].analyses[analysis_label].result['integral_after_deconvolution'])