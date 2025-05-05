from waffles.data_classes.WaveformSet import *
from waffles.data_classes.Waveform import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft, fftshift # For the deconvolution
from waffles.utils.baseline.baseline import SBaseline # Baseline computation from Henrique
from tqdm import tqdm  # Importa tqdm per la barra di avanzamento

def integral(wf_adcs,baseline, start=50,stop=100, time_step_ns=16, time_offset=0):
    return time_step_ns * (((stop - start + 1) * baseline) - np.sum(wf_adcs[start - time_offset: stop + 1 - time_offset]))

def from_sigma_to_cutoff(sigma):
    npoints = 1024//2 # The fft has half the length of the waveforms
    FFTFreq = 32.5
    binwidth = FFTFreq/npoints
    fc_in_ticks = sigma * np.sqrt(np.log(2))
    return fc_in_ticks*binwidth

def from_cutoff_to_sigma(cutoff_frequency):
    npoints = 1024//2 # The fft has half the length of the waveforms
    FFTFreq = 32.5
    binwidth = FFTFreq/npoints
    fc_in_ticks = cutoff_frequency/binwidth
    return fc_in_ticks/np.sqrt(np.log(2))

def normalize_signal_chastGPT(signal, peak_index):
    min_val, max_val = np.min(signal), np.max(signal)
    norm_signal = 2 * (signal - min_val) / (max_val - min_val) - 1  # Normalizzazione [-1,1]
    
    shift = peak_index - np.argmax(norm_signal)  # Calcoliamo lo shift necessario
    return np.roll(norm_signal, shift)  # Shiftiamo il segnale

def normalize_signal(signal, peak_index=50):
    base = np.mean(signal[-250:-100])
    min_val, max_val = np.min(signal-base), np.max(signal-base)
    norm_signal = (signal - base) / max_val
    
    #shift = peak_index - np.argmax(norm_signal)  # Calcoliamo lo shift necessario
    return norm_signal


def gaus(x, sigma=20):
    return np.exp(-(x)**2/2/sigma**2)

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


# INPUT DATA 
NP04_wfset_filepath = f"/afs/cern.ch/work/a/anbalbon/public/np04_beam_pickles/set_A/set_A_self_15files109.pkl"
maritza_template_folder = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft'
Larsoft_daphne_channel_map_path = '/afs/cern.ch/work/a/anbalbon/public/template_larsoft/PDHD_PDS_ChannelMap.csv' #By federico


# CHANNEL SELECTION
apa = 2
endpoint = 109
daq_channel = 7 
n_wf = 0
vendor = fbk_or_hpk(endpoint,daq_channel)

# Gaussian filter
sigma_list = [10, 15, 20, 25, 30]
cutoff_list = [1, 1.5, 2, 2.5, 3, 3.5]

# Reading Maritza template
print('Searchig for maritza template...')
df = pd.read_csv(Larsoft_daphne_channel_map_path, sep=",")
daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
daphne_to_offline = dict(zip(daphne_channels, df['offline_ch']))
offline_to_daphne = dict(zip(df['offline_ch'],daphne_channels))
daphne_channel = daq_channel + 100*endpoint
apa_template_folder  = next((f for f in Path(maritza_template_folder).glob("*APA2*") if f.is_dir()), None)
martiza_template_file = next(apa_template_folder.glob(f"*APA{apa}_CH{daphne_to_offline[daphne_channel]}*.txt"), None)
with open(martiza_template_file, "r") as file:
    maritza_values = [float(line.strip()) for line in file]
maritza_template = np.array(maritza_values)
print('done\n')

# Reading wfset and beam selection
print('Reading waveform pickles file...')
with open(NP04_wfset_filepath, 'rb') as f:
    wfset = pickle.load(f) 
    print('done\n')
    
print('Beam event and channel section...')    
wfset_ch_beam = WaveformSet.from_filtered_WaveformSet(WaveformSet.from_filtered_WaveformSet(wfset, beam_self_trigger_filter), channel_filter, end=endpoint, ch=daq_channel)
print('done\n')

print(f"Wfset : {len(wfset.waveforms)}")
print(f"Wfset selected (beam and ch): {len(wfset_ch_beam.waveforms)}")

run_A = { "1": 27343, "2": 27355, "3": 27361, "5": 27367, "7": 27374}
# run_A = {"2": 27355}
for energy, run in run_A.items():
    try: 
        wfset_ch_beam_run = WaveformSet.from_filtered_WaveformSet(wfset_ch_beam, run_filter,run)
    except:
        continue
    
    print(f"Energy {energy}, wfset: {len(wfset_ch_beam_run.waveforms)}")
    APA_pdf_file = PdfPages(f"/afs/cern.ch/work/a/anbalbon/private/waffles/src/waffles/np04_analysis/lightyield_vs_energy/output/APA{apa}_END{endpoint}_CH{daq_channel}_{energy}GeV_deconvolution.pdf")

    max_index = len(wfset_ch_beam_run.waveforms)
    if max_index>15:
        n = 15
    else:
        n = max_index
        
    for n_wf in tqdm(range(n)): #for n_wf in tqdm(range(len(wfset_ch_beam_run.waveforms)), desc="Processing waveforms", unit="wf"):
        my_waveform = wfset_ch_beam_run.waveforms[n_wf]  
            
        # Removing baseline
        baseliner = SBaseline()
        baseliner.binsbase       = np.linspace(0, 2**14-1, 2**14)
        baseliner.threshold      = 6
        baseliner.wait           = 25
        baseliner.minimumfrac    = 0.166666
        baseliner.baselinestart  = 0
        baseliner.baselinefinish = 50

        base, optimal = baseliner.wfset_baseline(my_waveform)
        #print(f"Baseline evaluation status: {optimal} \t Value: {base}")
        
        new_wf_adcs = my_waveform.adcs.astype(float) - base
        new_wf_adcs = -new_wf_adcs
        
        # First deconvolution
        signal_fft = np.fft.fft(new_wf_adcs)
        maritza_template_fft = np.fft.fft(maritza_template, n=len(new_wf_adcs))
        martiza_deconvolved_fft = signal_fft / maritza_template_fft    
        martiza_deconvolved_wf = np.fft.ifft(martiza_deconvolved_fft).real  
        
        fig, axs = plt.subplots(len(cutoff_list), 2, figsize=(12, 10))  # 5 righe, 2 colonne
        fig.suptitle(f'Waveform {n_wf}', fontsize=14)

               
        for idx, cutoff_frequency in enumerate(cutoff_list):
            # Creating Gaussian filter
            _x = np.linspace(0, 1024, 1024, endpoint=False)
            sigma = from_cutoff_to_sigma(cutoff_frequency)
            filter_gaus = np.array([gaus(x, sigma=sigma) for x in _x])
            
            # Gaussian filter on wfset in frequency domain
            filtered_martiza_deconvolved_fft = martiza_deconvolved_fft * filter_gaus
            filtered_martiza_deconvolved_wf = np.fft.ifft(filtered_martiza_deconvolved_fft).real  

            # Gaussian filter on template in frequency domain
            filtered_martiza_template_fft = maritza_template_fft * filter_gaus
            filtered_martiza_template = np.fft.ifft(filtered_martiza_template_fft).real  

            # Plot waveform status
            axs[idx, 0].set_title(f'Waveform status (Cutoff = {cutoff_frequency} MHz , Sigma = {sigma:.2f})')
            axs[idx, 0].plot(normalize_signal(new_wf_adcs,np.argmax(new_wf_adcs)), color='blue', label='Original') 
            axs[idx, 0].plot(normalize_signal(maritza_template,np.argmax(new_wf_adcs)), color='orange', label='Template')
            #axs[idx, 0].plot(normalize_signal(filtered_martiza_template,np.argmax(new_wf_adcs)), color='gold', label='Filtered maritza template')
            axs[idx, 0].plot(normalize_signal(filtered_martiza_deconvolved_wf,np.argmax(new_wf_adcs)), color='red', label=f'Deconv + filter {integral(new_wf_adcs,base)}')
            axs[idx, 0].set_xlabel('Time ticks')
            axs[idx, 0].set_ylabel('Amplitude')
            axs[idx, 0].legend(fontsize=5, loc='upper right', frameon=False)
            axs[idx, 0].set_xlim(0, 400) 
            # axs[idx, 0].plot(new_wf_adcs, color='blue', label='Original') 
            # axs[idx, 0].plot(maritza_template, color='orange', label='Template')
            # axs[idx, 0].plot(filtered_martiza_deconvolved_wf, color='red', label='Deconv + filter')
            

            # Spettro in frequenza
            frequencies = np.fft.fftfreq(1024, d=16 * 1e-9 * 1e+6)[:1024//2+1]
            frequencies[-1] = -frequencies[-1]
            axs[idx, 1].set_title(f'Frequency spectrum (Cutoff = {cutoff_frequency} MHz , Sigma = {sigma:.2f})')
            axs[idx, 1].axvline(x=cutoff_frequency, color='black', linestyle="--", linewidth=0.5, label=f'Cutoff = {cutoff_frequency:.2f} MHz')
            axs[idx, 1].plot(frequencies, np.abs(signal_fft)[0:1024//2+1], color='blue', linestyle="--", label='Original')
            axs[idx, 1].plot(frequencies, np.abs(maritza_template_fft)[0:1024//2+1], color='orange', linestyle="--", label='Template')
            #axs[idx, 1].plot(frequencies, np.abs(filtered_martiza_template_fft)[0:1024//2+1], color='gold', linestyle="--", label='Filtered maritza template')
            axs[idx, 1].plot(frequencies, np.abs(martiza_deconvolved_fft)[0:1024//2+1], color='green', linestyle="--", label='Deconvolved')
            axs[idx, 1].plot(frequencies, np.abs(filtered_martiza_deconvolved_fft)[0:1024//2+1], color='red', linestyle="--", label='Deconv + filter')
            axs[idx, 1].set_yscale("log")
            axs[idx, 1].set_xscale("log")
            axs[idx, 1].set_xlabel('Frequency (MHz)')
            axs[idx, 1].set_ylabel('FFT')
            axs[idx, 1].set_ylim(1e-2, None)
            axs[idx, 1].legend(fontsize=4, loc='upper right', frameon=True)

        fig.subplots_adjust(hspace=0.8, wspace=0.3)
        APA_pdf_file.savefig(fig)
        plt.close(fig)  

    APA_pdf_file.close()

