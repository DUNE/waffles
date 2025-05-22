import json
import click
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statistics

from waffles.input_output.hdf5_structured import load_structured_waveformset
import waffles.plotting.drawing_tools as draw

plotwfms = False 
nadcthrs = 8 # number of ADC thresholds
countupcross = np.zeros((8,), dtype=np.int32)
delta_ADC = np.zeros((8,), dtype=np.float32)
countwfms = 0
channelofinterest = 30
delta_ADC[0] = 25 # normally depends on channel, set at roughly SPE amplitude
for ithres in range(nadcthrs):
    delta_ADC[ithres] = delta_ADC[0] + 30*ithres

############################
# Before black blanket cover
# Stored: /pnfs/dune/persistent/users/weishi/PDVDNoiseHunt
############################
# cosmic run, self triggered (before cover NP02)
#filepath="/pnfs/dune/persistent/users/weishi/PDVDNoiseHunt/processed_np02vd_raw_run036362_0000_df-s04-d0_dw_0_20250507T145213.hdf5.copied_structured.hdf5"
# random trigger no led
#filepath="processed_np02vd_raw_run036019_0000_df-s04-d0_dw_0_20250425T090046.hdf5_structured.hdf5"

############################
# After black blanket cover
# Stored: /pnfs/dune/persistent/users/weishi/PDVDNoiseHunt
############################
# RANDOM TRIGGER (Cover PrM fibers + temperature monitor light 1500nm on, camera on with light off, Bi source)
#filepath="processed_np02vd_raw_run036400_0000_df-s04-d0_dw_0_20250512T121635.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036412_0000_df-s04-d0_dw_0_20250513T142358.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036413_0000_df-s04-d0_dw_0_20250513T144818.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036414_0000_df-s04-d0_dw_0_20250513T145205.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036419_0000_df-s04-d0_dw_0_20250513T151530.hdf5.copied_structured.hdf5"
# RANDOM TRIGGER (Remove PrM fibers cover + temperature monitor light 1500nm off, camera on with light off, Bi source)
#filepath="processed_np02vd_raw_run036434_0000_df-s04-d0_dw_0_20250514T084924.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036435_0000_df-s04-d0_dw_0_20250514T085249.hdf5.copied_structured.hdf5"
# RANDOM TRIGGER (Remove PrM fibers cover + temperature monitor light 1500nm off and fibers unplugged and flange covered, camera on with light off, Bi source)
#filepath="processed_np02vd_raw_run036445_0000_df-s04-d0_dw_0_20250514T093130.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036449_0000_df-s04-d0_dw_0_20250514T094448.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036463_0000_df-s04-d0_dw_0_20250514T103210.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036457_0000_df-s04-d0_dw_0_20250514T101218.hdf5.copied_structured.hdf5"
# RANDOM TRIGGER (Manhole not covered)
#filepath="processed_np02vd_raw_run036481_0000_df-s04-d0_dw_0_20250515T085932.hdf5.copied_structured.hdf5"
#filepath="processed_np02vd_raw_run036482_0000_df-s04-d0_dw_0_20250515T090254.hdf5.copied_structured.hdf5"
# RANDOM TRIGGER (Manhole covered with copper foil and black blankets)
#filepath="processed_np02vd_raw_run036489_0000_df-s04-d0_dw_0_20250515T103041.hdf5.copied_structured.hdf5"
filepath="/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-VD/check_light_leakage/processed_np02vd_raw_run036498_0000_df-s04-d0_dw_0_20250515T105742.hdf5.copied_structured.hdf5"

# Cosmic trigger
# /pnfs/dune/persistent/users/weishi/PDVDNoiseHunt/processed_np02vd_raw_run036401_0000_df-s04-d0_dw_0_20250512T122851.hdf5.copied_structured.hdf5
#filepath="processed_np02vd_raw_run036405_0000_df-s04-d0_dw_0_20250513T123222.hdf5.copied_structured.hdf5" - same condition as cosmic run 36362
#filepath="processed_np02vd_raw_run036401_0000_df-s04-d0_dw_0_20250512T122851.hdf5.copied_structured.hdf5"

wfset = load_structured_waveformset(str(filepath))

print("file path: ", filepath)
print("run number: ", wfset.runs)
print("available channels: ", wfset.available_channels)
print("tot waveforms from all channels: ", len(wfset.waveforms))
print("1st wfm attributes: ", vars(wfset.waveforms[0]))
print("1st wfm adcs: ", wfset.waveforms[0].adcs)
print("1st wfm number of ticks: ", len(wfset.waveforms[0].adcs))
print("1st wfm channel: ", wfset.waveforms[0].channel)

BaselineADCAllWfms = []

for iwfm in range(len(wfset.waveforms)):

        # TCO side lower XA closer to cathode, VD style DVDC-DVDM
        if iwfm % 10000 == 0:
            print(iwfm)

        if wfset.waveforms[iwfm].channel == channelofinterest:
            countwfms = countwfms + 1
            # find baseline of the wfm
            baseline_ADC = 0.
            baseline_ADC = statistics.mode(wfset.waveforms[iwfm].adcs)

            # Fill baseline ADC of all waveforms in the data (a distribution of baselines)
            BaselineADCAllWfms.append(baseline_ADC)

            # Loop over points in the waveform
            for itick in range(len(wfset.waveforms[iwfm].adcs)):

                if itick < 1023: # max 1024 ticks
                    # Count up-crossings in each waveform
                    for ithres in range(nadcthrs):
                        if( ( wfset.waveforms[iwfm].adcs[itick] - (baseline_ADC+delta_ADC[ithres]) )<0 and ( wfset.waveforms[iwfm].adcs[itick+1] - (baseline_ADC+delta_ADC[ithres]) )>0 ):
                            countupcross[ithres] += 1

            if iwfm < 300 and plotwfms == True:
                # plot the wfm individually
                xaxis = [x for x in range(len(wfset.waveforms[iwfm].adcs))]
                plt.plot(xaxis, wfset.waveforms[iwfm].adcs)
                plt.savefig("plots/"+str(wfset.runs)+"_ch_"+str(channelofinterest)+"wfm_"+str(iwfm)+".pdf")
                plt.clf() # important to clear figure
                plt.close()

print("iwfm at end: ", iwfm)
print("tot wfms in channel ", channelofinterest, ": ", countwfms)

print("========== Rate report =========== ")
for ithres in range(nadcthrs):
    print("Counts at delta ADC ", delta_ADC[ithres], ": ", countupcross[ithres])

for ithres in range(nadcthrs):
    print("Counts/10us at delta ADC ", delta_ADC[ithres], ": ", countupcross[ithres]*10000/(countwfms*(len(wfset.waveforms[iwfm].adcs)*16)) )

'''
Baselines_allwfms = [(x) for x in BaselineADCAllWfms]
plt.hist(Baselines_allwfms, range=(0,10000), bins=1000)
plt.xlabel('ADC')
plt.draw()
plt.savefig("./"+str(wfset.runs)+"_ch_"+str(channelofinterest)+"_baselines.pdf")
plt.clf() # important to clear figure
plt.close()
'''