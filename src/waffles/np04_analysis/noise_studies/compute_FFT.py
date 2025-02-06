# comment

import waffles
import yaml
import pickle
import numpy as np
import pandas as pd
import waffles.input.raw_hdf5_reader as reader

######################################################################
############### HARDCODE #############################################

channel_map_file = "./PDHD_PDS_ChannelMap.csv"


# run_vgain_dict = {
#     31913: 3192, 31914: 3192, 31915: 2926, 31916: 2926,
#     31917: 2793, 31918: 2793, 31919: 2660, 31920: 2660,
#     31921: 2527, 31922: 2527, 31923: 2394, 31924: 2394,
#     31925: 2261, 31926: 2261, 31927: 2128, 31928: 2128,
#     31929: 2128, 31930: 1995, 31931: 1995, 31932: 1862,
#     31933: 1862, 31934: 1729, 31935: 1729, 31936: 1596,
#     31937: 1596, 31938: 1463, 31939: 1463, 31940: 1330,
#     31941: 1330, 31942: 1197, 31943: 1197, 31944: 1064,
#     31945: 1064, 31946: 931,  31947: 931
# }


out_path = "/eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/analysis/"
######################################################################


######################################################################
############### FUNCTION IMPLEMENTATION ##############################
def read_waveformset(filepath_folder: str, run: int, full_stat = True) -> waffles.WaveformSet:
    filepath_file = filepath_folder + "0" + str(run) + ".txt"
    filepath = reader.get_filepaths_from_rucio(filepath_file)

    if full_stat:
        wfset = reader.WaveformSet_from_hdf5_file(filepath[0])
        for fp in filepath[1:]:
            ws = reader.WaveformSet_from_hdf5_file(fp)
            wfset.merge(ws)
    else:
        wfset = reader.WaveformSet_from_hdf5_file(filepath[0])

    return wfset


def allow_ep_wfs(waveform: waffles.Waveform, endpoint) -> bool:
    return waveform.endpoint == endpoint

def allow_channel_wfs(waveform: waffles.Waveform, channel: int) -> bool:
    return waveform.channel == channel

def create_float_waveforms(waveforms: waffles.Waveform) -> None:
    for wf in waveforms:
        wf.adcs_float = wf.adcs.astype(np.float64)

def sub_baseline_to_wfs(waveforms: waffles.Waveform, prepulse_ticks: int):
    norm = 1./prepulse_ticks
    for wf in waveforms:
        baseline = np.sum(wf.adcs_float[:prepulse_ticks])*norm
        wf.adcs_float -= baseline
        wf.adcs_float *= -1
######################################################################





######################################################################
############### MAIN #################################################
# path = "/eos/home-f/fegalizz/ProtoDUNE_HD/Noise_Studies/files/"
# runs = [31920,31928, 31932, 31936]
# files = [path+"wfset_"+str(run)+".pkl" for run in runs]


if __name__ == "__main__":

    # Setup variables according to the noise_run_info.yaml file
    with open("./noise_run_info.yaml", 'r') as stream:
        run_info = yaml.safe_load(stream)

    filepath_folder = run_info.get("filepath_folder")
    run_vgain_dict  = run_info.get("run_vgain_dict", {})

    # Setup variables according to the user_config.yaml file
    with open("./user_config.yaml", 'r') as stream:
        user_config = yaml.safe_load(stream)

    out_path  = user_config.get("out_path")
    full_stat = user_config.get("full_stat")
    runs      = user_config.get("user_runs", [])
    if (len(runs) == 0):
        runs = user_config.get("all_noise_runs", [])
        if (len(runs) == 0):
            print("No runs to analyze")
            exit()

    df = pd.read_csv(channel_map_file, sep=",")
    daphne_channels = df['daphne_ch'].values + 100*df['endpoint'].values
    daphne_to_offline = dict(zip(daphne_channels, df['offline_ch']))


    # create a csv file with conlumns: run, endpoint, channel, rms,
    my_csv_file = open(out_path+"Noise_Studies_Results.csv", "w")


    # for file, run in zip(files,runs):
    #     print("Reading file: ", file)
    #     with open(f'{file}', 'rb') as f:
    #         wfset_run = pickle.load(f)

    for run in runs:
        print("Reading run: ", run)
        wfset_run = read_waveformset(filepath_folder, run, full_stat=full_stat)
        endpoints = wfset_run.get_set_of_endpoints()

        for ep in endpoints:
            print("Endpoint: ", ep)
            wfset_ep = waffles.WaveformSet.from_filtered_WaveformSet(wfset_run, allow_ep_wfs, ep)

            ep_ch_dict = wfset_ep.get_run_collapsed_available_channels()
            channels = list(ep_ch_dict[ep])

            for ch in channels:
                print("Channel: ", ch)
                wfset_ch = waffles.WaveformSet.from_filtered_WaveformSet(wfset_ep, allow_channel_wfs, ch)
                # check if the channel is in the daphne_to_offline dictionary
                channel = np.uint16(np.uint16(ep)*100+np.uint16(ch))
                if channel not in daphne_to_offline:
                    print(f"Channel {channel} not in the daphne_to_offline dictionary")
                    continue
                offline_ch = daphne_to_offline[channel]
        
                wfs = wfset_ch.waveforms
                create_float_waveforms(wfs)
                sub_baseline_to_wfs(wfs, 1024)

                norm = 1./len(wfs)
                fft2_avg = np.zeros(1024)
                rms = 0.

                # Compute the average FFT of the wfs.adcs_float
                for wf in wfs:
                    rms += np.std(wf.adcs_float)
                    fft  = np.fft.fft(wf.adcs_float)
                    fft2 = np.abs(fft)
                    fft2_avg += fft2

                fft2_avg = fft2_avg*norm
                rms = rms*norm

                vgain = run_vgain_dict[run]
                # print rms in a csv file
                my_csv_file.write(f"{run},{vgain},{ep},{ch},{offline_ch},{rms}\n")
                # print fft2_avg in a file.txt
                np.savetxt(out_path+"/FFT_txt/fft_run_"+str(run)+"_vgain_"+str(vgain)+"_ch_"+str(channel)+"_offlinech_"+str(offline_ch)+".txt", fft2_avg[0:513])
