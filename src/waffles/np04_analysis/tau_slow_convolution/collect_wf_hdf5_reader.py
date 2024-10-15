import pickle
import waffles.input.raw_hdf5_reader as reader
from waffles.data_classes.Waveform import Waveform
from waffles.data_classes.WaveformSet import WaveformSet
import argparse
from glob import glob
import os
import waffles

class Collector:
    def __init__(self, mtype:str = "purity", endpoint = 112, doall:bool = True, runs:list = [], forceit = False, rucio_files_path="./", output_path="./"):
        """This class is used to extract data from hdf5 files to waffles pickles.
        To be used in the context of convolution fit.

        Here I set the endpoint I want and which runs I want to skip
        The runs to be processed should be in the folder `rucio_files_path`/`mtype`_combo_files/ where mtype is `purity`, `led` or `beam`
        The files should follow this pather: `025171.txt` 

        """

        self.doall = True
        self.runs = runs
        self.mtype = mtype
        self.maxwvfs = 1e6
        self.forceit = forceit
        self.endpoint = endpoint

        self.channels_of_endpoint = {
            112: [11220, 11221, 11222, 11223, 11224, 11225, 11226, 11227, 11230, 11231, 11232, 11235, 11237],
            111: [11111, 11113, 11114, 11116, 11117, 11115, 11112, 11110],
        }
        for k, v in self.channels_of_endpoint.items():
            self.channels_of_endpoint[k] = [ x - k*100 if (x-k*100)>0 else x for x in v ]

        # This runs are skipped for endpoint 111 as I found no waveform there...
        self.skipthisled = [ 26114, 26139, 26156, 26075 ]
        self.skipthispurity = [ 26078, 26116, 26141, 26145, 26147, 26149, 26152, 26154, 26161, 26163, 26165, 26167 ]
        self.skipthisbeam = [ 28518 ]


        self.rucio_filepath = sorted(glob(f"{rucio_files_path}/{self.mtype}_combo_files/*.txt"))
        self.output_path = output_path
        if not os.path.isdir(self.output_path):
            raise Exception(f"Please, create the output folder: {self.output_path}")
        if not self.rucio_filepath:
            raise Exception(f"Aparentely, no run found in {self.mtype}_combo_files... ")


    def allow_certain_endpoints_channels(self, waveform: Waveform, allowed_endpoints:list, allowed_channels:list) -> bool:
        if waveform.endpoint in allowed_endpoints:
            if waveform.channel in allowed_channels:
                return True
        return False

    def collect(self):

        for f in self.rucio_filepath:
            runnumber = os.path.basename(f).strip(".txt")

            
            pickname = f"{self.output_path}/{self.endpoint}/wfset_run{runnumber}.pkl"
            if self.runs:
                if int(runnumber) not in self.runs:
                    continue
            if (int(runnumber) in self.skipthisled or int(runnumber) in self.skipthispurity or int(runnumber) in self.skipthisbeam) and self.endpoint == 111:
                continue
            if not self.forceit and os.path.isfile(pickname):
                print("Already there...", runnumber)
                continue

            filepaths = reader.get_filepaths_from_rucio(f)
            wfset_ch = 0
            allfiles = len(filepaths)
            for i, fpath in enumerate(filepaths):
                print(f"Processing {i}/{allfiles}")
                wfset = 0
                try:
                    wfset = reader.WaveformSet_from_hdf5_file( fpath,                            # path to the root file
                                                               read_full_streaming_data = False, # self-triggered (False) data
                                                               allowed_endpoints = [str(self.endpoint)],      # 
                                                             )                                   # subsample the data reading (read each 2 entries)
                except Exception as error:
                    print(error)
                    print("some error reading file.. ")
                    continue

                print('Filtering dataset...')
                tmpwfset = 0
                try:
                    tmpwfset = WaveformSet.from_filtered_WaveformSet( wfset, self.allow_certain_endpoints_channels, [self.endpoint], self.channels_of_endpoint[self.endpoint])
                except Exception as error:
                    print(error)
                    print('Type: ', error.__class__.__name__)
                    print('Continuing...')
                    continue
                if wfset_ch==0:
                    wfset_ch = tmpwfset
                else:
                    wfset_ch.merge(tmpwfset)
                    if len(wfset_ch.waveforms) > self.maxwvfs:
                        break
                print('wset len: ', len(wfset_ch.waveforms))
            if wfset_ch == 0:
                print("No data for these channels and endpoint :( ")
                continue
            print('Saving dataset...')

            with open(pickname, "wb") as f:
                pickle.dump(wfset_ch, f)



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('-r','--runs', type=int, nargs="+", help="Keep empty for all, or put the runs you want to be processed")
    parse.add_argument('-t','--mtype', type=str, help="What run list to be used (purity, led or beam)", default="purity")
    parse.add_argument('-e','--endpoint', type=int, help="Which endpoint", default=112)
    parse.add_argument('-f','--forceit', action="store_true", help='Overwrite...')
    args = vars(parse.parse_args())


    pathofwaffles = os.path.dirname(waffles.__file__)
    c = Collector(
        rucio_files_path=f"{pathofwaffles}/np04_analysis/tau_slow_convolution/runlists/files/",
        output_path="./rawdata/waffles_tau_slow_protoDUNE_HD/",
        **args)
    c.collect()




    
