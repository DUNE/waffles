
import pickle
import waffles.input.raw_hdf5_reader as reader

rucio_filepath = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/029297.txt"
filepaths = reader.get_filepaths_from_rucio(rucio_filepath)
print(filepaths)
wfset = reader.WaveformSet_from_hdf5_file( filepaths[1],                     # path to the root file
                                           read_full_streaming_data = False, # self-triggered (False) data
                                           #truncate_wfs_to_minimum  = True,
                                         )                                   # subsample the data reading (read each 2 entries)
with open("wfset_??.pkl", "wb") as f:
    pickle.dump(wfset, f)

