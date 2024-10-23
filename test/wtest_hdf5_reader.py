import pickle
import waffles.input.raw_hdf5_reader as reader

print("Ok")
rucio_filepath = "/eos/experiment/neutplatform/protodune/dune/hd-protodune/1a/ec/np04hd_raw_run030003_0000_dataflow0_datawriter_0_20241014T152553.hdf5"
#filepaths = reader.get_filepaths_from_rucio(rucio_filepath)
#print(filepaths)


wfset = reader.WaveformSet_from_hdf5_file( rucio_filepath,                     # path to the root file
                                           read_full_streaming_data = False, # self-triggered (False) data
                                         )                                   # subsample the data reading (read each 2 entries)
with open("wfset_30003.pkl", "wb") as f:
    pickle.dump(wfset, f)
