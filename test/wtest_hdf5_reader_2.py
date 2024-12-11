import pickle
import waffles.input.raw_hdf5_reader as reader

print("Ok")

rucio_filepath = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/027367.txt"

filepaths = reader.get_filepaths_from_rucio(rucio_filepath)
#print(filepaths)


count = 0

for fp in filepaths:
  print(fp)
  wfset = reader.WaveformSet_from_hdf5_file( fp,                     # path to the root file
                                             read_full_streaming_data = False, # self-triggered (False) data
                                            )
  
  with open("/eos/home-f/fegalizz/public/to_Anna/wfset_27367_"+str(count)+".pkl", "wb") as f:
      pickle.dump(wfset, f)
  count += 1
