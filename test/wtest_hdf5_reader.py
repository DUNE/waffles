import pickle
import re
import waffles.input.raw_hdf5_reader as reader

rucio_filepath = "/eos/experiment/neutplatform/protodune/dune/hd-protodune/1a/ec/np04hd_raw_run030003_0000_dataflow0_datawriter_0_20241014T152553.hdf5"

match = re.search(r"run(\d{6})", rucio_filepath)

wfset = reader.WaveformSet_from_hdf5_file(
  rucio_filepath,                    
  read_full_streaming_data=False     
)

output_filename = f"wfset_{run_number}.pkl"
with open(output_filename, "wb") as f:
  pickle.dump(wfset, f)