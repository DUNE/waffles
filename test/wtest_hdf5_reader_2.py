import os
import pickle
import re
import waffles.input.raw_hdf5_reader as reader

rucio_filepath = (
    "/eos/experiment/neutplatform/protodune/experiments/"
    "ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/027367.txt"
)

match = re.search(r"(\d{5,6})\.txt", rucio_filepath)

filepaths = reader.get_filepaths_from_rucio(rucio_filepath)

for fp in filepaths:
    print(f"Processing file: {fp}")
    wfset = reader.WaveformSet_from_hdf5_file(
        fp,                              
        read_full_streaming_data=False 
    )

    base_name = os.path.basename(fp).replace(".hdf5", "")
    output_filename = (
        f"/eos/home-f/fegalizz/public/to_Anna/wfset_{run_number}_{base_name}.pkl"
    )

    with open(output_filename, "wb") as f:
        pickle.dump(wfset, f)

    print(f"WaveformSet saved to {output_filename}")