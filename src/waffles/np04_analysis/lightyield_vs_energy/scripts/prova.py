# # LETTURA STRUCTURED HDF5 FILES
# import h5py
# from waffles.input_output.hdf5_structured import * 
# filepath = "/afs/cern.ch/work/a/anbalbon/public/reading_beamrun_NEW/run027343/processed_np04hd_raw_run027343_0019_dataflow2_datawriter_0_20240621T135147.hdf5_structured.hdf5"
# wfset=load_structured_waveformset(filepath)
# print(len(wfset.waveforms))

# # LETTURA FILE PKL 
import pickle
filepath = "/eos/user/a/anbalbon/reading_beamrun_NEW/run027343/prova_290825.pkl"
with open(filepath, 'rb') as f:
    wfset = pickle.load(f)
print(len(wfset.waveforms))

