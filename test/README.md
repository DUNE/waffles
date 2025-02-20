# Use of 'hdf5_to_hdf5_converter.py'

This Python script reads the complete HDF5 file provided, saves the waveformsets in a compressed HDF5 format (Gzip), and reads it again.

### To execute it:
```bash
python3 hdf5_to_hdf5_converter.py --runs <run_number>
```

## From a rucio filepath

If you retrieve the run from Rucio, remember to do the following beforehand:
```bash
cd <repos_dir>/waffles/scripts/
python get_rucio.py --runs <run_number>
```
You may need to update `<rucio_filepath>` in the script, depending on the corresponding Rucio filepath.

## From a downloaded file in a specific filepath 

If you have downloaded the HDF5 file to a specific filepath, you must provide `<filepath>`.

In both cases, `<det>` must be provided in `reader.WaveformSet_from_hdf5_file`, depending on the detector ID. 

### Output:
- A compressed HDF5 file with the corresponding waveformsets.  
- The file size.  
- The time taken.  

# Use of 'hdf5_to_severals_converter.py'

This Python script reads the complete HDF5 file provided and saves the waveformsets in different formats: Pickle, Hickle, no compressed HDF5 and compressed HDF5 files (Gzip, LZF). 

### To execute it:
```bash
python3 hdf5_to_severals_converter.py --runs <run_number>
```

## From a rucio filepath

As in `hdf5_to_hdf5_converter.py`, to find the corresponding Rucio filepath for the run, you must do the following beforehand:
```bash
cd <repos_dir>/waffles/scripts/
python get_rucio.py --runs <run_number>
```
You may need to update `<rucio_filepath>` in the script, depending on the corresponding Rucio filepath.

## From a download file in a specific filepath 

If you have downloaded the HDF5 file to a specific filepath, you must provide `<filepath>`.

In both cases, `<det>` must be provided in `reader.WaveformSet_from_hdf5_file`, depending on the detector ID.

### Output:
- The waveformsets saved in different formats.  
- The file sizes.  
- The time taken in each case.