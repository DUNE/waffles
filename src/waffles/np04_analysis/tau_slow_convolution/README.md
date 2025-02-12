# How to run

To do the analysis, you need to run three codes:

1. ./collect_wf_hdf5_reader.py
2. ./extract_selection.py
3. ./convfit.py

## collect_wf_hdf5_reader.py

The code gets hdf5 files and create WaveformSet's with selected endpoints and channels. If `runs` is not given, it will go over all files in `./runlists/files/`. 

Example of usage:
```bash
python  collect_wf_hdf5_reader.py --runs 29229 -t beam -e 111
```

Check `--help` for inputs

## extract_selection.py

For the saved pickle files, in `rawdata/waffles_tau_slow_protoDUNE_HD` the "spe" templates or LAr responses will be retrieved. The script `./runselection.sh` makes things much faster... BUT CAREFUL, because you need the `csv` files that are in `np4_data`.

Example of usage:
```bash
python extract_selection.py --runs 25171 -r -rl purity -ch 11114 11116
```

Check `--help` for inputs


## convfit.py

Performs the convolution fit :) 

It will automatically save data, unless argument `--no-save` is passed. The script `./runconvfit.sh` makes things faster.. but again CAREFUL.

Example of usage:

```bash
python convfit.py --runs 25171 -ch 11114 -rl purity -ft -tt 0 -ns new_analysis -i --scan 6'
```

Have fun :D 
