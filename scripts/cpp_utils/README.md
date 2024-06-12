# CPP UTILS

In this folder you can find several cpp macros that can be used to quickly analyse the acquired data.

## 0. Compile

The first step (only the first time you use them) is to compile the macros inside the hdf5libs git repository.
For that you need to run inside an updated daq environment:

```bash
sh compile_decoder.sh
```

which will clone the repository inside your environment and compile the macros for you to use them.

## 1. Decode

Now you are able to run the `HDF5toROOT_decoder` from wherever folder you need. It is expecting the raw `hdf5` file location as input. For getting this path you can run `python get_rucio.py` to either read a txt file stored in `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths``or generate it with your `rucio` credentials.

Usage example:
```bash
HDF5toROOT_decoder /eos/experiment/neutplatform/protodune/dune/hd-protodune/f7/2a/np04hd_raw_run026912_0000_dataflow0_datawriter_0_20240612T081722.hdf5
```

To have a look at the statistics of the duplicated waveforms you need to run, in a similar way:
```bash
HDF5LIBS_duplications /eos/experiment/neutplatform/protodune/dune/hd-protodune/f7/2a/np04hd_raw_run026912_0000_dataflow0_datawriter_0_20240612T081722.hdf5
```

## 2. Plot

For having a look at you data you can run the `plotsAPA.C` macro (no need to compile).
Usage example:
```bash
root plotsAPA.C("run_26912_0_dataflow0_datawriter_0_decode.root",26912) -b -q
```
This will produce an output `run_XXXXX.root` with the plots stored inside.