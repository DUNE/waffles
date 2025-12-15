import click, inquirer

from waffles.utils.utils import print_colored
import waffles.input_output.raw_hdf5_reader as reader  # path utilities (rucio)
from waffles.input_output.waveform_loader import load_waveforms
from waffles.input_output.persistence_utils import WaveformSet_to_file

@click.command(help=f"\033[34mSave the WaveformSet object in a pickle file for easier loading.\n\033[0m")
@click.option("--run",   default = None, help="Run number to process", type=str)
@click.option("--debug", default = True, help="Debug flag")
def main(run, debug):
    '''
    Script to process peak/pedestal variables and save the WaveformSet object + unidimensional variables in a pickle file.

    Args:
        - run (int): Run number to be analysed. I can also be a list of runs separated by commas.
    Example: python 01Process.py --run 123456 or --run 123456,123457
    '''
    if run is None: 
        q = [ inquirer.Text("run", message="Please provide the run(s) number(s) to be analysed, separated by commas:)")]
        run_list = inquirer.prompt(q)["run"].split(",")
    else:
        run_list = run.split(",")
    
    for r in run_list:
        rucio_filepath = f"/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/{str(r).zfill(6)}.txt"
        if debug: 
            print_colored(f"Processing {str(r).zfill(6)}...", color="DEBUG")
        
        filepaths = reader.get_filepaths_from_rucio(rucio_filepath)
        if len(filepaths) > 5:
            print_colored(f"This run has {len(filepaths)} hdf5 files. \n {filepaths[:5]}", color="WARNING")
            file_lim = [ inquirer.Text("file_lim", message="How many of them do we process?") ]
            file_lim = inquirer.prompt(file_lim)["file_lim"]
        else: 
            file_lim = len(filepaths)
        
        # Read waveforms using the unified loader and merge
        wfset = None
        for fp in filepaths[:int(file_lim)]:
            wf = load_waveforms(fp, det="HD_PDS", force_raw=True)
            if wfset is None:
                wfset = wf
            elif wf is not None:
                wfset.merge(wf)
        # TODO: subsample the data reading (read each 2 entries)

        if debug:
            print_colored("Saving the WaveformSet object as structured HDF5...", color="DEBUG")
        WaveformSet_to_file(
            waveform_set=wfset,
            output_filepath=f"../data/{str(r).zfill(6)}_full_wfset_raw.hdf5",
            overwrite=True,
            format="hdf5",
            compression="gzip",
            compression_opts=5,
            structured=True,
        )
        
        print_colored(f"\nDone! WaveformSet saved in {str(r).zfill(6)}_full_wfset_raw.hdf5\n", color="SUCCESS")
        

if __name__ == "__main__":
    main()
