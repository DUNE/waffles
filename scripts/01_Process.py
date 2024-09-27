import click, pickle

from waffles.utils.utils import print_colored
import waffles.input.raw_hdf5_reader as reader

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
        print_colored("Please provide a run(s) number(s) to be analysed, separated by commas:)", color="yellow", styles=["bold"])
        run = [int(input("Run(s) number(s): "))]
    if len(run)!=1: runs_list = list(map(int, list(run.split(","))))
    else: runs_list = run
    
    for r in runs_list:
        rucio_filepath = f"/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/{str(r).zfill(6)}.txt"
        if debug: 
            print_colored(f"Processing {str(r).zfill(6)}...", color="DEBUG")
        
        filepaths = reader.get_filepaths_from_rucio(rucio_filepath)
        if len(filepaths) > 5:
            print_colored(f"This run has {len(filepaths)} hdf5 files. \n {filepaths[:5]}", color="WARNING")
            file_lim = input("How many of them do we process? ")
        else: 
            file_lim = len(filepaths)
        
        wfset = reader.WaveformSet_from_hdf5_files( filepaths[:int(file_lim)],        # path to the root file
                                                    read_full_streaming_data = False, # self-triggered (False) data
                                                  )                                   # TODO: subsample the data reading (read each 2 entries)

        if debug: 
            print_colored("Saving the WaveformSet object in a pickle file...", color="DEBUG")
        with open(f"{str(r).zfill(6)}_full_wfset_raw.pkl", "wb") as f:
            pickle.dump(wfset, f)
        
        print_colored(f"\nDone! WaveformSet saved in {str(r).zfill(6)}_wfset.pkl\n", color="SUCCESS")
        

if __name__ == "__main__":
    main()