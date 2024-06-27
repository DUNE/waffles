import os, click, subprocess, platform, shlex

@click.command()
@click.option("--runs", help="Run number to be analysed")
def main(runs):
    '''
    Basic standalone script to convert HDF5 files to .npy files.

    Args:
        - run (int): Run number to be analysed
    Example: python get_rucio.py --run 123456
    '''

    ## Check if the run number is provided ##
    if runs is None: 
        print("\033[35mPlease provide a run(s) number(s) to be analysed, separated by commas:)\033[0m")
        runs = [int(input("Run(s) number(s): "))]
    if len(runs)!=1: runs_list = list(map(int, list(runs.split(","))))
    else: runs_list = runs
    
    current_path = os.getcwd()
    saving_path = "/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/1_rucio_paths/"
    ## Loop over the runs ##
    for one_run in runs_list:
        print(f"\033[94m\nGetting the path for run {one_run}:\033[0m")
        # Check if the files are stored in /eos/
        if os.path.exists(f"{saving_path}{str(one_run).zfill(6)}.txt"):
            print(f"\033[92mFound the file {saving_path}{str(one_run).zfill(6)}.txt\n\033[0m")
            with open(f"{saving_path}{str(one_run).zfill(6)}.txt", "r") as f: print(f.read())
        # If not, get the rucio paths
        else:
            print(f"\033[35m\nYou are the first one looking for this file. Let's get the rucio paths!.\033[0m")
            homepath = os.environ['HOME']

            # Check if the current OS is CentOS 7
            if 'centos-7' in platform.platform():
                print(f"\033[92mYou are running on CentOS 7. No need to ssh o lxplus7.\033[0m")
                # If it is CentOS 7, just run the script
                get_rucio = f"bash get_protodunehd_files.sh local cern {one_run}" 
                subprocess.call(shlex.split(get_rucio), shell=False)
                print(f"[WARNING] Inside lxplus7 the file will be saved in {homepath}/{one_run}.txt and not moved to {saving_path}{one_run}.txt\n")
            
            # If not --> Run the SSH command/enter a container (needs to be already in lxplus)
            else:
                # print(f"Connecting to lxplus7 to get rucio paths :)\n") # NO MORE LXPLUS7
                # ssh_command = f'ssh -t {username}@lxplus7.cern.ch "source {current_path}/get_protodunehd_files.sh local cern {one_run}"'
                username = os.environ['USER']
                print(f"Starting a SL7 container to get rucio paths :)\n")
                sl7_command = f'/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer exec -f -B \
                /cvmfs,/afs/cern.ch/user/{username[0]}/{username},/tmp,/etc/hostname,/etc/hosts,/etc/krb5.conf,/run/user/ /cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest \
                sh {current_path}/get_protodunehd_files.sh local cern {one_run}'
                subprocess.run(shlex.split(sl7_command), shell=False)

                one_run = str(one_run).zfill(6)
                print("\n")
                with open(f"{homepath}/{one_run}.txt", "r") as f:
                    print(f.read())
                print("\n")

                subprocess.call(shlex.split(f"mv {homepath}/{one_run}.txt {saving_path}{one_run}.txt"), shell=False)


if __name__ == "__main__":
    main()