import os, re, click, subprocess, shlex, inquirer

def save_output(homepath, saving_path, one_run):
    one_run = str(one_run).zfill(6)
    try: 
        print("\n")
        with open(f"{homepath}/{one_run}.txt", "r") as f:
            output = f.read()
            print(output)
        print("\n")
        subprocess.call(shlex.split(f"mv {homepath}/{one_run}.txt {saving_path}{one_run}.txt"), shell=False)
    except FileNotFoundError:
        output = ""
        print("\033[35mNo files found for this run number.\033[0m")
    
    return output

@click.command()
@click.option("--runs", help="Run number to be analysed")
@click.option("--max-files", default=30, type=int, help="Maximum number of files to save")
def main(runs, max_files):
    '''
    Basic standalone script to convert HDF5 files to .npy files.

    Args:
        - run (int): Run number to be analysed
        - max-files (int, optional): Maximum number of files to save
    Example: python get_rucio.py --run 123456 --max-files 10
    '''

    ## Check if the run number is provided ##
    if runs is None: 
        print("\033[35mPlease provide a run(s) number(s) to be analysed, separated by commas:)\033[0m")
        runs = [int(input("Run(s) number(s): "))]
    if len(runs)!=1: runs_list = list(map(int, list(runs.split(","))))
    else: runs_list = runs
    print(runs)
    
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
            # print(f"\033[35m\nYou are the first one looking for this file. Let's get the rucio paths!.\033[0m")
            # homepath = os.environ['HOME']

            # get_rucio = f"bash get_protodunehd_files.sh local cern {one_run}" #This looks for local files in CERN computers
            # subprocess.call(shlex.split(get_rucio), shell=False)
            # output = save_output(homepath, saving_path, one_run)
            
            # if output == "":
                get_sites = f"rucio list-file-replicas hd-protodune:hd-protodune_{one_run} | head -n {max_files}" #This looks for local files in CERN computers
                sites_list = subprocess.check_output(shlex.split(get_sites), shell=False)
                sites_list = sites_list.decode('utf-8')  # Decode bytes to string
                
                rse_pattern = re.compile(r'\|\s+(\w+):')
                rse_values = set(rse_pattern.findall(sites_list))
                list_rse = list(rse_values)
                list_rse.remove("RSE")
                
                #question to choose the site with inquirer
                question = [ inquirer.Checkbox("RSE", message=f"Choose the RSE to save the rucio paths of your run", choices=list_rse) ]
                user_input = inquirer.prompt(question)["RSE"]
                
                # Extract and save associated REPLICA paths
                replica_pattern = re.compile(rf'{user_input[0]}: (.+)')
                replica_paths = replica_pattern.findall(sites_list)

                # Apply max-files limit
                if max_files is not None:
                    replica_paths = replica_paths[:max_files]

                with open(f"{saving_path}{str(one_run).zfill(6)}.txt", "w") as f:
                    for path in replica_paths:
                        cleaned_path = path.strip().rstrip('|')
                        f.write(cleaned_path + "\n")

                print(f"Saved {len(replica_paths)} REPLICA paths for RSE {user_input[0]} to {saving_path}{str(one_run).zfill(6)}.txt")

if __name__ == "__main__":
    main()