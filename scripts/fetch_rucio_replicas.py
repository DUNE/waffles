import os
import subprocess
import shlex
import getpass
import click

# Define the path where the output files should be saved
SAVING_PATH = ""

def is_rucio_active():
    """Checks if Rucio is already active by running `rucio whoami`."""
    try:
        subprocess.run("rucio whoami", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("\033[92mRucio is already set up. Skipping environment setup.\033[0m")
        return True
    except subprocess.CalledProcessError:
        print("\033[93mRucio is not set up. Initializing the environment...\033[0m")
        return False

def setup_rucio_environment():
    """Sets up the Rucio environment if not already active."""
    if is_rucio_active():
        return  # Skip setup if Rucio is already running

    print("\033[94mSetting up the Rucio environment...\033[0m")

    # Prompt for Kerberos authentication
    username = input("Enter your @FNAL.GOV username: ")
    password = getpass.getpass("Please enter your password: ")

    # Full command string to set up the environment, authenticate, and persist
    setup_cmd = f"""
    source /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh && \
    spack load r-m-dd-config experiment=dune && \
    spack load kx509 && \
    echo "{password}" | kinit {username}@FNAL.GOV && \
    kx509 && \
    export RUCIO_ACCOUNT={username} && \
    voms-proxy-init -rfc -noregen -voms=dune:/dune/Role=Analysis -valid 120:00 && \
    export UPS_OVERRIDE="-H Linux64bit+3.10-2.17" && \
    setup ifdhc && \
    rucio whoami
    """

    # Run in a single shell session
    env_command = f'bash -c "{setup_cmd}"'
    result = subprocess.run(env_command, shell=True, executable="/bin/bash")

    if result.returncode == 0:
        print("\033[92mRucio authentication successful!\033[0m")
    else:
        print("\033[31mRucio setup failed. Please check your environment settings.\033[0m")
        exit(1)

def fetch_rucio_replicas(run_number, max_files):
    """Fetches Rucio file replicas for a given run, selects all files from the first found realm (protocol + domain), and saves the results."""

    run_str = str(run_number).zfill(6)
    output_file = f"{SAVING_PATH}{run_str}.txt"

    print(f"\033[94mFetching Rucio replicas for run {run_str}...\033[0m")

    # Construct the Rucio command
    rucio_command = f"rucio list-file-replicas --pfns hd-protodune:hd-protodune_{run_number}"

    try:
        # Run the command and capture the output
        result = subprocess.run(rucio_command, shell=True, check=True, text=True, capture_output=True, executable="/bin/bash")
        lines = result.stdout.strip().split("\n")

        if not lines:
            print(f"\033[31mNo file replicas found for run {run_str}.\033[0m")
            return

        # Identify the first "realm" (protocol + domain)
        first_realm = None
        for line in lines:
            if "://" in line:
                first_realm = line.split("/")[2]  # Extracts domain (e.g., eospublic.cern.ch)
                break

        if not first_realm:
            print("\033[31mNo valid realm found in the Rucio output.\033[0m")
            return

        print(f"\033[94mFiltering results from the first realm: {first_realm}\033[0m")

        # Filter only lines that match the first realm (protocol + domain)
        filtered_lines = [line for line in lines if first_realm in line]

        # Limit to max_files
        selected_lines = filtered_lines[:max_files]

        # Save to file
        with open(output_file, "w") as f:
            f.write("\n".join(selected_lines) + "\n")

        print(f"\033[92mSaved {len(selected_lines)} Rucio paths from {first_realm} to {output_file}\033[0m")

    except subprocess.CalledProcessError as e:
        print(f"\033[31mError fetching Rucio replicas: {e}\033[0m")

@click.command()
@click.option("--runs", required=True, help="Comma-separated run numbers (e.g., 28676,28677)")
@click.option("--max-files", required=True, type=int, help="Maximum number of files to save per run")
def main(runs, max_files):
    """Main function to set up Rucio and fetch file replicas."""
    # Set up the Rucio environment (only if needed)
    setup_rucio_environment()

    # Parse run numbers
    run_list = [run.strip() for run in runs.split(",")]

    # Process each run
    for run in run_list:
        fetch_rucio_replicas(run, max_files)

    # Cleanup authentication
    os.system("kdestroy")
    print("\033[92mSession complete. Kerberos authentication destroyed.\033[0m")

if __name__ == "__main__":
    main()