"""
This script allows to visualize the waveforms for a given run. 

In particular, it performs the following steps:

1. Connects to a remote server via SSH (with password or private key).
2. Searches for .hdf5 files for a given run number.
3. Downloads the selected file in the current folder.
4. Updates a JSON configuration and runs an external processing script (07_save_structured_from_config.py).
5. Loads the processed structured HDF5 waveform.
6. Analyzes the waveforms using a basic analysis class.
7. Plots the results (TCO and non-TCO membranes). 

"""

import os
import getpass
import json
import paramiko
import subprocess

import numpy as np
import plotly.graph_objects as pgo
import plotly.subplots as psu
import h5py

from waffles.input_output.hdf5_structured import load_structured_waveformset
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.IPDict import IPDict
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map
from waffles.plotting.plot import plot_ChannelWsGrid


# ------------------------------------------------------------------------------
# SSH UTILS
# ------------------------------------------------------------------------------

def connect_ssh(hostname, port, username, private_key_path=None, password=None):
    """Establish an SSH connection with optional private key or password."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if private_key_path:
        key = paramiko.RSAKey.from_private_key_file(private_key_path, password=password)
        client.connect(hostname, port=port, username=username, pkey=key)
    elif password:
        client.connect(hostname, port=port, username=username, password=password)
    else:
        raise ValueError("Either private_key_path or password must be provided")
    return client


def list_files(ssh_client, remote_path, run_number):
    """List remote .hdf5 files matching the run number."""
    cmd = f"ls {remote_path}/np02vd_raw_run{run_number:06d}_*.hdf5"
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    files = stdout.read().decode('utf-8').splitlines()
    if stderr.read().decode('utf-8'):
        raise RuntimeError("Error listing files on remote server")
    return files


def download_files(ssh_client, remote_files, local_dir):
    """Download specified files from remote server via SFTP."""
    sftp = ssh_client.open_sftp()
    os.makedirs(local_dir, exist_ok=True)
    downloaded_files = []
    for remote_file in remote_files:
        local_file = os.path.join(local_dir, os.path.basename(remote_file))
        sftp.get(remote_file, local_file)
        print(f"Downloaded: {remote_file} -> {local_file}")
        downloaded_files.append(os.path.basename(remote_file))
    sftp.close()
    return downloaded_files


# ------------------------------------------------------------------------------
# PROCESSING UTILS
# ------------------------------------------------------------------------------

def update_config_and_run(run_number, hdf5_filename):
    """Create a config file and run the processing script."""
    txt_filename = f"{run_number}.txt"
    with open(txt_filename, "w") as f:
        f.write(hdf5_filename + "\n")
    print(f"Text file '{txt_filename}' created.")

    with open("config.json") as f:
        config = json.load(f)
    config["run"] = run_number
    with open("temp_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("Running 07_save_structured_from_config.py ...")
    subprocess.run(["python3", "07_save_structured_from_config.py", "--config", "temp_config.json"], check=True)
    print("Processing complete.")


# ------------------------------------------------------------------------------
# WAVEFORM ANALYSIS & PLOTTING
# ------------------------------------------------------------------------------

def print_waveform_timing_info(wfset):
    """Logs min/max timestamp and the approximate time delta."""
    timestamps = [wf.timestamp for wf in wfset.waveforms]
    if not timestamps:
        print("No waveforms found!")
        return
    a, b = np.min(timestamps), np.max(timestamps)
    print(f"Min timestamp: {a}, Max timestamp: {b}, Δt: {b - a} ticks")
    print(f"Δt in seconds: {(b - a) * 16e-9:.3e} s")
    print(f"Light travels: {(3e5)*(b - a)*16e-9:.2f} Km (approx)")


def analyze_waveforms(wfset, label="standard", starting_tick=50, width=70):
    """Performs a basic waveform analysis."""
    baseline_limits = [0, 50, 900, 1000]
    input_params = IPDict(
        baseline_limits=baseline_limits,
        int_ll=starting_tick,
        int_ul=starting_tick + width,
        amp_ll=starting_tick,
        amp_ul=starting_tick + width
    )
    checks_kwargs = dict(points_no=wfset.points_per_wf)
    print("Running waveform analysis...")
    wfset.analyse(
        label,
        BasicWfAna,
        input_params,
        *[],
        analysis_kwargs={},
        checks_kwargs=checks_kwargs,
        overwrite=True
    )


def create_channel_grids(wfset, bins=115, domain=(-10000., 50000.)):
    """Creates TCO and non-TCO ChannelWsGrid dictionaries from a WaveformSet."""
    return {
        "TCO": ChannelWsGrid(
            mem_geometry_map[2],
            wfset,
            compute_calib_histo=False,
            bins_number=bins,
            domain=np.array(domain),
            variable='integral',
            analysis_label=''
        ),
        "nTCO": ChannelWsGrid(
            mem_geometry_map[1],
            wfset,
            compute_calib_histo=False,
            bins_number=bins,
            domain=np.array(domain),
            variable='integral',
            analysis_label=''
        ),
    }


def plot_single_grid(grid, title="Grid Plot", save_path=None):
    """Plots a single ChannelWsGrid in overlay mode."""
    figure = psu.make_subplots(rows=4, cols=2)
    plot_ChannelWsGrid(
        figure=figure,
        channel_ws_grid=grid,
        share_x_scale=True,
        share_y_scale=True,
        mode='overlay',
        wfs_per_axes=50
    )
    figure.update_layout(
        title={'text': title, 'font': {'size': 24}},
        width=1000,
        height=800,
        template="plotly_white",
        showlegend=True
    )
    if save_path:
        figure.write_html(str(save_path))
        print(f"Saved: {save_path}")
    else:
        figure.show()


def plot_processed_file(path, max_waveforms=2000, label="standard"):
    """Load, analyze and plot waveform data from a structured HDF5 file."""
    print(f"Loading structured data from: {path}")
    wfset = load_structured_waveformset(path, max_waveforms=max_waveforms)
    print_waveform_timing_info(wfset)
    analyze_waveforms(wfset, label=label)
    grids = create_channel_grids(wfset)
    plot_single_grid(grids["TCO"], title="TCO")
    plot_single_grid(grids["nTCO"], title="nTCO")


# ------------------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------------------

def main():
    default_hostname = "np04-srv-004"
    hostname = input(f"Enter the hostname [default: {default_hostname}]: ").strip() or default_hostname
    port = 22
    username = input("Enter the username: ").strip()
    use_key = input("Are you using an SSH key (yes/no)? ").strip().lower() == "yes"
    private_key_path = None
    password = None

    if use_key:
        private_key_path = input("Enter the path to the private key: ").strip()
        if input("Does the key require a passphrase (yes/no)? ").strip().lower() == "yes":
            password = getpass.getpass("Enter the passphrase: ")
    else:
        password = getpass.getpass("Enter your password: ")

    run_number = int(input("Enter the run number: "))
    remote_path = "/data0"
    local_dir = "."

    try:
        ssh_client = connect_ssh(hostname, port, username, private_key_path, password)
        print("Connected to remote server.")
        files = list_files(ssh_client, remote_path, run_number)
        if not files:
            print("No files found.")
            return

        for i, f in enumerate(files):
            print(f"[{i}] {f}")

        selected_file = files[0] 
        downloaded = download_files(ssh_client, [selected_file], local_dir)
        ssh_client.close()

        run_str = f"{run_number:06d}" 
        update_config_and_run(run_str, downloaded[0])

        # Search for structured processed file and plot it
        files_in_dir = os.listdir(os.getcwd())
        structured_files = [f for f in files_in_dir if f.startswith(f"processed_np02vd_raw_run{run_str}_")]
        if not structured_files:
            print(f"⚠️ No processed structured file found for run {run_number}.")
            return
        path = os.path.join(os.getcwd(), structured_files[0])
        plot_processed_file(path)

    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()
