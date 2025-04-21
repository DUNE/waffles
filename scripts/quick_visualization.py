import os
import getpass
import json
import paramiko
import subprocess

import waffles

import numpy as np
import plotly.graph_objects as pgo
import pickle
from waffles.plotting.plot import plot_ChannelWsGrid
import plotly.subplots as psu
import gc
import h5py
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.IPDict import IPDict
from waffles.input_output.hdf5_structured import load_structured_waveformset


# from waffles.data_classes.WafflesAnalysis import WafflesAnalysis, BaseInputParams
# from waffles.data_classes.WaveformAdcs import WaveformAdcs
# from waffles.data_classes.Waveform import Waveform
# from waffles.data_classes.WaveformSet import WaveformSet
import waffles.np02_data.ProtoDUNE_VD_maps 
import waffles.np04_data.ProtoDUNE_HD_APA_maps 
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
# from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_map, cat_geometry_map
# from waffles.input_output.pickle_hdf5_reader import WaveformSet_from_hdf5_pickle


from pathlib import Path
import plotly.subplots as psu
from waffles.np02_data.ProtoDUNE_VD_maps import mem_geometry_tco, mem_geometry_nontco
from waffles.data_classes.ChannelWsGrid import ChannelWsGrid
from waffles.data_classes.BasicWfAna import BasicWfAna
from waffles.data_classes.IPDict import IPDict
from waffles.plotting.plot import plot_ChannelWsGrid

# ------------------------------------------------------------------------------
# SINGLE-RUN UTILS
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
            mem_geometry_tco,
            wfset,
            compute_calib_histo=False,
            bins_number=bins,
            domain=np.array(domain),
            variable='integral',
            analysis_label=''
        ),
        "nTCO": ChannelWsGrid(
            mem_geometry_nontco,
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

def plot_single_run(filepath, max_waveforms=2000, label="standard"):
    """
    Loads, analyzes, and plots TCO & nTCO from a single structured HDF5 file.
    Modify if you prefer 'average' or 'heatmap' instead of 'overlay'.
    """
    print(f"Loading single run from: {filepath}")
    wfset = load_structured_waveformset(filepath, max_waveforms=max_waveforms)
    print_waveform_timing_info(wfset)
    analyze_waveforms(wfset, label=label)
    grids = create_channel_grids(wfset)

    # Plot TCO
    plot_single_grid(grids["TCO"], title=f" TCO")
    # Plot nTCO
    plot_single_grid(grids["nTCO"], title=f" nTCO")


def connect_ssh(hostname, port, username, private_key_path=None, password=None):
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
    cmd = f"ls {remote_path}/np02vd_raw_run{run_number:06d}_*.hdf5"
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    files = stdout.read().decode('utf-8').splitlines()
    if stderr.read().decode('utf-8'):
        raise RuntimeError("Error listing files on remote server")
    return files


def download_files(ssh_client, remote_files, local_dir):
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

def inspect_hdf5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n📂 Inspeccionando archivo: {file_path}\n")

            found_wfset = False

            def visit_func(name, obj):
                nonlocal found_wfset
                print(f"{name}: {type(obj)}")
                if isinstance(obj, h5py.Dataset):
                    print(f"  📏 Shape: {obj.shape}, 🧬 Dtype: {obj.dtype}")
                if name.endswith('wfset'):
                    found_wfset = True

            f.visititems(visit_func)

            if found_wfset:
                print("\n✅ Dataset 'wfset' encontrado en el archivo.")
            else:
                print("\n⚠️ No se encontró ningún dataset llamado 'wfset'.")

    except Exception as e:
        print(f"❌ Error al abrir el archivo HDF5: {e}")
        
def update_config_and_run(run_number, hdf5_filename):
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

    # === Analyse and Plot ===
    files_in_dir = os.listdir(os.getcwd())
    structured_files = [f for f in files_in_dir if f.startswith(f"processed_np02vd_raw_run{run_number}_")]
    if not structured_files:
        print(f"⚠️ No processed structured file found for run {run_number}.")
        return
    path = os.path.join(os.getcwd(), structured_files[0])
    plot_single_run(path)
    # print('path:',path)
    # #inspect_hdf5_file(path)
    # # print(wfset)
    # print(f'loading new wfset')
    # wfset = load_structured_waveformset(path)
    # print(f'new waveforset loaded')
    # print(f'wfset object type:{type(wfset)}')
    # print(f'wfset.waveforms[0] object type:{type(wfset.waveforms[0])}')
    
    # a = np.min([wf.timestamp for wf in wfset.waveforms])
    # print(f'min ts = {a}')
    # endpoints = list({ep for r in wfset.available_channels for ep in wfset.available_channels[r]})
    # print(f'available endpoints {endpoints}')

    # def allow_endpoints(waveform : waffles.Waveform, my_endpoints: list) -> bool:
    #     if int(waveform.timestamp-a)<int(3e9) :
    #         if waveform.endpoint in my_endpoints :
    #             return True
    #     else:
    #         return False

    # wfset = WaveformSet.from_filtered_WaveformSet(wfset, allow_endpoints, endpoints)
    # if not wfset.waveforms:
    #     print("⚠️ No waveforms after filtering.")
    #     return
    
    # print (f'waveforms saved: {len(wfset.waveforms)}')

    # ipdict = IPDict(baseline_limits=[0, 50, 900, 1000], int_ll=50, int_ul=120, amp_ll=50, amp_ul=120)
    # wfset.analyse("standard", BasicWfAna, ipdict, analysis_kwargs={"int_ul": wfset.points_per_wf - 1, 
    #                 "prominence": 50, "rel_height": 0.5, "width": [0, 75], "return_peaks_properties": True},
    #               checks_kwargs={"points_no": wfset.points_per_wf})

    # grid = ChannelWsGrid(np02_data.ProtoDUNE_VD_maps.mem_geometry_map[1], wfset, compute_calib_histo=False,
    #                      bins_number=115, domain=np.array((-10000., 50000.)), variable='integral', analysis_label='standard')

    # fig = psu.make_subplots(rows=4, cols=2)
    # plot_ChannelWsGrid(figure=fig, channel_ws_grid=grid, share_x_scale=True, share_y_scale=True, mode='overlay', wfs_per_axes=50)
    # fig.update_layout(title="Waveforms from membrane non-TCO", width=1100, height=1200, template="ggplot2", showlegend=True)
    # html_output = f"wfs_nontco_{run_number}.html"
    # pio.write_html(fig, file=html_output, auto_open=True)
    # print(f"📊 HTML visualization saved and opened: {html_output}")


def main():
    hostname = input("Enter the hostname (e.g., np04-srv-004): ").strip()
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

        selected_file = files[0]  # take first
        downloaded = download_files(ssh_client, [selected_file], local_dir)
        ssh_client.close()

        update_config_and_run(f"{run_number:06d}", downloaded[0])

    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()