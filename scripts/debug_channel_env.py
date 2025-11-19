#!/usr/bin/env python3
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

def describe_module(name):
    try:
        module = importlib.import_module(name)
    except Exception as err:
        print(f"[X] {name}: failed to import ({err})")
        return None
    print(f"[✓] {name}: {module.__file__}")
    return module

def git_info(path):
    path = Path(path).resolve()
    while path != path.parent and not (path / ".git").exists():
        path = path.parent
    if not (path / ".git").exists():
        print("    (not inside a git checkout)")
        return
    try:
        desc = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        print(f"    git repo: {path}")
        print(f"    commit : {desc}")
    except subprocess.CalledProcessError:
        print("    git info unavailable")

print("Python executable:", sys.executable)
print()

mod_utils = describe_module("rawdatautils.unpack.utils")
if mod_utils:
    git_info(mod_utils.__file__)
    print("    frame_obj:", mod_utils.DAPHNEEthStreamUnpacker.frame_obj)
    print("    get_det_data_all first line:")
    print("       ", inspect.getsource(mod_utils.DAPHNEStreamUnpacker.get_det_data_all).splitlines()[0])
    print()

mod_reader = describe_module("waffles.input_output.daphne_eth_reader")
if mod_reader:
    git_info(mod_reader.__file__)
    print("    load_daphne_eth_waveforms snippet:")
    snippet = inspect.getsource(mod_reader.load_daphne_eth_waveforms).splitlines()[160:210]
    for line in snippet:
        print("       ", line)
