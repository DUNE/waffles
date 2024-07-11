# WAFFLES
PDS Working group tool intended for **Waveform Analysis Framework For Light Emission Studies (WAFFLES)**

Check our [documentation](https://waffles.readthedocs.io/en/latest/index.html) to learn all the details on how to use, contribute and develop with WAFFLES. [![Documentation Status](https://readthedocs.org/projects/waffles/badge/?version=latest)](https://waffles.readthedocs.io/en/latest/?badge=latest)


## Quick Start

Clone the repository and install the dependencies:

```bash
git clone https://github.com/DUNE/waffles.git 
cd waffles
git checkout -b <your_branch_name>
```

Create a virtual environment and install the dependencies:

```bash
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
pip install -r docs/requirements.txt
```

Add the `ROOT` library to your environment by adding these lines at the end of the `bin/activate` file:

```bash
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.32.02/x86_64-almalinux9.4-gcc114-opt/bin/thisroot.sh
export JUPYTER_CONFIG_DIR=$VIRTUAL_ENV
```

To deactivate the environment just run `deactivate` in your terminal.

## Examples

Check `/eos/experiment/neutplatform/protodune/experiments/ProtoDUNE-II/PDS_Commissioning/waffles/0_TUTORIAL` to take some different types of runs for testing the library.

For more details see the notebooks in the [examples](https://waffles.readthedocs.io/en/latest/examples/4_Examples.html) section of the documentation.