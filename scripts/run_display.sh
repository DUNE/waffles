#!/bin/bash

# RUN: sh scripts/run_display.sh
# OUTPUT: display app in your browser by following the generated link


current_dir=$(basename "$PWD") # Get the basename of the current directory
if [ "$current_dir" == "scripts" ]; then # Check if the current directory is 'scripts'
    cd ..
fi

pip install .
cd /afs/cern.ch/work/l/lperez/ProtoDUNE-HD/waffles/src/waffles/plotting/display

python3 - <<EOF
from np04_display import Display
Display().run_app()
EOF