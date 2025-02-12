#!/bin/bash

type=${1:-purity}
maxprocess=${2:-3}

if [ "$ot" != -t ]; then
    runlist=$( tail -n +2 runlists/${type}_runs.csv | cut -d , -f 1 )
fi

echo ${runlist[@]} | xargs -P${maxprocess} -n1 bash -c 'python extract_selection.py --runs "$0" -rl '${type}' -ch 11114 -f'

echo ${runlist[@]} | xargs -P${maxprocess} -n1 bash -c 'python extract_selection.py --runs "$0" -rl '${type}' -ch 11225 11227 -f'


wait && echo "All done..."
