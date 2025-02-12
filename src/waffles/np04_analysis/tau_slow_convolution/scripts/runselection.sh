#!/bin/bash

type=${1:-purity}
<<<<<<< HEAD
ot=${2:--r}
maxprocess=${3:-3}

if [ "$ot" != -t ]; then
    runlist=$( tail -n +2 runlists/${type}_runs.csv | cut -d , -f 1 )
else
    runlist=$( tail -n +2 runlists/${type}_runs.csv | cut -d , -f 5 | sort | uniq )
    ot=-t
fi

echo ${runlist[@]} | xargs -P${maxprocess} -n1 bash -c 'python extract_selection.py --runs "$0" '${ot}' -rl '${type}' -ch 11114 11116 -f'

echo ${runlist[@]} | xargs -P${maxprocess} -n1 bash -c 'python extract_selection.py --runs "$0" '${ot}' -rl '${type}' -ch 11225 11227 -f'
=======
maxprocess=${2:-3}

if [ "$ot" != -t ]; then
    runlist=$( tail -n +2 runlists/${type}_runs.csv | cut -d , -f 1 )
fi

echo ${runlist[@]} | xargs -P${maxprocess} -n1 bash -c 'python extract_selection.py --runs "$0" -rl '${type}' -ch 11114 -f'

echo ${runlist[@]} | xargs -P${maxprocess} -n1 bash -c 'python extract_selection.py --runs "$0" -rl '${type}' -ch 11225 11227 -f'
>>>>>>> main


wait && echo "All done..."
