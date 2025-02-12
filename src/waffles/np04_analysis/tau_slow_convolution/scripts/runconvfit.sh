#!/bin/bash

type=${1:-purity}
maxprocess=${2:-4}

runlist=$( tail -n +2 runlists/${type}_runs.csv | cut -d , -f 1 )

channels=( 11225 11227 11114 )
# channels=( 11114 )
for ch in ${channels[@]}; do
    echo "channel ${ch}"
    # echo ${runlist[@]} | xargs -P${maxprocess} -n1 bash -c 'python convfit.py --runs "$0" -ch '${ch}' -rl '${type}
    # echo ${runlist[@]} | xargs -P${maxprocess} -n1 bash -c 'python convfit.py --runs "$0" -ch '${ch}' -rl '${type}' -ft -tt 26261'
    echo ${runlist[@]} | xargs -P${maxprocess} -n1 bash -c 'python convfit.py --runs "$0" -ch '${ch}' -rl '${type}' -ft -tt 29177 -ns new_template_inter -i --scan 6'
    echo ${runlist[@]} | xargs -P${maxprocess} -n1 bash -c 'python convfit.py --runs "$0" -ch '${ch}' -rl '${type}' -ft -tt 0 -ns ultimate_inter -i  --scan 6'

done
