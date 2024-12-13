#!/bin/bash

USER=$( whoami )

. /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh
spack load kx509
kx509
voms-proxy-init -rfc -noregen -voms=dune:/dune/Role=Analysis -valid 120:00

export PS2='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u\[\033[00m\]@\[\033[01;31m\]\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer shell --shell=/bin/bash -B /cvmfs,/opt,/run/user,/etc/hostname,/etc/hosts,/etc/krb5.conf,/eos/user/h/hvieirad --env "PS1=$PS2" --env "PATH"=$PATH --env "LD_LIBRARY_PATH"=$LD_LIBRARY_PATH --ipc --pid /cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest




