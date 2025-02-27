#!/bin/bash

config=$1
id=${2:-'null'}
params=${*:3}

if [ "$id" == 'null' ]; then
    id=$(date '+%Y-%m-%d_%H-%M-%S')
fi

DEBUG=true python -m src.main +experiment=${config} hydra.run.dir=./outputs/${config}/${id} mode=train hydra.job.name=train wandb.activated=false checkpointing.save=false ${params}
